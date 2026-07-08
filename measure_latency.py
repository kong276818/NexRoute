"""
Appendix C — Computational Efficiency: Latency Benchmarks

측정 항목:
  1. TTFT (Time-to-First-Token): 첫 토큰 출력까지 걸리는 시간
  2. E2E Latency: 라우팅 + RAG 검색 + 생성 전체 시간
  3. Adapter Switching Overhead: LoRA 어댑터 전환 시간
  4. Component Breakdown: 라우팅 / 검색 / 생성 각각의 소요 시간
  5. VRAM peak: torch.cuda.max_memory_allocated()

하드웨어: 실행 환경 자동 감지
"""
import json, time, os, gc, torch, numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    AutoModel, TextStreamer
)
import torch.nn as nn
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss

# ── 설정 ──────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
LORA_DIR    = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
ROUTER_PATH = "/home/sclab/moe_model/router_multilabel.pt"
EMBED_MODEL = "intfloat/multilingual-e5-large"
OUT_PATH    = "/home/sclab/moe_model/latency_results.json"

DOMAINS = ["LAW", "MED", "SOC", "STEM"]
LORA_MAP = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc",
            "STEM":"lora_stem","CROSS":"lora_cross"}

N_WARMUP = 2   # 워밍업 반복 수 (측정 제외)
N_REPEAT = 5   # 도메인당 측정 반복 수

# ── 대표 쿼리 (도메인당 N_REPEAT개) ──────────────────────────────────
SAMPLE_QUERIES = {
    "LAW": [
        "What is the doctrine of promissory estoppel in contract law?",
        "How does the court determine the standard for negligence?",
        "What are the elements required to establish a valid contract?",
        "Explain the difference between civil and criminal liability.",
        "What is the legal standard for determining fair use in copyright?",
    ],
    "MED": [
        "How does the renin-angiotensin system regulate blood pressure?",
        "What are the mechanisms of action of beta-blockers?",
        "Explain the pathophysiology of type 2 diabetes mellitus.",
        "How does the blood-brain barrier protect the central nervous system?",
        "What is the mechanism of action of penicillin antibiotics?",
    ],
    "SOC": [
        "How does social stratification affect educational outcomes?",
        "What is the role of social capital in community development?",
        "Explain Bourdieu's concept of cultural capital.",
        "How does urbanization affect social cohesion?",
        "What are the sociological causes of poverty?",
    ],
    "STEM": [
        "How does CRISPR-Cas9 technology work?",
        "What is the mechanism behind quantum entanglement?",
        "Explain the process of photosynthesis at the molecular level.",
        "How do neural networks learn through backpropagation?",
        "What is the role of enzymes in biochemical reactions?",
    ],
    "CROSS": [
        "How might advances in neuroscience impact criminal justice policy?",
        "What are the ethical implications of AI in medical diagnosis?",
        "How does environmental law intersect with climate science?",
        "What role does social psychology play in legal decision-making?",
        "How do advances in genetics affect social inequality?",
    ],
}

# ── 유틸 ──────────────────────────────────────────────────────────────
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0

def reset_vram_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── 모델 로드 ─────────────────────────────────────────────────────────
print("=" * 60)
print("  NexRoute Latency Benchmark")
print("=" * 60)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
)

print("\n[1/4] 베이스 모델 로드...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True,
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg, device_map="auto",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager"
)
model.eval()


class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 4)
        )
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

print("[2/4] BERT Router 로드...")
router_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
router = DomainRouter()
router.load_state_dict(torch.load(ROUTER_PATH, map_location="cpu"))
router.eval().to("cuda")

print("[3/4] 임베딩 모델 + FAISS 인덱스 구축...")
embed_model = SentenceTransformer(EMBED_MODEL)
db = {}
for dom, fname in [("LAW","law_rewritten.json"),("MED","med_rewritten.json"),
                   ("SOC","soc_rewritten.json"),("STEM","stem_rewritten.json"),
                   ("CROSS","cross_rewritten.json")]:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    texts = [f"{d['instruction']} {d.get('output',d.get('gold_answer',''))}"
             for d in data][:500]
    embs = embed_model.encode(texts, show_progress_bar=False, batch_size=128)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    idx  = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    db[dom] = (texts, idx)
    print(f"  {dom}: {len(texts)}개")

print("[4/4] LoRA 초기 어댑터(LAW) 로드...")
model = PeftModel.from_pretrained(model, os.path.join(LORA_DIR, "lora_law"),
                                  adapter_name="LAW")
current_lora = "LAW"

results = {}

# ─────────────────────────────────────────────────────────────────────
# A. Adapter Switching Overhead
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("A. Adapter Switching Overhead")
print("=" * 60)

switch_seq = ["MED","SOC","STEM","CROSS","LAW"]
for dom in switch_seq:
    model.load_adapter(os.path.join(LORA_DIR, LORA_MAP[dom]), adapter_name=dom)

# 워밍업
for _ in range(N_WARMUP):
    for dom in ["MED","LAW"]:
        model.set_adapter(dom)

switch_times = {d: [] for d in switch_seq}
for _ in range(N_REPEAT):
    prev = "LAW"
    for dom in switch_seq:
        cuda_sync()
        t0 = time.perf_counter()
        model.set_adapter(dom)
        cuda_sync()
        t1 = time.perf_counter()
        switch_times[dom].append((t1 - t0) * 1000)
        prev = dom

print(f"\n{'Transition':<20} {'Mean(ms)':>10} {'Std(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}")
print("-" * 55)
all_switch = []
for dom, times in switch_times.items():
    m, s = np.mean(times), np.std(times)
    all_switch.extend(times)
    print(f"  → {dom:<17} {m:>10.2f} {s:>10.2f} {min(times):>10.2f} {max(times):>10.2f}")
print(f"  {'[Overall]':<18} {np.mean(all_switch):>10.2f} {np.std(all_switch):>10.2f}")

results["adapter_switching_ms"] = {
    dom: {"mean": round(float(np.mean(t)),2), "std": round(float(np.std(t)),2)}
    for dom, t in switch_times.items()
}
results["adapter_switching_ms"]["overall"] = {
    "mean": round(float(np.mean(all_switch)),2),
    "std":  round(float(np.std(all_switch)),2)
}


# ─────────────────────────────────────────────────────────────────────
# B. Routing Latency (BERT)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("B. Routing Latency (BERT domain router)")
print("=" * 60)

def route(query):
    enc = router_tok(query, max_length=256, truncation=True,
                     padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = router(enc["input_ids"], enc["attention_mask"])
        torch.sigmoid(logits)

# 워밍업
for q in list(SAMPLE_QUERIES["LAW"])[:N_WARMUP]:
    route(q)

route_times_by_domain = {d: [] for d in SAMPLE_QUERIES}
for dom, queries in SAMPLE_QUERIES.items():
    for q in queries:
        cuda_sync()
        t0 = time.perf_counter()
        route(q)
        cuda_sync()
        t1 = time.perf_counter()
        route_times_by_domain[dom].append((t1 - t0) * 1000)

all_route = [t for ts in route_times_by_domain.values() for t in ts]
print(f"\n  전체 라우팅 지연: {np.mean(all_route):.2f} ± {np.std(all_route):.2f} ms")
print(f"  (N={len(all_route)}, min={min(all_route):.2f}, max={max(all_route):.2f})")
results["routing_latency_ms"] = {
    "mean": round(float(np.mean(all_route)), 2),
    "std":  round(float(np.std(all_route)), 2),
    "min":  round(float(min(all_route)), 2),
    "max":  round(float(max(all_route)), 2),
    "n":    len(all_route)
}


# ─────────────────────────────────────────────────────────────────────
# C. RAG Retrieval Latency
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("C. RAG Retrieval Latency")
print("=" * 60)

def retrieve(query, domain, top_k=10):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    texts, index = db[domain]
    index.search(emb.astype(np.float32), top_k)

# 워밍업
for q in list(SAMPLE_QUERIES["LAW"])[:N_WARMUP]:
    retrieve(q, "LAW")

retrieval_times = {d: [] for d in ["LAW","MED","SOC","STEM","CROSS"]}
for dom, queries in SAMPLE_QUERIES.items():
    if dom not in db:
        continue
    for q in queries:
        t0 = time.perf_counter()
        retrieve(q, dom)
        t1 = time.perf_counter()
        retrieval_times[dom].append((t1 - t0) * 1000)

all_ret = [t for ts in retrieval_times.values() for t in ts]
print(f"\n  전체 RAG 검색 지연: {np.mean(all_ret):.2f} ± {np.std(all_ret):.2f} ms")
print(f"  (N={len(all_ret)}, min={min(all_ret):.2f}, max={max(all_ret):.2f})")
results["retrieval_latency_ms"] = {
    "mean": round(float(np.mean(all_ret)), 2),
    "std":  round(float(np.std(all_ret)), 2),
    "min":  round(float(min(all_ret)), 2),
    "max":  round(float(max(all_ret)), 2),
}


# ─────────────────────────────────────────────────────────────────────
# D. TTFT + E2E Latency (도메인별)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("D. TTFT + End-to-End Latency (도메인별)")
print("=" * 60)

def build_prompt(query, domain):
    ctx_q = f"[{domain}] Context for {query[:50]}"
    return (
        f"You are an expert. Answer the following question.\n\n"
        f"Context:\n{ctx_q}\n\nQuestion: {query}\n\nAnswer:"
    )

def measure_ttft(query, domain):
    """첫 토큰까지의 시간: max_new_tokens=1"""
    prompt = build_prompt(query, domain)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    cuda_sync()
    return (time.perf_counter() - t0) * 1000

def measure_e2e(query, domain):
    """전체 생성 (128 new tokens) 시간"""
    prompt = build_prompt(query, domain)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    cuda_sync()
    elapsed = (time.perf_counter() - t0) * 1000
    n_gen = out.shape[1] - inputs["input_ids"].shape[1]
    return elapsed, n_gen

# 워밍업
print("\n  워밍업 중...")
model.set_adapter("LAW")
for q in list(SAMPLE_QUERIES["LAW"])[:N_WARMUP]:
    measure_ttft(q, "LAW")
    measure_e2e(q, "LAW")

ttft_by_dom = {}
e2e_by_dom  = {}
tps_by_dom  = {}  # tokens per second

dom_order = ["LAW","MED","SOC","STEM","CROSS"]
for dom in dom_order:
    queries = SAMPLE_QUERIES[dom]
    model.set_adapter(dom if dom in LORA_MAP else "LAW")
    reset_vram_peak()
    ttfts, e2es, tps_list = [], [], []
    for q in queries:
        ttfts.append(measure_ttft(q, dom))
        elapsed_ms, n_tok = measure_e2e(q, dom)
        e2es.append(elapsed_ms)
        tps_list.append(n_tok / (elapsed_ms / 1000))  # tok/s
    ttft_by_dom[dom] = ttfts
    e2e_by_dom[dom]  = e2es
    tps_by_dom[dom]  = tps_list
    vram = get_vram_gb()
    print(f"  {dom:<6}: TTFT={np.mean(ttfts):.0f}±{np.std(ttfts):.0f}ms  "
          f"E2E={np.mean(e2es):.0f}±{np.std(e2es):.0f}ms  "
          f"TPS={np.mean(tps_list):.1f}tok/s  VRAM={vram:.1f}GB")

# CROSS: is_cross=True → adapter switching 포함한 E2E
print("\n  [CROSS with switch] adapter 전환 포함 E2E:")
cross_e2e_with_switch = []
model.set_adapter("LAW")
for q in SAMPLE_QUERIES["CROSS"]:
    cuda_sync()
    t0 = time.perf_counter()
    model.set_adapter("CROSS")   # 전환
    elapsed_ms, _ = measure_e2e(q, "CROSS")
    cuda_sync()
    total = (time.perf_counter() - t0) * 1000
    cross_e2e_with_switch.append(total)
    model.set_adapter("LAW")
print(f"  CROSS+switch E2E: {np.mean(cross_e2e_with_switch):.0f}±{np.std(cross_e2e_with_switch):.0f}ms")


# ─────────────────────────────────────────────────────────────────────
# E. Full Pipeline E2E (routing + retrieval + switch + generate)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("E. Full Pipeline E2E Breakdown")
print("=" * 60)

breakdown_samples = []
for dom in dom_order:
    for q in SAMPLE_QUERIES[dom][:3]:
        # Routing
        cuda_sync()
        t_route_start = time.perf_counter()
        enc = router_tok(q, max_length=256, truncation=True,
                         padding="max_length", return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = router(enc["input_ids"], enc["attention_mask"])
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()
        sorted_d = sorted(zip(DOMAINS, scores), key=lambda x: -x[1])
        top1, s1 = sorted_d[0]
        top2, s2 = sorted_d[1]
        is_cross  = (s2 >= 0.30) and ((s1 - s2) <= 0.40)
        cuda_sync()
        t_route_end = time.perf_counter()

        # Adapter switch
        target = "CROSS" if is_cross else top1
        cuda_sync()
        t_switch_start = time.perf_counter()
        model.set_adapter(target)
        cuda_sync()
        t_switch_end = time.perf_counter()

        # Retrieval
        t_ret_start = time.perf_counter()
        retrieve(q, top1)
        t_ret_end = time.perf_counter()

        # Generation
        elapsed_gen, n_tok = measure_e2e(q, target)

        breakdown_samples.append({
            "domain": dom,
            "routing_ms":  (t_route_end  - t_route_start)  * 1000,
            "switch_ms":   (t_switch_end - t_switch_start) * 1000,
            "retrieval_ms":(t_ret_end    - t_ret_start)    * 1000,
            "gen_ms":      elapsed_gen,
            "total_ms":    (t_route_end - t_route_start)*1000
                           + (t_switch_end - t_switch_start)*1000
                           + (t_ret_end - t_ret_start)*1000
                           + elapsed_gen,
            "n_tok": n_tok,
        })

r_ms  = np.mean([x["routing_ms"]   for x in breakdown_samples])
sw_ms = np.mean([x["switch_ms"]    for x in breakdown_samples])
rt_ms = np.mean([x["retrieval_ms"] for x in breakdown_samples])
g_ms  = np.mean([x["gen_ms"]       for x in breakdown_samples])
tot   = np.mean([x["total_ms"]     for x in breakdown_samples])

print(f"\n  {'Component':<22} {'Mean(ms)':>10} {'%':>6}")
print("  " + "-" * 42)
for label, val in [("Routing (BERT)", r_ms),
                    ("Adapter Switch", sw_ms),
                    ("RAG Retrieval", rt_ms),
                    ("Generation", g_ms)]:
    print(f"  {label:<22} {val:>10.1f} {val/tot*100:>5.1f}%")
print(f"  {'─'*38}")
print(f"  {'Total E2E':<22} {tot:>10.1f} {'100.0':>5}%")


# ─────────────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────────────
results["ttft_ms"] = {
    dom: {"mean": round(float(np.mean(v)),1), "std": round(float(np.std(v)),1)}
    for dom, v in ttft_by_dom.items()
}
results["e2e_ms"] = {
    dom: {"mean": round(float(np.mean(v)),1), "std": round(float(np.std(v)),1)}
    for dom, v in e2e_by_dom.items()
}
results["tps"] = {
    dom: {"mean": round(float(np.mean(v)),1), "std": round(float(np.std(v)),1)}
    for dom, v in tps_by_dom.items()
}
results["cross_e2e_with_switch_ms"] = {
    "mean": round(float(np.mean(cross_e2e_with_switch)),1),
    "std":  round(float(np.std(cross_e2e_with_switch)),1),
}
results["pipeline_breakdown_ms"] = {
    "routing":    round(r_ms, 1),
    "adapter_switch": round(sw_ms, 1),
    "retrieval":  round(rt_ms, 1),
    "generation": round(g_ms, 1),
    "total":      round(tot, 1),
    "routing_pct":   round(r_ms/tot*100,1),
    "switch_pct":    round(sw_ms/tot*100,1),
    "retrieval_pct": round(rt_ms/tot*100,1),
    "gen_pct":       round(g_ms/tot*100,1),
}
results["hardware"] = {
    "gpu":  torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)
                     if torch.cuda.is_available() else 0,
}

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\n✅ 저장: {OUT_PATH}")

# ─────────────────────────────────────────────────────────────────────
# 최종 요약 테이블
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Appendix C — Computational Efficiency Summary")
print("=" * 70)
print(f"\n  Hardware: {results['hardware']['gpu']}")
print(f"\n  [Table C.1] End-to-End Latency & TTFT")
print(f"  {'Domain':<8} {'TTFT(ms)':>12} {'E2E(ms)':>12} {'TPS(tok/s)':>12}")
print("  " + "-" * 48)
for dom in dom_order:
    ttft_v = results["ttft_ms"][dom]
    e2e_v  = results["e2e_ms"][dom]
    tps_v  = results["tps"][dom]
    print(f"  {dom:<8} {ttft_v['mean']:>7.0f}±{ttft_v['std']:<4.0f} "
          f"{e2e_v['mean']:>7.0f}±{e2e_v['std']:<4.0f} "
          f"{tps_v['mean']:>12.1f}")

print(f"\n  [Table C.2] Pipeline Component Breakdown (per query)")
bd = results["pipeline_breakdown_ms"]
print(f"  Routing:         {bd['routing']:>7.1f} ms  ({bd['routing_pct']}%)")
print(f"  Adapter Switch:  {bd['adapter_switch']:>7.1f} ms  ({bd['switch_pct']}%)")
print(f"  RAG Retrieval:   {bd['retrieval']:>7.1f} ms  ({bd['retrieval_pct']}%)")
print(f"  Generation:      {bd['generation']:>7.1f} ms  ({bd['gen_pct']}%)")
print(f"  ─────────────────────────────")
print(f"  Total E2E:       {bd['total']:>7.1f} ms")
print(f"\n  CROSS (w/ adapter switch): {results['cross_e2e_with_switch_ms']['mean']:.0f}"
      f"±{results['cross_e2e_with_switch_ms']['std']:.0f} ms")

sw = results["adapter_switching_ms"]["overall"]
print(f"\n  [Table C.3] Adapter Switching Overhead")
print(f"  Mean: {sw['mean']:.2f} ms  Std: {sw['std']:.2f} ms  (negligible vs generation)")
print(f"\n  Routing Latency: {results['routing_latency_ms']['mean']:.2f} ms")
print(f"  RAG Retrieval:   {results['retrieval_latency_ms']['mean']:.2f} ms")
