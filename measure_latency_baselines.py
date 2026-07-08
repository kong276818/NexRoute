"""
Latency Benchmark — Naive RAG 7B & Adaptive RAG 7B
eval_baselines_500.py와 완전히 동일한 구조 사용
(단일 flat FAISS, max_new_tokens=300, temperature=0.7)

측정 항목:
  - Routing & Retrieval (ms): 라우팅 판단 + FAISS 검색
  - TTFT (ms): 첫 토큰 도달 시간 (라우팅+검색+생성 첫 토큰 포함)
  - Throughput (tok/s): 생성 토큰 수 / 생성 소요 시간

출력: latency_baseline_results.json
"""
import json, os, time, datetime, torch, faiss, numpy as np, re
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TextIteratorStreamer)
from sentence_transformers import SentenceTransformer
import threading

BASE_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
EMBED_MODEL = "intfloat/multilingual-e5-large"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_PATH    = "/home/sclab/moe_model/latency_baseline_results.json"

KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':  ['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':  ['social','society','culture','economic','political','community','psychology','policy'],
    'STEM': ['science','physics','chemical','biology','math','engineering','technology','energy'],
}
SYSTEMS = {
    'LAW':  "You are a legal expert.",
    'MED':  "You are a medical expert.",
    'SOC':  "You are a social science expert.",
    'STEM': "You are a science expert.",
}

def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

# ── 모델 로드 ────────────────────────────────────────────────────────────
print("[1/4] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_MODEL)
print("[1/4] 완료")

print("[2/4] 단일 통합 FAISS 인덱스 구축 (eval_baselines_500.py 동일)...")
files = {"LAW":"law.json","MED":"med.json","SOC":"soc.json",
         "STEM":"stem.json","CROSS":"cross_rewritten.json"}
all_texts = []
for domain, fname in files.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    for d in data:
        out = d.get("output", d.get("gold_answer",""))
        if any(k in out for k in ["The correct answer","All answers:"]):
            continue
        all_texts.append(f"{d['instruction']} {out}")
    print(f"  {domain}: 로드 완료")

embs = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
flat_index = faiss.IndexFlatIP(embs.shape[1])
flat_index.add(embs.astype(np.float32))
print(f"[2/4] FAISS 인덱스 완료: {len(all_texts)}개")

print("[3/4] Qwen2.5-7B 로드...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True,
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager"
)
model.eval()
print("[3/4] 모델 로드 완료")

print("[4/4] GT 데이터 로드...")
with open(GT_PATH) as f:
    gt_data = json.load(f)
print(f"[4/4] {len(gt_data)}개 쿼리 로드")

def retrieve_flat(query, top_k=10):
    emb = embed_model.encode([query], show_progress_bar=False)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    _, I = flat_index.search(emb.astype(np.float32), top_k)
    return [all_texts[i][:200] for i in I[0]]

def measure_one(query, routing):
    """
    routing: "none"    → Naive RAG
             "keyword" → Adaptive RAG (keyword routing)
    반환: (rr_ms, ttft_ms, throughput_tps)
    """
    t0 = time.perf_counter()

    # ── Routing & Retrieval ───────────────────────────────────
    if routing == "keyword":
        domain = keyword_route(query)
        system = SYSTEMS.get(domain, "You are a helpful expert.")
    else:
        system = "You are a helpful expert."

    context = retrieve_flat(query)
    torch.cuda.synchronize()
    t_rr_end = time.perf_counter()
    rr_ms = (t_rr_end - t0) * 1000

    # ── 프롬프트 구성 (eval_baselines_500.py 동일) ───────────────
    ctx_text = "\n".join(context)
    prompt = (
        f"{system}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form.\n\n"
        f"Answer:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)

    # ── TTFT via TextIteratorStreamer ────────────────────────────
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                   skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=300, temperature=0.7,
                      do_sample=True, pad_token_id=tokenizer.eos_token_id,
                      streamer=streamer)

    t_query = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    ttft_ms = None
    tokens = []
    t_gen_start = None

    for tok in streamer:
        if ttft_ms is None:
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t_query) * 1000
            t_gen_start = time.perf_counter()
        tokens.append(tok)

    thread.join()
    torch.cuda.synchronize()
    t_gen_end = time.perf_counter()

    answer = "".join(tokens).strip()
    n_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
    gen_duration = t_gen_end - t_gen_start if t_gen_start else 1e-9
    throughput = n_tokens / gen_duration if gen_duration > 0 else 0

    return rr_ms, ttft_ms or 0.0, throughput


def benchmark(routing, name, n_warmup=3, n_samples=500):
    print(f"\n{'='*60}")
    print(f"측정: {name}  (routing={routing})")
    print(f"{'='*60}")

    queries = [d.get("instruction","") for d in gt_data[:n_samples]]

    # 워밍업
    print(f"워밍업 ({n_warmup}개)...")
    for i in range(n_warmup):
        measure_one(queries[i], routing)
        print(f"  warmup [{i+1}/{n_warmup}] 완료")

    # 본 측정
    print(f"본 측정 ({n_samples}개)...")
    rr_list, ttft_list, tps_list = [], [], []
    for i, q in enumerate(queries):
        rr, ttft, tps = measure_one(q, routing)
        rr_list.append(rr); ttft_list.append(ttft); tps_list.append(tps)
        if (i+1) % 50 == 0:
            print(f"  [{i+1:>3}/{n_samples}]  RR={np.mean(rr_list):.1f}ms"
                  f"  TTFT={np.mean(ttft_list):.1f}ms"
                  f"  TPS={np.mean(tps_list):.1f}tok/s")

    avg_rr   = round(float(np.mean(rr_list)),  1)
    avg_ttft = round(float(np.mean(ttft_list)), 1)
    avg_tps  = round(float(np.mean(tps_list)),  1)
    std_rr   = round(float(np.std(rr_list)),   1)
    std_ttft = round(float(np.std(ttft_list)), 1)
    std_tps  = round(float(np.std(tps_list)),  1)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Routing & Retrieval: {avg_rr} ± {std_rr} ms")
    print(f"  TTFT:                {avg_ttft} ± {std_ttft} ms")
    print(f"  Throughput:          {avg_tps} ± {std_tps} tok/s")
    print(f"{'='*60}")

    return {
        "name": name, "routing": routing, "n": n_samples,
        "routing_retrieval_ms": {"avg": avg_rr, "std": std_rr},
        "ttft_ms":              {"avg": avg_ttft, "std": std_ttft},
        "throughput_toks_per_sec": {"avg": avg_tps, "std": std_tps},
    }


results = {}
results["Naive RAG 7B"]    = benchmark("none",    "Naive RAG 7B")
results["Adaptive RAG 7B"] = benchmark("keyword", "Adaptive RAG 7B")

results["_meta"] = {
    "script":    "measure_latency_baselines.py",
    "timestamp": datetime.datetime.now().isoformat(),
    "model":     BASE_7B,
    "embed":     EMBED_MODEL,
    "n_samples": 500,
    "note":      "NexRoute 7B은 latency_full_results.json 참조 (RR=32.9ms, TTFT=418.1ms, TPS=12.5)",
}

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ 저장: {OUT_PATH}")

print("\n\n=== 논문 Table latency 값 ===")
nexroute = {"rr": 32.9, "ttft": 418.1, "tps": 12.5}
for name, r in results.items():
    if name == "_meta": continue
    print(f"{name:<20}  RR={r['routing_retrieval_ms']['avg']}ms"
          f"  TTFT={r['ttft_ms']['avg']}ms"
          f"  TPS={r['throughput_toks_per_sec']['avg']}tok/s")
print(f"{'NexRoute 7B':<20}  RR={nexroute['rr']}ms"
      f"  TTFT={nexroute['ttft']}ms"
      f"  TPS={nexroute['tps']}tok/s")
