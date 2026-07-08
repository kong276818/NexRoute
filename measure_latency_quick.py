"""
Latency Quick Benchmark — Naive RAG 7B / Adaptive RAG 7B / NexRoute 7B
N=100, warmup=3
"""
import json, os, time, datetime, torch, faiss, numpy as np, threading
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TextIteratorStreamer)
from sentence_transformers import SentenceTransformer
from pipeline import DualMoEPipeline, FEW_SHOT, DOMAIN_SYSTEM
from peft import PeftModel

BASE_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
EMBED_MODEL = "intfloat/multilingual-e5-large"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
LORA_DIR    = "/home/sclab/moe_model/lora_weights_7b_v3"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_PATH    = "/home/sclab/moe_model/latency_quick_results.json"
N           = 100
N_WARMUP    = 3

KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':  ['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':  ['social','society','culture','economic','political','community','psychology','policy'],
    'STEM': ['science','physics','chemical','biology','math','engineering','technology','energy'],
}
SYSTEMS = {
    'LAW': "You are a legal expert.", 'MED': "You are a medical expert.",
    'SOC': "You are a social science expert.", 'STEM': "You are a science expert.",
}

def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

# ── GT 로드 ──────────────────────────────────────────────────────────────
with open(GT_PATH) as f:
    gt_data = json.load(f)
queries = [d["instruction"] for d in gt_data[:N + N_WARMUP]]
print(f"쿼리 로드: {len(queries)}개 (warmup {N_WARMUP} + 측정 {N})")

# ── 공통 임베딩 + FAISS ───────────────────────────────────────────────────
print("\n[1] 임베딩 모델 + FAISS 구축...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer(EMBED_MODEL)

files = {"LAW":"law.json","MED":"med.json","SOC":"soc.json",
         "STEM":"stem.json","CROSS":"cross_rewritten.json"}
all_texts = []
for domain, fname in files.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    for d in data:
        out = d.get("output", d.get("gold_answer",""))
        if any(k in out for k in ["The correct answer","All answers:"]): continue
        all_texts.append(f"{d['instruction']} {out}")

embs = embed_model.encode(all_texts, show_progress_bar=False, batch_size=64)
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
flat_index = faiss.IndexFlatIP(embs.shape[1])
flat_index.add(embs.astype(np.float32))
print(f"  FAISS 완료: {len(all_texts)}개")

def retrieve_flat(query, top_k=10):
    emb = embed_model.encode([query], show_progress_bar=False)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    _, I = flat_index.search(emb.astype(np.float32), top_k)
    return [all_texts[i][:200] for i in I[0]]

# ── 7B 모델 로드 ──────────────────────────────────────────────────────────
print("\n[2] Qwen2.5-7B 로드 (4bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True,
                                          local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="cuda:0",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager")
model.eval()
print("  7B 로드 완료")

# ── 측정 함수 ─────────────────────────────────────────────────────────────
def measure(query, prompt_fn, max_new_tokens=300):
    t0 = time.perf_counter()
    prompt, rr_done = prompt_fn(query)
    torch.cuda.synchronize()
    t_rr = time.perf_counter()
    rr_ms = (t_rr - t0) * 1000

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                   skip_special_tokens=True)
    gen_kw = dict(**inputs, max_new_tokens=max_new_tokens, temperature=0.7,
                  do_sample=True, pad_token_id=tokenizer.eos_token_id,
                  streamer=streamer)

    t_q = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=gen_kw)
    thread.start()

    ttft_ms = None; tokens = []; t_first = None
    for tok in streamer:
        if ttft_ms is None:
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t_q) * 1000
            t_first = time.perf_counter()
        tokens.append(tok)
    thread.join()
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    answer = "".join(tokens).strip()
    n_tok = len(tokenizer.encode(answer, add_special_tokens=False))
    dur = t_end - t_first if t_first else 1e-9
    tps = n_tok / dur if dur > 0 else 0
    return rr_ms, ttft_ms or 0.0, tps

def naive_prompt(query):
    ctx = retrieve_flat(query)
    ctx_text = "\n".join(ctx)
    prompt = (f"You are a helpful expert.\n\nContext:\n{ctx_text}\n\n"
              f"Question: {query}\n\n"
              f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
    return prompt, True

def adaptive_prompt(query):
    domain = keyword_route(query)
    system = SYSTEMS.get(domain, "You are a helpful expert.")
    ctx = retrieve_flat(query)
    ctx_text = "\n".join(ctx)
    prompt = (f"{system}\n\nContext:\n{ctx_text}\n\n"
              f"Question: {query}\n\n"
              f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
    return prompt, True

def run_benchmark(name, prompt_fn, max_new_tokens=300):
    print(f"\n{'='*55}")
    print(f"측정: {name}  (N={N}, warmup={N_WARMUP})")
    print(f"{'='*55}")

    print(f"워밍업...")
    for i in range(N_WARMUP):
        measure(queries[i], prompt_fn, max_new_tokens)
        print(f"  [{i+1}/{N_WARMUP}] 완료")

    rr_list, ttft_list, tps_list = [], [], []
    for i, q in enumerate(queries[N_WARMUP: N_WARMUP + N]):
        rr, ttft, tps = measure(q, prompt_fn, max_new_tokens)
        rr_list.append(rr); ttft_list.append(ttft); tps_list.append(tps)
        if (i+1) % 20 == 0:
            print(f"  [{i+1:>3}/{N}]  RR={np.mean(rr_list):.1f}ms"
                  f"  TTFT={np.mean(ttft_list):.1f}ms"
                  f"  TPS={np.mean(tps_list):.1f}tok/s")

    avg = lambda x: round(float(np.mean(x)), 1)
    std = lambda x: round(float(np.std(x)), 1)
    result = {
        "routing_retrieval_ms":    {"avg": avg(rr_list),   "std": std(rr_list)},
        "ttft_ms":                 {"avg": avg(ttft_list), "std": std(ttft_list)},
        "throughput_toks_per_sec": {"avg": avg(tps_list),  "std": std(tps_list)},
        "n": N,
    }
    print(f"\n  RR   = {result['routing_retrieval_ms']['avg']} ± {result['routing_retrieval_ms']['std']} ms")
    print(f"  TTFT = {result['ttft_ms']['avg']} ± {result['ttft_ms']['std']} ms")
    print(f"  TPS  = {result['throughput_toks_per_sec']['avg']} ± {result['throughput_toks_per_sec']['std']} tok/s")
    return result

# ── 실행 ─────────────────────────────────────────────────────────────────
results = {}
results["Naive RAG 7B"]    = run_benchmark("Naive RAG 7B",    naive_prompt,    300)
results["Adaptive RAG 7B"] = run_benchmark("Adaptive RAG 7B", adaptive_prompt, 300)

# NexRoute 7B: LoRA 어댑터 추가 로드 후 측정
print("\n[3] NexRoute 7B 어댑터 로드 (set_adapter 전환만 측정)...")
LORA_MAP = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc",
            "STEM":"lora_stem","CROSS":"lora_cross"}
model = PeftModel.from_pretrained(model, os.path.join(LORA_DIR,"lora_cross"),
                                  adapter_name="CROSS")
for dom, lname in LORA_MAP.items():
    if dom == "CROSS": continue
    model.load_adapter(os.path.join(LORA_DIR, lname), adapter_name=dom)
model.set_adapter("CROSS")
cur_lora = "CROSS"

# BERT 라우터
import torch as _torch
from transformers import BertTokenizerFast, BertModel
ROUTER_PATH = "/home/sclab/moe_model/router_multilabel.pt"
BERT_MODEL  = "bert-base-uncased"
DOMAINS_LIST = ["LAW","MED","SOC","STEM","CROSS"]
bert_tok = BertTokenizerFast.from_pretrained(BERT_MODEL)
bert_mdl = BertModel.from_pretrained(BERT_MODEL).to("cuda:0")
router_head = _torch.load(ROUTER_PATH, map_location="cuda:0")
bert_mdl.eval(); router_head.eval()

# 도메인 파티션 FAISS (pipeline.py 동일)
print("  도메인 파티션 FAISS 구축...")
domain_texts = {}; domain_index = {}
for domain, fname in files.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    txts = []
    for d in data:
        out = d.get("output", d.get("gold_answer",""))
        if any(k in out for k in ["The correct answer","All answers:"]): continue
        txts.append(f"{d['instruction']} {out}")
    emb2 = embed_model.encode(txts, show_progress_bar=False, batch_size=64)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    idx2 = faiss.IndexFlatIP(emb2.shape[1])
    idx2.add(emb2.astype(np.float32))
    domain_texts[domain] = txts
    domain_index[domain] = idx2
print("  완료")

CROSS_THRESHOLD = 0.6
def bert_route(query):
    enc = bert_tok(query, return_tensors="pt", truncation=True,
                   max_length=128).to("cuda:0")
    with _torch.no_grad():
        out = bert_mdl(**enc)
        logits = router_head(out.last_hidden_state[:,0,:])
        probs  = _torch.sigmoid(logits)[0].cpu().numpy()
    top2 = np.argsort(probs)[::-1][:2]
    d1, d2 = DOMAINS_LIST[top2[0]], DOMAINS_LIST[top2[1]]
    p1, p2 = probs[top2[0]], probs[top2[1]]
    is_cross = (p1 < CROSS_THRESHOLD) or (p2 > CROSS_THRESHOLD * 0.8 and p1 - p2 < 0.2)
    domains = [d1, d2] if is_cross else [d1]
    return domains, is_cross

def retrieve_domain(query, domains, is_cross, top_k=10):
    results = []
    for dom in (domains if is_cross else [domains[0]]):
        txts = domain_texts.get(dom, all_texts)
        idx  = domain_index.get(dom, flat_index)
        emb  = embed_model.encode([query], show_progress_bar=False)
        emb  = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        _, I = idx.search(emb.astype(np.float32), top_k)
        results += [{"domain": dom, "text": txts[i]} for i in I[0]]
    return results

def nexroute_prompt(query):
    global cur_lora
    domains, is_cross = bert_route(query)
    ctx = retrieve_domain(query, domains, is_cross)
    ctx_text = "\n".join([f"[{r['domain']}] {r['text'][:200]}" for r in ctx])
    lora_target = "CROSS" if is_cross else domains[0]
    if cur_lora != lora_target:
        model.set_adapter(lora_target)
        cur_lora = lora_target
    if is_cross:
        system = (f"You are a multidisciplinary expert specializing in "
                  f"{' and '.join(domains)}. Integrate knowledge from both domains.")
        few_shot = FEW_SHOT["CROSS"]
    else:
        system = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")
        few_shot = FEW_SHOT.get(domains[0], FEW_SHOT["STEM"])
    domain_instruction = "provide a comprehensive analytical answer with specific mechanisms and implications"
    prompt = (f"{system}\n\nHere is an example of a high-quality answer:\n{few_shot}\n\n"
              f"Context:\n{ctx_text}\n\nQuestion: {query}\n\n"
              f"Following the example style above, {domain_instruction}. "
              f"Do NOT use multiple choice format. Write in flowing paragraph form only.\n\nAnswer:")
    return prompt, True

results["NexRoute 7B"] = run_benchmark("NexRoute 7B", nexroute_prompt, 512)

# ── 저장 + 출력 ───────────────────────────────────────────────────────────
results["_meta"] = {
    "script": "measure_latency_quick.py",
    "timestamp": datetime.datetime.now().isoformat(),
    "n_samples": N, "n_warmup": N_WARMUP,
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("\n\n" + "="*65)
print(f"{'Model':<20} {'RR (ms)':>10} {'TTFT (ms)':>10} {'TPS (tok/s)':>12}")
print("-"*65)
for name in ["Naive RAG 7B", "Adaptive RAG 7B", "NexRoute 7B"]:
    r = results[name]
    print(f"{name:<20} {r['routing_retrieval_ms']['avg']:>10.1f}"
          f" {r['ttft_ms']['avg']:>10.1f}"
          f" {r['throughput_toks_per_sec']['avg']:>12.1f}")
print("="*65)
print(f"\n✅ 저장: {OUT_PATH}")
