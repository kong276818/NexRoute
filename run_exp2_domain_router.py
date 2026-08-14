"""
run_exp2_domain_router.py
==========================
Experiment 2: Domain router necessity (4 routing conditions)
  - predicted       : NexRoute router prediction (re-use v01_gen)
  - oracle_domain   : gold_domain routing (new inference)
  - always_cross    : CROSS adapter + cross-retrieval (top1+top2) for all (new inference)
  - always_single_best: top1 adapter + single-partition retrieval for all (new inference)

Metrics: ROUGE-L, BERTScore, BLEU-4
Cost metrics: passages retrieved, prompt tokens, latency per sample

Scope: clean-300 (sample_id 0-299)
Output: results/exp2_domain_router.json
"""
import os, json, csv, gc, time
import numpy as np
import torch
from collections import Counter

np.random.seed(42); torch.manual_seed(42)

BASE_7B   = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR  = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR  = "/home/sclab/moe_model/processed_v2"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV = "/home/sclab/moe_model/preds_500.csv"
V01_GEN   = "/home/sclab/moe_model/results/ablation/v01_gen_500.json"
KB_CACHE  = "/home/sclab/moe_model/results/ablation/kb_cache"
EMBED_NAME= "intfloat/multilingual-e5-large"
OUT_PATH  = "/home/sclab/moe_model/results/exp2_domain_router.json"
TOP_K     = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] DEVICE={DEVICE}")

DOMAIN_SYSTEM = {
    "LAW":   "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":   "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":   "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM":  "You are a science and engineering expert. Answer accurately based on scientific principles.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains to answer accurately.",
}
DOMAIN_FILES = {
    "LAW":"law_rewritten.json","MED":"med_rewritten.json",
    "SOC":"soc_rewritten.json","STEM":"stem_rewritten.json","CROSS":"cross_rewritten.json",
}
# Use predicted function scaffold for all conditions (held constant)
DEFAULT_SCAFFOLD = "Based on the following evidence, provide a comprehensive and accurate answer to the question. Draw directly on the retrieved context and synthesize the key points:"

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("[1] 데이터 로드...")
gt_list = json.load(open(GT_PATH))
preds_info = {}
gold_domain_map = {}
gold_function_map = {}
with open(PREDS_CSV) as f:
    for r in csv.DictReader(f):
        sid = int(r['sample_id'])
        preds_info[sid] = {
            'pred_domain': r.get('pred_domain_nexroute','CROSS').strip() or 'CROSS',
            'is_cross':    r.get('is_cross','').strip().lower() == 'true',
            'top1':        r.get('top1','SOC').strip() or 'SOC',
            'top2':        r.get('top2','STEM').strip() or 'STEM',
        }
        if r.get('gold_domain','').strip():
            gold_domain_map[sid] = r['gold_domain'].strip()
        if r.get('gold_function','').strip():
            gold_function_map[sid] = r['gold_function'].strip()

v01_by_id = {int(x['sample_id']): x for x in json.load(open(V01_GEN))}

eval_ids = list(range(300))
print(f"  Eval set: {len(eval_ids)}, CROSS: {sum(1 for sid in eval_ids if preds_info.get(sid,{}).get('is_cross'))}")

# Identify single-best domain overall (mode across all domains in eval set)
single_domain_counts = Counter(
    preds_info[sid]['top1'] for sid in eval_ids if not preds_info.get(sid,{}).get('is_cross'))
# For always_single_best: use top1 predicted domain (NOT oracle); for cross queries, still use top1
print(f"  Domain dist (top1): {Counter(preds_info[sid]['top1'] for sid in eval_ids)}")

# ── KB ─────────────────────────────────────────────────────────────────────────
print("[2] KB 로드...")
import faiss as faiss_lib
from sentence_transformers import SentenceTransformer

domain_texts = {}; domain_index = {}
for dom in DOMAIN_FILES:
    txt_path = os.path.join(KB_CACHE, f"{dom}_texts.json")
    emb_path = os.path.join(KB_CACHE, f"{dom}_embs.npy")
    texts = json.load(open(txt_path)) if os.path.exists(txt_path) else []
    if os.path.exists(emb_path):
        embs = np.load(emb_path).astype(np.float32)
    else:
        enc = SentenceTransformer(EMBED_NAME, device=DEVICE)
        embs = enc.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        del enc; gc.collect()
    idx = faiss_lib.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    domain_texts[dom] = texts
    domain_index[dom] = idx

embed_model = SentenceTransformer(EMBED_NAME, device=DEVICE)

def retrieve_cross(query, top1, top2):
    emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    results = []
    for dom in [top1, top2]:
        if dom not in domain_index: continue
        per = max(1, TOP_K // 2)
        D, I = domain_index[dom].search(emb, per)
        for score, i in zip(D[0], I[0]):
            results.append((float(score), f"[{dom}] {domain_texts[dom][i]}"))
    results.sort(reverse=True)
    return [t for _, t in results[:TOP_K]]

def retrieve_single(query, domain):
    if domain not in domain_index: return []
    emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = domain_index[domain].search(emb, TOP_K)
    return [domain_texts[domain][i] for i in I[0]]

# ── LLM ────────────────────────────────────────────────────────────────────────
print("[3] LLM 로드 (4-bit)...")
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True)
lora_map  = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc","STEM":"lora_stem","CROSS":"lora_cross"}
cur_lora  = [None]

def switch_lora(domain):
    if cur_lora[0] == domain: return
    path = os.path.join(LORA_DIR, lora_map[domain])
    if cur_lora[0] is None:
        global base_model
        base_model = PeftModel.from_pretrained(base_model, path, adapter_name=domain)
    else:
        base_model.load_adapter(path, adapter_name=domain)
        base_model.set_adapter(domain)
    cur_lora[0] = domain

def generate_answer(query, context, adapter_domain):
    switch_lora(adapter_domain)
    ctx_text = "\n".join([c[:300] for c in context])
    prompt = f"Here is context:\n{ctx_text}\n\nQuestion: {query}\n\n{DEFAULT_SCAFFOLD}\n\nAnswer:"
    system = DOMAIN_SYSTEM.get(adapter_domain, DOMAIN_SYSTEM["CROSS"])
    messages = [{"role":"system","content":system}, {"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp  = tokenizer(text, return_tensors="pt").to(base_model.device)
    n_prompt = int(inp.input_ids.shape[1])
    t0   = time.perf_counter()
    with torch.no_grad():
        out = base_model.generate(**inp, max_new_tokens=300, temperature=0.1, do_sample=True,
                                  pad_token_id=tokenizer.eos_token_id)
    lat = time.perf_counter() - t0
    ans = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return ans, round(lat, 3), n_prompt

# ── Resume ─────────────────────────────────────────────────────────────────────
done = {}
if os.path.exists(OUT_PATH):
    prev = json.load(open(OUT_PATH))
    done = {r['sample_id']: r for r in prev.get('results', [])}
    print(f"[resume] {len(done)} already done")

results = list(done.values())
todo = [sid for sid in eval_ids if sid not in done]
print(f"[4] Generating {len(todo)} × 3 new conditions (oracle_domain, always_cross, always_single_best)...")

for n, sid in enumerate(todo):
    item = gt_list[sid]
    query = item.get('instruction','').strip() or item.get('query','').strip()
    gold  = item.get('gold_answer','').strip()
    info  = preds_info[sid]
    gold_dom = gold_domain_map.get(sid, 'CROSS')

    # predicted: re-use v01
    ans_predicted = v01_by_id.get(sid, {}).get('pred_answer', '')

    # oracle_domain: route by gold_domain
    oracle_is_cross = (gold_dom == 'CROSS')
    if oracle_is_cross:
        ctx_oracle = retrieve_cross(query, info['top1'], info['top2'])
        oracle_adapter = 'CROSS'
    else:
        ctx_oracle = retrieve_single(query, gold_dom)
        oracle_adapter = gold_dom
    ans_oracle, lat_oracle, ptok_oracle = generate_answer(query, ctx_oracle, oracle_adapter)

    # always_cross: CROSS adapter + cross retrieval (top1+top2 from predictor)
    ctx_always_cross = retrieve_cross(query, info['top1'], info['top2'])
    ans_always_cross, lat_ac, ptok_ac = generate_answer(query, ctx_always_cross, 'CROSS')

    # always_single_best: top1 predicted adapter + single partition
    top1_dom = info['top1']
    ctx_single = retrieve_single(query, top1_dom)
    ans_single, lat_single, ptok_single = generate_answer(query, ctx_single, top1_dom)

    # Cost info
    row = {
        'sample_id':         sid,
        'domain':            item.get('expected_domain',''),
        'gold_domain':       gold_dom,
        'pred_domain':       info['pred_domain'],
        'is_cross':          info['is_cross'],
        'top1':              info['top1'],
        'top2':              info['top2'],
        'gold_answer':       gold,
        'predicted':         ans_predicted,
        'oracle_domain':     ans_oracle,
        'always_cross':      ans_always_cross,
        'always_single_best':ans_single,
        'cost_oracle_domain':    {'is_cross': oracle_is_cross, 'passages': len(ctx_oracle), 'ptokens': ptok_oracle, 'latency': lat_oracle},
        'cost_always_cross':     {'is_cross': True,  'passages': len(ctx_always_cross), 'ptokens': ptok_ac, 'latency': lat_ac},
        'cost_always_single_best':{'is_cross': False, 'passages': len(ctx_single), 'ptokens': ptok_single, 'latency': lat_single},
    }
    results.append(row)
    done[sid] = row

    if (n+1) % 20 == 0 or (n+1) == len(todo):
        print(f"  [{n+1}/{len(todo)}] sid={sid} gold_dom={gold_dom} pred={info['pred_domain']} lat_oracle={lat_oracle:.1f}s lat_ac={lat_ac:.1f}s lat_single={lat_single:.1f}s")
        json.dump({'partial': (n+1) < len(todo), 'results': results}, open(OUT_PATH,'w'), indent=2, ensure_ascii=False)

json.dump({'partial': False, 'n': len(results), 'results': results},
          open(OUT_PATH,'w'), indent=2, ensure_ascii=False)
print(f"\n[done] Saved: {OUT_PATH}")
