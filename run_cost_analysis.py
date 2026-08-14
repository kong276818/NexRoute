"""
run_cost_analysis.py
=====================
Compute cost metrics for Experiment 2 domain routing conditions.
Reads exp2_domain_router.json for passages/ptokens/latency already collected.
Runs a 10-sample VRAM profiling pass per condition to estimate peak VRAM.

Conditions: predicted, oracle_domain, always_cross, always_single_best
Baseline for VRAM profiling: use existing results to pick representative samples.

Output: results/exp2_cost_summary.json
"""
import json, csv, os, gc, time, re
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] DEVICE={DEVICE}")

EXP2_PATH = "/home/sclab/moe_model/results/exp2_domain_router.json"
OUT_PATH  = "/home/sclab/moe_model/results/exp2_cost_summary.json"

BASE_7B   = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR  = "/home/sclab/moe_model/lora_weights_7b_v3"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV = "/home/sclab/moe_model/preds_500.csv"
KB_CACHE  = "/home/sclab/moe_model/results/ablation/kb_cache"
EMBED_NAME= "intfloat/multilingual-e5-large"
TOP_K     = 10
N_VRAM_SAMPLES = 10  # per condition for VRAM profiling

# ── Load Exp2 results ──────────────────────────────────────────────────────────
print("[1] Loading Exp2 results...")
data = json.load(open(EXP2_PATH))
results = data.get('results', [])
print(f"  n={len(results)}")

CONDS = ['predicted', 'oracle_domain', 'always_cross', 'always_single_best']
COST_KEYS = {'oracle_domain': 'cost_oracle_domain',
             'always_cross':  'cost_always_cross',
             'always_single_best': 'cost_always_single_best'}

# ── Aggregate existing cost metrics ───────────────────────────────────────────
print("[2] Aggregating passages / ptokens / latency...")
cost_stats = {}

# 'predicted' cost: re-use v01 ctx settings (domain routing)
# We need to infer: predicted routes is_cross → cross retrieval, else single
# These are the same as what v01 used - we need to compute from preds_csv
gt_list = json.load(open(GT_PATH))
preds_info = {}
with open(PREDS_CSV) as f:
    for r in csv.DictReader(f):
        sid = int(r['sample_id'])
        preds_info[sid] = {
            'pred_domain': r.get('pred_domain_nexroute','CROSS').strip() or 'CROSS',
            'is_cross':    r.get('is_cross','').strip().lower() == 'true',
        }

# For predicted: passages = TOP_K always (either cross or single partition gives TOP_K)
# ptokens: not directly available for predicted (re-use from v01 gen if available)
# Estimate predicted ptokens from always_cross ptokens (same context size)
pred_ptokens = []
pred_latencies = []
for r in results:
    sid = r['sample_id']
    info = preds_info.get(sid, {})
    # predicted passages: always TOP_K (10)
    if 'cost_always_cross' in r:
        # Use always_cross ptokens as proxy for predicted (same TOP_K)
        pred_ptokens.append(r['cost_always_cross']['ptokens'])
    if 'cost_always_cross' in r:
        pred_latencies.append(r['cost_always_cross']['latency'])

cost_stats['predicted'] = {
    'passages_mean':     TOP_K,
    'passages_sd':       0.0,
    'ptokens_mean':      round(float(np.mean(pred_ptokens)), 1) if pred_ptokens else None,
    'ptokens_sd':        round(float(np.std(pred_ptokens)), 1) if pred_ptokens else None,
    'latency_mean_s':    round(float(np.mean(pred_latencies)), 2) if pred_latencies else None,
    'latency_sd_s':      round(float(np.std(pred_latencies)), 2) if pred_latencies else None,
    'note': 'ptokens/latency estimated from always_cross (same TOP_K=10)',
}

for cond, key in COST_KEYS.items():
    rows = [r[key] for r in results if key in r]
    if not rows:
        cost_stats[cond] = {'n': 0}
        continue
    passages  = [r['passages'] for r in rows]
    ptokens   = [r['ptokens'] for r in rows]
    latencies = [r['latency'] for r in rows]
    cost_stats[cond] = {
        'n': len(rows),
        'passages_mean':   round(float(np.mean(passages)), 2),
        'passages_sd':     round(float(np.std(passages)), 2),
        'ptokens_mean':    round(float(np.mean(ptokens)), 1),
        'ptokens_sd':      round(float(np.std(ptokens)), 1),
        'latency_mean_s':  round(float(np.mean(latencies)), 2),
        'latency_sd_s':    round(float(np.std(latencies)), 2),
    }

print(f"\n  {'Condition':<22} {'Passages':>9} {'PTokens':>9} {'Latency(s)':>11}")
for cond in CONDS:
    s = cost_stats.get(cond, {})
    print(f"  {cond:<22} {s.get('passages_mean','-'):>9} {s.get('ptokens_mean','-'):>9} {s.get('latency_mean_s','-'):>11}")

# ── VRAM profiling ─────────────────────────────────────────────────────────────
print("\n[3] VRAM profiling (10 samples × 4 conditions)...")
from sentence_transformers import SentenceTransformer
import faiss as faiss_lib
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DOMAIN_FILES = {"LAW":"law_rewritten.json","MED":"med_rewritten.json",
                "SOC":"soc_rewritten.json","STEM":"stem_rewritten.json","CROSS":"cross_rewritten.json"}
DOMAIN_SYSTEM = {
    "LAW": "You are a legal expert.", "MED": "You are a medical expert.",
    "SOC": "You are a social science expert.", "STEM": "You are a science and engineering expert.",
    "CROSS": "You are a multidisciplinary expert.",
}
SCAFFOLD = "Based on the following evidence, provide a comprehensive and accurate answer:"

domain_texts = {}; domain_index = {}
for dom in DOMAIN_FILES:
    txt_path = os.path.join(KB_CACHE, f"{dom}_texts.json")
    emb_path = os.path.join(KB_CACHE, f"{dom}_embs.npy")
    texts = json.load(open(txt_path)) if os.path.exists(txt_path) else []
    embs = np.load(emb_path).astype(np.float32) if os.path.exists(emb_path) else None
    if embs is None: continue
    idx = faiss_lib.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    domain_texts[dom] = texts; domain_index[dom] = idx

embed_model = SentenceTransformer(EMBED_NAME, device=DEVICE)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE_7B, quantization_config=bnb,
                                              device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True)
lora_map = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc","STEM":"lora_stem","CROSS":"lora_cross"}
cur_lora = [None]

def switch_lora(domain):
    if cur_lora[0] == domain: return
    path = os.path.join(LORA_DIR, lora_map[domain])
    if cur_lora[0] is None:
        global model
        model = PeftModel.from_pretrained(model, path, adapter_name=domain)
    else:
        model.load_adapter(path, adapter_name=domain)
        model.set_adapter(domain)
    cur_lora[0] = domain

def profile_one(query, ctx, adapter):
    switch_lora(adapter)
    ctx_text = "\n".join([c[:300] for c in ctx])
    prompt = f"Context:\n{ctx_text}\n\nQuestion: {query}\n\n{SCAFFOLD}\n\nAnswer:"
    msgs = [{"role":"system","content":DOMAIN_SYSTEM[adapter]},{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tokenizer(text, return_tensors="pt").to(model.device)
    n_prompt = int(inp.input_ids.shape[1])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=300, temperature=0.1, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
    lat = time.perf_counter() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    return lat, vram_peak, n_prompt

# Sample N_VRAM_SAMPLES from results
sample_rows = results[:N_VRAM_SAMPLES]
vram_stats = {cond: [] for cond in CONDS}

for r in sample_rows:
    sid = r['sample_id']
    item = gt_list[sid]
    query = item.get('instruction','').strip() or item.get('query','').strip()
    info = preds_info.get(sid, {})
    gold_dom = r.get('gold_domain','CROSS')

    # predicted: use NexRoute routing (replicate)
    pred_dom = info.get('pred_domain','CROSS')
    is_cross_pred = info.get('is_cross', False)
    if is_cross_pred:
        emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        ctx_pred = []
        for dom in [r.get('top1','SOC'), r.get('top2','STEM')]:
            if dom in domain_index:
                D, I = domain_index[dom].search(emb, max(1, TOP_K//2))
                ctx_pred.extend([f"[{dom}] {domain_texts[dom][i]}" for i in I[0]])
        adapter_pred = 'CROSS'
    else:
        emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        if pred_dom in domain_index:
            D, I = domain_index[pred_dom].search(emb, TOP_K)
            ctx_pred = [domain_texts[pred_dom][i] for i in I[0]]
        else: ctx_pred = []
        adapter_pred = pred_dom
    lat, vram, ntok = profile_one(query, ctx_pred, adapter_pred)
    vram_stats['predicted'].append({'vram_gb': round(vram,2), 'latency': round(lat,2), 'ptokens': ntok})

    # oracle_domain
    oracle_is_cross = (gold_dom == 'CROSS')
    if oracle_is_cross:
        emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        ctx_oracle = []
        for dom in [r.get('top1','SOC'), r.get('top2','STEM')]:
            if dom in domain_index:
                D, I = domain_index[dom].search(emb, max(1, TOP_K//2))
                ctx_oracle.extend([f"[{dom}] {domain_texts[dom][i]}" for i in I[0]])
        adapter_oracle = 'CROSS'
    else:
        emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        if gold_dom in domain_index:
            D, I = domain_index[gold_dom].search(emb, TOP_K)
            ctx_oracle = [domain_texts[gold_dom][i] for i in I[0]]
        else: ctx_oracle = []
        adapter_oracle = gold_dom
    lat, vram, ntok = profile_one(query, ctx_oracle, adapter_oracle)
    vram_stats['oracle_domain'].append({'vram_gb': round(vram,2), 'latency': round(lat,2), 'ptokens': ntok})

    # always_cross
    emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    ctx_ac = []
    for dom in [r.get('top1','SOC'), r.get('top2','STEM')]:
        if dom in domain_index:
            D, I = domain_index[dom].search(emb, max(1, TOP_K//2))
            ctx_ac.extend([f"[{dom}] {domain_texts[dom][i]}" for i in I[0]])
    lat, vram, ntok = profile_one(query, ctx_ac, 'CROSS')
    vram_stats['always_cross'].append({'vram_gb': round(vram,2), 'latency': round(lat,2), 'ptokens': ntok})

    # always_single_best
    top1_dom = r.get('top1', info.get('pred_domain','SOC'))
    emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    if top1_dom in domain_index:
        D, I = domain_index[top1_dom].search(emb, TOP_K)
        ctx_single = [domain_texts[top1_dom][i] for i in I[0]]
    else: ctx_single = []
    lat, vram, ntok = profile_one(query, ctx_single, top1_dom)
    vram_stats['always_single_best'].append({'vram_gb': round(vram,2), 'latency': round(lat,2), 'ptokens': ntok})

# Aggregate VRAM
print(f"\n  {'Condition':<22} {'VRAM_peak(GB)':>14} {'Latency(s)':>11}")
for cond in CONDS:
    rows = vram_stats[cond]
    if not rows: continue
    avg_vram = np.mean([r['vram_gb'] for r in rows])
    avg_lat  = np.mean([r['latency'] for r in rows])
    print(f"  {cond:<22} {avg_vram:>14.2f} {avg_lat:>11.2f}")
    cost_stats[cond]['vram_peak_gb_mean'] = round(float(avg_vram), 2)
    cost_stats[cond]['vram_peak_gb_sd']   = round(float(np.std([r['vram_gb'] for r in rows])), 2)

out = {
    'n_results': len(results),
    'n_vram_samples': N_VRAM_SAMPLES,
    'top_k_passages': TOP_K,
    'cost_summary': cost_stats,
    'vram_per_sample': vram_stats,
}
json.dump(out, open(OUT_PATH,'w'), indent=2)
print(f"\n[done] Saved: {OUT_PATH}")
