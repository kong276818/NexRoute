"""
run_exp1_oracle_scaffold.py
============================
Experiment 1: Function scaffold oracle reconstruction (4 conditions)
  - oracle  : label_func_type() → scaffold; OTHER → generic scaffold
  - predicted: B_predicted from v2 gen (V01 DeBERTa output, re-used)
  - generic : function-agnostic scaffold (length-matched)
  - none    : E_none from v2 gen (V02 output, re-used)

Only oracle and generic need new inference (300 samples each).
predicted and none are loaded from existing gen files.

Output: results/exp1_scaffold_oracle.json
"""
import os, json, csv, gc, re, time, pickle
import numpy as np
import torch
from pathlib import Path
from collections import Counter

np.random.seed(42); torch.manual_seed(42)

BASE_7B   = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR  = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR  = "/home/sclab/moe_model/processed_v2"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV = "/home/sclab/moe_model/preds_500.csv"
V01_GEN   = "/home/sclab/moe_model/results/ablation/v01_gen_500.json"
V02_GEN   = "/home/sclab/moe_model/results/ablation/v02_gen_500.json"
KB_CACHE  = "/home/sclab/moe_model/results/ablation/kb_cache"
EMBED_NAME= "intfloat/multilingual-e5-large"
OUT_PATH  = "/home/sclab/moe_model/results/exp1_scaffold_oracle.json"
TOP_K     = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] DEVICE={DEVICE}")

# ── label_func_type ────────────────────────────────────────────────────────────
def label_func_type(text):
    t = text.lower().strip()
    if re.search(r'^what (is|are|was|were|do|does|did)\b|^which (of|is|are)\b|\bdefin|\bmeaning of\b|\bwhat does .+ mean\b', t):
        return 'WHAT'
    if re.search(r'^how (do|does|did|can|to|is|are|would)\b|\bsteps? (to|for)\b|\bmechanism of\b|\bprocedure\b|\bprocess of\b', t):
        return 'HOW'
    if re.search(r'^why (is|are|was|do|does|did|would|can)\b|\bcause(s)? of\b|\breason(s)? for\b|\bwhat causes\b|\bwhat led to\b', t):
        return 'WHY'
    if re.search(r'\bcompare\b|\bdifference(s)? between\b|\bversus\b|\bvs\.?\b|\bcontrast\b|\bdistinguish\b|\bhow .+ differ\b', t):
        return 'COMPARE'
    return 'OTHER'

SCAFFOLDS = {
    "WHAT":    "Based on the following evidence, identify and define the key facts. Do not infer beyond the provided context:",
    "HOW":     "Using the following evidence, describe the mechanism or procedure in a step-by-step logical flow:",
    "WHY":     "Based on the following evidence, construct a definitive causal chain. Explicitly explain the underlying reasons connecting premise to outcome:",
    "COMPARE": "Using the following evidence, contrast the entities in the query. Highlight structural similarities and key differences in a comparative framework:",
    "GENERIC": "Based on the following evidence, provide a comprehensive and accurate answer to the question. Draw directly on the retrieved context and synthesize the key points:",
}

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

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("[1] 데이터 로드...")
gt_list    = json.load(open(GT_PATH))
rows_by_id = {}
gold_func  = {}
preds_info = {}
with open(PREDS_CSV) as f:
    for r in csv.DictReader(f):
        sid = int(r['sample_id'])
        rows_by_id[sid] = r
        if r.get('gold_function','').strip():
            gold_func[sid] = r['gold_function'].strip()
        preds_info[sid] = {
            'pred_domain': r.get('pred_domain_nexroute','CROSS'),
            'is_cross':    r.get('is_cross','').strip().lower() == 'true',
            'top1':        r.get('top1','SOC'),
            'top2':        r.get('top2','STEM'),
        }

v01_by_id = {int(x['sample_id']): x for x in json.load(open(V01_GEN))}
v02_by_id = {int(x['sample_id']): x for x in json.load(open(V02_GEN))}

# Clean-300: idx 0-299
eval_ids  = list(range(300))
print(f"  Eval set: {len(eval_ids)} samples")

# Compute oracle labels
oracle_labels = {}
for sid in eval_ids:
    q = gt_list[sid].get('instruction','').strip() or gt_list[sid].get('query','').strip()
    oracle_labels[sid] = label_func_type(q)

n_other   = sum(1 for v in oracle_labels.values() if v == 'OTHER')
n_valid   = len(oracle_labels) - n_other
print(f"  Oracle label dist: {Counter(oracle_labels.values())}")
print(f"  Valid (non-OTHER): {n_valid}/300 = {n_valid/300*100:.1f}%")
match_gold = sum(1 for sid in eval_ids if oracle_labels[sid] != 'OTHER' and oracle_labels.get(sid) == gold_func.get(sid,''))
print(f"  Oracle == gold (non-OTHER): {match_gold}/300")

# ── KB 캐시 ────────────────────────────────────────────────────────────────────
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

def retrieve(query, lora_target, is_cross, top1, top2):
    emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    if is_cross:
        results = []
        for dom in [top1, top2]:
            if dom not in domain_index: continue
            per = max(1, TOP_K // 2)
            D, I = domain_index[dom].search(emb, per)
            for score, i in zip(D[0], I[0]):
                results.append((float(score), f"[{dom}] {domain_texts[dom][i]}"))
        results.sort(reverse=True)
        return [t for _, t in results[:TOP_K]]
    else:
        if lora_target not in domain_index: return []
        D, I = domain_index[lora_target].search(emb, TOP_K)
        return [domain_texts[lora_target][i] for i in I[0]]

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

def generate_answer(query, context, scaffold_label, lora_target):
    switch_lora(lora_target)
    ctx_text = "\n".join([c[:300] for c in context])
    instruction = SCAFFOLDS.get(scaffold_label, SCAFFOLDS["GENERIC"])
    prompt = f"Here is context:\n{ctx_text}\n\nQuestion: {query}\n\n{instruction}\n\nAnswer:"
    system = DOMAIN_SYSTEM.get(lora_target, DOMAIN_SYSTEM["CROSS"])
    messages = [{"role":"system","content":system}, {"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp  = tokenizer(text, return_tensors="pt").to(base_model.device)
    t0   = time.perf_counter()
    with torch.no_grad():
        out = base_model.generate(**inp, max_new_tokens=300, temperature=0.1, do_sample=True,
                                  pad_token_id=tokenizer.eos_token_id)
    lat = time.perf_counter() - t0
    ans = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    prompt_tokens = inp.input_ids.shape[1]
    return ans, lat, prompt_tokens

# ── Resume ─────────────────────────────────────────────────────────────────────
done = {}
if os.path.exists(OUT_PATH):
    prev = json.load(open(OUT_PATH))
    done = {r['sample_id']: r for r in prev.get('results', [])}
    print(f"[resume] {len(done)} already done")

results = list(done.values())
todo = [sid for sid in eval_ids if sid not in done]
print(f"[4] Generating {len(todo)} samples × 2 new conditions (oracle, generic)...")

for n, sid in enumerate(todo):
    item = gt_list[sid]
    query = item.get('instruction','').strip() or item.get('query','').strip()
    gold  = item.get('gold_answer','').strip()
    info  = preds_info[sid]

    ctx = retrieve(query, info['pred_domain'], info['is_cross'], info['top1'], info['top2'])
    lora_target = info['pred_domain']

    oracle_lbl = oracle_labels[sid]
    oracle_scaffold = oracle_lbl if oracle_lbl != 'OTHER' else 'GENERIC'

    # oracle condition
    ans_oracle, lat_oracle, tok_oracle = generate_answer(query, ctx, oracle_scaffold, lora_target)
    # generic condition
    ans_generic, lat_generic, tok_generic = generate_answer(query, ctx, 'GENERIC', lora_target)

    # re-use from existing gen files
    ans_predicted = v01_by_id.get(sid, {}).get('pred_answer', '')
    ans_none      = v02_by_id.get(sid, {}).get('pred_answer', '')

    row = {
        'sample_id':     sid,
        'domain':        item.get('expected_domain', ''),
        'is_cross':      info['is_cross'],
        'gold_function': gold_func.get(sid, ''),
        'oracle_label':  oracle_lbl,
        'gold_answer':   gold,
        'oracle':        ans_oracle,
        'generic':       ans_generic,
        'predicted':     ans_predicted,
        'none':          ans_none,
        'latency_oracle':  round(lat_oracle,3),
        'latency_generic': round(lat_generic,3),
        'ptokens_oracle':  tok_oracle,
        'ptokens_generic': tok_generic,
    }
    results.append(row)
    done[sid] = row

    if (n+1) % 20 == 0 or (n+1) == len(todo):
        print(f"  [{n+1}/{len(todo)}] sid={sid} oracle_lbl={oracle_lbl} lat_oracle={lat_oracle:.1f}s lat_generic={lat_generic:.1f}s")
        json.dump({'partial': (n+1) < len(todo), 'results': results}, open(OUT_PATH,'w'), indent=2, ensure_ascii=False)

json.dump({'partial': False, 'n': len(results), 'oracle_dist': dict(Counter(oracle_labels.values())),
           'results': results}, open(OUT_PATH,'w'), indent=2, ensure_ascii=False)
print(f"\n[done] Saved: {OUT_PATH}")
