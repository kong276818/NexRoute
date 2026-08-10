"""
run_function_axis_ablation.py
=============================
Function Axis Controlled Ablation — 5-way scaffold comparison

목표: function scaffold의 causal 효과를 oracle / predicted / wrong / random / none
     5가지 조건으로 측정

조건:
  A_oracle    : gold function label (preds_500.csv gold_function) → scaffold 적용
  B_predicted : TF-IDF router prediction → scaffold 적용 (predicted wrong 포함)
  C_wrong     : gold label의 cyclic shift → 의도적 오분류 scaffold
  D_random    : seed-fixed random label → scaffold 적용
  E_none      : scaffold 없음 (도메인 instruction만, V02 조건)

평가 대상: Clean-300 (idx 0-299), gold_function 있는 샘플 (300개 중 일부)
출력: results/function_axis_ablation_clean300.json
"""
import os, json, csv, random, pickle, torch, numpy as np
import re
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import faiss
import nltk
nltk.download('punkt_tab', quiet=True)

# ── 경로 ─────────────────────────────────────────────────────────────────────
BASE_7B    = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR   = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV  = "/home/sclab/moe_model/preds_500.csv"
FUNC_ROUTER= "/home/sclab/moe_model/routers/func_router/pipeline.pkl"
EMBED_NAME = "intfloat/multilingual-e5-large"
OUT_PATH   = "/home/sclab/moe_model/results/function_axis_ablation_clean300.json"
SEED       = 42
TOP_K      = 10

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

FUNCTION_LABELS = ["WHAT", "HOW", "WHY", "COMPARE"]
CYCLIC_WRONG = {"WHAT":"HOW", "HOW":"WHY", "WHY":"COMPARE", "COMPARE":"WHAT",
                "OTHER":"WHAT"}

SCAFFOLDS = {
    "WHAT":    "Based on the following evidence, identify and define the key facts. Do not infer beyond the provided context:",
    "HOW":     "Using the following evidence, describe the mechanism or procedure in a step-by-step logical flow:",
    "WHY":     "Based on the following evidence, construct a definitive causal chain. Explicitly explain the underlying reasons connecting premise to outcome:",
    "COMPARE": "Using the following evidence, contrast the entities in the query. Highlight structural similarities and key differences in a comparative framework:",
}

DOMAIN_SYSTEM = {
    "LAW":   "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":   "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":   "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM":  "You are a science and engineering expert. Answer accurately based on scientific principles.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains to answer accurately.",
}

FEW_SHOT = {
    "LAW": "Q: What is the doctrine of promissory estoppel?\nA: Promissory estoppel is a legal doctrine that prevents a party from withdrawing a promise when another party has reasonably relied on it to their detriment.",
    "MED": "Q: How does the renin-angiotensin system regulate blood pressure?\nA: The renin-angiotensin system regulates blood pressure through a cascade of enzymatic reactions initiated when the kidneys detect low blood pressure or reduced sodium levels.",
    "SOC": "Q: How does social stratification affect educational outcomes?\nA: Social stratification significantly influences educational outcomes by creating unequal access to resources, opportunities, and cultural capital.",
    "STEM": "Q: How does CRISPR-Cas9 technology work?\nA: CRISPR-Cas9 is a revolutionary gene-editing technology that uses a guide RNA to direct the Cas9 endonuclease to specific DNA sequences.",
    "CROSS": "Q: How might advances in neuroscience impact criminal justice policy?\nA: Advances in neuroscience are profoundly reshaping criminal justice policy by providing empirical insights into the biological underpinnings of behavior.",
}

DOMAIN_FILES = {
    "LAW": "law_rewritten.json", "MED": "med_rewritten.json",
    "SOC": "soc_rewritten.json", "STEM": "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("[*] GT 로드...")
gt_list = json.load(open(GT_PATH))

print("[*] preds_500.csv 로드...")
routing_map = {}
func_gold_map = {}
with open(PREDS_CSV) as f:
    for row in csv.DictReader(f):
        idx = int(row['sample_id'])
        routing_map[idx] = {
            'pred_domain': row['pred_domain_nexroute'],
            'is_cross':    row['is_cross'].strip().lower() == 'true',
            'top1': row['top1'], 'top2': row['top2'],
        }
        if row.get('gold_function', '').strip():
            func_gold_map[idx] = row['gold_function'].strip()

# Clean-300 중 gold_function 있는 샘플 (idx 0-299)
eval_items = [(i, gt_list[i]) for i in range(300) if i in func_gold_map]
print(f"  Clean-300 gold_function 있는 샘플: {len(eval_items)}")

# ── Function Router 로드 ─────────────────────────────────────────────────────
print("[*] TF-IDF function router 로드...")
with open(FUNC_ROUTER, 'rb') as f:
    func_router = pickle.load(f)

# ── Embedding 모델 & FAISS 구축 ───────────────────────────────────────────────
print("[*] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_NAME, device="cuda")

print("[*] FAISS 인덱스 구축...")
domain_texts = {}; domain_index = {}
for domain, fname in DOMAIN_FILES.items():
    data = json.load(open(os.path.join(DATA_DIR, fname)))
    texts = []
    for d in data:
        out = d.get("output", d.get("gold_answer", ""))
        if any(k in out for k in ["The correct answer", "All answers:"]): continue
        texts.append(f"{d['instruction']} {out}")
    embs = embed_model.encode(texts, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True)
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs.astype(np.float32))
    domain_texts[domain] = texts; domain_index[domain] = idx
    print(f"  {domain}: {len(texts)}")

def retrieve(query, domains, is_cross):
    emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    if is_cross:
        results = []
        per = max(1, TOP_K // len(domains))
        for dom in domains:
            if dom not in domain_index: continue
            D, I = domain_index[dom].search(emb, per)
            for score, i in zip(D[0], I[0]):
                results.append((float(score), f"[{dom}] {domain_texts[dom][i]}"))
        results.sort(reverse=True)
        return [t for _, t in results[:TOP_K]]
    else:
        dom = domains[0]
        if dom not in domain_index: return []
        D, I = domain_index[dom].search(emb, TOP_K)
        return [domain_texts[dom][i] for i in I[0]]

# ── LLM 로드 ─────────────────────────────────────────────────────────────────
print("[*] LLM 로드 (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_7B, quantization_config=bnb,
                                                   device_map="auto", trust_remote_code=True)
tokenizer   = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True)

lora_map = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc",
            "STEM":"lora_stem","CROSS":"lora_cross"}
current_lora = [None]

def switch_lora(domain):
    global base_model
    if current_lora[0] == domain: return
    lora_path = os.path.join(LORA_DIR, lora_map[domain])
    if current_lora[0] is None:
        base_model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=domain)
    else:
        base_model.load_adapter(lora_path, adapter_name=domain)
        base_model.set_adapter(domain)
    current_lora[0] = domain

def generate(prompt, system, max_new_tokens=512):
    messages = [{"role":"system","content":system},
                {"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def build_prompt(query, context, func_label):
    ctx_text = "\n".join([c[:200] for c in context])
    if func_label and func_label in SCAFFOLDS:
        instruction = SCAFFOLDS[func_label]
    else:
        instruction = "provide a comprehensive analytical answer with specific mechanisms and implications"
    return (f"Here is context:\n{ctx_text}\n\n"
            f"Question: {query}\n\n"
            f"{instruction}\n\nAnswer:")

# ── 실험 루프 ─────────────────────────────────────────────────────────────────
rng = random.Random(SEED)
rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
results = []

print(f"\n[*] 실험 시작: {len(eval_items)}개 샘플 × 5 조건")
for step, (idx, item) in enumerate(eval_items):
    query = item["instruction"]
    gold  = item["gold_answer"]
    r     = routing_map.get(idx, {})
    pred_domain = r.get('pred_domain', 'STEM')
    is_cross    = r.get('is_cross', False)
    top1 = r.get('top1', pred_domain)
    top2 = r.get('top2', pred_domain)
    domains = [top1, top2] if is_cross else [top1]

    lora_target = "CROSS" if is_cross else top1
    switch_lora(lora_target)
    system = DOMAIN_SYSTEM.get(lora_target, DOMAIN_SYSTEM["CROSS"])

    context = retrieve(query, domains, is_cross)
    gold_func = func_gold_map[idx]
    pred_func = func_router.predict([query])[0]  # TF-IDF prediction
    wrong_func = CYCLIC_WRONG.get(gold_func, "HOW")
    rand_func  = rng.choice(FUNCTION_LABELS)

    row = {
        "idx": idx, "gold_function": gold_func, "pred_function": pred_func,
        "wrong_function": wrong_func, "random_function": rand_func,
        "is_cross": is_cross, "domain": item.get("expected_domain", "?"),
        "correct_prediction": (pred_func == gold_func),
    }

    for cond, func_label in [
        ("A_oracle",    gold_func),
        ("B_predicted", pred_func),
        ("C_wrong",     wrong_func),
        ("D_random",    rand_func),
        ("E_none",      None),
    ]:
        prompt = build_prompt(query, context, func_label)
        ans = generate(prompt, system)
        rl = rscorer.score(gold, ans)['rougeL'].fmeasure * 100
        row[cond] = {"answer": ans, "rouge_l": rl}

    results.append(row)

    # 중간 보고
    if (step+1) % 10 == 0:
        print(f"  [{step+1}/{len(eval_items)}] 완료")
        partial = {}
        for cond in ["A_oracle","B_predicted","C_wrong","D_random","E_none"]:
            vals = [r[cond]["rouge_l"] for r in results]
            partial[cond] = np.mean(vals)
        print(f"  Current avg ROUGE-L: " + ", ".join(f"{k}={v:.2f}" for k,v in partial.items()))

        # 중간 저장
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        json.dump({"partial": True, "n_completed": len(results), "results": results},
                  open(OUT_PATH, 'w'), indent=2, ensure_ascii=False)

# ── 최종 집계 ─────────────────────────────────────────────────────────────────
conds = ["A_oracle","B_predicted","C_wrong","D_random","E_none"]
cond_labels = {"A_oracle":"Oracle","B_predicted":"Predicted","C_wrong":"Wrong",
               "D_random":"Random","E_none":"None (baseline)"}

summary = {}
for cond in conds:
    vals = np.array([r[cond]["rouge_l"] for r in results])
    cross = np.array([r[cond]["rouge_l"] for r in results if r['is_cross']])
    single = np.array([r[cond]["rouge_l"] for r in results if not r['is_cross']])
    summary[cond] = {
        "all_mean": float(np.mean(vals)),
        "cross_mean": float(np.mean(cross)) if len(cross) else None,
        "single_mean": float(np.mean(single)) if len(single) else None,
    }

print("\n" + "="*60)
print("Function Axis Ablation 최종 결과")
print("="*60)
print(f"{'Condition':<20} {'All ROUGE-L':>12} {'CROSS':>8} {'Single':>8}")
for cond in conds:
    s = summary[cond]
    cross_str = f"{s['cross_mean']:.2f}" if s['cross_mean'] else "  n/a"
    single_str = f"{s['single_mean']:.2f}" if s['single_mean'] else "  n/a"
    print(f"  {cond_labels[cond]:<18} {s['all_mean']:>12.2f} {cross_str:>8} {single_str:>8}")

out = {"partial": False, "n_total": len(results), "summary": summary, "results": results}
json.dump(out, open(OUT_PATH, 'w'), indent=2, ensure_ascii=False)
print(f"\n저장: {OUT_PATH}")
