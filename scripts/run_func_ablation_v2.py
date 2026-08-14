"""
run_func_ablation_v2.py
=======================
Function Axis Controlled Ablation — 5-way scaffold comparison (enhanced)

조건:
  A_oracle    : gold function label (gt_dataset_500 expected_func)
  B_predicted : V01 gen file 재사용 (DeBERTa 라우터 예측 기준, 동일 pipeline)
  C_wrong     : gold label 사이클 시프트 (WHAT→HOW→WHY→COMPARE→WHAT)
  D_random    : seed-fixed random label
  E_none      : V02 gen file 재사용 (scaffold 없음)

B_predicted = V01 gen 재사용 이유:
  TF-IDF func router가 아닌 DeBERTa 라우터 기준 결과가 필요하기 때문.
  V01 gen file이 정확히 이 조건 (DeBERTa predicted function label + scaffold + LoRA).

E_none = V02 gen 재사용 이유:
  V02가 scaffold 제거만 한 NexRoute variant이므로 E_none에 해당.

A_oracle / C_wrong / D_random: 신규 추론 필요 (300 샘플)

평가 범위: clean 300 (sample_id 0~299, gold_function 있는 전체)

측정:
  - ROUGE-L, BERTScore F1, BLEU-4, 응답 길이(단어/문장)
  - Hallucination (NLI, 추론 후 별도 패스)

출력:
  results/function_axis_ablation_v2_gen.json     (생성 결과)
  results/function_axis_ablation_v2_scores.json  (메트릭 요약)
"""
import os, json, csv, random, pickle, torch, gc
import numpy as np
from pathlib import Path

random.seed(42); np.random.seed(42); torch.manual_seed(42)

BASE_7B   = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR  = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR  = "/home/sclab/moe_model/processed_v2"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV = "/home/sclab/moe_model/preds_500.csv"
V01_GEN   = "/home/sclab/moe_model/results/ablation/v01_gen_500.json"
V02_GEN   = "/home/sclab/moe_model/results/ablation/v02_gen_500.json"
KB_CACHE  = "/home/sclab/moe_model/results/ablation/kb_cache"
OUT_GEN   = "/home/sclab/moe_model/results/function_axis_ablation_v2_gen.json"
OUT_SCORE = "/home/sclab/moe_model/results/function_axis_ablation_v2_scores.json"
SEED      = 42
TOP_K     = 10
EMBED_NAME = "intfloat/multilingual-e5-large"

FUNCTION_LABELS = ["WHAT", "HOW", "WHY", "COMPARE"]
CYCLIC_WRONG = {"WHAT":"HOW", "HOW":"WHY", "WHY":"COMPARE", "COMPARE":"WHAT"}
DOMAIN_FILES = {
    "LAW":"law_rewritten.json", "MED":"med_rewritten.json",
    "SOC":"soc_rewritten.json", "STEM":"stem_rewritten.json",
    "CROSS":"cross_rewritten.json",
}
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] DEVICE={DEVICE}")

# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
print("[1] 데이터 로드...")
gt_list = json.load(open(GT_PATH))

preds_by_id = {}
gold_func_map = {}
with open(PREDS_CSV) as f:
    for row in csv.DictReader(f):
        sid = int(row['sample_id'])
        preds_by_id[sid] = row
        if row.get('gold_function','').strip():
            gold_func_map[sid] = row['gold_function'].strip()

v01_by_id = {int(x['sample_id']): x for x in json.load(open(V01_GEN))}
v02_by_id = {int(x['sample_id']): x for x in json.load(open(V02_GEN))}

# 평가 대상: sample_id 0-299 중 gold_function 있는 것 (전부 300개)
eval_ids = sorted([sid for sid in range(300) if sid in gold_func_map])
print(f"  평가 샘플: {len(eval_ids)}개")

# ─── 이미 완료된 결과 로드 (재개 지원) ────────────────────────────────────────
done_results = {}
if os.path.exists(OUT_GEN):
    prev = json.load(open(OUT_GEN))
    if isinstance(prev, dict) and 'results' in prev:
        for r in prev['results']:
            done_results[r['sample_id']] = r
        print(f"  재개: {len(done_results)}개 이미 완료")

remaining_ids = [sid for sid in eval_ids if sid not in done_results]
print(f"  신규 추론 필요: {len(remaining_ids)}개")

# ─── KB 캐시 로드 + FAISS 구축 ────────────────────────────────────────────────
print("[2] KB 로드 (캐시 우선)...")
import faiss as faiss_lib
from sentence_transformers import SentenceTransformer

cache_valid = os.path.exists(os.path.join(KB_CACHE, "meta.json"))
if cache_valid:
    print("  캐시 HIT")
    enc_tmp = None
else:
    print("  캐시 MISS — 임베딩 계산...")
    enc_tmp = SentenceTransformer(EMBED_NAME, device=DEVICE)

domain_texts = {}; domain_index = {}
for dom, fname in DOMAIN_FILES.items():
    raw = json.load(open(os.path.join(DATA_DIR, fname)))
    texts = [d.get("output", d.get("gold_answer","")) for d in raw]
    texts = [t for t in texts if len(t) > 20]

    emb_path = os.path.join(KB_CACHE, f"{dom}_embs.npy")
    if cache_valid and os.path.exists(emb_path):
        embs = np.load(emb_path).astype(np.float32)
        print(f"  {dom}: cache {len(texts)} texts")
    else:
        embs = enc_tmp.encode(texts, batch_size=256, normalize_embeddings=True,
                              show_progress_bar=False).astype(np.float32)

    idx = faiss_lib.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    domain_texts[dom] = texts
    domain_index[dom] = idx

if enc_tmp is not None:
    del enc_tmp; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

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

# ─── LLM 로드 ─────────────────────────────────────────────────────────────────
print("[3] LLM 로드 (4-bit)...")
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="auto", trust_remote_code=True)
tokenizer  = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True)
lora_map   = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc",
              "STEM":"lora_stem","CROSS":"lora_cross"}
current_lora = [None]

def switch_lora(domain):
    if current_lora[0] == domain: return
    lora_path = os.path.join(LORA_DIR, lora_map[domain])
    if current_lora[0] is None:
        global base_model
        base_model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=domain)
    else:
        base_model.load_adapter(lora_path, adapter_name=domain)
        base_model.set_adapter(domain)
    current_lora[0] = domain

def generate(query, context, func_label, lora_target):
    switch_lora(lora_target)
    ctx_text = "\n".join([c[:300] for c in context])
    if func_label and func_label in SCAFFOLDS:
        instruction = SCAFFOLDS[func_label]
    else:
        instruction = "provide a comprehensive analytical answer with specific mechanisms and implications"
    prompt = (f"Here is context:\n{ctx_text}\n\n"
              f"Question: {query}\n\n"
              f"{instruction}\n\nAnswer:")
    system = DOMAIN_SYSTEM.get(lora_target, DOMAIN_SYSTEM["CROSS"])
    messages = [{"role":"system","content":system},
                {"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=512,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# ─── 신규 추론 루프 ────────────────────────────────────────────────────────────
rng = random.Random(SEED)
# 전체 random assignments 미리 결정 (재현성)
rand_map = {}
for sid in eval_ids:
    rand_map[sid] = rng.choice(FUNCTION_LABELS)

results = list(done_results.values())

print(f"\n[4] 신규 추론 시작: {len(remaining_ids)}개 샘플 × 3 조건 (oracle/wrong/random)")
for step, sid in enumerate(remaining_ids):
    gt_item  = gt_list[sid]
    pred_row = preds_by_id.get(sid, {})
    gold_func = gold_func_map[sid]
    wrong_func = CYCLIC_WRONG.get(gold_func, "HOW")
    rand_func  = rand_map[sid]

    # 라우팅 정보 (V01 기준 — 동일 context 보장)
    v01_item    = v01_by_id.get(sid, {})
    pred_domain = v01_item.get('pred_domain', pred_row.get('pred_domain_nexroute', 'STEM'))
    is_cross    = v01_item.get('is_cross', pred_row.get('is_cross','false').lower()=='true')
    top1 = pred_row.get('top1', pred_domain)
    top2 = pred_row.get('top2', pred_domain)
    lora_target = "CROSS" if is_cross else (top1 if top1 in lora_map else pred_domain)
    if lora_target not in lora_map: lora_target = 'CROSS'

    query = gt_item['instruction']
    gold  = gt_item['gold_answer']
    context = retrieve(query, lora_target, is_cross, top1, top2)

    row = {
        "sample_id": sid,
        "domain": gt_item.get("expected_domain","?"),
        "is_cross": is_cross,
        "gold_function": gold_func,
        "wrong_function": wrong_func,
        "random_function": rand_func,
        "pred_function_v01": v01_item.get('pred_function','?'),
        "gold_answer": gold,
        "A_oracle": generate(query, context, gold_func, lora_target),
        "C_wrong":   generate(query, context, wrong_func, lora_target),
        "D_random":  generate(query, context, rand_func, lora_target),
        # B/E from existing gen files
        "B_predicted": v01_by_id.get(sid, {}).get('pred_answer',''),
        "E_none":       v02_by_id.get(sid, {}).get('pred_answer',''),
    }
    results.append(row)

    if (step+1) % 10 == 0:
        print(f"  [{step+1}/{len(remaining_ids)}] 완료 (total done={len(results)})")
        json.dump({"partial": True, "n_completed": len(results), "results": results},
                  open(OUT_GEN, 'w'), indent=2, ensure_ascii=False)

# B/E 채우기: done_results에 B/E 없는 경우
for r in results:
    sid = r['sample_id']
    if 'B_predicted' not in r or not r['B_predicted']:
        r['B_predicted'] = v01_by_id.get(sid,{}).get('pred_answer','')
    if 'E_none' not in r or not r['E_none']:
        r['E_none'] = v02_by_id.get(sid,{}).get('pred_answer','')

json.dump({"partial": False, "n_total": len(results), "results": results},
          open(OUT_GEN, 'w'), indent=2, ensure_ascii=False)
print(f"\n[4] 생성 완료. 저장: {OUT_GEN}")

# ─── 메트릭 계산 ──────────────────────────────────────────────────────────────
print("\n[5] 메트릭 계산 (ROUGE-L, BERTScore, BLEU-4, 길이)...")

del base_model; gc.collect()
if DEVICE == "cuda": torch.cuda.empty_cache()

import nltk; nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import sacrebleu

rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# BERTScore
from bert_score import score as bs_score

CONDS = ["A_oracle","B_predicted","C_wrong","D_random","E_none"]
COND_LABELS = {
    "A_oracle": "Oracle (gold func)",
    "B_predicted": "Predicted (V01, DeBERTa)",
    "C_wrong": "Wrong (cyclic shift)",
    "D_random": "Random",
    "E_none": "None / no scaffold (V02)",
}

# Collect texts for BERTScore (batch)
all_preds = {c: [] for c in CONDS}
all_refs  = []
all_sids  = []

for r in results:
    all_refs.append(r['gold_answer'])
    all_sids.append(r['sample_id'])
    for c in CONDS:
        all_preds[c].append(r.get(c,'') or '')

print("  BERTScore 계산 중...")
bs_results = {}
for cond in CONDS:
    preds_list = all_preds[cond]
    P, R, F = bs_score(preds_list, all_refs, lang="en", model_type="roberta-large",
                        device=DEVICE, verbose=False, batch_size=16)
    bs_results[cond] = F.numpy().tolist()
    print(f"  {cond}: BERTScore mean={np.mean(bs_results[cond]):.4f}")

# Compute per-sample metrics
print("  ROUGE-L / BLEU-4 / 길이 계산...")
per_sample = []
for i, r in enumerate(results):
    sid    = r['sample_id']
    gold   = r['gold_answer']
    domain = r.get('domain','?')
    is_cross = r.get('is_cross', False)

    row_metrics = {"sample_id": sid, "domain": domain, "is_cross": is_cross,
                   "gold_function": r.get('gold_function',''),
                   "pred_function_v01": r.get('pred_function_v01','')}

    for cond in CONDS:
        pred = r.get(cond,'') or ''
        rl = rscorer.score(gold, pred)['rougeL'].fmeasure * 100 if pred else 0.0
        bleu = sacrebleu.sentence_bleu(pred, [gold]).score if pred else 0.0
        words = len(pred.split())
        sents = len(sent_tokenize(pred)) if pred else 0
        bs_f = bs_results[cond][i]
        row_metrics[cond] = {
            "rouge_l": round(rl,3),
            "bertscore_f": round(float(bs_f*100),3),
            "bleu4": round(bleu,3),
            "n_words": words,
            "n_sents": sents,
        }
    per_sample.append(row_metrics)

# ─── 집계 ─────────────────────────────────────────────────────────────────────
print("\n[6] 집계...")

def aggregate(rows_subset, cond):
    if not rows_subset: return {}
    metrics = rows_subset[0][cond].keys()
    return {m: round(np.mean([r[cond][m] for r in rows_subset if cond in r]),3)
            for m in metrics}

summary = {}
for cond in CONDS:
    all_rows    = [r for r in per_sample if cond in r]
    cross_rows  = [r for r in all_rows if r.get('is_cross')]
    single_rows = [r for r in all_rows if not r.get('is_cross')]
    summary[cond] = {
        "label": COND_LABELS[cond],
        "n": len(all_rows),
        "overall": aggregate(all_rows, cond),
        "cross": aggregate(cross_rows, cond),
        "single": aggregate(single_rows, cond),
    }

print("\n" + "="*70)
print("Function Axis Ablation 결과 요약 (clean-300, gold_function 보유 샘플)")
print("="*70)
print(f"{'Condition':<30} {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8} {'Words':>7} {'Sents':>7}")
for cond in CONDS:
    s = summary[cond]['overall']
    print(f"  {COND_LABELS[cond]:<28} {s.get('rouge_l',0):>8.2f} {s.get('bertscore_f',0):>10.2f} "
          f"{s.get('bleu4',0):>8.2f} {s.get('n_words',0):>7.0f} {s.get('n_sents',0):>7.1f}")

print("\n[CROSS subset]")
for cond in CONDS:
    s = summary[cond]['cross']
    if s:
        print(f"  {COND_LABELS[cond]:<28} ROUGE-L={s.get('rouge_l',0):.2f}  "
              f"BERTScore={s.get('bertscore_f',0):.2f}  BLEU-4={s.get('bleu4',0):.2f}")

print("\n[Single-domain subset]")
for cond in CONDS:
    s = summary[cond]['single']
    if s:
        print(f"  {COND_LABELS[cond]:<28} ROUGE-L={s.get('rouge_l',0):.2f}  "
              f"BERTScore={s.get('bertscore_f',0):.2f}  BLEU-4={s.get('bleu4',0):.2f}")

out = {
    "n_samples": len(results),
    "eval_scope": "clean-300 with gold_function (sample_id 0-299)",
    "B_predicted_source": "v01_gen_500.json (DeBERTa dual-axis router prediction)",
    "E_none_source": "v02_gen_500.json (V02, w/o function scaffold)",
    "summary": summary,
    "per_sample": per_sample,
}
json.dump(out, open(OUT_SCORE, 'w'), indent=2, ensure_ascii=False)
print(f"\n저장: {OUT_SCORE}")
print("블록 2 생성+메트릭 완료. Hallucination은 별도 패스 필요.")
