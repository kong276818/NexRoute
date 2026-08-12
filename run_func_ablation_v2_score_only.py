"""
run_func_ablation_v2_score_only.py
===================================
생성 결과(function_axis_ablation_v2_gen.json)에서 메트릭만 계산.
모델 로드 없이 ROUGE-L, BERTScore, BLEU-4만 계산.
"""
import json, gc
import numpy as np
import torch

GEN_PATH  = "/home/sclab/moe_model/results/function_axis_ablation_v2_gen.json"
OUT_SCORE = "/home/sclab/moe_model/results/function_axis_ablation_v2_scores.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] DEVICE={DEVICE}")

data = json.load(open(GEN_PATH))
results = data["results"]
print(f"[load] n_samples={len(results)}")

import nltk; nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bs_score

rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

CONDS = ["A_oracle","B_predicted","C_wrong","D_random","E_none"]
COND_LABELS = {
    "A_oracle":    "Oracle (gold func)",
    "B_predicted": "Predicted (V01, DeBERTa)",
    "C_wrong":     "Wrong (cyclic shift)",
    "D_random":    "Random",
    "E_none":      "None / no scaffold (V02)",
}

all_refs  = [r['gold_answer'] for r in results]
all_preds = {c: [r.get(c,'') or '' for r in results] for c in CONDS}

print("  BERTScore 계산 중...")
bs_results = {}
for cond in CONDS:
    P, R, F = bs_score(all_preds[cond], all_refs, lang="en",
                       model_type="roberta-large",
                       device=DEVICE, verbose=False, batch_size=16)
    bs_results[cond] = F.numpy().tolist()
    print(f"  {cond}: BERTScore mean={np.mean(bs_results[cond]):.4f}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("  ROUGE-L / BLEU-4 / 길이 계산...")
per_sample = []
for i, r in enumerate(results):
    sid      = r['sample_id']
    gold     = r['gold_answer']
    domain   = r.get('domain', '?')
    is_cross = r.get('is_cross', False)

    row = {"sample_id": sid, "domain": domain, "is_cross": is_cross,
           "gold_function": r.get('gold_function',''),
           "pred_function_v01": r.get('pred_function_v01','')}

    for cond in CONDS:
        pred = r.get(cond,'') or ''
        rl   = rscorer.score(gold, pred)['rougeL'].fmeasure * 100 if pred else 0.0
        bleu = sacrebleu.sentence_bleu(pred, [gold]).score if pred else 0.0
        row[cond] = {
            "rouge_l":     round(rl, 3),
            "bertscore_f": round(float(bs_results[cond][i]) * 100, 3),
            "bleu4":       round(bleu, 3),
            "n_words":     len(pred.split()),
            "n_sents":     len(sent_tokenize(pred)) if pred else 0,
        }
    per_sample.append(row)

def aggregate(rows, cond):
    if not rows: return {}
    metrics = rows[0][cond].keys()
    return {m: round(np.mean([r[cond][m] for r in rows if cond in r]), 3)
            for m in metrics}

summary = {}
for cond in CONDS:
    all_r    = [r for r in per_sample if cond in r]
    cross_r  = [r for r in all_r if r.get('is_cross')]
    single_r = [r for r in all_r if not r.get('is_cross')]
    summary[cond] = {
        "label":   COND_LABELS[cond],
        "n":       len(all_r),
        "overall": aggregate(all_r, cond),
        "cross":   aggregate(cross_r, cond),
        "single":  aggregate(single_r, cond),
    }

print("\n" + "="*70)
print("Function Axis Ablation v2 — 메트릭 요약")
print("="*70)
print(f"{'Condition':<30} {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8} {'Words':>7}")
for cond in CONDS:
    s = summary[cond]['overall']
    print(f"  {COND_LABELS[cond]:<28} {s.get('rouge_l',0):>8.2f} "
          f"{s.get('bertscore_f',0):>10.2f} {s.get('bleu4',0):>8.2f} "
          f"{s.get('n_words',0):>7.0f}")

print("\n[CROSS]")
for cond in CONDS:
    s = summary[cond]['cross']
    if s:
        print(f"  {COND_LABELS[cond]:<28} ROUGE-L={s.get('rouge_l',0):.2f}  "
              f"BERTScore={s.get('bertscore_f',0):.2f}  BLEU-4={s.get('bleu4',0):.2f}")

print("\n[Single]")
for cond in CONDS:
    s = summary[cond]['single']
    if s:
        print(f"  {COND_LABELS[cond]:<28} ROUGE-L={s.get('rouge_l',0):.2f}  "
              f"BERTScore={s.get('bertscore_f',0):.2f}  BLEU-4={s.get('bleu4',0):.2f}")

out = {
    "n_samples":          len(results),
    "eval_scope":         "clean-300 with gold_function (sample_id 0-299)",
    "B_predicted_source": "v01_gen_500.json (DeBERTa dual-axis router)",
    "E_none_source":      "v02_gen_500.json (V02, w/o function scaffold)",
    "summary":            summary,
    "per_sample":         per_sample,
}
json.dump(out, open(OUT_SCORE, 'w'), indent=2, ensure_ascii=False)
print(f"\n저장: {OUT_SCORE}")
