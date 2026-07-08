"""
기존 ood_results.csv의 generated_answer / gold_answer로
bert-large-uncased BERTScore를 재계산해서 덮어씁니다.
"""

import csv
import os
import torch
from bert_score import BERTScorer

OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(OUT_DIR, "ood_results.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "ood_summary.csv")

RESULT_COLS = ["id", "benchmark", "gold_domain", "system", "question",
               "gold_answer", "generated_answer", "pred_domain",
               "rougeL", "bertscore", "bleu4"]

print("[*] BERTScorer 로드 (bert-large-uncased)...")
bs = BERTScorer(
    model_type="bert-large-uncased",
    lang="en",
    rescale_with_baseline=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print("[*] BERTScorer 준비 완료")

# 기존 결과 로드
with open(RESULT_PATH, encoding="utf-8") as f:
    records = list(csv.DictReader(f))

print(f"[*] {len(records)}개 레코드 로드")

# valid 레코드만 배치 계산
valid = [r for r in records if r["generated_answer"] not in ("[FAILED]", "")]
print(f"[*] 유효 레코드: {len(valid)}개 — BERTScore 계산 중...")

preds = [r["generated_answer"] for r in valid]
refs  = [r["gold_answer"]      for r in valid]

_, _, F = bs.score(preds, refs)
scores = [round(float(f) * 100, 4) for f in F]

for r, s in zip(valid, scores):
    r["bertscore"] = s

# 저장
with open(RESULT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
    writer.writeheader()
    writer.writerows(records)
print(f"[*] 결과 저장: {RESULT_PATH}")

# 집계
from collections import defaultdict
import numpy as np

groups = defaultdict(lambda: {"rougeL": [], "bertscore": [], "bleu4": [],
                               "routing_correct": [], "n": 0})
for r in valid:
    key = (r["benchmark"], r["system"])
    g   = groups[key]
    g["rougeL"].append(float(r["rougeL"]))
    g["bertscore"].append(float(r["bertscore"]))
    g["bleu4"].append(float(r["bleu4"]))
    g["n"] += 1
    if r["system"] == "NexRoute-7B":
        g["routing_correct"].append(
            1 if r["pred_domain"] == r["gold_domain"] else 0
        )

print("\n" + "=" * 75)
print(f"  {'Benchmark':<12} {'System':<14} {'N':>4}  "
      f"{'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8} {'RouteAcc':>10}")
print("-" * 75)

summary_rows = []
for (bench, system), g in sorted(groups.items()):
    avg_r  = round(float(np.mean(g["rougeL"])),    2)
    avg_b  = round(float(np.mean(g["bertscore"])), 2)
    avg_bl = round(float(np.mean(g["bleu4"])),     2)
    acc    = (round(float(np.mean(g["routing_correct"])) * 100, 1)
              if g["routing_correct"] else "—")

    print(f"  {bench:<12} {system:<14} {g['n']:>4}  "
          f"{avg_r:>8.2f} {avg_b:>10.2f} {avg_bl:>8.2f} {str(acc):>10}")

    summary_rows.append({
        "benchmark":   bench,
        "system":      system,
        "n":           g["n"],
        "rouge_l":     avg_r,
        "bertscore":   avg_b,
        "bleu4":       avg_bl,
        "routing_acc": acc if isinstance(acc, float) else "",
    })

print("=" * 75)

with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["benchmark", "system", "n",
                       "rouge_l", "bertscore", "bleu4", "routing_acc"]
    )
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"[*] 집계 저장: {SUMMARY_PATH}")
print("\n[완료]")
