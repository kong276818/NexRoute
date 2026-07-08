import json
import torch
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)

RESULT_PATH = "/home/sclab/moe_model/eval_results.json"

# ── 데이터 로드 ───────────────────────────────────
with open(RESULT_PATH) as f:
    data = json.load(f)

# GENERATION_FAILED 제거
data = [x for x in data if "[GENERATION_FAILED]" not in x.get("pred_answer", "")]
print(f"평가 샘플: {len(data)}개")

golds = [x["gold_answer"] for x in data]
preds = [x["pred_answer"] for x in data]

# ── 1. ROUGE-L ────────────────────────────────────
print("\n[1/4] ROUGE-L 계산 중...")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_scores = [
    scorer.score(g, p)["rougeL"].fmeasure
    for g, p in zip(golds, preds)
]
rouge_avg = np.mean(rouge_scores)

# ── 2. BERTScore ──────────────────────────────────
print("[2/4] BERTScore 계산 중...")
P, R, F1 = bert_score(
    preds, golds,
    lang="en",
    model_type="bert-base-uncased",
    verbose=False
)
bertscore_avg = F1.mean().item()

# ── 3. BLEU-4 ─────────────────────────────────────
print("[3/4] BLEU-4 계산 중...")
smooth = SmoothingFunction().method1
bleu_scores = []
for g, p in zip(golds, preds):
    ref   = [nltk.word_tokenize(g.lower())]
    hyp   = nltk.word_tokenize(p.lower())
    bleu_scores.append(sentence_bleu(ref, hyp, smoothing_function=smooth))
bleu_avg = np.mean(bleu_scores)

# ── 4. CROSS 라우팅 정확도 ────────────────────────
print("[4/4] CROSS 라우팅 분석 중...")
cross_gt   = [x for x in data if "CROSS" in x.get("domain_gt", "")]
cross_pred = [x for x in data if x.get("is_cross", False)]
total      = len(data)

# GT에 CROSS 라벨 없으면 → pred 기준으로만 분석
cross_detected     = len(cross_pred)
cross_ratio        = cross_detected / total * 100

# 도메인별 성능
domain_scores = defaultdict(lambda: {"rouge": [], "bert": [], "bleu": []})
for i, item in enumerate(data):
    domain = item["domain_pred"][0] if not item["is_cross"] else "CROSS"
    domain_scores[domain]["rouge"].append(rouge_scores[i])
    domain_scores[domain]["bert"].append(F1[i].item())
    domain_scores[domain]["bleu"].append(bleu_scores[i])

# ── 결과 출력 ─────────────────────────────────────
print("\n" + "="*60)
print("              전체 평가 결과")
print("="*60)
print(f"{'지표':<20} {'점수':>10}")
print("-"*35)
print(f"{'ROUGE-L':<20} {rouge_avg:>10.4f}")
print(f"{'BERTScore F1':<20} {bertscore_avg:>10.4f}")
print(f"{'BLEU-4':<20} {bleu_avg:>10.4f}")
print(f"{'CROSS 감지율':<20} {cross_ratio:>9.1f}%")
print(f"{'CROSS 샘플 수':<20} {cross_detected:>10}")

print("\n" + "="*60)
print("              도메인별 BERTScore F1")
print("="*60)
print(f"{'도메인':<10} {'샘플수':>6} {'ROUGE-L':>10} {'BERTScore':>10} {'BLEU-4':>10}")
print("-"*50)
for domain in ["LAW", "MED", "SOC", "STEM", "CROSS"]:
    if domain not in domain_scores:
        continue
    d = domain_scores[domain]
    n = len(d["rouge"])
    r = np.mean(d["rouge"])
    b = np.mean(d["bert"])
    bl = np.mean(d["bleu"])
    print(f"{domain:<10} {n:>6} {r:>10.4f} {b:>10.4f} {bl:>10.4f}")

# ── 결과 저장 ─────────────────────────────────────
eval_summary = {
    "total_samples":   len(data),
    "cross_detected":  cross_detected,
    "cross_ratio":     round(cross_ratio, 2),
    "overall": {
        "rouge_l":    round(rouge_avg, 4),
        "bertscore":  round(bertscore_avg, 4),
        "bleu_4":     round(bleu_avg, 4),
    },
    "by_domain": {
        domain: {
            "n":         len(v["rouge"]),
            "rouge_l":   round(np.mean(v["rouge"]), 4),
            "bertscore": round(np.mean(v["bert"]), 4),
            "bleu_4":    round(np.mean(v["bleu"]), 4),
        }
        for domain, v in domain_scores.items()
    }
}

with open("/home/sclab/moe_model/eval_summary.json", "w") as f:
    json.dump(eval_summary, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료: /home/sclab/moe_model/eval_summary.json")