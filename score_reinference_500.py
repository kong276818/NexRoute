"""
score_reinference_500.py
========================
재추론 결과 채점: ROUGE-L, BERTScore, BLEU-4
출력:
  scores_500_{system}.csv  — 샘플별 지표
  reinference_summary.json — 집계 (Clean-300 / Exposed-200 / CROSS / Single)
  memorization_gap.json    — NexRoute vs 베이스라인 평균 gap

규칙: 모든 수치는 파일에서 계산; 하드코딩 금지
"""
import os, json, csv
import numpy as np
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

GT_PATH  = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_DIR  = "/home/sclab/moe_model/reinference_results"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEMS  = ["naive_rag", "tfidf_router", "hyde", "crag", "nexroute"]

CLEAN_INDICES   = set(range(0,   300))   # 인덱스 0-299 (LoRA 독립)
EXPOSED_INDICES = set(range(300, 500))   # 인덱스 300-499 (LoRA 노출)

with open(GT_PATH) as f:
    gt_list = json.load(f)


def compute_row_metrics(gold: str, pred: str) -> dict:
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = rscorer.score(gold, pred)["rougeL"].fmeasure * 100
    bleu = sentence_bleu(
        [nltk.word_tokenize(gold.lower())],
        nltk.word_tokenize(pred.lower()),
        smoothing_function=SmoothingFunction().method1
    ) * 100
    return {"rouge_l": round(rl, 4), "bleu4": round(bleu, 4)}


def compute_bertscore_batch(golds: list, preds: list) -> list:
    _, _, F1 = bert_score_fn(
        preds, golds,
        lang="en",
        model_type="roberta-large",
        rescale_with_baseline=False,
        verbose=False,
        device=DEVICE,
    )
    return [round(float(v) * 100, 4) for v in F1]


def agg(values: list) -> dict:
    arr = [v for v in values if v is not None]
    if not arr:
        return {"mean": None, "n": 0}
    return {"mean": round(float(np.mean(arr)), 4), "n": len(arr)}


summary = {}

for sys_name in SYSTEMS:
    pred_path  = os.path.join(OUT_DIR, f"preds_500_{sys_name}.json")
    score_path = os.path.join(OUT_DIR, f"scores_500_{sys_name}.csv")

    if not os.path.exists(pred_path):
        print(f"[SKIP] {sys_name}: {pred_path} 없음")
        continue

    with open(pred_path) as f:
        preds = json.load(f)

    print(f"\n[▶] {sys_name}  ({len(preds)} samples)")

    # ── 배치 BERTScore ────────────────────────────────────────────────────────
    valid_rows = []
    for item in preds:
        idx  = item["idx"]
        gt   = gt_list[idx]
        gold = item.get("gold_answer", gt.get("gold_answer", "")).strip()
        pred = item.get("pred_answer", "").strip()
        if not gold or not pred or pred in ("[FAILED]", "[GENERATION_FAILED]"):
            valid_rows.append(None)
        else:
            valid_rows.append((idx, gold, pred,
                               item.get("expected_domain", gt.get("expected_domain","")),
                               item.get("is_cross", gt.get("expected_domain","") == "CROSS")))

    valid_only = [r for r in valid_rows if r is not None]
    golds = [r[1] for r in valid_only]
    pred_texts = [r[2] for r in valid_only]

    print(f"  BERTScore 계산 ({len(valid_only)}개)...")
    bs_scores = compute_bertscore_batch(golds, pred_texts)
    bs_map    = {r[0]: bs for r, bs in zip(valid_only, bs_scores)}

    # ── 샘플별 CSV 저장 ───────────────────────────────────────────────────────
    rows = []
    for item in preds:
        idx       = item["idx"]
        gt_item   = gt_list[idx]
        gold      = item.get("gold_answer", gt_item.get("gold_answer", "")).strip()
        pred      = item.get("pred_answer", "").strip()
        domain    = item.get("expected_domain", gt_item.get("expected_domain", ""))
        is_cross  = (domain == "CROSS")
        split     = "clean" if idx in CLEAN_INDICES else "exposed"

        if not gold or not pred or pred in ("[FAILED]", "[GENERATION_FAILED]"):
            rows.append({"idx": idx, "domain": domain, "is_cross": is_cross,
                         "split": split, "rouge_l": None, "bertscore": None,
                         "bleu4": None, "failed": True})
            continue

        rm  = compute_row_metrics(gold, pred)
        bs  = bs_map.get(idx)
        rows.append({
            "idx": idx, "domain": domain, "is_cross": is_cross,
            "split": split, "rouge_l": rm["rouge_l"], "bertscore": bs,
            "bleu4": rm["bleu4"], "failed": False
        })

    with open(score_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx","domain","is_cross","split",
                                               "rouge_l","bertscore","bleu4","failed"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {score_path}")

    # ── 집계 ─────────────────────────────────────────────────────────────────
    def group_metrics(mask_fn):
        sub = [r for r in rows if mask_fn(r) and not r["failed"]]
        return {
            "rouge_l":   agg([r["rouge_l"]   for r in sub]),
            "bertscore": agg([r["bertscore"] for r in sub]),
            "bleu4":     agg([r["bleu4"]     for r in sub]),
        }

    summary[sys_name] = {
        "overall":          group_metrics(lambda r: True),
        "clean":            group_metrics(lambda r: r["split"] == "clean"),
        "exposed":          group_metrics(lambda r: r["split"] == "exposed"),
        "cross":            group_metrics(lambda r: r["is_cross"]),
        "single":           group_metrics(lambda r: not r["is_cross"]),
        "clean_cross":      group_metrics(lambda r: r["split"]=="clean"   and r["is_cross"]),
        "clean_single":     group_metrics(lambda r: r["split"]=="clean"   and not r["is_cross"]),
        "exposed_cross":    group_metrics(lambda r: r["split"]=="exposed" and r["is_cross"]),
        "exposed_single":   group_metrics(lambda r: r["split"]=="exposed" and not r["is_cross"]),
        "n_total":   len(rows),
        "n_valid":   sum(1 for r in rows if not r["failed"]),
        "n_failed":  sum(1 for r in rows if r["failed"]),
    }

    for g, m in summary[sys_name].items():
        if isinstance(m, dict) and "rouge_l" in m:
            rl  = m["rouge_l"]["mean"]
            bs  = m["bertscore"]["mean"]
            n   = m["rouge_l"]["n"]
            print(f"    {g:<18} n={n:<5} RL={rl}  BS={bs}")

# ── 메모리제이션 gap 계산 ──────────────────────────────────────────────────────
print("\n[*] 메모리제이션 gap 계산...")
gap_results = {}

def get_gap(sys_name: str, metric: str = "bertscore") -> dict | None:
    s = summary.get(sys_name)
    if not s:
        return None
    exp_val   = s.get("exposed",     {}).get(metric, {}).get("mean")
    clean_val = s.get("clean",       {}).get(metric, {}).get("mean")
    if exp_val is None or clean_val is None:
        return None
    return {"exposed": exp_val, "clean": clean_val,
            "gap": round(exp_val - clean_val, 4)}

baselines = ["naive_rag", "tfidf_router", "hyde", "crag"]
for metric in ["bertscore", "rouge_l"]:
    nexroute_gap = get_gap("nexroute", metric)
    baseline_gaps = [get_gap(b, metric) for b in baselines if get_gap(b, metric) is not None]

    if nexroute_gap and baseline_gaps:
        avg_baseline_gap = round(np.mean([g["gap"] for g in baseline_gaps]), 4)
        net_memorization = round(nexroute_gap["gap"] - avg_baseline_gap, 4)
        gap_results[metric] = {
            "nexroute_gap":       nexroute_gap,
            "baseline_avg_gap":   avg_baseline_gap,
            "baseline_individual": {b: get_gap(b, metric) for b in baselines},
            "net_memorization_effect": net_memorization,
        }
        print(f"  [{metric}] NexRoute gap={nexroute_gap['gap']:+.4f}  "
              f"Baseline avg gap={avg_baseline_gap:+.4f}  "
              f"Net memorization={net_memorization:+.4f}")

# ── 저장 ────────────────────────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "reinference_summary.json")
gap_path     = os.path.join(OUT_DIR, "memorization_gap.json")

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
with open(gap_path, "w") as f:
    json.dump(gap_results, f, indent=2)

print(f"\n[✓] 저장:")
print(f"  {summary_path}")
print(f"  {gap_path}")

# ── 최종 요약 테이블 ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"{'System':<16}  {'Clean RL':>8} {'Clean BS':>9}  {'Exposed RL':>10} {'Exposed BS':>10}  {'Gap BS':>7}")
print("-"*80)
for sys_name in SYSTEMS:
    s = summary.get(sys_name)
    if not s:
        continue
    c_rl = s["clean"]["rouge_l"]["mean"]
    c_bs = s["clean"]["bertscore"]["mean"]
    e_rl = s["exposed"]["rouge_l"]["mean"]
    e_bs = s["exposed"]["bertscore"]["mean"]
    gap  = gap_results.get("bertscore", {}).get("nexroute_gap" if sys_name=="nexroute" else None, {})
    g_bs = round(e_bs - c_bs, 2) if e_bs and c_bs else None
    marker = " ◀" if sys_name == "nexroute" else ""
    print(f"  {sys_name:<16} {c_rl or 'N/A':>8} {c_bs or 'N/A':>9}  {e_rl or 'N/A':>10} {e_bs or 'N/A':>10}  {g_bs or 'N/A':>7}{marker}")
print("="*80)
