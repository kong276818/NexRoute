"""
eval_metrics.py — Phase 2: ROUGE-L, BERTScore(roberta-large), BLEU-4, FActScore proxy
Reads all variant generation files, computes metrics per variant × domain, saves CSV + JSON.

FActScore approximation: splits generated answer into sentences; for each sentence,
checks support from gold_answer using BERTScore similarity (cosine on roberta-large
[CLS] embeddings). Reports supported-sentence ratio as FActScore proxy.
This is NOT full FActScore (no atomic decomposition, no external KB retrieval).
It is a reference-grounded sentence-level faithfulness proxy only.
"""

import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer as rs

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

GT_PATH  = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
GEN_DIR  = "/home/sclab/moe_model/results/full_ablation/generations"
OUT_DIR  = "/home/sclab/moe_model/results/full_ablation/metrics"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_DOMAINS = ["LAW", "MED", "SOC", "STEM", "CROSS"]

# ── Load ground-truth instruction→domain map ─────────────────────────────
gt_data = json.load(open(GT_PATH))
instr2domain = {g["instruction"].strip(): g["expected_domain"] for g in gt_data}

# ── Metrics setup ─────────────────────────────────────────────────────────
rouge = rs.RougeScorer(["rougeL"], use_stemmer=True)
smooth = SmoothingFunction().method1

print("Loading BERTScorer (roberta-large, rescale_with_baseline) ...")
bs = BERTScorer(
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print("BERTScorer ready")


def rouge_l(ref: str, hyp: str) -> float:
    return rouge.score(ref, hyp)["rougeL"].fmeasure


def bleu4(ref: str, hyp: str) -> float:
    ref_toks = [nltk.word_tokenize(ref.lower())]
    hyp_toks = nltk.word_tokenize(hyp.lower())
    return sentence_bleu(ref_toks, hyp_toks, smoothing_function=smooth)


# ── FActScore proxy ────────────────────────────────────────────────────────
def factscore_proxy(preds: list[str], refs: list[str]) -> list[float]:
    """
    For each (pred, ref) pair:
    - Split pred into sentences
    - Check each sentence against ref using BERTScore F1 (roberta-large)
    - A sentence is "supported" if max BERTScore F1 vs any ref sentence ≥ 0.75
    - Returns proportion of supported sentences per sample

    NOTE: this is a PROXY, not full FActScore. It checks sentence-level
    grounding in the gold reference, not factual verification against a KB.
    """
    scores = []
    for pred, ref in zip(preds, refs):
        pred_sents = nltk.sent_tokenize(pred)
        ref_sents  = nltk.sent_tokenize(ref)
        if not pred_sents or not ref_sents:
            scores.append(0.0)
            continue
        # BERTScore each pred sentence against all ref sentences
        n_supported = 0
        for ps in pred_sents:
            # replicate ps × len(ref_sents) to batch BERTScore
            _, _, F = bs.score([ps] * len(ref_sents), ref_sents)
            if float(F.max()) >= 0.75:
                n_supported += 1
        scores.append(n_supported / len(pred_sents))
    return scores


# ── Scan generation files ─────────────────────────────────────────────────
gen_files = sorted(f for f in os.listdir(GEN_DIR) if f.endswith(".json"))
print(f"\nFound {len(gen_files)} generation files: {gen_files}")

all_results = {}   # variant_name → per-domain metrics dict

for fname in gen_files:
    vname = fname.replace(".json", "")
    data  = json.load(open(os.path.join(GEN_DIR, fname)))

    # Normalize: ensure expected_domain field
    for item in data:
        if "expected_domain" not in item:
            # Variant 1 reuse: look up from gt map
            q = item.get("query", item.get("instruction", "")).strip()
            item["expected_domain"] = instr2domain.get(q, "UNKNOWN")
        # Filter failures
        if "[GENERATION_FAILED]" in item.get("pred_answer", ""):
            item["pred_answer"] = ""

    valid  = [d for d in data if d.get("pred_answer", "").strip()]
    failed = len(data) - len(valid)
    print(f"\n[{vname}]  N={len(data)}, failed={failed}, valid={len(valid)}")

    # Compute metrics globally then per domain
    by_domain = defaultdict(lambda: {"golds": [], "preds": []})
    for item in valid:
        dom = item["expected_domain"]
        by_domain[dom]["golds"].append(item["gold_answer"])
        by_domain[dom]["preds"].append(item["pred_answer"])

    variant_metrics = {}
    all_golds, all_preds = [], []

    for dom, pairs in by_domain.items():
        golds, preds = pairs["golds"], pairs["preds"]
        all_golds.extend(golds)
        all_preds.extend(preds)

    # BERTScore in one batch (more efficient)
    print("  Computing BERTScore ...")
    _, _, F_all = bs.score(all_preds, all_golds)
    bs_map = {i: float(F_all[i]) for i in range(len(all_preds))}

    print("  Computing FActScore proxy ...")
    factscore_all = factscore_proxy(all_preds, all_golds)

    # Map back to domains
    offset = 0
    all_metrics = {"OVERALL": {"n": 0, "rouge_l": [], "bertscore": [],
                                "bleu_4": [], "factscore_proxy": []}}
    for dom in ALL_DOMAINS:
        if dom not in by_domain:
            continue
        golds, preds = by_domain[dom]["golds"], by_domain[dom]["preds"]
        n = len(golds)
        rl  = [rouge_l(g, p) for g, p in zip(golds, preds)]
        bl  = [bleu4(g, p) for g, p in zip(golds, preds)]
        bsc = [bs_map[offset + i] for i in range(n)]
        fsc = factscore_all[offset: offset + n]
        offset += n

        all_metrics[dom] = {
            "n":               n,
            "rouge_l":         round(float(np.mean(rl)), 4),
            "bertscore":       round(float(np.mean(bsc)), 4),
            "bleu_4":          round(float(np.mean(bl)), 4),
            "factscore_proxy": round(float(np.mean(fsc)), 4),
        }
        for lst, key in [(rl, "rouge_l"), (bsc, "bertscore"),
                          (bl, "bleu_4"), (fsc, "factscore_proxy")]:
            all_metrics["OVERALL"][key] = all_metrics["OVERALL"].get(key, []) + lst
        all_metrics["OVERALL"]["n"] += n

    # Finalize OVERALL
    ov = all_metrics["OVERALL"]
    for key in ["rouge_l", "bertscore", "bleu_4", "factscore_proxy"]:
        ov[key] = round(float(np.mean(ov[key])), 4) if ov[key] else 0.0

    variant_metrics = all_metrics
    all_results[vname] = variant_metrics

    print(f"  OVERALL → ROUGE-L={ov['rouge_l']:.4f}  "
          f"BERTScore={ov['bertscore']:.4f}  "
          f"BLEU-4={ov['bleu_4']:.4f}  "
          f"FActScore~={ov['factscore_proxy']:.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────
json.dump(all_results, open(f"{OUT_DIR}/eval_metrics_all.json", "w"),
          indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT_DIR}/eval_metrics_all.json")

# ── Save CSV ──────────────────────────────────────────────────────────────
metrics = ["rouge_l", "bertscore", "bleu_4", "factscore_proxy"]
rows    = []
for vname, vm in all_results.items():
    for dom in ["OVERALL"] + ALL_DOMAINS:
        if dom not in vm:
            continue
        row = {"variant": vname, "domain": dom}
        row.update({m: vm[dom].get(m, "") for m in metrics})
        row["n"] = vm[dom]["n"]
        rows.append(row)

with open(f"{OUT_DIR}/eval_metrics_all.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "domain", "n"] + metrics)
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved: {OUT_DIR}/eval_metrics_all.csv")

# ── Print markdown Table A ────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("  TABLE A  —  ROUGE-L / BERTScore / BLEU-4 / FActScore~ (per variant × domain)")
print("=" * 80)
header = (f"{'Variant':<42} {'Domain':<7} {'N':>4}  "
          f"{'ROUGE-L':>8} {'BERTSc':>8} {'BLEU-4':>8} {'FActSc~':>8}")
print(header)
print("-" * len(header))
for vname, vm in all_results.items():
    for dom in ["OVERALL"] + ALL_DOMAINS:
        if dom not in vm:
            continue
        d = vm[dom]
        print(f"  {vname:<40} {dom:<7} {d['n']:>4}  "
              f"{d['rouge_l']:>8.4f} {d['bertscore']:>8.4f} "
              f"{d['bleu_4']:>8.4f} {d['factscore_proxy']:>8.4f}")
    print()
