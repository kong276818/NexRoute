"""
proxy_calibration_entropy.py — Phase 4
Router confidence calibration: Shannon entropy vs. routing accuracy.

Runs DeBERTa router on all 500 GT samples.
For each sample, computes:
  H_domain = -Σ p_i * log(p_i)   (Shannon entropy of normalized domain scores)

Splits into:
  - correctly_routed: pred_domain == expected_domain
  - misrouted:        pred_domain != expected_domain

Tests: do misrouted samples have significantly higher entropy?
If yes → entropy can be used as an abstention/human-review trigger.

NOTE: This is a POST-HOC analysis, not a deployed abstention mechanism.
Function Axis does NOT have a learned softmax output in the current
pipeline (it is a deterministic rule-based mapping), so only Domain Axis
entropy is analyzed here.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from transformers import AutoModel, AutoTokenizer

GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
ROUTER_PT = "/home/sclab/moe_model/router_deberta_best.pt"
BERT_NAME = "microsoft/deberta-large"
OUT_PATH  = "/home/sclab/moe_model/results/full_ablation/calibration_entropy.json"

DOMAINS = ["LAW", "MED", "SOC", "STEM"]
TAU, DELTA = 0.30, 0.40


class NexRouteRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(BERT_NAME)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(512, 4),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.encoder(**kw)
        return self.classifier(self.dropout(out.last_hidden_state[:, 0, :]))


def shannon_entropy(probs: np.ndarray) -> float:
    """H = -Σ p log(p), with p normalized via softmax."""
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))


print("Loading DeBERTa router ...")
device = torch.device("cuda")
router = NexRouteRouter().to(device)
state  = torch.load(ROUTER_PT, map_location=device)
if any(k.startswith("module.") for k in state.keys()):
    state = {k[7:]: v for k, v in state.items()}
router.load_state_dict(state, strict=False)
router.eval()
tok = AutoTokenizer.from_pretrained(BERT_NAME)
print("Router ready")

gt = json.load(open(GT_PATH))
print(f"GT samples: {len(gt)}")

records = []
for item in gt:
    query  = item["instruction"].strip()
    true_d = item["expected_domain"]

    enc = tok(query, max_length=256, truncation=True,
               padding="max_length", return_tensors="pt").to(device)
    kw  = {"input_ids": enc["input_ids"],
           "attention_mask": enc["attention_mask"]}
    if "token_type_ids" in enc:
        kw["token_type_ids"] = enc["token_type_ids"]

    with torch.no_grad():
        raw = torch.sigmoid(router(**kw)).squeeze().cpu().numpy()  # shape [4]

    # Softmax-normalise for entropy (convert sigmoid scores → probability dist)
    probs    = np.exp(raw) / np.exp(raw).sum()
    h_domain = shannon_entropy(probs)

    sorted_d = sorted(zip(DOMAINS, raw), key=lambda x: -x[1])
    s1, s2   = sorted_d[0][1], sorted_d[1][1]
    is_cross = bool(s2 >= TAU and s1 - s2 <= DELTA)
    pred_d   = "CROSS" if is_cross else sorted_d[0][0]

    correct = (pred_d == true_d)

    records.append({
        "query":        query,
        "true_domain":  true_d,
        "pred_domain":  pred_d,
        "correct":      correct,
        "raw_scores":   {d: round(float(v), 6) for d, v in zip(DOMAINS, raw)},
        "probs":        {d: round(float(p), 6) for d, p in zip(DOMAINS, probs)},
        "entropy":      round(h_domain, 6),
        "is_cross":     is_cross,
    })

# ── Aggregate ─────────────────────────────────────────────────────────────
correct_h   = [r["entropy"] for r in records if r["correct"]]
misrouted_h = [r["entropy"] for r in records if not r["correct"]]

n_correct   = len(correct_h)
n_misrouted = len(misrouted_h)
acc         = n_correct / len(records) * 100

print(f"\nOverall accuracy:  {acc:.1f}%")
print(f"Correctly routed:  {n_correct}  (mean H={np.mean(correct_h):.4f})")
print(f"Misrouted:         {n_misrouted}  (mean H={np.mean(misrouted_h):.4f})")

# Mann-Whitney U test (non-parametric, more robust than t-test for entropy)
if misrouted_h:
    stat, pval = stats.mannwhitneyu(
        misrouted_h, correct_h, alternative="greater")
    print(f"Mann-Whitney U:    U={stat:.1f}, p={pval:.4f}")
    significant = pval < 0.05
    direction_holds = np.mean(misrouted_h) > np.mean(correct_h)
else:
    stat, pval, significant, direction_holds = None, None, None, None

# ── Per-domain breakdown ───────────────────────────────────────────────────
per_domain = {}
for dom in ["LAW", "MED", "SOC", "STEM", "CROSS"]:
    subset = [r for r in records if r["true_domain"] == dom]
    if not subset:
        continue
    c_h = [r["entropy"] for r in subset if r["correct"]]
    m_h = [r["entropy"] for r in subset if not r["correct"]]
    per_domain[dom] = {
        "n":             len(subset),
        "n_correct":     len(c_h),
        "n_misrouted":   len(m_h),
        "mean_H_correct":   round(float(np.mean(c_h)), 4) if c_h else None,
        "mean_H_misrouted": round(float(np.mean(m_h)), 4) if m_h else None,
    }
    print(f"  {dom}: correct H={per_domain[dom]['mean_H_correct']}, "
          f"misrouted H={per_domain[dom]['mean_H_misrouted']}")

# ── Entropy percentile thresholds ─────────────────────────────────────────
# If we set an abstention threshold at H > P90 of ALL samples, what fraction
# of misrouted samples fall above it?
all_H      = [r["entropy"] for r in records]
p90_thresh = float(np.percentile(all_H, 90))
above_p90_misrouted  = sum(1 for r in records
                            if not r["correct"] and r["entropy"] > p90_thresh)
above_p90_correct    = sum(1 for r in records
                            if r["correct"] and r["entropy"] > p90_thresh)
abstention_precision = (above_p90_misrouted /
                        (above_p90_misrouted + above_p90_correct)
                        if (above_p90_misrouted + above_p90_correct) else 0)
abstention_recall    = (above_p90_misrouted / n_misrouted
                        if n_misrouted else 0)

print(f"\nP90 entropy threshold: {p90_thresh:.4f}")
print(f"  Above P90 → misrouted: {above_p90_misrouted}, "
      f"correct: {above_p90_correct}")
print(f"  Precision (of flagged):  {abstention_precision:.4f}")
print(f"  Recall (of misrouted):   {abstention_recall:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────
output = {
    "note": (
        "Domain Axis Shannon entropy (H = -Σ p log p on softmax-normalised "
        "sigmoid scores). Function Axis has no learned softmax output "
        "(deterministic rule-based), so only Domain Axis entropy is analysed. "
        "This is a post-hoc analysis; no deployed abstention mechanism is implied."
    ),
    "overall": {
        "n_total":          len(records),
        "accuracy_pct":     round(acc, 1),
        "n_correct":        n_correct,
        "n_misrouted":      n_misrouted,
        "mean_H_correct":   round(float(np.mean(correct_h)), 4),
        "mean_H_misrouted": round(float(np.mean(misrouted_h)), 4) if misrouted_h else None,
        "mannwhitney_U":    float(stat) if stat is not None else None,
        "mannwhitney_p":    float(pval) if pval is not None else None,
        "significant_p05":  significant,
        "direction_holds (misrouted H > correct H)": direction_holds,
    },
    "entropy_thresholding": {
        "threshold":               round(p90_thresh, 4),
        "threshold_percentile":    90,
        "above_threshold_misrouted": above_p90_misrouted,
        "above_threshold_correct":   above_p90_correct,
        "abstention_precision":      round(abstention_precision, 4),
        "abstention_recall_of_misrouted": round(abstention_recall, 4),
    },
    "per_domain":  per_domain,
    "per_sample":  records,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)): return o.item()
        if isinstance(o, np.floating): return float(o)
        return super().default(o)

json.dump(output, open(OUT_PATH, "w"), indent=2, ensure_ascii=False, cls=_Enc)
print(f"\nSaved: {OUT_PATH}")

# ── Print Table C ─────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  TABLE C  —  Calibration Entropy (correctly-routed vs. misrouted)")
print("  NOTE: Post-hoc analysis only; not a deployed abstention mechanism")
print("=" * 65)
print(f"{'Domain':<8} {'N':>4} {'N_corr':>7} {'N_mis':>6} "
      f"{'H_correct':>10} {'H_misrouted':>12}")
print("-" * 60)
for dom in ["LAW", "MED", "SOC", "STEM", "CROSS"]:
    if dom not in per_domain:
        continue
    d = per_domain[dom]
    hc = f"{d['mean_H_correct']:.4f}" if d['mean_H_correct'] is not None else "N/A"
    hm = f"{d['mean_H_misrouted']:.4f}" if d['mean_H_misrouted'] is not None else "N/A"
    print(f"  {dom:<6} {d['n']:>4} {d['n_correct']:>7} {d['n_misrouted']:>6} "
          f"{hc:>10} {hm:>12}")
print("-" * 60)
ov = output["overall"]
hm_str = f"{ov['mean_H_misrouted']:.4f}" if ov['mean_H_misrouted'] else "N/A"
print(f"  {'ALL':<6} {ov['n_total']:>4} {ov['n_correct']:>7} "
      f"{ov['n_misrouted']:>6} {ov['mean_H_correct']:>10.4f} {hm_str:>12}")
print()
p_str = f"{ov['mannwhitney_p']:.4f}" if ov['mannwhitney_p'] is not None else "N/A"
print(f"  Mann-Whitney U={ov['mannwhitney_U']},  p={p_str},  "
      f"significant={ov['significant_p05']}")
print(f"  Direction (misrouted H > correct H): "
      f"{ov['direction_holds (misrouted H > correct H)']}")
et = output["entropy_thresholding"]
print(f"\n  P90 abstention threshold = {et['threshold']:.4f}")
print(f"  Precision={et['abstention_precision']:.4f},  "
      f"Recall(of misrouted)={et['abstention_recall_of_misrouted']:.4f}")
