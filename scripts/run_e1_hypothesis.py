"""
E1 Hypothesis Test: Function Heterogeneity × Scaffold Contribution
Proposition 1 verification using existing v01/v02 generation results (no GPU required).

v01 = NexRoute full (with function scaffold)
v02 = NexRoute without function scaffold (NEUTRAL scaffold)

Key question: Does the ROUGE-L gap (v01 - v02) increase when function
heterogeneity within a domain is higher? (Proposition 1 prediction)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = "results/ablation"
OUTPUT_DIR = "results/e1_hypothesis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
v01 = pd.read_csv(f"{RESULTS_DIR}/v01_scores_per_sample.csv")
v02 = pd.read_csv(f"{RESULTS_DIR}/v02_scores_per_sample.csv")

# Align columns
v02 = v02.rename(columns={"bertscore_f": "bertscore_f_v02", "rouge_l": "rouge_l_v02", "bleu4": "bleu4_v02"})
v01 = v01.rename(columns={"bertscore_f": "bertscore_f_v01", "rouge_l": "rouge_l_v01", "bleu4": "bleu4_v01"})

merged = pd.merge(
    v01[["sample_id", "domain", "is_cross_gt", "is_clean300", "pred_function",
         "rouge_l_v01", "bertscore_f_v01"]],
    v02[["sample_id", "rouge_l_v02", "bertscore_f_v02"]],
    on="sample_id"
)
merged["scaffold_gain_rl"] = merged["rouge_l_v01"] - merged["rouge_l_v02"]

print(f"Merged samples: {len(merged)}")
print(f"Function distribution:\n{merged['pred_function'].value_counts()}\n")

# ---------------------------------------------------------------------------
# 2. Per-function analysis
# ---------------------------------------------------------------------------
print("=" * 60)
print("Per-function ROUGE-L: v01 (scaffold) vs v02 (no scaffold)")
print("=" * 60)

func_rows = []
for func in ["WHAT", "HOW", "WHY", "COMPARE"]:
    sub = merged[merged["pred_function"] == func]
    n = len(sub)
    rl_v01 = sub["rouge_l_v01"].mean()
    rl_v02 = sub["rouge_l_v02"].mean()
    gain = sub["scaffold_gain_rl"].mean()
    _, pval = stats.ttest_rel(sub["rouge_l_v01"], sub["rouge_l_v02"])
    func_rows.append({
        "Function": func, "N": n,
        "v01_RL": round(rl_v01, 3), "v02_RL": round(rl_v02, 3),
        "Gain": round(gain, 3), "p_value": round(pval, 4)
    })
    print(f"  {func:8s} (n={n:3d}): v01={rl_v01:.3f}, v02={rl_v02:.3f}, "
          f"gain={gain:+.3f}, p={pval:.4f}")

func_df = pd.DataFrame(func_rows)
func_df.to_csv(f"{OUTPUT_DIR}/per_function_scaffold_gain.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Per-domain function heterogeneity → scaffold gain
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Per-domain: Function Entropy vs Mean Scaffold Gain")
print("=" * 60)

domain_rows = []
for dom in ["LAW", "MED", "SOC", "STEM", "CROSS"]:
    sub = merged[merged["domain"] == dom]
    if len(sub) < 5:
        continue
    counts = sub["pred_function"].value_counts(normalize=True)
    probs = counts.values
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    gain = sub["scaffold_gain_rl"].mean()
    n = len(sub)
    domain_rows.append({
        "Domain": dom, "N": n,
        "FuncEntropy": round(entropy, 4),
        "MeanGain_RL": round(gain, 4)
    })
    print(f"  {dom:6s} (n={n:3d}): entropy={entropy:.4f}, mean_gain={gain:+.4f}")

domain_df = pd.DataFrame(domain_rows)
domain_df.to_csv(f"{OUTPUT_DIR}/domain_entropy_gain.csv", index=False)

# ---------------------------------------------------------------------------
# 4. CROSS query ratio sweep
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CROSS Ratio Sweep: scaffold gain as CROSS% increases")
print("=" * 60)

non_cross = merged[~merged["is_cross_gt"]]
cross = merged[merged["is_cross_gt"]]

sweep_rows = []
for cross_ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    n_cross = int(len(non_cross) * cross_ratio / max(1 - cross_ratio, 1e-6))
    n_cross = min(n_cross, len(cross))
    sub_cross = cross.sample(n=n_cross, random_state=42) if n_cross > 0 else cross.iloc[:0]
    subset = pd.concat([non_cross, sub_cross])
    actual_ratio = sub_cross.shape[0] / len(subset) if len(subset) > 0 else 0
    gain = subset["scaffold_gain_rl"].mean()
    sweep_rows.append({"target_cross_ratio": cross_ratio,
                        "actual_cross_ratio": round(actual_ratio, 3),
                        "mean_gain_rl": round(gain, 4),
                        "n_total": len(subset)})
    print(f"  CROSS {cross_ratio:.0%}: gain={gain:+.4f} (n={len(subset)})")

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(f"{OUTPUT_DIR}/cross_ratio_sweep.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

# 5a. Per-function gain bar chart
ax = axes[0]
bars = ax.bar(func_df["Function"], func_df["Gain"],
              color=["steelblue" if g > 0 else "tomato" for g in func_df["Gain"]])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("(a) Scaffold Gain by Function Type\n(V01 − V02 ROUGE-L)", fontsize=11)
ax.set_xlabel("Function Type")
ax.set_ylabel("Mean ROUGE-L Gain")
for bar, row in zip(bars, func_df.itertuples()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"n={row.N}", ha="center", fontsize=8)

# 5b. Domain entropy vs gain scatter
ax = axes[1]
ax.scatter(domain_df["FuncEntropy"], domain_df["MeanGain_RL"], s=80, color="steelblue")
for _, row in domain_df.iterrows():
    ax.annotate(row["Domain"], (row["FuncEntropy"], row["MeanGain_RL"]),
                textcoords="offset points", xytext=(5, 3), fontsize=9)
m, b = np.polyfit(domain_df["FuncEntropy"], domain_df["MeanGain_RL"], 1)
xs = np.linspace(domain_df["FuncEntropy"].min(), domain_df["FuncEntropy"].max(), 50)
ax.plot(xs, m * xs + b, "r--", linewidth=1)
ax.set_title("(b) Function Entropy vs Scaffold Gain\n(Per Domain)", fontsize=11)
ax.set_xlabel("Function Entropy (bits)")
ax.set_ylabel("Mean ROUGE-L Gain (V01−V02)")

# 5c. CROSS ratio sweep
ax = axes[2]
ax.plot(sweep_df["target_cross_ratio"] * 100, sweep_df["mean_gain_rl"],
        marker="o", color="steelblue")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("(c) Scaffold Gain vs CROSS Query Ratio", fontsize=11)
ax.set_xlabel("CROSS Query Ratio (%)")
ax.set_ylabel("Mean ROUGE-L Gain (V01−V02)")

fig.savefig(f"{OUTPUT_DIR}/e1_hypothesis_figure.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{OUTPUT_DIR}/e1_hypothesis_figure.pdf", bbox_inches="tight")
print(f"\nFigure saved to {OUTPUT_DIR}/e1_hypothesis_figure.png")

# ---------------------------------------------------------------------------
# 6. Summary stats for paper
# ---------------------------------------------------------------------------
overall_gain = merged["scaffold_gain_rl"].mean()
_, overall_p = stats.ttest_rel(merged["rouge_l_v01"], merged["rouge_l_v02"])
cross_gain = cross["scaffold_gain_rl"].mean() if len(cross) > 0 else float("nan")
single_gain = non_cross["scaffold_gain_rl"].mean() if len(non_cross) > 0 else float("nan")

summary = {
    "overall_mean_gain_rl": round(overall_gain, 4),
    "overall_p_value": round(float(overall_p), 6),
    "cross_domain_mean_gain_rl": round(cross_gain, 4),
    "single_domain_mean_gain_rl": round(single_gain, 4),
    "per_function": func_rows,
    "per_domain": domain_rows,
    "cross_ratio_sweep": sweep_rows
}

with open(f"{OUTPUT_DIR}/e1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("E1 SUMMARY (for paper)")
print("=" * 60)
print(f"Overall scaffold gain:       {overall_gain:+.3f} RL (p={overall_p:.4f})")
print(f"CROSS-domain scaffold gain:  {cross_gain:+.3f} RL")
print(f"Single-domain scaffold gain: {single_gain:+.3f} RL")
print(f"\nResults saved to {OUTPUT_DIR}/")
