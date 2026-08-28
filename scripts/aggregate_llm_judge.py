"""
Multi-Model Judge Evaluation — Aggregate Analysis

Usage:
  cd /home/sclab/moe_model
  python scripts/aggregate_llm_judge.py

Input:  results/llm_judge/score_sheet.csv
Output: results/llm_judge/aggregate_results.csv
        results/llm_judge/law_diagnostic.csv
        (printed: preference analysis, judge agreement)

Statistical note:
  Primary unit = query pair (n=25), not judge × query (n=100).
  Within each query, judge scores are averaged first, then query-level
  statistics are computed to avoid pseudo-replication.
"""

import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

INPUT  = "results/llm_judge/score_sheet.csv"
OUT_MAIN = "results/llm_judge/aggregate_results.csv"
OUT_LAW  = "results/llm_judge/law_diagnostic.csv"

SEED   = 20260827
N_BOOT = 10000

rng = np.random.default_rng(SEED)

METRICS = [
    "factual",
    "task",
    "reasoning",
    "completeness",
    "relevance",
    "overall",
]
METRICS_OPTIONAL = ["synthesis", "grounding"]


# --------------------------------------------------------------------------
# Load and validate
# --------------------------------------------------------------------------

df = pd.read_csv(INPUT)

required_cols = [
    "sample_id", "domain", "is_cross",
    "judge_provider", "system_A", "system_B",
    "A_factual", "A_task", "A_reasoning",
    "A_completeness", "A_relevance", "A_overall",
    "B_factual", "B_task", "B_reasoning",
    "B_completeness", "B_relevance", "B_overall",
    "preference",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in score_sheet.csv: {missing}")

# Drop rows where mandatory scores are not yet filled
core_score_cols = [c for c in df.columns if c.startswith("A_") or c.startswith("B_")]
df_filled = df.dropna(subset=["A_factual", "B_factual", "preference"]).copy()

if len(df_filled) == 0:
    print("No filled rows found in score_sheet.csv. Fill in judge scores first.")
    raise SystemExit(0)

print(f"Loaded {len(df_filled)} filled evaluations from {INPUT}")
print(f"  Judges: {df_filled['judge_provider'].unique().tolist()}")
print(f"  Samples: {df_filled['sample_id'].nunique()}")


# --------------------------------------------------------------------------
# Determine which optional metrics are present
# --------------------------------------------------------------------------

active_metrics = list(METRICS)
for m in METRICS_OPTIONAL:
    if f"A_{m}" in df_filled.columns and df_filled[f"A_{m}"].notna().any():
        active_metrics.append(m)

print(f"  Active metrics: {active_metrics}")


# --------------------------------------------------------------------------
# Convert A/B scores → NexRoute/Baseline scores
# --------------------------------------------------------------------------

def to_system_scores(row, metric):
    a_col, b_col = f"A_{metric}", f"B_{metric}"
    a_val = row.get(a_col, np.nan)
    b_val = row.get(b_col, np.nan)
    if pd.isna(a_val) or pd.isna(b_val):
        return np.nan, np.nan
    if row["system_A"] == "NexRoute":
        return float(a_val), float(b_val)
    else:
        return float(b_val), float(a_val)


long_rows = []
for _, row in df_filled.iterrows():
    for metric in active_metrics:
        nex_score, base_score = to_system_scores(row, metric)
        if np.isnan(nex_score):
            continue
        long_rows.append({
            "sample_id":     row["sample_id"],
            "domain":        row["domain"],
            "is_cross":      int(row["is_cross"]),
            "judge_provider": row["judge_provider"],
            "metric":        metric,
            "NexRoute":      nex_score,
            "Baseline":      base_score,
        })

long_df = pd.DataFrame(long_rows)


# --------------------------------------------------------------------------
# Step 1: Average judges within each (sample, metric) → query-level scores
# --------------------------------------------------------------------------

sample_df = (
    long_df
    .groupby(["sample_id", "domain", "is_cross", "metric"], as_index=False)
    [["NexRoute", "Baseline"]]
    .mean()
)
sample_df["delta"] = sample_df["NexRoute"] - sample_df["Baseline"]


# --------------------------------------------------------------------------
# Bootstrap CI helper
# --------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    means = [np.mean(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


# --------------------------------------------------------------------------
# Step 2: Main results table (all samples)
# --------------------------------------------------------------------------

summary_rows = []
for metric, g in sample_df.groupby("metric"):
    nex   = g["NexRoute"].to_numpy(float)
    base  = g["Baseline"].to_numpy(float)
    delta = nex - base

    lo, hi = bootstrap_ci(delta)

    try:
        _, p = wilcoxon(nex, base, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p = np.nan

    summary_rows.append({
        "metric":         metric,
        "n_samples":      len(g),
        "nexroute_mean":  round(nex.mean(), 3),
        "baseline_mean":  round(base.mean(), 3),
        "mean_delta":     round(delta.mean(), 3),
        "ci95_low":       round(lo, 3),
        "ci95_high":      round(hi, 3),
        "wilcoxon_p":     round(p, 4) if not np.isnan(p) else np.nan,
    })

summary = pd.DataFrame(summary_rows)
print("\n=== MAIN RESULTS (all samples, n_samples = query pairs) ===")
print(summary.to_string(index=False))
summary.to_csv(OUT_MAIN, index=False)
print(f"\nSaved to {OUT_MAIN}")


# --------------------------------------------------------------------------
# Step 3: CROSS-only synthesis (secondary endpoint)
# --------------------------------------------------------------------------

if "synthesis" in active_metrics:
    cross_df = sample_df[
        (sample_df["is_cross"] == 1) & (sample_df["metric"] == "synthesis")
    ].copy()

    if len(cross_df) > 0:
        nex_c  = cross_df["NexRoute"].to_numpy(float)
        base_c = cross_df["Baseline"].to_numpy(float)
        delta_c = nex_c - base_c
        lo_c, hi_c = bootstrap_ci(delta_c)

        try:
            _, p_c = wilcoxon(nex_c, base_c, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p_c = np.nan

        print(f"\n=== CROSS-DOMAIN SYNTHESIS (n={len(cross_df)} CROSS query pairs) ===")
        print(f"  NexRoute mean : {nex_c.mean():.3f}")
        print(f"  Baseline mean : {base_c.mean():.3f}")
        print(f"  Delta         : {delta_c.mean():.3f}")
        print(f"  95% CI        : [{lo_c:.3f}, {hi_c:.3f}]")
        print(f"  Wilcoxon p    : {p_c:.4f}")
    else:
        print("\nNo CROSS synthesis scores found.")


# --------------------------------------------------------------------------
# Step 4: Pairwise preference analysis
# --------------------------------------------------------------------------

pref_rows = []
for _, row in df_filled.iterrows():
    pref = str(row["preference"]).strip().upper()
    if pref == "TIE":
        winner = "TIE"
    elif pref == "A":
        winner = row["system_A"]
    elif pref == "B":
        winner = row["system_B"]
    else:
        continue
    pref_rows.append({
        "sample_id":     row["sample_id"],
        "judge_provider": row["judge_provider"],
        "winner":        winner,
    })

pref_df = pd.DataFrame(pref_rows)

print("\n=== RAW JUDGE PREFERENCE (% of all judge evaluations) ===")
print(pref_df["winner"].value_counts(normalize=True).mul(100).round(1).to_string())


def majority_vote(x):
    counts = x.value_counts()
    if len(counts) == 0:
        return "TIE"
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "TIE"
    return counts.index[0]


majority = (
    pref_df
    .groupby("sample_id")["winner"]
    .apply(majority_vote)
)

print(f"\n=== SAMPLE-LEVEL MAJORITY PREFERENCE (n={len(majority)} query pairs) ===")
print(majority.value_counts(normalize=True).mul(100).round(1).to_string())


# --------------------------------------------------------------------------
# Step 5: Inter-judge agreement (Spearman)
# --------------------------------------------------------------------------

print("\n=== INTER-JUDGE SCORE AGREEMENT (mean pairwise Spearman rho) ===")
for metric in active_metrics:
    temp = long_df[long_df["metric"] == metric]
    pivot = temp.pivot_table(
        index="sample_id",
        columns="judge_provider",
        values="NexRoute",
    )
    judges = pivot.columns.tolist()
    correlations = []
    for j1, j2 in itertools.combinations(judges, 2):
        pair = pivot[[j1, j2]].dropna()
        if len(pair) < 3:
            continue
        rho, _ = spearmanr(pair[j1], pair[j2])
        if not np.isnan(rho):
            correlations.append(rho)
    if correlations:
        print(f"  {metric:<18s}: rho = {np.mean(correlations):.3f}  (n_pairs={len(correlations)})")
    else:
        print(f"  {metric:<18s}: insufficient data")


# --------------------------------------------------------------------------
# Step 6: LAW diagnostic
# --------------------------------------------------------------------------

law_df = sample_df[sample_df["domain"].str.upper() == "LAW"].copy()

if len(law_df) > 0:
    print(f"\n=== LAW DIAGNOSTIC (n={law_df['sample_id'].nunique()} LAW query pairs) ===")
    law_summary = (
        law_df
        .groupby("metric")[["NexRoute", "Baseline", "delta"]]
        .mean()
        .round(3)
    )
    print(law_summary.to_string())
    law_summary.to_csv(OUT_LAW)
    print(f"Saved to {OUT_LAW}")
else:
    print("\nNo LAW samples found.")


# --------------------------------------------------------------------------
# Step 7: Composite scores (pre-registered, per spec section 21)
# --------------------------------------------------------------------------

TASK_LEVEL_METRICS = [m for m in ["factual", "task", "reasoning", "completeness", "relevance"]
                      if m in active_metrics]

if len(TASK_LEVEL_METRICS) > 0:
    composite_df = (
        sample_df[sample_df["metric"].isin(TASK_LEVEL_METRICS)]
        .groupby("sample_id")[["NexRoute", "Baseline"]]
        .mean()
    )
    composite_df["delta"] = composite_df["NexRoute"] - composite_df["Baseline"]
    lo_c, hi_c = bootstrap_ci(composite_df["delta"].to_numpy())
    try:
        _, p_c = wilcoxon(
            composite_df["NexRoute"].to_numpy(),
            composite_df["Baseline"].to_numpy(),
            zero_method="wilcox", alternative="two-sided"
        )
    except ValueError:
        p_c = np.nan

    print(f"\n=== COMPOSITE: Task-Level Quality (mean of {TASK_LEVEL_METRICS}) ===")
    print(f"  NexRoute : {composite_df['NexRoute'].mean():.3f}")
    print(f"  Baseline : {composite_df['Baseline'].mean():.3f}")
    print(f"  Delta    : {composite_df['delta'].mean():.3f}  95%CI [{lo_c:.3f}, {hi_c:.3f}]")
    print(f"  Wilcoxon p: {p_c:.4f}" if not np.isnan(p_c) else "  Wilcoxon p: N/A")

print("\nDone.")
