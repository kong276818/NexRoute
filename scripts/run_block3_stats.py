"""
run_block3_stats.py
===================
블록 3 — 통계 복구 (CPU 전용)

3-1. 생성 품질: NexRoute vs 각 7B baseline, paired bootstrap
3-2. Hallucination: cluster bootstrap
3-3. Ablation (Table 8): V01 vs 각 variant, paired bootstrap
"""
import json, csv, os, sys
import numpy as np
from pathlib import Path

np.random.seed(42)
N_BOOT = 10000

BASE = Path("/home/sclab/moe_model")
RECOMP_DIR = BASE / "results/recomputed"
ABLATION_DIR = BASE / "results/ablation"
OUT = BASE / "results/block3_stats.json"

# ─── Bootstrap 유틸 ────────────────────────────────────────────────────────────
def paired_bootstrap(a, b, n=N_BOOT, stat="mean"):
    """Paired bootstrap: stat(b) - stat(a). Returns (obs_diff, ci_lo, ci_hi, p)."""
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    diff = b - a
    obs  = np.mean(diff)
    boot = np.array([np.mean(np.random.choice(diff, len(diff), replace=True))
                     for _ in range(n)])
    ci_lo = np.percentile(boot, 2.5)
    ci_hi = np.percentile(boot, 97.5)
    p = np.mean(boot <= 0) if obs >= 0 else np.mean(boot >= 0)
    p = 2 * min(p, 1-p)  # two-tailed
    return float(obs), float(ci_lo), float(ci_hi), float(p)

def cohens_d(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    pooled = np.sqrt((np.std(a,ddof=1)**2 + np.std(b,ddof=1)**2) / 2)
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else 0.0

def cluster_bootstrap_hal(per_sample_rows, group_a_col, group_b_col, n=N_BOOT):
    """
    Cluster bootstrap for hallucination (answer-level clusters).
    Returns (obs_diff, ci_lo, ci_hi, p) where diff = hal_b - hal_a (pp).
    """
    rows_a = [(int(r.get('n_sents',0) or 0), int(r.get('n_contra',0) or 0))
              for r in per_sample_rows]
    # For hallucination, each answer is a cluster; resample answers
    ns_a   = np.array([r[0] for r in rows_a])
    nc_a   = np.array([r[1] for r in rows_a])
    # group_b requires separate CSV for variant b
    # Compute overall rates
    obs_a = 100 * nc_a.sum() / ns_a.sum() if ns_a.sum() else 0
    return obs_a  # placeholder — real cluster boot done per comparison below

def sample_hal(csv_path, domain_filter=None, cross_filter=None, excl_law=False):
    """Load hallucination per-sample CSV, return (ns_array, nc_array, hal_pct, rows).
    Handles both 'idx' and 'sample_id' column names; sorts by ID for alignment."""
    rows = list(csv.DictReader(open(csv_path)))
    # Normalize id column
    for r in rows:
        if 'idx' in r and 'sample_id' not in r:
            r['sample_id'] = r['idx']
    rows.sort(key=lambda r: int(r.get('sample_id', r.get('idx', 0))))
    if domain_filter:
        rows = [r for r in rows if r.get('expected_domain','').upper() == domain_filter.upper()]
    if cross_filter is not None:
        rows = [r for r in rows if (str(r.get('is_cross','')).lower() in ('true','1')) == cross_filter]
    if excl_law:
        rows = [r for r in rows if r.get('expected_domain','').upper() != 'LAW']
    ns = np.array([int(r.get('n_sents',0) or 0) for r in rows])
    nc = np.array([int(r.get('n_contra',0) or 0) for r in rows])
    hal = 100 * nc.sum() / ns.sum() if ns.sum() else 0.0
    return ns, nc, hal, rows

def bootstrap_hal_diff(ns_a, nc_a, ns_b, nc_b, n=N_BOOT):
    """Cluster bootstrap: hal_b - hal_a (pp)."""
    assert len(ns_a) == len(ns_b)
    obs_a = 100 * nc_a.sum() / ns_a.sum() if ns_a.sum() else 0
    obs_b = 100 * nc_b.sum() / ns_b.sum() if ns_b.sum() else 0
    obs_diff = obs_b - obs_a

    diffs = []
    idx = np.arange(len(ns_a))
    for _ in range(n):
        boot_idx = np.random.choice(idx, len(idx), replace=True)
        ba_ns = ns_a[boot_idx]; ba_nc = nc_a[boot_idx]
        bb_ns = ns_b[boot_idx]; bb_nc = nc_b[boot_idx]
        ha = 100 * ba_nc.sum() / ba_ns.sum() if ba_ns.sum() else 0
        hb = 100 * bb_nc.sum() / bb_ns.sum() if bb_ns.sum() else 0
        diffs.append(hb - ha)

    diffs = np.array(diffs)
    ci_lo = np.percentile(diffs, 2.5)
    ci_hi = np.percentile(diffs, 97.5)
    p = np.mean(diffs <= 0) if obs_diff >= 0 else np.mean(diffs >= 0)
    p = 2 * min(p, 1-p)
    return float(obs_diff), float(ci_lo), float(ci_hi), float(p)

# ─── 3-1. 생성 품질: NexRoute(reinference) vs 7B baselines ────────────────────
print("=" * 60)
print("3-1. 생성 품질 통계 (NexRoute reinference vs 7B baselines)")
print("=" * 60)

def load_scores(csv_path, clean300=False, cross_only=False, single_only=False):
    rows = list(csv.DictReader(open(csv_path)))
    if clean300:
        rows = [r for r in rows if str(r.get('is_clean300',r.get('is_clean',''))).lower() in ('true','1')]
    if cross_only:
        rows = [r for r in rows if str(r.get('is_cross_gt',r.get('is_cross',''))).lower() in ('true','1')]
    if single_only:
        rows = [r for r in rows if str(r.get('is_cross_gt',r.get('is_cross',''))).lower() not in ('true','1')]
    return rows

nexroute_full = load_scores(RECOMP_DIR / "nexroute_reinference_scores.csv")
nexroute_c300 = load_scores(RECOMP_DIR / "nexroute_reinference_scores.csv", clean300=True)

BASELINES = {
    "naive_rag": RECOMP_DIR / "naive_rag_scores_per_sample.csv",
    "tfidf_router": RECOMP_DIR / "tfidf_router_scores_per_sample.csv",
    "crag": RECOMP_DIR / "crag_scores_per_sample.csv",
    "hyde": RECOMP_DIR / "hyde_scores_per_sample.csv",
}

stats_31 = {}
subsets = [("full500", False, False, False), ("clean300", True, False, False),
           ("clean300_cross", True, True, False), ("clean300_single", True, False, True)]

all_p_values = []  # for Holm correction

for bl_name, bl_path in BASELINES.items():
    stats_31[bl_name] = {}
    for subset_name, c300, cross, single in subsets:
        bl_rows = load_scores(bl_path, clean300=c300, cross_only=cross, single_only=single)
        nx_rows = load_scores(RECOMP_DIR/"nexroute_reinference_scores.csv",
                              clean300=c300, cross_only=cross, single_only=single)
        if len(bl_rows) != len(nx_rows):
            print(f"  WARN: {bl_name}/{subset_name} size mismatch {len(nx_rows)} vs {len(bl_rows)}")
            continue

        # Sort by sample_id for pairing
        nx_rows.sort(key=lambda r: int(r.get('sample_id',0)))
        bl_rows.sort(key=lambda r: int(r.get('sample_id',0)))

        subset_stats = {}
        for metric, col in [("rouge_l","rouge_l"), ("bertscore","bertscore_f"), ("bleu4","bleu4")]:
            a = [float(r.get(col,0) or 0) for r in bl_rows]
            b = [float(r.get(col,0) or 0) for r in nx_rows]
            obs, lo, hi, p = paired_bootstrap(a, b)
            d = cohens_d(a, b)
            subset_stats[metric] = {
                "baseline_mean": round(np.mean(a),3),
                "nexroute_mean": round(np.mean(b),3),
                "diff": round(obs,3), "ci95": [round(lo,3), round(hi,3)],
                "p_raw": round(p,4), "cohens_d": round(d,3),
            }
            all_p_values.append(("3-1", bl_name, subset_name, metric, p))
        stats_31[bl_name][subset_name] = subset_stats

# Holm correction (per subset: 4 baselines × 3 metrics = 12 tests)
# Do separately for each subset
for subset_name, c300, cross, single in subsets:
    pairs = [(bl, m) for bl in BASELINES for m in ["rouge_l","bertscore","bleu4"]]
    ps = [stats_31[bl][subset_name][m]["p_raw"]
          for bl, m in pairs if subset_name in stats_31.get(bl,{})]
    if not ps: continue
    sorted_pairs = sorted(zip(ps, range(len(ps))), key=lambda x: x[0])
    k = len(ps)
    holm_ps = [None]*k
    for rank, (p_val, orig_idx) in enumerate(sorted_pairs):
        holm_ps[orig_idx] = min(1.0, p_val * (k - rank))
    for (bl, m), hp in zip(pairs, holm_ps):
        if subset_name in stats_31.get(bl,{}) and m in stats_31[bl].get(subset_name,{}):
            stats_31[bl][subset_name][m]["p_holm"] = round(hp,4) if hp is not None else None

# Print summary
print(f"\nNexRoute (reinf) vs baselines, clean-300 ROUGE-L:")
print(f"{'Baseline':<15} {'Δ':>8} {'95%CI':>18} {'p_raw':>8} {'p_holm':>8} {'d':>7}")
for bl in BASELINES:
    if "clean300" in stats_31.get(bl,{}):
        s = stats_31[bl]["clean300"].get("rouge_l",{})
        print(f"  {bl:<13} {s.get('diff',0):>+8.3f} "
              f"[{s.get('ci95',[0,0])[0]:>+7.3f},{s.get('ci95',[0,0])[1]:>+7.3f}] "
              f"{s.get('p_raw',1):>8.4f} {s.get('p_holm',1) or 1:>8.4f} "
              f"{s.get('cohens_d',0):>7.3f}")

# ─── 3-2. Hallucination: NexRoute vs baselines ────────────────────────────────
print("\n" + "=" * 60)
print("3-2. Hallucination 통계")
print("=" * 60)

HAL_CSVS = {
    "v01": ABLATION_DIR / "v01_hallucination_per_sample.csv",
}

# Find baseline hallucination CSVs
for name in ["naive_rag","tfidf_router","crag","hyde"]:
    paths = list((ABLATION_DIR).glob(f"{name}*hallucination*per_sample*.csv"))
    if not paths:
        paths = list(BASE.rglob(f"{name}*hallucination*per_sample*.csv"))
    if paths:
        HAL_CSVS[name] = paths[0]

print(f"  Hallucination CSVs found: {list(HAL_CSVS.keys())}")

stats_32 = {}
for scope_name, cross_f, excl in [
    ("overall", None, False),
    ("cross", True, False),
    ("single", False, False),
    ("excl_law", None, True),
]:
    scope_stats = {}
    if "v01" in HAL_CSVS:
        ns_v01, nc_v01, hal_v01, _ = sample_hal(HAL_CSVS["v01"],
                                                  cross_filter=cross_f, excl_law=excl)
    else:
        print("  WARN: v01 hallucination CSV not found"); continue

    for bl in [k for k in HAL_CSVS if k != "v01"]:
        ns_bl, nc_bl, hal_bl, _ = sample_hal(HAL_CSVS[bl], cross_filter=cross_f, excl_law=excl)
        if len(ns_v01) != len(ns_bl):
            print(f"  WARN: v01 vs {bl} size mismatch {len(ns_v01)} vs {len(ns_bl)}")
            continue
        diff, lo, hi, p = bootstrap_hal_diff(ns_bl, nc_bl, ns_v01, nc_v01)
        scope_stats[bl] = {
            "baseline_hal": round(hal_bl,2),
            "nexroute_hal": round(hal_v01,2),
            "diff_pp": round(diff,3),  # negative = nexroute lower = better
            "ci95": [round(lo,3), round(hi,3)],
            "p": round(p,4),
        }
    stats_32[scope_name] = scope_stats

print(f"\n  NexRoute(V01) vs baselines, hallucination diff (negative=NexRoute better):")
for scope_name, data in stats_32.items():
    print(f"  [{scope_name}]")
    for bl, s in data.items():
        sig = "***" if s['p']<0.001 else ("**" if s['p']<0.01 else ("*" if s['p']<0.05 else "ns"))
        print(f"    {bl:<15} NexRoute={s['nexroute_hal']}%  baseline={s['baseline_hal']}%  "
              f"Δ={s['diff_pp']:+.2f}pp  p={s['p']:.4f}{sig}")

# ─── 3-3. Ablation: V01 vs variants ────────────────────────────────────────────
print("\n" + "=" * 60)
print("3-3. Ablation 통계 (V01 vs V02~V05)")
print("=" * 60)

ABLATION_SCORE_CSVS = {
    "v01": ABLATION_DIR / "v01_scores_per_sample.csv",
    "v02": ABLATION_DIR / "v02_scores_per_sample.csv",
    "v03": ABLATION_DIR / "v03_scores_per_sample.csv",
    "v04": ABLATION_DIR / "v04_scores_per_sample.csv",
    "v05": ABLATION_DIR / "v05_scores_per_sample.csv",
}

def load_ablation_scores(csv_path, subset=None):
    rows = list(csv.DictReader(open(csv_path)))
    # is_cross might be 'is_cross_gt' or 'is_cross_pred' or 'is_cross'
    def is_cross_val(r):
        for k in ('is_cross_gt','is_cross','is_cross_pred'):
            if k in r: return str(r[k]).lower() in ('true','1')
        return False
    if subset == "cross":
        rows = [r for r in rows if is_cross_val(r)]
    elif subset == "single":
        rows = [r for r in rows if not is_cross_val(r)]
    rows.sort(key=lambda r: int(r.get('sample_id', r.get('idx', 0))))
    return rows

stats_33_gen = {}
for variant in ["v02","v03","v04","v05"]:
    stats_33_gen[variant] = {}
    if not (ABLATION_DIR / f"{variant}_scores_per_sample.csv").exists():
        print(f"  MISSING: {variant}")
        continue
    for subset_name in ["overall","cross","single"]:
        sub = None if subset_name=="overall" else subset_name
        rows_v01 = load_ablation_scores(ABLATION_SCORE_CSVS["v01"], subset=sub)
        rows_vx  = load_ablation_scores(ABLATION_SCORE_CSVS[variant], subset=sub)
        if len(rows_v01) != len(rows_vx):
            print(f"  WARN: v01 vs {variant}/{subset_name} size {len(rows_v01)} vs {len(rows_vx)}")
            continue

        sub_stats = {}
        for metric, col in [("rouge_l","rouge_l"), ("bertscore","bertscore_f"), ("bleu4","bleu4")]:
            a = [float(r.get(col,0) or 0) for r in rows_v01]
            b = [float(r.get(col,0) or 0) for r in rows_vx]
            obs, lo, hi, p = paired_bootstrap(a, b)
            d = cohens_d(a, b)
            sub_stats[metric] = {
                "v01_mean": round(np.mean(a),3),
                "variant_mean": round(np.mean(b),3),
                "diff": round(obs,3),
                "ci95": [round(lo,3), round(hi,3)],
                "p": round(p,4),
                "cohens_d": round(d,3),
            }
        stats_33_gen[variant][subset_name] = sub_stats

print(f"\n  Generation quality (overall), diff = variant - V01 (negative = degradation):")
print(f"  {'Variant':<6} {'ROUGE-L Δ':>11} {'CI95':>18} {'p':>8} {'BERTScore Δ':>12}")
for var in ["v02","v03","v04","v05"]:
    if "overall" in stats_33_gen.get(var,{}):
        rl = stats_33_gen[var]["overall"]["rouge_l"]
        bs = stats_33_gen[var]["overall"]["bertscore"]
        sig_rl = "***" if rl['p']<0.001 else ("**" if rl['p']<0.01 else ("*" if rl['p']<0.05 else "ns"))
        print(f"  {var:<6} {rl['diff']:>+11.3f} [{rl['ci95'][0]:>+7.3f},{rl['ci95'][1]:>+7.3f}] "
              f"{rl['p']:>8.4f}{sig_rl}  {bs['diff']:>+12.3f}")

# Ablation hallucination
stats_33_hal = {}
ABLATION_HAL_CSVS = {
    "v01": ABLATION_DIR / "v01_hallucination_per_sample.csv",
    "v02": ABLATION_DIR / "v02_hallucination_per_sample.csv",
    "v03": ABLATION_DIR / "v03_hallucination_per_sample.csv",
    "v04": ABLATION_DIR / "v04_hallucination_per_sample.csv",
    "v05": ABLATION_DIR / "v05_hallucination_per_sample.csv",
}

print(f"\n  Hallucination (cluster bootstrap), diff = variant - V01 (positive = more hal):")
print(f"  {'Variant':<6} {'V01 hal':>8} {'Var hal':>8} {'Δ pp':>8} {'CI95':>18} {'p':>8}")
for variant in ["v02","v03","v04","v05"]:
    stats_33_hal[variant] = {}
    if not (ABLATION_DIR / f"{variant}_hallucination_per_sample.csv").exists():
        print(f"  MISSING: {variant}"); continue
    for scope_name, cross_f, excl in [
        ("overall", None, False),
        ("cross", True, False),
        ("single", False, False),
    ]:
        ns_v01, nc_v01, hal_v01, _ = sample_hal(ABLATION_HAL_CSVS["v01"],
                                                  cross_filter=cross_f, excl_law=False)
        ns_vx,  nc_vx,  hal_vx, _  = sample_hal(ABLATION_HAL_CSVS[variant],
                                                   cross_filter=cross_f, excl_law=False)
        if len(ns_v01) != len(ns_vx):
            print(f"  WARN: {variant}/{scope_name} size mismatch"); continue
        diff, lo, hi, p = bootstrap_hal_diff(ns_v01, nc_v01, ns_vx, nc_vx)
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
        stats_33_hal[variant][scope_name] = {
            "v01_hal": round(hal_v01,3),
            "variant_hal": round(hal_vx,3),
            "diff_pp": round(diff,3),
            "ci95": [round(lo,3), round(hi,3)],
            "p": round(p,4),
        }
        if scope_name == "overall":
            print(f"  {variant:<6} {hal_v01:>8.2f}% {hal_vx:>8.2f}% {diff:>+8.3f} "
                  f"[{lo:>+7.3f},{hi:>+7.3f}] {p:>8.4f}{sig}")

# ─── 저장 ─────────────────────────────────────────────────────────────────────
output = {
    "3-1_generation_quality": stats_31,
    "3-2_hallucination": stats_32,
    "3-3_ablation_gen": stats_33_gen,
    "3-3_ablation_hal": stats_33_hal,
}
json.dump(output, open(OUT, 'w'), indent=2)
print(f"\n저장: {OUT}")
print("\n블록 3 완료.")
