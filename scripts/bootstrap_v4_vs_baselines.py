"""
bootstrap_v4_vs_baselines.py
============================
NexRoute v4 vs 기존 4개 베이스라인 paired bootstrap 유의성 검정.
Clean-300 기준. Holm 보정 적용.

출력: reinference_results/v4_bootstrap_results.json
"""
import csv, json, os
import numpy as np

OUT_DIR = "reinference_results"
CLEAN_INDICES = set(range(0, 300))
N_BOOT = 10000
rng = np.random.default_rng(42)


def load_scores(path, domain_filter=None, split_filter="clean"):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["failed"] == "True":
                continue
            idx = int(r["idx"])
            if idx not in CLEAN_INDICES:
                continue
            if r["domain"] == "LAW":
                domain = "LAW"
            elif r["is_cross"] == "True":
                domain = "CROSS"
            else:
                domain = "SINGLE"

            if domain_filter and domain != domain_filter:
                continue
            rows[idx] = {
                "rouge_l":   float(r["rouge_l"]) if r["rouge_l"] else None,
                "bertscore": float(r["bertscore"]) if r["bertscore"] else None,
                "bleu4":     float(r["bleu4"]) if r["bleu4"] else None,
                "domain":    domain,
            }
    return rows


def paired_bootstrap(scores_a, scores_b, metric="rouge_l", n_boot=N_BOOT):
    pairs = []
    for idx in scores_a:
        if idx in scores_b:
            va = scores_a[idx].get(metric)
            vb = scores_b[idx].get(metric)
            if va is not None and vb is not None:
                pairs.append((va, vb))
    if not pairs:
        return None

    a_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    obs   = np.mean(a_arr) - np.mean(b_arr)

    # bootstrap
    diffs = []
    n = len(pairs)
    for _ in range(n_boot):
        idx_b = rng.integers(0, n, n)
        diffs.append(np.mean(a_arr[idx_b]) - np.mean(b_arr[idx_b]))
    diffs = np.array(diffs)

    # two-sided p-value (shift under H0)
    shifted = diffs - np.mean(diffs)
    p = float(np.mean(np.abs(shifted) >= np.abs(obs)))

    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    return {
        "delta": round(float(obs), 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "p": round(max(p, 1/n_boot), 5),
        "n_pairs": len(pairs),
        "mean_a": round(float(np.mean(a_arr)), 4),
        "mean_b": round(float(np.mean(b_arr)), 4),
    }


def holm_correct(pvals):
    n = len(pvals)
    sorted_pairs = sorted(enumerate(pvals), key=lambda x: x[1])
    corrected = [None] * n
    for rank, (orig_idx, p) in enumerate(sorted_pairs):
        corrected[orig_idx] = min(1.0, p * (n - rank))
    # monotonicity
    sorted_corrected = [corrected[i] for i, _ in sorted(enumerate(corrected), key=lambda x: pvals[x[0]])]
    running_max = 0
    for i, val in enumerate(sorted_corrected):
        running_max = max(running_max, val)
        sorted_corrected[i] = running_max
    # map back
    sorted_idx = [i for i, _ in sorted(enumerate(pvals), key=lambda x: pvals[x[0]])]
    result = [None] * n
    for rank, orig_idx in enumerate(sorted_idx):
        result[orig_idx] = round(sorted_corrected[rank], 5)
    return result


# ── 파일 로드 ──────────────────────────────────────────────────────────────────
baseline_files = {
    "naive_rag":   os.path.join(OUT_DIR, "scores_500_naive_rag.csv"),
    "tfidf_router": os.path.join(OUT_DIR, "scores_500_tfidf_router.csv"),
    "hyde":        os.path.join(OUT_DIR, "scores_500_hyde.csv"),
    "crag":        os.path.join(OUT_DIR, "scores_500_crag.csv"),
}
v4_file   = os.path.join(OUT_DIR, "scores_500_nexroute_lawv4.csv")
orig_file = os.path.join(OUT_DIR, "scores_500_nexroute.csv")

results = {}

for scope, domain_filter in [
    ("overall_clean",  None),
    ("cross_clean",    "CROSS"),
    ("single_clean",   "SINGLE"),
    ("law_clean",      "LAW"),
]:
    print(f"\n=== {scope} ===")
    v4_scores   = load_scores(v4_file,   domain_filter)
    orig_scores = load_scores(orig_file, domain_filter)

    scope_results = {}

    # v4 vs each baseline
    pvals = []
    comparison_list = []
    for metric in ["rouge_l", "bertscore"]:
        for bl_name, bl_file in baseline_files.items():
            bl_scores = load_scores(bl_file, domain_filter)
            res = paired_bootstrap(v4_scores, bl_scores, metric)
            comparison_list.append((metric, bl_name, res))
            pvals.append(res["p"] if res else 1.0)

    corrected = holm_correct(pvals)

    for i, (metric, bl_name, res) in enumerate(comparison_list):
        if res is None:
            continue
        res["p_holm"] = corrected[i]
        key = f"v4_vs_{bl_name}_{metric}"
        scope_results[key] = res
        sig = "***" if res["p_holm"] < 0.001 else ("**" if res["p_holm"] < 0.01 else ("*" if res["p_holm"] < 0.05 else "ns"))
        print(f"  v4 vs {bl_name:<14} {metric:<10} Δ={res['delta']:+.4f}  "
              f"95%CI=[{res['ci_95'][0]:+.4f},{res['ci_95'][1]:+.4f}]  "
              f"p={res['p']:.4f}  p_holm={res['p_holm']:.4f}  {sig}")

    # v4 vs orig
    for metric in ["rouge_l", "bertscore"]:
        res = paired_bootstrap(v4_scores, orig_scores, metric)
        if res:
            key = f"v4_vs_nexroute_orig_{metric}"
            scope_results[key] = res
            print(f"  v4 vs nexroute_orig {metric:<10} Δ={res['delta']:+.4f}  "
                  f"95%CI=[{res['ci_95'][0]:+.4f},{res['ci_95'][1]:+.4f}]  "
                  f"p={res['p']:.5f}")

    results[scope] = scope_results

out_path = os.path.join(OUT_DIR, "v4_bootstrap_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[+] 저장: {out_path}")
