"""
nli_length_normalized.py
========================
기존 NLI per-sample CSV에서 길이 정규화된 contradiction rate 계산.
(총 contradictions / 총 sentences) 방식으로 응답 길이 편향 제거.

출력:
  results/nli_length_normalized.json  – 시스템별 정규화 통계
  콘솔에 페이퍼용 테이블 출력

실행:
    python scripts/nli_length_normalized.py
"""
import csv, json
import numpy as np
from scipy import stats

CLEAN_IDXS = set(range(300))

SYSTEMS = {
    "NexRoute":    "results/ablation/v01_hallucination_per_sample.csv",
    "NaiveRAG":    "results/ablation/naive_rag_hallucination_per_sample.csv",
    "TF-IDF":      "results/ablation/tfidf_router_hallucination_per_sample.csv",
    "HyDE":        "results/ablation/hyde_hallucination_per_sample.csv",
    "CRAG":        "results/ablation/crag_hallucination_per_sample.csv",
}

def load(path, domain_filter=None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            idx = int(r['idx'])
            if idx not in CLEAN_IDXS:
                continue
            if domain_filter == "excl_law" and r['expected_domain'] == 'LAW':
                continue
            if domain_filter == "law" and r['expected_domain'] != 'LAW':
                continue
            if r.get('n_sents', '') == '' or int(r['n_sents']) == 0:
                continue
            rows.append({
                'idx': idx,
                'domain': r['expected_domain'],
                'n_sents': int(r['n_sents']),
                'n_contra': int(r['n_contra']),
                'hal_rate': float(r['hal_rate']),
            })
    return rows

def normalized_rate(rows):
    """총 contradictions / 총 sentences (길이 정규화; %)."""
    total_s = sum(r['n_sents'] for r in rows)
    total_c = sum(r['n_contra'] for r in rows)
    return (total_c / total_s * 100) if total_s > 0 else 0.0

def mean_per_sample(rows):
    """per-sample hal_rate 평균 (hal_rate는 이미 % 단위)."""
    return np.mean([r['hal_rate'] for r in rows])

def bootstrap_ci(rows_a, rows_b, n_boot=5000, seed=42):
    """per-sample rate 기반 bootstrap 차이 CI (% 단위)."""
    rng = np.random.default_rng(seed)
    idx_a = {r['idx']: r['hal_rate'] for r in rows_a}
    idx_b = {r['idx']: r['hal_rate'] for r in rows_b}
    common = sorted(set(idx_a) & set(idx_b))
    if not common:
        return None, None
    a_arr = np.array([idx_a[i] for i in common])
    b_arr = np.array([idx_b[i] for i in common])
    diffs = []
    n = len(a_arr)
    for _ in range(n_boot):
        idx_boot = rng.integers(0, n, n)
        diffs.append(np.mean(a_arr[idx_boot]) - np.mean(b_arr[idx_boot]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

print("=" * 75)
print("NLI Contradiction Rate: 원본(per-sample mean) vs 길이 정규화(total/total)")
print("=" * 75)

results = {}
for scope, df in [("Clean-300 all", None), ("Clean-300 excl-LAW", "excl_law"), ("LAW only", "law")]:
    print(f"\n--- {scope} ---")
    print(f"{'System':<12} {'n_samps':>7} {'n_sents':>8} {'per-samp%':>10} {'norm%':>8}")
    scope_data = {}
    for name, path in SYSTEMS.items():
        rows = load(path, domain_filter=df if df else None)
        raw  = mean_per_sample(rows)
        norm = normalized_rate(rows)
        n_s  = sum(r['n_sents'] for r in rows)
        print(f"  {name:<12} {len(rows):>7} {n_s:>8} {raw:>10.2f} {norm:>8.2f}")
        scope_data[name] = {'rows': rows, 'raw': raw, 'norm': norm, 'n_sents': n_s}
    results[scope] = {k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in scope_data.items()}

    # Wilcoxon NexRoute vs each baseline
    nr_rows = load(SYSTEMS["NexRoute"], domain_filter=df if df else None)
    print(f"\n  Wilcoxon (NexRoute vs baseline, per-sample rates, % unit):")
    for bl_name in ["NaiveRAG", "TF-IDF", "HyDE", "CRAG"]:
        bl_rows = load(SYSTEMS[bl_name], domain_filter=df if df else None)
        idx_nr = {r['idx']: r['hal_rate'] for r in nr_rows}
        idx_bl = {r['idx']: r['hal_rate'] for r in bl_rows}
        common = sorted(set(idx_nr) & set(idx_bl))
        a_arr = np.array([idx_nr[i] for i in common])
        b_arr = np.array([idx_bl[i] for i in common])
        diff = a_arr - b_arr
        if np.all(diff == 0):
            p = 1.0
        else:
            _, p = stats.wilcoxon(a_arr, b_arr, alternative='two-sided', zero_method='zsplit')
        delta_raw = np.mean(a_arr) - np.mean(b_arr)
        ci_lo, ci_hi = bootstrap_ci(nr_rows, bl_rows)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    vs {bl_name:<10} Δ={delta_raw:+.2f}pp  CI=[{ci_lo:+.2f},{ci_hi:+.2f}]  p={p:.4f}  {sig}")

# JSON 저장
out = {}
for scope, sdata in results.items():
    out[scope] = sdata
with open("results/nli_length_normalized.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[+] 저장: results/nli_length_normalized.json")
