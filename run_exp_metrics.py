"""
run_exp_metrics.py
===================
Compute ROUGE-L, BERTScore, BLEU-4 + paired bootstrap (n=10000) + Holm correction
for Experiment 1 and Experiment 2 outputs.

Experiment 1 baseline: 'predicted' (NexRoute with DeBERTa scaffold)
Experiment 2 baseline: 'predicted' (NexRoute with predicted routing)

Outputs:
  results/exp1_metrics.json
  results/exp2_metrics.json
"""
import json, gc, warnings
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
np.random.seed(42)

DEVICE = "cuda"
N_BOOT = 10_000
ALPHA  = 0.05

# ── helpers ────────────────────────────────────────────────────────────────────
def compute_rouge_l(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return np.array([sc.score(r, p)['rougeL'].fmeasure * 100 for p, r in zip(preds, refs)])

def compute_bleu4(preds, refs):
    import sacrebleu
    return np.array([sacrebleu.sentence_bleu(p, [r]).score for p, r in zip(preds, refs)])

def compute_bertscore(preds, refs):
    from bert_score import score as bs_score
    import torch
    P, R, F = bs_score(preds, refs, lang="en", model_type="roberta-large",
                       device=DEVICE, verbose=False, batch_size=16)
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return F.numpy() * 100

def paired_bootstrap(scores_a, scores_b, n_boot=N_BOOT):
    """Two-sided paired bootstrap test: H0: mean(a)==mean(b). Returns p-value."""
    n = len(scores_a)
    observed = np.mean(scores_a) - np.mean(scores_b)
    centered_a = scores_a - np.mean(scores_a)
    centered_b = scores_b - np.mean(scores_b)
    count = 0
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        diff = np.mean(centered_a[idx]) - np.mean(centered_b[idx])
        if abs(diff) >= abs(observed):
            count += 1
    return count / n_boot

def bootstrap_ci(scores, n_boot=N_BOOT, ci=0.95):
    means = [np.mean(np.random.choice(scores, len(scores))) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [(1-ci)/2*100, (1+ci)/2*100])
    return round(lo, 3), round(hi, 3)

def analyze(gen_path, conditions, baseline_cond, out_path):
    data = json.load(open(gen_path))
    results = data.get('results', data) if isinstance(data, dict) else data
    print(f"\n[{gen_path}] n={len(results)}")

    refs = [r['gold_answer'] for r in results]

    # Compute all metrics
    print("  BERTScore...")
    bs_all = {}
    for cond in conditions:
        preds = [r.get(cond,'') or '' for r in results]
        bs_all[cond] = compute_bertscore(preds, refs)

    print("  ROUGE-L / BLEU-4...")
    rl_all = {}; bl_all = {}
    for cond in conditions:
        preds = [r.get(cond,'') or '' for r in results]
        rl_all[cond] = compute_rouge_l(preds, refs)
        bl_all[cond] = compute_bleu4(preds, refs)

    # Bootstrap p-values vs baseline (Holm correction)
    non_base = [c for c in conditions if c != baseline_cond]
    raw_ps = {'rouge_l':{}, 'bertscore':{}, 'bleu4':{}}
    for cond in non_base:
        raw_ps['rouge_l'][cond]  = paired_bootstrap(rl_all[cond], rl_all[baseline_cond])
        raw_ps['bertscore'][cond]= paired_bootstrap(bs_all[cond], bs_all[baseline_cond])
        raw_ps['bleu4'][cond]    = paired_bootstrap(bl_all[cond], bl_all[baseline_cond])

    def holm_correct(p_dict):
        keys = list(p_dict.keys())
        pvals = [p_dict[k] for k in keys]
        if not pvals: return {}
        _, corrected, _, _ = multipletests(pvals, method='holm')
        return {k: float(round(p, 4)) for k, p in zip(keys, corrected)}

    corrected = {
        'rouge_l':  holm_correct(raw_ps['rouge_l']),
        'bertscore':holm_correct(raw_ps['bertscore']),
        'bleu4':    holm_correct(raw_ps['bleu4']),
    }

    # Aggregate
    summary = {}
    for cond in conditions:
        n = len(results)
        rl  = rl_all[cond]; bs = bs_all[cond]; bl = bl_all[cond]
        ci_rl = bootstrap_ci(rl); ci_bs = bootstrap_ci(bs); ci_bl = bootstrap_ci(bl)
        summary[cond] = {
            'n': n,
            'rouge_l':     round(float(np.mean(rl)), 3),
            'rouge_l_ci':  list(ci_rl),
            'bertscore':   round(float(np.mean(bs)), 3),
            'bertscore_ci':list(ci_bs),
            'bleu4':       round(float(np.mean(bl)), 3),
            'bleu4_ci':    list(ci_bl),
        }
        if cond != baseline_cond:
            summary[cond]['p_rouge_l_holm']  = corrected['rouge_l'].get(cond,'')
            summary[cond]['p_bertscore_holm']= corrected['bertscore'].get(cond,'')
            summary[cond]['p_bleu4_holm']    = corrected['bleu4'].get(cond,'')

    # Print table
    print(f"\n  {'Condition':<22} {'ROUGE-L':>9} {'CI':>15} {'BERTScore':>10} {'CI':>15} {'p(ROUGE,Holm)':>14}")
    for cond in conditions:
        s  = summary[cond]
        p  = s.get('p_rouge_l_holm', '-')
        ci_rl = f"[{s['rouge_l_ci'][0]:.2f},{s['rouge_l_ci'][1]:.2f}]"
        ci_bs = f"[{s['bertscore_ci'][0]:.2f},{s['bertscore_ci'][1]:.2f}]"
        star = ('**' if isinstance(p,float) and p < 0.01 else
                '*' if isinstance(p,float) and p < 0.05 else
                '†' if isinstance(p,float) and p < 0.10 else '')
        print(f"  {cond:<22} {s['rouge_l']:>9.3f} {ci_rl:>15} {s['bertscore']:>10.3f} {ci_bs:>15} {p!s:>12} {star}")

    out = {'gen_path': gen_path, 'baseline': baseline_cond, 'conditions': conditions,
           'n_bootstrap': N_BOOT, 'alpha': ALPHA, 'summary': summary,
           'raw_pvalues': {m: {k: round(v,4) for k,v in d.items()} for m,d in raw_ps.items()},
           'holm_pvalues': corrected}
    json.dump(out, open(out_path,'w'), indent=2)
    print(f"  Saved: {out_path}")
    return out

# ── Experiment 1 ───────────────────────────────────────────────────────────────
print("="*60)
print("EXPERIMENT 1: Function scaffold oracle reconstruction")
exp1 = analyze(
    gen_path    = "/home/sclab/moe_model/results/exp1_scaffold_oracle.json",
    conditions  = ["predicted", "oracle", "generic", "none"],
    baseline_cond = "predicted",
    out_path    = "/home/sclab/moe_model/results/exp1_metrics.json",
)

# ── Experiment 2 ───────────────────────────────────────────────────────────────
print("="*60)
print("EXPERIMENT 2: Domain router necessity")
exp2 = analyze(
    gen_path    = "/home/sclab/moe_model/results/exp2_domain_router.json",
    conditions  = ["predicted", "oracle_domain", "always_cross", "always_single_best"],
    baseline_cond = "predicted",
    out_path    = "/home/sclab/moe_model/results/exp2_metrics.json",
)

# ── Exp 2 cost metrics ─────────────────────────────────────────────────────────
print("\n  [Exp 2 Cost Metrics]")
data2 = json.load(open("/home/sclab/moe_model/results/exp2_domain_router.json"))
results2 = data2.get('results', [])
cost_conds = ['always_cross', 'always_single_best', 'oracle_domain']
cost_labels = {'always_cross': 'always-CROSS', 'always_single_best': 'always-single-best', 'oracle_domain': 'oracle-domain'}
print(f"  {'Condition':<22} {'Passages':>9} {'PTokens':>9} {'Latency(s)':>11}")
for cond in cost_conds:
    key = f'cost_{cond}'
    rows = [r[key] for r in results2 if key in r]
    if not rows: continue
    avg_pass = np.mean([r['passages'] for r in rows])
    avg_ptok = np.mean([r['ptokens'] for r in rows])
    avg_lat  = np.mean([r['latency'] for r in rows])
    print(f"  {cost_labels[cond]:<22} {avg_pass:>9.1f} {avg_ptok:>9.1f} {avg_lat:>11.2f}")

print("\n[done] Metrics complete.")
