import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EVAL_PATH  = "/home/sclab/moe_model/eval_results_7b_v3_500.json"
LABELS     = ['LAW', 'MED', 'SOC', 'STEM', 'CROSS']

# ── 1. NexRoute 혼동행렬 (실측) ──────────────────────────────
with open(GT_PATH) as f:  gt = json.load(f)
with open(EVAL_PATH) as f: nx = json.load(f)
text_map = {d['instruction'].strip(): d for d in gt}

def get_pred_nx(e):
    preds = e.get('domain_pred', [])
    return 'CROSS' if e.get('is_cross', False) else (preds[0] if preds else 'UNKNOWN')

idx = {l: i for i, l in enumerate(LABELS)}
cm_nx = np.zeros((5, 5), dtype=int)
for e in nx:
    q = e.get('query', '').strip()
    gt_item = text_map.get(q)
    if not gt_item: continue
    t = gt_item.get('expected_domain', '')
    p = get_pred_nx(e)
    if t in idx and p in idx:
        cm_nx[idx[t], idx[p]] += 1

# ── 2. Adaptive RAG 혼동행렬 (키워드 라우팅 재현) ─────────────
KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract',
             'judge','attorney','liability','crime','rights'],
    'MED':  ['medical','patient','disease','treatment','clinical',
             'diagnosis','symptom','drug','doctor','health','therapy'],
    'SOC':  ['social','society','culture','economic','political',
             'community','psychology','sociology','behavior','cultural'],
    'STEM': ['science','physics','chemical','biology','math',
             'engineering','technology','equation','energy','cell'],
}

def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    best_score = max(scores.values())
    if best_score == 0:
        return 'SOC'  # 기본값
    sorted_d = sorted(scores.items(), key=lambda x: -x[1])
    # CROSS: top-2가 거의 비슷하면 CROSS로 판정
    if len(sorted_d) >= 2 and sorted_d[1][1] >= sorted_d[0][1] * 0.7 and sorted_d[0][1] >= 2:
        return 'CROSS'
    return sorted_d[0][0]

cm_ad = np.zeros((5, 5), dtype=int)
for d in gt:
    t = d.get('expected_domain', '')
    p = keyword_route(d.get('instruction', ''))
    if t in idx and p in idx:
        cm_ad[idx[t], idx[p]] += 1

# 퍼센트 정규화
def normalize(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums * 100

cm_ad_pct = normalize(cm_ad)
cm_nx_pct = normalize(cm_nx)

acc_ad = cm_ad.trace() / cm_ad.sum() * 100
acc_nx = cm_nx.trace() / cm_nx.sum() * 100

print(f"Adaptive RAG accuracy: {acc_ad:.2f}%")
print(f"NexRoute accuracy:     {acc_nx:.2f}%")
print("\nNexRoute confusion matrix:")
for i, l in enumerate(LABELS):
    print(f"  {l}: {list(cm_nx[i])}")

# ── 3. 시각화 ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 0.6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor='white')
fig.suptitle('Routing Confusion Matrices: Adaptive RAG vs.\ NexRoute 7B',
             fontsize=13, fontweight='bold', y=1.02)

# 컬러맵
cmap_ad = plt.cm.Blues
cmap_nx = LinearSegmentedColormap.from_list(
    'nx_purple',
    ['#F5F4FE', '#CECBF6', '#AFA9EC', '#8B85DC', '#6D64CC', '#3C3489'],
    N=256
)

def draw_cm(ax, cm_pct, cm_cnt, title, subtitle, cmap, thresh=45):
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100,
                   aspect='equal', interpolation='nearest')
    n = len(LABELS)
    for i in range(n):
        for j in range(n):
            v_pct = cm_pct[i, j]
            v_cnt = cm_cnt[i, j]
            tc = 'white' if v_pct > thresh else '#1a1a1a'
            fw = 'bold' if i == j else 'normal'
            fs = 10 if i == j else 8.5
            ax.text(j, i, f'{v_pct:.0f}%',
                    ha='center', va='center',
                    fontsize=fs, color=tc, fontweight=fw)
            if i == j:
                ax.text(j, i + 0.32, f'({v_cnt})',
                        ha='center', va='center',
                        fontsize=6.5, color=tc, alpha=0.85)

    for i in range(n + 1):
        ax.axhline(i - 0.5, color='white', lw=1.2)
        ax.axvline(i - 0.5, color='white', lw=1.2)

    # 대각선 강조
    for i in range(n):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1,
                              lw=2.0, edgecolor='white',
                              facecolor='none', zorder=3)
        ax.add_patch(rect)

    ax.set_xticks(range(n))
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_xlabel('Predicted Domain', fontsize=10, labelpad=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_ylabel('True Domain', fontsize=10, labelpad=5)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
    ax.text(0.5, 1.03, subtitle, transform=ax.transAxes,
            ha='center', fontsize=9, color='#555555', style='italic')
    ax.spines[:].set_visible(False)
    plt.colorbar(im, ax=ax, label='Routing Accuracy (%)', shrink=0.85)

draw_cm(ax1, cm_ad_pct, cm_ad,
        'Adaptive RAG',
        f'Total accuracy: {acc_ad:.1f}%',
        cmap_ad, thresh=45)

draw_cm(ax2, cm_nx_pct, cm_nx,
        'NexRoute 7B (Ours)',
        f'Total accuracy: {acc_nx:.1f}%',
        cmap_nx, thresh=45)

plt.tight_layout()
plt.savefig('/home/sclab/moe_model/fig3_confusion_matrix_real.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/home/sclab/moe_model/fig3_confusion_matrix_real.pdf',
            bbox_inches='tight', facecolor='white')
print('\n저장 완료: fig3_confusion_matrix_real.png/.pdf')