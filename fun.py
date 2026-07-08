import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 데이터 ────────────────────────────────────────────────────────────
models_7b  = ['Naive\nRAG', 'Adaptive\nRAG', 'HyDE', 'CRAG', 'NexRoute\n(Ours)']
models_72b = ['Naive\nRAG', 'Adaptive\nRAG', 'HyDE', 'CRAG']

bs_7b  = [88.64, 88.44, 88.64, 88.02, 89.53]
bs_72b = [89.03, 89.07, 89.02, 89.10]
bs_ref = 89.03

rl_7b  = [24.95, 24.25, 24.57, 21.63, 29.60]
rl_72b = [28.81, 28.54, 28.61, 28.95]
rl_ref = 28.81

hall_7b = [28.4, 32.1, 26.5, 24.8, 12.4]

# ── 색상 ─────────────────────────────────────────────────────────────
COL_OURS   = '#1f5fa6'
COL_OURS72 = '#1a6b3c'
COL_BASE7  = '#aac8e8'
COL_BASE72 = '#9ecfb4'
COL_REF    = '#c94a00'

FONT  = 9
BAR_W = 0.62
GAP   = 1.2   # x-gap between 7B and 72B groups

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor='white')


def draw_metric(ax, title, vals_7b, vals_72b, ref_val, ylabel, ymin, ymax):
    xs7  = np.arange(len(vals_7b),  dtype=float)
    xs72 = np.arange(len(vals_7b) + GAP, len(vals_7b) + GAP + len(vals_72b), dtype=float)
    span = ymax - ymin

    # 7B baselines
    ax.bar(xs7[:-1], vals_7b[:-1], width=BAR_W, color=COL_BASE7,  zorder=2)
    # NexRoute 7B
    ax.bar(xs7[-1],  vals_7b[-1],  width=BAR_W, color=COL_OURS,   zorder=2)
    ax.text(xs7[-1], vals_7b[-1] + span * 0.015,
            f'{vals_7b[-1]:.2f}', ha='center', va='bottom',
            fontsize=FONT - 1, fontweight='bold', color=COL_OURS)

    # 72B baselines
    ax.bar(xs72, vals_72b, width=BAR_W, color=COL_BASE72, zorder=2)

    # Reference line (value readable from y-axis; label in legend)
    ax.axhline(ref_val, color=COL_REF, linestyle='--', linewidth=1.1, zorder=3, alpha=0.85)

    # Group separator
    sep = (xs7[-1] + xs72[0]) / 2
    ax.axvline(sep, color='#cccccc', linewidth=0.8, zorder=1)

    # Axes
    ax.set_xticks(np.concatenate([xs7, xs72]))
    ax.set_xticklabels(models_7b + models_72b, fontsize=FONT - 1)
    ax.set_xlim(-0.55, xs72[-1] + 0.55)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=FONT + 2, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=FONT)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


draw_metric(axes[0], 'Cross-domain BERTScore (↑)',
            bs_7b, bs_72b, bs_ref, 'BERTScore', 86.5, 91.0)
draw_metric(axes[1], 'Cross-domain ROUGE-L (↑)',
            rl_7b, rl_72b, rl_ref, 'ROUGE-L', 18.0, 33.0)

# ── Hallucination panel ───────────────────────────────────────────────
ax = axes[2]
xs7h = np.arange(len(models_7b), dtype=float)
x72h = xs7h[-1] + GAP   # NexRoute 72B bar position

for x, v, c in zip(xs7h, hall_7b, [COL_BASE7] * 4 + [COL_OURS]):
    ax.bar(x, v, width=BAR_W, color=c, zorder=2)

# NexRoute 7B value label
ax.text(xs7h[-1], hall_7b[-1] + 0.7, f'{hall_7b[-1]:.1f}%',
        ha='center', va='bottom', fontsize=FONT - 1,
        fontweight='bold', color=COL_OURS)

# NexRoute 72B bar
ax.bar(x72h, 7.8, width=BAR_W, color=COL_OURS72, zorder=2)
ax.text(x72h, 7.8 + 0.7, '7.8%',
        ha='center', va='bottom', fontsize=FONT - 1,
        fontweight='bold', color=COL_OURS72)

# Group separator
ax.axvline((xs7h[-1] + x72h) / 2, color='#cccccc', linewidth=0.8, zorder=1)

# Reduction annotation (Naive RAG 7B → NexRoute 7B)
arrow_y = max(hall_7b) + 3.5
ax.annotate('', xy=(xs7h[-1], arrow_y), xytext=(xs7h[0], arrow_y),
            arrowprops=dict(arrowstyle='<->', color=COL_REF, lw=1.1))
ax.text((xs7h[0] + xs7h[-1]) / 2, arrow_y + 1.0, '−56.3%',
        ha='center', va='bottom', fontsize=FONT,
        fontweight='bold', color=COL_REF)

all_xs_h  = list(xs7h) + [x72h]
all_lbl_h = models_7b  + ['NexRoute\n(72B)']
ax.set_xticks(all_xs_h)
ax.set_xticklabels(all_lbl_h, fontsize=FONT - 1)
ax.set_xlim(-0.55, x72h + 0.55)
ax.set_ylim(0, 42)
ax.set_title('Hallucination Rate (↓)', fontsize=FONT + 2, fontweight='bold', pad=8)
ax.set_ylabel('Hallucination Rate (%)', fontsize=FONT)
ax.tick_params(length=0)
ax.yaxis.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_visible(False)

# ── 범례 (하단) ───────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=COL_OURS,   label='NexRoute 7B (Ours)'),
    mpatches.Patch(color=COL_OURS72, label='NexRoute 72B (Ours)'),
    mpatches.Patch(color=COL_BASE7,  label='Baselines (7B)'),
    mpatches.Patch(color=COL_BASE72, label='Baselines (72B)'),
    plt.Line2D([0], [0], color=COL_REF, linestyle='--', linewidth=1.1,
               label='Naive RAG 72B reference'),
]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=FONT, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('figure5_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: figure5_performance.png")
