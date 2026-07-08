import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

labels = ['LAW', 'MED', 'SOC', 'STEM', 'CROSS']

cm_adaptive = np.array([
    [92, 2, 2, 2, 2],
    [ 4,84, 4, 4, 4],
    [ 8, 8,68, 8, 8],
    [ 4, 3, 3,87, 3],
    [ 2, 1, 1, 1,95],
], dtype=float)

cm_nexroute = np.array([
    [99, 1, 0, 0, 0],
    [ 4,85, 4, 4, 3],
    [ 6, 6,76, 6, 6],
    [ 5, 5, 5,80, 5],
    [ 1, 1, 1, 0,97],
], dtype=float)

cmap = LinearSegmentedColormap.from_list(
    'teal_navy',
    ['#f8fafc', '#e2f4f0', '#7ecfc0', '#2a9d8f', '#1a4a6e', '#0d2b45']
)

fig = plt.figure(figsize=(16, 7), facecolor='#ffffff')
gs  = fig.add_gridspec(1, 3, width_ratios=[11, 2.2, 11],
                       left=0.06, right=0.96,
                       top=0.92, bottom=0.13,
                       wspace=0.02)

ax_l = fig.add_subplot(gs[0])
ax_m = fig.add_subplot(gs[1])
ax_r = fig.add_subplot(gs[2])
ax_m.axis('off')

def draw_cm(ax, cm, title, subtitle, total_acc, show_ylabel=True):
    ax.set_facecolor('#ffffff')
    n      = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / n * 100

    ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100,
              aspect='equal', interpolation='nearest')

    for i in range(5):
        for j in range(5):
            val = cm_pct[i, j]
            txt = f'{val:.0f}%'
            if i == j:
                fc, fs, fw = 'white', 15, 700
            else:
                fc = '#94a3b8' if val < 10 else '#475569'
                fs, fw = 10.5, 400
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=fs, fontweight=fw, color=fc)

    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=10.5, color='#64748b', fontweight='500')
    ax.set_yticks(range(5))
    if show_ylabel:
        ax.set_yticklabels(labels, fontsize=10.5, color='#64748b', fontweight='500')
        ax.set_ylabel('True Label', fontsize=10.5, color='#94a3b8',
                      labelpad=12, fontstyle='italic')
    else:
        ax.set_yticklabels([])

    ax.set_xlabel('Predicted Label', fontsize=10.5, color='#94a3b8',
                  labelpad=12, fontstyle='italic')
    ax.tick_params(length=0, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.text(0.5, 1.10, title,
            transform=ax.transAxes, ha='center',
            fontsize=16, fontweight='bold', color='#0f172a')
    ax.text(0.5, 1.045, subtitle,
            transform=ax.transAxes, ha='center',
            fontsize=10, color='#94a3b8', fontstyle='italic')
    ax.text(0.5, 1.000, f'Total accuracy  {total_acc:.1f}%',
            transform=ax.transAxes, ha='center',
            fontsize=10.5, color='#2a9d8f', fontweight='600')

draw_cm(ax_l, cm_adaptive,
        'Adaptive RAG', 'TF-IDF + LR keyword routing', 85.2,
        show_ylabel=True)
draw_cm(ax_r, cm_nexroute,
        'NexRoute  (Ours)', 'Fine-tuned DeBERTa-large dual-axis', 87.4,
        show_ylabel=False)

ORANGE = '#f4a261'
for i in range(5):
    ax_l.add_patch(patches.Rectangle(
        (i-0.5, i-0.5), 1, 1,
        fill=False, edgecolor=ORANGE, linewidth=1.8,
        linestyle=(0, (5, 3)), zorder=5
    ))
    ax_r.add_patch(patches.Rectangle(
        (i-0.5, i-0.5), 1, 1,
        fill=False, edgecolor=ORANGE, linewidth=2.2,
        linestyle='-', zorder=5
    ))

pill = patches.FancyBboxPatch(
    (0.08, 0.28), 0.84, 0.44,
    boxstyle='round,pad=0.04',
    facecolor='#f1f5f9', edgecolor='#e2e8f0',
    linewidth=1, zorder=1,
    transform=ax_m.transAxes
)
ax_m.add_patch(pill)

ax_m.annotate('',
    xy=(0.82, 0.505), xytext=(0.18, 0.505),
    xycoords='axes fraction', textcoords='axes fraction',
    arrowprops=dict(arrowstyle='->', color='#2a9d8f',
                    lw=2.2, mutation_scale=16)
)

ax_m.text(0.5, 0.625, '+2.2%p',
          transform=ax_m.transAxes, ha='center', va='center',
          fontsize=13, fontweight='bold', color='#0f172a',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#d1fae5',
                    edgecolor='none'))

details = [
    ('SOC',  '68→76%', '+8.0%p', '#f4a261'),
    ('STEM', '87→80%', '−7.0%p', '#94a3b8'),
]
y = 0.40
for name, change, delta, color in details:
    ax_m.text(0.5, y, f'{name}  {delta}',
              transform=ax_m.transAxes, ha='center',
              fontsize=9, fontweight='600', color=color)
    ax_m.text(0.5, y - 0.07, change,
              transform=ax_m.transAxes, ha='center',
              fontsize=8.5, color='#94a3b8')
    y -= 0.20

ax_m.set_xlim(0, 1)
ax_m.set_ylim(0, 1)

plt.savefig('figure4_confusion_matrix.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: figure4_confusion_matrix.png")