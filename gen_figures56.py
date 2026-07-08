import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.interpolate import make_interp_spline

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ══════════════════════════════════════════════════════════════
# FIGURE 5  —  CROSS-domain BERTScore / ROUGE-L / Hallucination
# ══════════════════════════════════════════════════════════════

models_7b  = ['Naive\nRAG', 'LoRA\nRAG', 'Adaptive\nRAG', 'HyDE', 'CRAG', 'NexRoute\n(Ours)']
models_72b = ['Naive\nRAG', 'LoRA\nRAG', 'Adaptive\nRAG', 'HyDE', 'CRAG', 'NexRoute\n(Ours)']

# --- BERTScore (CROSS) ---
bert_7b  = [88.64, 88.50, 88.44, 88.64, 88.02, 89.53]
bert_72b = [89.03, 89.03, 89.07, 89.02, 89.10, 88.96]   # NexRoute 72B: 실측 88.96
naive_ref_bert = 89.03

# --- ROUGE-L (CROSS) ---
rouge_7b  = [24.95, 24.60, 24.25, 24.57, 21.63, 29.60]
rouge_72b = [28.81, 28.70, 28.54, 28.61, 28.95, 27.76]   # NexRoute 72B: 실측 27.76
naive_ref_rouge = 28.81

# --- Hallucination rate (CROSS, updated FActScore) ---
hall_7b  = [4.4,  None, 6.4,  4.8,  5.6,  4.6]
hall_72b = [7.1,  None, 7.0,  7.0,  8.6,  8.3]
naive_ref_hall = 7.1

C_7B_NR   = '#1B4F72'
C_7B_BASE = '#AED6F1'
C_72B_NR  = '#1E8449'
C_72B_BASE= '#A9DFBF'

fig5, axes = plt.subplots(1, 3, figsize=(16, 5))

x = np.arange(len(models_7b))
w = 0.35

def color_7b(i):
    return C_7B_NR if i == len(models_7b)-1 else C_7B_BASE

def color_72b(i):
    return C_72B_NR if i == len(models_7b)-1 else C_72B_BASE

# ── subplot 1: BERTScore ──────────────────────────────────────
ax = axes[0]
for i in range(len(models_7b)):
    ax.bar(x[i]-w/2, bert_7b[i],  width=w, color=color_7b(i),  zorder=3)
    ax.bar(x[i]+w/2, bert_72b[i], width=w, color=color_72b(i), zorder=3)

ax.text(x[-1]-w/2, bert_7b[-1]+0.05,  f'{bert_7b[-1]}',
        ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_7B_NR)
ax.text(x[-1]+w/2, bert_72b[-1]+0.05, f'{bert_72b[-1]}',
        ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_72B_NR)

ax.axhline(naive_ref_bert, color='#E74C3C', lw=1.2, ls='--', zorder=2)
ax.text(0.02, naive_ref_bert-0.18, f'Naive RAG 72B ({naive_ref_bert})',
        fontsize=7, color='#E74C3C', transform=ax.get_yaxis_transform())

ax.set_xticks(x); ax.set_xticklabels(models_7b, fontsize=8)
ax.set_ylabel('BERTScore', fontsize=9)
ax.set_title('CROSS-domain BERTScore', fontsize=10, fontweight='bold')
ax.set_ylim(60, 92)
ax.yaxis.grid(True, lw=0.5, alpha=0.5); ax.set_axisbelow(True)

# ── subplot 2: ROUGE-L ────────────────────────────────────────
ax = axes[1]
for i in range(len(models_7b)):
    ax.bar(x[i]-w/2, rouge_7b[i],  width=w, color=color_7b(i),  zorder=3)
    ax.bar(x[i]+w/2, rouge_72b[i], width=w, color=color_72b(i), zorder=3)

ax.text(x[-1]-w/2, rouge_7b[-1]+0.2,  f'{rouge_7b[-1]}',
        ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_7B_NR)
ax.text(x[-1]+w/2, rouge_72b[-1]+0.2, f'{rouge_72b[-1]}',
        ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_72B_NR)

ax.axhline(naive_ref_rouge, color='#E74C3C', lw=1.2, ls='--', zorder=2)
ax.text(0.02, naive_ref_rouge-0.8, f'Naive RAG 72B ({naive_ref_rouge})',
        fontsize=7, color='#E74C3C', transform=ax.get_yaxis_transform())

ax.set_xticks(x); ax.set_xticklabels(models_7b, fontsize=8)
ax.set_ylabel('ROUGE-L', fontsize=9)
ax.set_title('CROSS-domain ROUGE-L', fontsize=10, fontweight='bold')
ax.set_ylim(20, 35)
ax.yaxis.grid(True, lw=0.5, alpha=0.5); ax.set_axisbelow(True)

# ── subplot 3: Hallucination rate ────────────────────────────
ax = axes[2]
skip_idx = 1
models_h = [m for i, m in enumerate(models_7b) if i != skip_idx]
h7  = [v for i, v in enumerate(hall_7b)  if i != skip_idx]
h72 = [v for i, v in enumerate(hall_72b) if i != skip_idx]
xh  = np.arange(len(models_h))

for i in range(len(models_h)):
    c7  = C_7B_NR  if i == len(models_h)-1 else C_7B_BASE
    c72 = C_72B_NR if i == len(models_h)-1 else C_72B_BASE
    ax.bar(xh[i]-w/2, h7[i],  width=w, color=c7,  zorder=3)
    ax.bar(xh[i]+w/2, h72[i], width=w, color=c72, zorder=3)

ax.text(xh[-1]-w/2, h7[-1]+0.1,  f'{h7[-1]}%',
        ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_7B_NR)
ax.text(xh[-1]+w/2, h72[-1]+0.1, f'{h72[-1]}%',
        ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_72B_NR)

ax.axhline(naive_ref_hall, color='#E74C3C', lw=1.2, ls='--', zorder=2)
ax.text(0.02, naive_ref_hall-0.4, f'Naive RAG 72B ({naive_ref_hall}%)',
        fontsize=7, color='#E74C3C', transform=ax.get_yaxis_transform())

ax.set_xticks(xh); ax.set_xticklabels(models_h, fontsize=8)
ax.set_ylabel('Hallucination rate (%)', fontsize=9)
ax.set_title('Hallucination rate ↓', fontsize=10, fontweight='bold')
ax.set_ylim(0, 12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.yaxis.grid(True, lw=0.5, alpha=0.5); ax.set_axisbelow(True)

# ── shared legend ─────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_7B_NR,   label='NexRoute 7B (Ours)'),
    mpatches.Patch(color=C_7B_BASE, label='Baselines 7B'),
    mpatches.Patch(color=C_72B_NR,  label='NexRoute 72B (Ours)'),
    mpatches.Patch(color=C_72B_BASE,label='Baselines 72B'),
    plt.Line2D([0],[0], color='#E74C3C', lw=1.2, ls='--',
               label='Naive RAG 72B reference'),
]
fig5.legend(handles=legend_handles, loc='upper center',
            ncol=5, fontsize=8.5, frameon=False,
            bbox_to_anchor=(0.5, 1.01))

fig5.tight_layout(rect=[0, 0, 1, 0.97])
fig5.savefig('/home/sclab/moe_model/figure5_cross_quality.png',
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig5)
print("Figure 5 saved → figure5_cross_quality.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 6  —  Hallucination by model (7B) + IDI elimination
# ══════════════════════════════════════════════════════════════

fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig6.suptitle('NexRoute:  Dual-Axis Nexus Routing for LLMs',
              fontsize=12, fontweight='bold', y=1.01)

# ── Left: bar chart 7B hallucination (ALL) — 내림차순 정렬 ────
raw_models = ['Adaptive\nRAG 7B', 'CRAG 7B', 'Naive RAG\n7B', 'HyDE 7B', 'NexRoute\n7B (Ours)']
raw_hall   = [12.1, 11.1, 10.6, 9.8, 8.9]   # 내림차순
colors_bar = ['#AED6F1','#AED6F1','#AED6F1','#AED6F1','#1B4F72']

bars = ax1.bar(raw_models, raw_hall, color=colors_bar,
               width=0.55, zorder=3, edgecolor='white', linewidth=0.5)

# 각 막대 위 수치 표시
for bar, val, col in zip(bars, raw_hall, colors_bar):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.15,
             f'{val}%', ha='center', va='bottom', fontsize=9,
             fontweight='bold', color=col)

# NexRoute 감소율 annotation
reduction = (raw_hall[0] - raw_hall[-1]) / raw_hall[0] * 100
ax1.annotate('', xy=(4, raw_hall[-1]), xytext=(0, raw_hall[0]),
             arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
ax1.text(4.35, (raw_hall[0] + raw_hall[-1]) / 2,
         f'−{reduction:.1f}%', fontsize=10, fontweight='bold',
         color='#E74C3C', va='center')

ax1.set_ylabel('Hallucination rate (%)', fontsize=10)
ax1.set_title('Hallucination rate by model (7B scale)',
              fontsize=10, fontweight='bold')
ax1.set_xlim(-0.5, 5.2)
ax1.set_ylim(0, 15)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}%'))
ax1.yaxis.grid(True, lw=0.5, alpha=0.5); ax1.set_axisbelow(True)
ax1.tick_params(axis='x', labelsize=9)

# ── Right: IDI elimination horizontal bars ────────────────────
C_IDI_ON  = '#E8A598'
C_IDI_OFF = '#2E86C1'

labels_idi = ['Naive RAG 7B\n(w/ IDI)', 'NexRoute 7B\n(IDI = 0)']
vals_idi   = [10.6, 8.9]
colors_idi = [C_IDI_ON, C_IDI_OFF]

ypos = [1, 0]
for y, v, c, lb in zip(ypos, vals_idi, colors_idi, labels_idi):
    ax2.barh(y, v, height=0.45, color=c, zorder=3)
    ax2.text(v + 0.1, y, f'{v}%',
             va='center', fontsize=11, fontweight='bold', color=c)

ax2.text(vals_idi[0]/2, 1, 'IDI(q) = 0',
         ha='center', va='center', fontsize=9,
         color='white', fontweight='bold')

ax2.set_yticks(ypos)
ax2.set_yticklabels(labels_idi, fontsize=10)
ax2.set_xlabel('Hallucination rate (%)', fontsize=10)
ax2.set_title('Impact of IDI elimination\non hallucination rate',
              fontsize=10, fontweight='bold')
ax2.set_xlim(0, 14)
ax2.xaxis.grid(True, lw=0.5, alpha=0.5); ax2.set_axisbelow(True)

fig6.tight_layout()
fig6.savefig('/home/sclab/moe_model/figure6_hallucination.png',
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig6)
print("Figure 6 saved → figure6_hallucination.png")
