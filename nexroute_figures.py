import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ══════════════════════════════════════════════════════════════
# FIGURE 5  —  CROSS-domain BERTScore / ROUGE-L / Hallucination
# ══════════════════════════════════════════════════════════════

models_7b = ['Naive\nRAG', 'Adaptive\nRAG', 'HyDE', 'CRAG', 'NexRoute\n(Ours)']

# --- BERTScore (CROSS) ---
bert_7b  = [88.64, 88.44, 88.64, 88.02, 89.53]
bert_72b = [88.64, 89.07, 89.02, 89.10, 90.96]
naive_ref_bert = 88.03

# --- ROUGE-L (CROSS) ---
rouge_7b  = [24.95, 24.25, 24.57, 21.63, 29.60]
rouge_72b = [28.81, 28.54, 28.61, 28.95, 30.76]
naive_ref_rouge = 28.81

C_7B_NR   = '#1B4F72'
C_7B_BASE = '#AED6F1'
C_72B_NR  = '#1E8449'
C_72B_BASE= '#A9DFBF'

fig5, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(models_7b))
w = 0.35

def c7(i, n): return C_7B_NR  if i == n-1 else C_7B_BASE
def c72(i, n): return C_72B_NR if i == n-1 else C_72B_BASE

# ── subplot 1: BERTScore ──────────────────────────────────────
ax = axes[0]
n = len(models_7b)
for i in range(n):
    ax.bar(x[i]-w/2, bert_7b[i],  width=w, color=c7(i, n),  zorder=3)
    ax.bar(x[i]+w/2, bert_72b[i], width=w, color=c72(i, n), zorder=3)

# NexRoute 레이블: 7B 왼쪽, 72B 오른쪽
ax.annotate(f'{bert_7b[-1]}',
            xy=(x[-1]-w/2, bert_7b[-1]), xycoords='data',
            xytext=(-8, 3), textcoords='offset points',
            ha='right', va='bottom', fontsize=9, fontweight='bold', color=C_7B_NR)
ax.annotate(f'{bert_72b[-1]}',
            xy=(x[-1]+w/2, bert_72b[-1]), xycoords='data',
            xytext=(8, 3), textcoords='offset points',
            ha='left', va='bottom', fontsize=9, fontweight='bold', color=C_72B_NR)

ax.axhline(naive_ref_bert, color='#E74C3C', lw=1.2, ls='--', zorder=2)
ax.text(0.02, naive_ref_bert-0.12, f'Naive RAG 72B ({naive_ref_bert})',
        fontsize=7, color='#E74C3C', transform=ax.get_yaxis_transform())

ax.set_xticks(x); ax.set_xticklabels(models_7b, fontsize=8)
ax.set_ylabel('BERTScore (RoBERTa-Large)', fontsize=9)
ax.set_title('CROSS-domain BERTScore', fontsize=10, fontweight='bold')
ax.set_ylim(86.5, 92.5)
ax.yaxis.grid(True, lw=0.5, alpha=0.5); ax.set_axisbelow(True)

# ── subplot 2: ROUGE-L ────────────────────────────────────────
ax = axes[1]
for i in range(n):
    ax.bar(x[i]-w/2, rouge_7b[i],  width=w, color=c7(i, n),  zorder=3)
    ax.bar(x[i]+w/2, rouge_72b[i], width=w, color=c72(i, n), zorder=3)

ax.annotate(f'{rouge_7b[-1]}',
            xy=(x[-1]-w/2, rouge_7b[-1]), xycoords='data',
            xytext=(-8, 3), textcoords='offset points',
            ha='right', va='bottom', fontsize=9, fontweight='bold', color=C_7B_NR)
ax.annotate(f'{rouge_72b[-1]}',
            xy=(x[-1]+w/2, rouge_72b[-1]), xycoords='data',
            xytext=(8, 3), textcoords='offset points',
            ha='left', va='bottom', fontsize=9, fontweight='bold', color=C_72B_NR)

ax.axhline(naive_ref_rouge, color='#E74C3C', lw=1.2, ls='--', zorder=2)
ax.text(0.02, naive_ref_rouge-0.55, f'Naive RAG 72B ({naive_ref_rouge})',
        fontsize=7, color='#E74C3C', transform=ax.get_yaxis_transform())

ax.set_xticks(x); ax.set_xticklabels(models_7b, fontsize=8)
ax.set_ylabel('ROUGE-L', fontsize=9)
ax.set_title('CROSS-domain ROUGE-L', fontsize=10, fontweight='bold')
ax.set_ylim(17.0, 34.0)
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
fig5.savefig('/home/sclab/moe_model/figure5_cross_quality_v2.png',
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig5)
print("Figure 5 v2 saved → figure5_cross_quality_v2.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 6  —  Hallucination by model (7B) + IDI elimination
# ══════════════════════════════════════════════════════════════

C_IDI_ON  = '#E8A598'
C_IDI_OFF = '#2E86C1'

fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig6.suptitle('NexRoute:  Dual-Axis Nexus Routing for LLMs',
              fontsize=12, fontweight='bold', y=1.01)

# ── Left: bar chart ───────────────────────────────────────────
models_all = ['Naive RAG\n7B', 'Adaptive\nRAG 7B',
              'HyDE 7B', 'CRAG 7B', 'NexRoute\n7B (Ours)']
hall_all   = [10.6, 12.1, 9.8, 11.1, 8.9]
colors_bar = ['#AED6F1','#AED6F1','#AED6F1','#AED6F1', C_7B_NR]

bars = ax1.bar(models_all, hall_all, color=colors_bar,
               width=0.55, zorder=3, edgecolor='white', linewidth=0.5)

for bar, val, col in zip(bars, hall_all, colors_bar):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.15,
             f'{val}%', ha='center', va='bottom', fontsize=9,
             fontweight='bold', color=col)

# 감소율: Naive RAG 7B 기준
naive_val   = hall_all[0]   # 10.6
nexroute_val = hall_all[-1]  # 8.9
reduction = (naive_val - nexroute_val) / naive_val * 100  # 16.0%

ax1.annotate('', xy=(4, nexroute_val + 0.2), xytext=(0, naive_val + 0.2),
             arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.8,
                             connectionstyle='arc3,rad=-0.25'))
ax1.text(2.0, max(hall_all) + 1.2,
         f'−{reduction:.1f}%', fontsize=11, fontweight='bold',
         color='#E74C3C', ha='center')

ax1.set_xlim(-0.5, 5.2)
ax1.set_ylim(0, 15)
ax1.set_ylabel('Hallucination rate (%)', fontsize=10)
ax1.set_title('Hallucination rate by model (7B scale)',
              fontsize=10, fontweight='bold')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax1.yaxis.grid(True, lw=0.5, alpha=0.5); ax1.set_axisbelow(True)
ax1.tick_params(axis='x', labelsize=9)

# ── Right: IDI elimination ────────────────────────────────────
labels_idi = ['Naive RAG 7B\n(w/ IDI)', 'NexRoute 7B\n(IDI = 0)']
vals_idi   = [10.6, 8.9]
colors_idi = [C_IDI_ON, C_IDI_OFF]
ypos = [1, 0]

for y, v, c in zip(ypos, vals_idi, colors_idi):
    ax2.barh(y, v, height=0.45, color=c, zorder=3)
    ax2.text(v + 0.1, y, f'{v}%', va='center', fontsize=11,
             fontweight='bold', color=c)

ax2.text(vals_idi[0] / 2, 1, 'IDI(q) = 0',
         ha='center', va='center', fontsize=9,
         color='white', fontweight='bold')

ax2.set_yticks(ypos); ax2.set_yticklabels(labels_idi, fontsize=10)
ax2.set_xlabel('Hallucination rate (%)', fontsize=10)
ax2.set_title('Impact of IDI elimination\non hallucination rate',
              fontsize=10, fontweight='bold')
ax2.set_xlim(0, 13)
ax2.xaxis.grid(True, lw=0.5, alpha=0.5); ax2.set_axisbelow(True)

fig6.tight_layout()
fig6.savefig('/home/sclab/moe_model/figure6_hallucination_v2.png',
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig6)
print("Figure 6 v2 saved → figure6_hallucination_v2.png")
