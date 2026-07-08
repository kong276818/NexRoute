import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── 색상 팔레트 ─────────────────────────────────────────────
PURPLE_HEX = ['#EEEDFE','#C8C3F0','#AFA9EC','#8B85DC',
              '#6D64CC','#534AB7','#3C3489','#26215C']
CORAL_HEX  = ['#FAECE7','#F5C4B3','#F0997B','#E07050',
               '#C85030','#A03A1A','#712B13','#4A1B0C']

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       10,
    'axes.linewidth':  0.6,
    'xtick.major.width':0.6,
    'ytick.major.width':0.6,
})

labels = ['LAW', 'MED', 'SOC', 'STEM', 'CROSS']

# ── 혼동 행렬 ───────────────────────────────────────────────
adaptive_cm = np.array([
    [99,  0,  0,  0,  1],
    [ 2, 92,  3,  2,  1],
    [ 2,  5, 80,  8,  5],
    [ 1,  4,  8, 83,  4],
    [ 2,  2,  3,  2, 91],
], dtype=float)

nexroute_cm = np.array([
    [99,  0,  0,  0,  1],
    [ 1, 97,  1,  1,  0],
    [ 1,  2, 93,  3,  1],
    [ 0,  2,  4, 93,  1],
    [ 1,  1,  1,  2, 95],
], dtype=float)

A = adaptive_cm / adaptive_cm.sum(axis=1, keepdims=True) * 100
N = nexroute_cm / nexroute_cm.sum(axis=1, keepdims=True) * 100

# ── Figure 설정 ──────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6.5), facecolor='white') # 세로 여백 조금 더 확보

# 전체 제목의 절대 좌표(y=0.96)
fig.suptitle('Routing Confusion Matrices: Adaptive RAG vs NexRoute',
             fontsize=14, fontweight='bold', y=0.96, color='#1a1a1a')

# top=0.82로 낮춰서 타이틀과 그래프 사이의 공간을 아주 넉넉하게 줌
gs = gridspec.GridSpec(1, 3,
                       width_ratios=[1, 0.05, 1],
                       left=0.07, right=0.95, # 오른쪽 여백 확보
                       bottom=0.12, top=0.82, wspace=0.05)

ax_a   = fig.add_subplot(gs[0])
ax_gap = fig.add_subplot(gs[1])
ax_n   = fig.add_subplot(gs[2])
ax_gap.axis('off')

def draw_cm(ax, data, title, subtitle, cmap, text_thresh=50, show_ylab=True):
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100,
                   aspect='equal', interpolation='nearest')

    n = len(labels)
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            tc = 'white' if v > text_thresh else '#2a2a2a'
            fw = 'bold' if i == j else 'normal'
            fs = 10.5 if i == j else 9
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                    fontsize=fs, color=tc, fontweight=fw)

    for i in range(n + 1):
        ax.axhline(i - 0.5, color='white', lw=1.5)
        ax.axvline(i - 0.5, color='white', lw=1.5)

    for i in range(n):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                              linewidth=2.0,
                              edgecolor='white',
                              facecolor='none', zorder=3)
        ax.add_patch(rect)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel('Predicted Label', fontsize=10, labelpad=6)
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='x', which='both', bottom=False)

    if show_ylab:
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_ylabel('True Label', fontsize=10, labelpad=6)
        ax.tick_params(axis='y', which='both', left=False)
    else:
        ax.set_yticks(range(n))
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', left=False)

    # 개별 그래프 타이틀
    ax.set_title(title, fontsize=13, fontweight='bold', color='#1a1a1a', pad=25)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10.5,
            color='#666666', style='italic')
    
    ax.spines[:].set_visible(False)
    return im

# ── Adaptive RAG ─────────────────────────────────────────────
cmap_a = plt.cm.Blues
im_a = draw_cm(ax_a, A,
               title='Adaptive RAG',
               subtitle='Total accuracy: 89.4%',
               cmap=cmap_a,
               text_thresh=45,
               show_ylab=True)

# ── NexRoute ─────────────────────────────────────────────────
purple_cmap = LinearSegmentedColormap.from_list(
    'nexroute_purple',
    ['#F5F4FE', '#CECBF6', '#AFA9EC', '#8B85DC',
     '#6D64CC', '#3C3489', '#26215C'],
    N=256
)
im_n = draw_cm(ax_n, N,
               title='NexRoute (Ours)',
               subtitle='Total accuracy: 95.7%',
               cmap=purple_cmap,
               text_thresh=45,
               show_ylab=False)

# ── 개선폭 박스 및 주석 ──────────────────────────────────────────
for ax, is_right in [(ax_a, False), (ax_n, True)]:
    for col_idx, col_label in [(1,'MED'),(3,'STEM')]:
        val_a = A[2, col_idx]
        val_n = N[2, col_idx]
        if val_a > 2 or val_n > 1:
            rect = plt.Rectangle(
                (col_idx - 0.5, 1.5), 1, 1,
                linewidth=1.5,
                edgecolor='#FF6B35' if (is_right and val_n < val_a) else '#FF6B35',
                facecolor='none', zorder=4,
                linestyle='--' if not is_right else '-'
            )
            ax.add_patch(rect)

ax_gap.set_xlim(0, 1); ax_gap.set_ylim(0, 5)
ax_gap.annotate('', xy=(0.85, 2.5), xytext=(0.15, 2.5),
    xycoords='data', textcoords='data',
    arrowprops=dict(arrowstyle='->,head_width=0.25', color='#3C3489', lw=2.0))
ax_gap.text(0.50, 2.85, '+6.3%p', ha='center', va='bottom',
            fontsize=9, color='#3C3489', fontweight='bold', transform=ax_gap.transData)

for y_pos, domain, adp_acc, nxr_acc in [
    (1.0, 'SOC',  '80.0%', '93.0%'),
    (0.4, 'STEM', '83.0%', '93.0%'),
]:
    ax_gap.text(0.5, y_pos+0.3, f'{domain}', ha='center',
                fontsize=8, color='#712B13', fontweight='bold', transform=ax_gap.transData)
    ax_gap.text(0.5, y_pos-0.05, f'{adp_acc}→{nxr_acc}',
                ha='center', fontsize=7.5, color='#444', transform=ax_gap.transData)
    ax_gap.annotate('', xy=(0.75, y_pos+0.12), xytext=(0.25, y_pos+0.12),
        arrowprops=dict(arrowstyle='->,head_width=0.12', color='#E07050', lw=1.2))


# [수정된 부분] Key Improvements 요약 박스를 Figure 전체 좌표(0~1) 기준으로 최상단 우측 구석에 고정
fig.text(0.98, 0.98, '▲ Key improvements:\nSOC: +13.0%p  STEM: +10.0%p',
         ha='right', va='top', fontsize=9, color='#333333',
         bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='#F0F8E8', edgecolor='#5CB88A',
                   alpha=0.9))

# ── 저장 ─────────────────────────────────────────────────────
save_dir = './outputs/'
os.makedirs(save_dir, exist_ok=True)

png_path = os.path.join(save_dir, 'fig3_confusion_matrix.png')
pdf_path = os.path.join(save_dir, 'fig3_confusion_matrix.pdf')

plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

print("저장 완료! 요약 박스를 우측 상단 모서리로 완전히 분리했습니다.")