import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── 실측 grid search 데이터 (논문 heatmap 값) ─────────────────────────────
tau_vals   = [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]
delta_vals = [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]

# 행=tau, 열=delta (논문 heatmap 수치)
grid = np.array([
    [97.4,97.3,97.3,97.2,97.2,97.2,97.2,97.1,97.1,96.8,96.8],
    [97.4,97.3,97.3,97.2,97.2,97.2,97.2,97.1,97.1,96.8,96.8],
    [97.4,97.3,97.3,97.2,97.2,97.2,97.2,97.1,97.1,96.8,96.8],
    [97.4,97.3,97.3,97.2,97.2,97.2,97.2,97.1,97.1,97.1,97.1],
    [97.4,97.3,97.3,97.2,97.2,97.2,97.2,97.2,97.2,97.2,97.2],
    [97.4,97.3,97.3,97.2,97.2,97.2,97.2,97.2,97.2,97.2,97.2],
    [97.4,97.3,97.3,97.3,97.3,97.3,97.3,97.3,97.3,97.3,97.3],
    [97.4,97.4,97.4,97.4,97.4,97.4,97.4,97.4,97.4,97.4,97.4],
    [97.5,97.5,97.5,97.5,97.5,97.5,97.5,97.5,97.5,97.5,97.5],
])

# τ sensitivity (δ=0.10 fixed) → delta 열 index 0
tau_curve  = grid[:, delta_vals.index(0.10)]   # δ=0.10 고정

# δ sensitivity (τ=0.50 fixed) → tau 행 index 8
delta_curve = grid[tau_vals.index(0.50), :]    # τ=0.50 고정

# ── Figure 설정 ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor='white')
gs  = fig.add_gridspec(2, 2,
                        height_ratios=[1, 1.4],
                        hspace=0.45, wspace=0.35,
                        left=0.07, right=0.76,
                        top=0.93, bottom=0.08)

ax_tau   = fig.add_subplot(gs[0, 0])
ax_delta = fig.add_subplot(gs[0, 1])
ax_heat  = fig.add_subplot(gs[1, :])

COL_LINE  = '#2a6db5'
COL_STAR  = '#d32f2f'
COL_GREEN = '#2e7d32'
COL_OPT   = '#ef5350'

# ── (1) τ sensitivity curve ───────────────────────────────────────────────
ax_tau.plot(tau_vals, tau_curve,
            'o-', color=COL_LINE, linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=2, zorder=3)

# optimal: argmax of tau_curve
opt_t_idx = int(np.argmax(tau_curve))
ax_tau.plot(tau_vals[opt_t_idx], tau_curve[opt_t_idx], '*', color=COL_STAR,
            markersize=14, zorder=5,
            label=f'Optimal $\\tau$={tau_vals[opt_t_idx]:.2f} ({tau_curve[opt_t_idx]:.2f}%)')
ax_tau.axvline(tau_vals[opt_t_idx], color=COL_OPT, linestyle='--',
               linewidth=1.2, alpha=0.6)

# adopted: marker는 δ=0.10 곡선 위(97.4%), label은 실제 채택값(τ=0.30, δ=0.40 → 97.2%)
adopted_t_curve_val = grid[tau_vals.index(0.30), delta_vals.index(0.10)]
adopted_t_label_val = grid[tau_vals.index(0.30), delta_vals.index(0.40)]
ax_tau.plot(0.30, adopted_t_curve_val, 'o', color=COL_GREEN,
            markersize=11, zorder=5,
            markerfacecolor=COL_GREEN,
            label=f'Adopted $\\tau$=0.30 ({adopted_t_label_val:.2f}%)')
ax_tau.axvline(0.30, color=COL_GREEN, linestyle='--',
               linewidth=1.2, alpha=0.6)

ax_tau.set_xlabel('$\\tau$ (min activation threshold)', fontsize=11)
ax_tau.set_ylabel('Routing accuracy (%)', fontsize=11)
ax_tau.set_title('$\\tau$ sensitivity ($\\delta$=0.10 fixed)',
                 fontsize=12, fontweight='bold')
ax_tau.set_xlim(0.08, 0.52)
ax_tau.set_ylim(96.5, 98.2)
ax_tau.set_xticks(tau_vals)
ax_tau.set_xticklabels([f'{v:.2f}' for v in tau_vals], fontsize=9)
# legend은 fig.legend()로 통합 — 개별 legend 없음
ax_tau.grid(True, alpha=0.3)
for sp in ax_tau.spines.values():
    sp.set_visible(False)

# ── (2) δ sensitivity curve ───────────────────────────────────────────────
ax_delta.plot(delta_vals, delta_curve,
              'o-', color=COL_LINE, linewidth=2, markersize=7,
              markerfacecolor='white', markeredgewidth=2, zorder=3)

# optimal: argmax of delta_curve
opt_d_idx = int(np.argmax(delta_curve))
ax_delta.plot(delta_vals[opt_d_idx], delta_curve[opt_d_idx], '*', color=COL_STAR,
              markersize=14, zorder=5,
              label=f'Optimal $\\delta$={delta_vals[opt_d_idx]:.2f} ({delta_curve[opt_d_idx]:.2f}%)')
ax_delta.axvline(delta_vals[opt_d_idx], color=COL_OPT, linestyle='--',
                 linewidth=1.2, alpha=0.6)

# adopted: δ=0.40 — marker는 τ=0.50 곡선 위, label은 실제 채택값(τ=0.30, δ=0.40)
adopted_d_curve_val = grid[tau_vals.index(0.50), delta_vals.index(0.40)]  # 97.5 (곡선 위치)
adopted_d_label_val = grid[tau_vals.index(0.30), delta_vals.index(0.40)]  # 97.2 (실제 채택값)
ax_delta.plot(0.40, adopted_d_curve_val,
              'o', color=COL_GREEN, markersize=11, zorder=5,
              markerfacecolor=COL_GREEN,
              label=f'Adopted $\\delta$=0.40 ({adopted_d_label_val:.2f}%)')
ax_delta.axvline(0.40, color=COL_GREEN, linestyle='--',
                 linewidth=1.2, alpha=0.6)

ax_delta.set_xlabel('$\\delta$ (max score gap)', fontsize=11)
ax_delta.set_ylabel('Routing accuracy (%)', fontsize=11)
ax_delta.set_title('$\\delta$ sensitivity ($\\tau$=0.50 fixed)',
                   fontsize=12, fontweight='bold')
ax_delta.set_xlim(0.08, 0.62)
ax_delta.set_ylim(96.5, 98.2)
ax_delta.set_xticks(delta_vals)
ax_delta.set_xticklabels([f'{v:.2f}' for v in delta_vals], fontsize=9)
# legend은 fig.legend()로 통합 — 개별 legend 없음
ax_delta.grid(True, alpha=0.3)
for sp in ax_delta.spines.values():
    sp.set_visible(False)

# ── (3) Joint heatmap ─────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    'blue_heatmap',
    ['#9ec9e2', '#6aadd5', '#3a87c0', '#1a5fa8', '#0d3a6e']
)

im = ax_heat.imshow(grid, cmap=cmap,
                    vmin=96.8, vmax=97.5,
                    aspect='auto', origin='upper')


# 셀 텍스트 — zorder=10으로 마커 위에 렌더링
for i in range(len(tau_vals)):
    for j in range(len(delta_vals)):
        val = grid[i, j]
        color = 'white' if val >= 97.2 else '#1a1a1a'
        ax_heat.text(j, i, f'{val:.1f}',
                     ha='center', va='center',
                     fontsize=8.5, color=color, zorder=10)

# heatmap markers: 셀 우측 상단 모서리에 배치 (숫자와 겹치지 않도록 offset)
ax_heat.plot(0.35, 7.65, '*', color=COL_STAR,  markersize=14, zorder=5,
             markeredgecolor='white', markeredgewidth=0.8)
ax_heat.plot(6.35, 3.65, 'o', color=COL_GREEN, markersize=10, zorder=5,
             markeredgecolor='white', markeredgewidth=0.8)

# 축
ax_heat.set_xticks(range(len(delta_vals)))
ax_heat.set_xticklabels([f'{v:.2f}' for v in delta_vals], fontsize=9)
ax_heat.set_yticks(range(len(tau_vals)))
ax_heat.set_yticklabels([f'{v:.2f}' for v in tau_vals], fontsize=9)
ax_heat.set_xlabel('$\\delta$ (max score gap)', fontsize=11)
ax_heat.set_ylabel('$\\tau$ (min activation threshold)', fontsize=11)

# 범례 타이틀 (동적 계산)
best_idx = np.unravel_index(np.argmax(grid), grid.shape)
best_tau_v   = tau_vals[best_idx[0]]
best_delta_v = delta_vals[best_idx[1]]
best_acc_v   = grid[best_idx]
paper_acc_v  = grid[tau_vals.index(0.30), delta_vals.index(0.40)]
title_str = (
    f'Joint grid search heatmap    '
    f'[$\\bigstar$] optimal: $\\tau$={best_tau_v:.2f}, $\\delta$={best_delta_v:.2f} $\\to$ {best_acc_v:.2f}%    '
    f'[$\\bullet$] paper: $\\tau$=0.30, $\\delta$=0.40 $\\to$ {paper_acc_v:.2f}%'
)
ax_heat.set_title(title_str, fontsize=10, pad=10)

# colorbar
cbar = plt.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)
cbar.set_label('Routing accuracy (%)', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# colorbar 눈금
cbar.set_ticks([96.8, 96.9, 97.0, 97.1, 97.2, 97.3, 97.4, 97.5])

# ── 통합 legend: 2줄로 figure 우측 상단에 1개만 표시 ──
from matplotlib.lines import Line2D
h_opt   = Line2D([0], [0], marker='*', color='w', markerfacecolor=COL_STAR,
                 markersize=14, markeredgecolor='white', markeredgewidth=0.5,
                 label='Optimal: $\\tau$=0.50, $\\delta$=0.10 (97.50%)')
h_adopt = Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_GREEN,
                 markersize=10, markeredgecolor='white', markeredgewidth=0.5,
                 label='Adopted: $\\tau$=0.30, $\\delta$=0.40 (97.20%)')
fig.legend(handles=[h_opt, h_adopt],
           loc='upper center',
           bbox_to_anchor=(0.5, 1.02),
           bbox_transform=fig.transFigure,
           borderaxespad=0, frameon=False,
           ncol=2,
           fontsize=10.5, markerscale=1.0,
           handlelength=1.5, handletextpad=0.8,
           columnspacing=2.0)

plt.savefig('figure2_sensitivity.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: figure2_sensitivity.png")