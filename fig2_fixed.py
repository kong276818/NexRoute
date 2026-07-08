import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── 캐시된 sensitivity 그리드 로드 ────────────────────────────────
with open('/home/sclab/moe_model/sensitivity_grid_result.json') as f:
    d = json.load(f)

grid   = np.array(d['acc_grid'])
taus   = d['tau_range']    # [0.10 ~ 0.50]
deltas = d['delta_range']  # [0.10 ~ 0.60]

best_idx   = np.unravel_index(grid.argmax(), grid.shape)
best_tau   = taus[best_idx[0]]
best_delta = deltas[best_idx[1]]
best_acc   = float(grid.max())

t_idx   = taus.index(0.30)
d_idx   = deltas.index(0.40)
sel_acc = float(grid[t_idx, d_idx])

print(f"Grid range: {grid.min():.1f}% ~ {grid.max():.1f}%")
print(f"Optimal:  τ={best_tau:.2f}, δ={best_delta:.2f} → {best_acc:.1f}%")
print(f"Adopted:  τ=0.30, δ=0.40 → {sel_acc:.1f}%")

# ── 색상 정의 ─────────────────────────────────────────────────────
C_BLUE  = '#1F77B4'
C_RED   = '#D62728'
C_GREEN = '#1B8A1E'
FONT    = 10

cmap = LinearSegmentedColormap.from_list(
    'nexroute',
    ['#F5F4FE', '#CECBF6', '#AFA9EC', '#8B85DC', '#6D64CC', '#3C3489'],
    N=256
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='white')

# ── Panel 1: τ sensitivity curve (δ=0.40 fixed) ──────────────────
ax = axes[0]
tau_curve = grid[:, d_idx]

ax.plot(taus, tau_curve, 'o-', color=C_BLUE, lw=2, ms=5, zorder=3)

# Red star: optimal τ
opt_t_idx = int(tau_curve.argmax())
ax.plot(taus[opt_t_idx], tau_curve[opt_t_idx], '*',
        color=C_RED, ms=14, zorder=5, markeredgecolor='white', markeredgewidth=0.8,
        label=f'Optimal τ={taus[opt_t_idx]:.2f} ({tau_curve[opt_t_idx]:.1f}%)')
ax.axvline(taus[opt_t_idx], color=C_RED, ls='--', lw=1.1, alpha=0.45)

# FIX 2: Green dot at adopted τ=0.30
ax.plot(0.30, tau_curve[t_idx], 'o',
        color=C_GREEN, ms=11, zorder=6,
        markeredgecolor='white', markeredgewidth=1.5,
        label=f'Adopted τ=0.30 ({tau_curve[t_idx]:.1f}%)')
ax.axvline(0.30, color=C_GREEN, ls=':', lw=1.2, alpha=0.6)

ax.set_xlabel('τ (min activation threshold)', fontsize=FONT)
ax.set_ylabel('Routing Accuracy (%)', fontsize=FONT)
ax.set_title('τ Sensitivity  (δ = 0.40 fixed)', fontsize=FONT + 1, fontweight='bold')
ax.set_ylim(grid.min() - 2, grid.max() + 1.5)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.tick_params(labelsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

# ── Panel 2: δ sensitivity curve (τ=0.30 fixed) ──────────────────
ax = axes[1]
delta_curve = grid[t_idx, :]

ax.plot(deltas, delta_curve, 'o-', color=C_BLUE, lw=2, ms=5, zorder=3)

# Red star: optimal δ
opt_d_idx = int(delta_curve.argmax())
ax.plot(deltas[opt_d_idx], delta_curve[opt_d_idx], '*',
        color=C_RED, ms=14, zorder=5, markeredgecolor='white', markeredgewidth=0.8,
        label=f'Optimal δ={deltas[opt_d_idx]:.2f} ({delta_curve[opt_d_idx]:.1f}%)')
ax.axvline(deltas[opt_d_idx], color=C_RED, ls='--', lw=1.1, alpha=0.45)

# FIX 2: Green dot at adopted δ=0.40
ax.plot(0.40, delta_curve[d_idx], 'o',
        color=C_GREEN, ms=11, zorder=6,
        markeredgecolor='white', markeredgewidth=1.5,
        label=f'Adopted δ=0.40 ({delta_curve[d_idx]:.1f}%)')
ax.axvline(0.40, color=C_GREEN, ls=':', lw=1.2, alpha=0.6)

ax.set_xlabel('δ (max score gap)', fontsize=FONT)
ax.set_ylabel('Routing Accuracy (%)', fontsize=FONT)
ax.set_title('δ Sensitivity  (τ = 0.30 fixed)', fontsize=FONT + 1, fontweight='bold')
ax.set_ylim(grid.min() - 2, grid.max() + 1.5)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.tick_params(labelsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

# ── Panel 3: Joint heatmap ────────────────────────────────────────
ax = axes[2]

# FIX 4: vmin=grid.min(), vmax=grid.max() → 실제 범위만 사용해 색상 대비 최대화
im = ax.imshow(
    grid, cmap=cmap, aspect='auto', origin='lower',
    extent=[deltas[0] - 0.025, deltas[-1] + 0.025,
            taus[0]   - 0.025, taus[-1]   + 0.025],
    vmin=grid.min(), vmax=grid.max()
)
cbar = plt.colorbar(im, ax=ax, label='Routing Accuracy (%)', shrink=0.96)
cbar.ax.tick_params(labelsize=8)

# FIX 1: Green solid dot (●) at adopted (τ=0.30, δ=0.40) — caption과 일치
ax.plot([0.40], [0.30], 'o',
        color=C_GREEN, ms=13, zorder=6,
        markeredgecolor='white', markeredgewidth=1.5,
        label=f'Adopted (τ=0.30, δ=0.40) ({sel_acc:.1f}%)')

# Red star at global optimum
ax.plot([best_delta], [best_tau], '*',
        color=C_RED, ms=14, zorder=7,
        markeredgecolor='white', markeredgewidth=0.8,
        label=f'Optimal (τ={best_tau:.2f}, δ={best_delta:.2f}) ({best_acc:.1f}%)')

ax.set_xlabel('δ (score gap threshold)', fontsize=FONT)
ax.set_ylabel('τ (activation threshold)', fontsize=FONT)
ax.set_title('Joint Sensitivity Heatmap', fontsize=FONT + 1, fontweight='bold')
ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
ax.tick_params(labelsize=8)

# ── 공통 타이틀 ───────────────────────────────────────────────────
fig.suptitle(
    f'NexRoute Routing Sensitivity Analysis  '
    f'(N = {d["n_validation"]} validation queries)',
    fontsize=12, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('/home/sclab/moe_model/fig2_sensitivity_final.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/home/sclab/moe_model/fig2_sensitivity_final.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved: fig2_sensitivity_final.png / .pdf")
