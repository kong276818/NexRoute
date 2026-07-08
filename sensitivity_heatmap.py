import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCORES_PATH = "/home/sclab/moe_model/sensitivity_validation_scores.json"
DOMAINS = ['LAW', 'MED', 'SOC', 'STEM']

with open(SCORES_PATH) as f:
    data = json.load(f)
print(f"검증 데이터: {len(data)}개")

# ── 그리드 정의 (논문 본문과 동일 범위) ────────────────────────
tau_range   = np.round(np.arange(0.10, 0.55, 0.05), 2)
delta_range = np.round(np.arange(0.10, 0.65, 0.05), 2)

acc_grid = np.zeros((len(tau_range), len(delta_range)))

for ti, tau in enumerate(tau_range):
    for di, delta in enumerate(delta_range):
        correct = 0
        for item in data:
            scores = item['domain_scores']
            sorted_d = sorted(scores.items(), key=lambda x: -x[1])
            s1, s2 = sorted_d[0][1], sorted_d[1][1]
            top1 = sorted_d[0][0]
            is_cross_pred = (s2 >= tau) and ((s1 - s2) <= delta)
            pred = 'CROSS' if is_cross_pred else top1
            true = item['expected_domain']
            if pred == true:
                correct += 1
        acc_grid[ti, di] = correct / len(data) * 100

# ── 최적점 / 선택값 비교 ─────────────────────────────────────
best_idx   = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
best_tau   = tau_range[best_idx[0]]
best_delta = delta_range[best_idx[1]]
best_acc   = acc_grid[best_idx]

cur_ti  = int(np.argmin(np.abs(tau_range - 0.30)))
cur_di  = int(np.argmin(np.abs(delta_range - 0.40)))
cur_acc = acc_grid[cur_ti, cur_di]

print(f"\nGlobal optimum : tau={best_tau:.2f}, delta={best_delta:.2f}, acc={best_acc:.2f}%")
print(f"Selected (0.30, 0.40): acc={cur_acc:.2f}%")
print(f"Gap: {best_acc - cur_acc:.2f}%p")
print(f"Heatmap range: {acc_grid.min():.2f}% ~ {acc_grid.max():.2f}%")

# ── 결과 저장 (논문 본문 수치 업데이트용) ───────────────────────
with open("/home/sclab/moe_model/sensitivity_grid_result.json", "w") as f:
    json.dump({
        "tau_range": tau_range.tolist(),
        "delta_range": delta_range.tolist(),
        "acc_grid": acc_grid.tolist(),
        "best_tau": float(best_tau), "best_delta": float(best_delta),
        "best_acc": float(best_acc),
        "selected_tau": 0.30, "selected_delta": 0.40,
        "selected_acc": float(cur_acc),
        "gap_pp": float(best_acc - cur_acc),
        "n_validation": len(data)
    }, f, indent=2)

# ── 히트맵 플롯 ──────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    'nexroute_purple',
    ['#F5F4FE', '#CECBF6', '#AFA9EC', '#8B85DC', '#6D64CC', '#3C3489'],
    N=256
)

fig, ax = plt.subplots(figsize=(8, 6.2), facecolor='white')
im = ax.imshow(
    acc_grid, cmap=cmap, aspect='auto', origin='lower',
    extent=[delta_range[0]-0.025, delta_range[-1]+0.025,
            tau_range[0]-0.025, tau_range[-1]+0.025],
    vmin=acc_grid.min(), vmax=acc_grid.max()
)
cbar = plt.colorbar(im, ax=ax, label='Routing Accuracy (%)')

ax.scatter([best_delta], [best_tau], color='#D4537E', marker='*', s=300,
           edgecolor='white', linewidth=1.2, zorder=5,
           label=f'Global optimum ({best_acc:.2f}%)')
ax.scatter([0.40], [0.30], color='white', marker='o', s=140,
           edgecolor='#3C3489', linewidth=2, zorder=5,
           label=f'Selected τ=0.30, δ=0.40 ({cur_acc:.2f}%)')

ax.set_xlabel('δ (score gap threshold)', fontsize=11)
ax.set_ylabel('τ (activation threshold)', fontsize=11)
ax.set_title(f'CROSS Detection Sensitivity (N={len(data)} validation queries)',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

plt.tight_layout()
plt.savefig('/home/sclab/moe_model/fig2_sensitivity_heatmap.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/home/sclab/moe_model/fig2_sensitivity_heatmap.pdf',
            bbox_inches='tight', facecolor='white')
print("\n저장 완료: /home/sclab/moe_model/fig2_sensitivity_heatmap.png/.pdf")

# ── 동률 지점 확인 (99.50%를 달성하는 다른 (tau,delta) 조합들) ──
tie_mask = np.isclose(acc_grid, best_acc)
tie_coords = np.argwhere(tie_mask)
print(f"\n전역 최적({best_acc:.2f}%)과 동률인 지점: {len(tie_coords)}개")
if len(tie_coords) <= 15:
    for ti, di in tie_coords:
        print(f"  tau={tau_range[ti]:.2f}, delta={delta_range[di]:.2f}")