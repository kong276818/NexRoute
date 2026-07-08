import json
import numpy as np

with open("/home/sclab/moe_model/sensitivity_grid_result.json") as f:
    result = json.load(f)

tau_range   = np.array(result["tau_range"])
delta_range = np.array(result["delta_range"])
acc_grid    = np.array(result["acc_grid"])
best_acc    = result["best_acc"]

tie_mask = np.isclose(acc_grid, best_acc)
tie_idx  = np.argwhere(tie_mask)

tau_tied   = sorted(set(tau_range[i] for i, j in tie_idx))
delta_tied = sorted(set(delta_range[j] for i, j in tie_idx))

print(f"동률 지점: {len(tie_idx)}/{acc_grid.size}개 ({len(tie_idx)/acc_grid.size*100:.1f}%)")
print(f"동률 영역 tau 범위:   {min(tau_tied):.2f} ~ {max(tau_tied):.2f}")
print(f"동률 영역 delta 범위: {min(delta_tied):.2f} ~ {max(delta_tied):.2f}")

# 선택값(0.30, 0.40)이 동률 영역 안에서 어디쯔음인지
print(f"\n선택값 (tau=0.30, delta=0.40)이 동률 영역 안에 있는가:")
is_inside = (0.30 in tau_tied) and (0.40 in delta_tied)
print(f"  -> {'예, 정확히 동률 지점' if is_inside else '동률 지점 아님'}")

# 동률 영역이 연속적인 사각형인지, 흩어져 있는지 확인
full_grid_in_range = len(tau_tied) * len(delta_tied)
print(f"\ntau_tied x delta_tied 전체 격자 칸수: {full_grid_in_range}")
print(f"실제 동률 칸수: {len(tie_idx)}")
print(f"-> {'연속 사각형에 가까움' if len(tie_idx) >= full_grid_in_range*0.8 else '흩어진 패턴 (불연속)'}")

# 동률 아닌 지점들 (성능 하락 지역) 확인
print(f"\n성능이 떨어지는 경계 확인:")
worst_idx = np.unravel_index(np.argmin(acc_grid), acc_grid.shape)
print(f"  최저점: tau={tau_range[worst_idx[0]]:.2f}, delta={delta_range[worst_idx[1]]:.2f}, acc={acc_grid[worst_idx]:.2f}%")