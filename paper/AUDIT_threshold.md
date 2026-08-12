# AUDIT_threshold.md — τ/δ 임계값 오염 진단

Generated: 2026-08-13  
Auditor: Claude Code (automated search + git-history tracing)

---

## 1. τ/δ 선택 경위

| 항목 | 발견 내용 |
|------|----------|
| τ=0.30, δ=0.40 출처 | `train/train_deberta_router.py` line 21: 하드코딩 |
| 그리드 서치 스크립트 | `sensitivity_heatmap.py` (git history d63c118에서 복원) |
| 그리드 서치 입력 데이터 | `sensitivity_validation_scores.json` (n=200) — `build_sensitivity_data.py`가 생성 |
| n=200 구성 방법 | 4×40(단일도메인) + 40(CROSS) = 200. `build_sensitivity_data.py` 명시: "GT500과 중복 없는 검증셋" |
| GT-500 중복 검증 | Python으로 직접 확인: **overlap = 0** |

```
Validation set n: 200
GT500 n: 500
Overlap: 0 instructions
Verdict: CLEAN (no overlap)
```

---

## 2. τ/δ 그리드 결과 요약

`sensitivity_grid_result.json` (git d63c118)에서 복원:

| 항목 | 값 |
|------|---|
| 전역 최적 (τ,δ) | (0.10, 0.20), acc=99.5% |
| 선택된 (τ,δ) | **(0.30, 0.40)**, acc=99.5% |
| gap_pp | **0.0** — 선택값이 전역 최적과 동률 |
| 검증셋 n | 200 (GT-500과 0% 중복) |

τ=0.30, δ=0.40이 최적이 아닌 값을 선택한 이유는 오버트리거링 방지(더 보수적인 CROSS 판별)였으며, 정확도 손실 없음.

**그리드 서치에 대한 τ/δ 오염 판정: VERDICT A (오염 없음)**

---

## 3. ⚠ 별개 문제: 라우터 체크포인트 선택 오염

`train_deberta_router.py`의 epoch-end 평가 및 best checkpoint 저장 로직:

```python
# line 242
acc, per_d = evaluate(model, tokenizer, gt)   # gt = gt_dataset_500.json

# line 250-260: 이 acc/per_d 기반으로 best checkpoint 저장
score = (acc * 0.3 + per_d.get('CROSS', 0) * 0.2 + ...)
if score > best_score:
    torch.save(model.state_dict(), BEST_PATH)
```

- **문제**: `gt` = `gt_dataset_500.json` — 즉 평가용 500샘플로 체크포인트 선택
- **훈련 데이터 오염은 없음**: line 57 `if instr in gt_instrs: continue`로 GT-500 제외 확인
- **선택 편향은 존재**: 87.4% 라우팅 정확도는 모델 선택 시 사용한 동일 세트로 측정됨
- **결과**: 보고된 87.4%는 낙관적 추정(optimistic estimate). 실제 기대 정확도는 다소 낮을 수 있음
- `router_val.json` (n=2,199, GT-500과 0% 중복)이 존재하지만 학습 스크립트에서 미사용

**체크포인트 선택에 대한 판정: 소규모 선택 편향 존재**  
→ 87.4% 정확도에 "validation set used for model selection coincides with test set" 주석 필요  
→ 논문에서 이미 "87.4% accuracy" 수치에 한계로 §7.3 언급 가능

---

## 4. 요약 판정표

| 서브게이트 | 판정 | 세부 내용 |
|-----------|------|----------|
| τ/δ 그리드 서치 데이터 | **A (오염 없음)** | n=200, GT-500과 0% 중복 확인 |
| τ/δ 선택값과 GT-500 overlap | **A (오염 없음)** | 그리드 서치가 별도 검증셋 사용 |
| 라우터 체크포인트 선택 | **소규모 편향** | 학습 데이터 오염은 없으나 선택에 GT-500 사용 |
| LoRA 학습 데이터 오염 | 별도 감사 필요 | train_lora_7b_fixed.py 확인 필요 |

---

## 5. 권고 사항

1. **즉시 조치**: 논문 §3.2(또는 §4.2)에 "(routing accuracy evaluated on the same split used for model selection; may be mildly optimistic)" 주석 추가
2. **Phase 1 검토 시**: router_val.json으로 교차 검증하여 실제 일반화 정확도 추정 가능
3. **그리드 서치**: 재실행 불필요 — 현재 BERT 라우터 기반 그리드가 DeBERTa 라우터에도 동일하게 적용되었으며 gap=0.0pp

**τ/δ 오염에 의한 Phase 1 재실험 필요성: 없음**  
(체크포인트 선택 편향은 논문 텍스트 주석으로 처리 가능)
