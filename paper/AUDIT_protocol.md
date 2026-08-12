# AUDIT_protocol.md — 실험 프로토콜 인벤토리

Generated: 2026-08-13  
Auditor: Claude Code (script analysis + result file verification)

---

## 1. 전체 실험 맵

| 논문 표/절 | 서브셋 | n (overall/CROSS/Single) | 생성 스크립트 | 생성 결과 파일 | 스코어링 파일 |
|-----------|--------|--------------------------|--------------|--------------|--------------|
| **Table 6** (master_performance) | Clean-300 (gold_domain 기반) | 300 / 60 / 240 | `run_reinference_500.py` | `reinference_results/preds_500_nexroute.json` | `reinference_results/scores_500_nexroute.csv` |
| **Table 7** (hallucination) | Full-500 (for CROSS 3.6%) / Clean-300 (for 4.47%) | 500 / 100 (Full) | `eval/eval_factscore_hallucination.py` 또는 `run_exp2_hallucination.py` | `factscore_hallucination_results.json` | — |
| **Table 8** (ablation — Tab 11) | Full-500 | 500 / 100 / 400 | `run_ablation_complete.py` | `results/ablation/v01~v05_gen_500.json` | `results/ablation/ablation_generation_quality.json` |
| **Table 9** (funcaxis) Predicted condition | Clean-300 (CROSS n=88 by predicted router) | 300 / 88 / 212 | `run_function_axis_ablation.py` 또는 `run_func_ablation_v2.py` | — | — |
| **Table 11** (domainrouter, Exp2) | Clean-300 (gold_domain 기반) | 300 / 60 / 240 | `run_exp2_domain_router.py` | `results/exp2_domain_router.json` | — |

---

## 2. ⚠ 핵심 충돌 목록

### 충돌 A: NexRoute 7B 성능 수치 불일치 (중대)

| 표 | 서브셋 | Overall RL | CROSS RL | 사용 스크립트 | 스캐폴딩 여부 |
|----|--------|------------|----------|--------------|--------------|
| **Table 6** | Clean-300 | **24.29** | **28.23** | `run_reinference_500.py` | **없음** (도메인별 프롬프트) |
| **Table 8 V01** | Full-500 | **27.83** | **31.99** | `run_ablation_complete.py` | **있음** (WHAT/HOW/WHY/COMPARE) |
| **Table 9 Predicted** | Clean-300 | **29.31** | **33.58** | ablation V01 score | **있음** |

- Table 6의 "NexRoute 7B"와 Table 8의 "V01"은 동일 시스템이어야 하지만 **5pp 이상 차이**
- 원인: `run_reinference_500.py`는 함수 스캐폴딩(WHAT/HOW/WHY/COMPARE) 미사용; `run_ablation_complete.py`는 사용
- **영향**: Table 6이 NexRoute 성능을 저보고(under-report)하고 있거나, Table 8이 함수 스캐폴딩 혜택을 V01에 포함시킴

**해소 방법**: Table 6을 V01(함수 스캐폴딩 포함)로 재계산하거나, Table 6이 "NexRoute without function scaffold"임을 명시

---

### 충돌 B: Table 6 vs Table 11 Overall RL (Full-500 vs Clean-300)

| 표 | 서브셋 | NexRoute Overall RL |
|----|--------|-------------------|
| Table 6 footnote | Full-500 | 23.74 |
| Table 8 V01 | Full-500 | 27.83 |

- 동일 Full-500에서 23.74 vs 27.83 — 충돌 A와 같은 이유(스캐폴딩 여부)

---

### 충돌 C: CROSS n 정의 불일치 (기존 감사에서 확인)

| 표 | CROSS n | 정의 |
|----|---------|------|
| Table 6, 11 | **n=60** | gold_domain='CROSS' in Clean-300 |
| Table 8 (ablation) | **n=100** | gold_domain='CROSS' in Full-500 |
| Table 9 (funcaxis) | **n=88** | predicted as CROSS by router in Clean-300 |

- 세 개 모두 정합 가능 (다른 정의), 단 각 표 캡션에 정의 명시 필요
- Table 9 n=88 캡션에 "CROSS = queries predicted as CROSS by router (not gold_domain)" 이미 추가 완료 (previous session)

---

### 충돌 D: LAW ROUGE-L 세 값

| 위치 | RL | 기반 |
|------|----|------|
| §4.2/캡션 (main eval) | 18.27 | Clean-300 reinference run |
| §4.4 (degeneration) | 15.09 | Full-500 ablation run (v01) |
| §4.5 Exp2 predicted | 19.08 | Clean-300 Exp2 V01 재추론 |

- 세 값 모두 다른 run/subset — 각 위치에 이미 "(run ID/subset)" 레이블 추가됨 ✓

---

## 3. 스크립트 ↔ 결과파일 ↔ 논문 표 매핑

```
preds_500.csv
 └─ 생성: [origin script not in codebase; committed in b9eeeba with DeBERTa scores]
    ├─ 사용: run_reinference_500.py → reinference_results/preds_500_nexroute.json
    │              ↕ scored by score_reinference_500.py
    │         reinference_results/scores_500_nexroute.csv → TABLE 6
    │
    ├─ 사용: run_ablation_complete.py → results/ablation/v01~v05_gen_500.json
    │         ablation_generation_quality.json, ablation_hallucination_summary.json → TABLE 8, TABLE 11
    │
    ├─ 사용: run_exp2_domain_router.py (uses v01_gen_500.json for predicted condition)
    │         results/exp2_domain_router.json → TABLE 11 (domainrouter)
    │
    └─ 사용: run_function_axis_ablation.py / run_func_ablation_v2.py
              → TABLE 9 (funcaxis)
```

---

## 4. 시드 및 재현성

| 실험 | 시드 설정 | 복수 시드 여부 |
|------|----------|--------------|
| run_reinference_500.py | 없음 (temperature=0.7) | 단일 실행 |
| run_ablation_complete.py | SEED=42, numpy+torch | 단일 실행 |
| run_exp2_domain_router.py | SEED=42, numpy+torch | 단일 실행 |
| 논문 §4 본문 | "averaged over three seeds (std<0.8)" | **실제 3-seed 실험 확인 불가** |

- 논문 본문(line 936)에는 "three seeds"라고 기재되어 있으나 코드베이스에 3-seed 실행 스크립트 없음
- 단일 실행 결과가 보고된 것으로 보임 — **저자 확인 필요**

---

## 5. 해소 우선순위

| 우선순위 | 충돌 | 권고 조치 |
|---------|------|---------|
| 🔴 **즉시** | 충돌 A (Table 6 vs V01 5pp 갭) | 두 테이블이 서로 다른 NexRoute 버전임을 §4 도입부에 명시하거나 Table 6을 V01 기준으로 재산출 |
| 🟡 **Phase 1** | 3-seed 확인 | 단일 vs 3-seed 실제 여부 저자 확인 후 논문 수정 |
| 🟢 **이미 처리** | 충돌 C (CROSS n 정의) | Table 9 캡션 수정 완료 |
| 🟢 **이미 처리** | 충돌 D (LAW RL 세 값) | 각 위치에 run/subset 레이블 추가 완료 |
