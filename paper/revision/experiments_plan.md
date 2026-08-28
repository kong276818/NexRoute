# STEP 4 — 실험 재설계 계획 (Experiments Plan)

## 현재 실험의 상태 평가

### 이미 충족된 항목
- ✅ E2 (Ablation): V02~V05 ablation이 Table 5에 완비 (두 축 각각 제거 결과)
- ✅ E3 (Routing behavior): Routing entropy, confusion matrix, misrouted sample analysis 존재
- ✅ E5 (통계적 유의성): 3 seed, 평균±표준편차, paired bootstrap + Holm correction
- ✅ E6 (비용): VRAM 14.8GB, latency 보고 (Supplementary S2)

### 반드시 추가/수정 필요한 항목
- ❌ E1 (가설 검증 실험): 존재하지 않음 → **최우선 추가**
- ❌ E4 (최신 강베이스라인): Typed-RAG 결과 미완성 (\placeholder), 추가 베이스라인 없음
- ❌ 외부 벤치마크: MedQA, MMLU-Law 결과 미완성 (\placeholder)
- ❌ LAW v4 adapter 결과: 미완성 (\placeholder)

---

## E1 — Hypothesis Test: Proposition 1 검증 실험

**목적**: "function heterogeneity가 증가할수록 single-axis routing의 품질 저하가 커진다"는 Proposition 1을 인위적으로 만든 실험으로 검증.

### 설계

**Factor 1: Function Heterogeneity** ($H$)
- 동일 도메인(예: MED) 내에서 What/How/Why/Compare 비율을 조절
- 설정: 균등 분포(H=high), 하나의 function 80% 집중(H=low), 두 function 50/50(H=medium)

**Factor 2: Domain Shift Intensity** (CROSS query 비율)
- 0%, 20%, 40%, 60%, 80% CROSS-domain 쿼리 비율 스윕

**Factor 3: Retrieval Partition** (ablation과 연계)
- Unified corpus (no partitioning) vs. Domain-partitioned

**Measured outcomes**:
- ROUGE-L gap: dual-axis vs. single-axis (domain-only)
- ROUGE-L gap: dual-axis vs. function-only (Typed-RAG)
- NLI contradiction rate

**데이터 소스**: 기존 eval 500샘플을 resampling하여 원하는 factor 분포로 서브셋 구성
(새로운 LLM generation 없이 기존 결과 재분석으로 가능)

### 실행 방법 (GPU 없이 가능한 버전)

```bash
# E1 스크립트 생성 후 실행:
# 1. preds_500.csv 또는 v01_gen_500.json에서 gold function label로 층화 샘플링
# 2. function 분포별 ROUGE-L 비교
python scripts/run_e1_hypothesis.py \
    --pred_file results/ablation/v01_gen_500.json \
    --gold_file processed_v2/gt_dataset_500.json \
    --output_dir results/e1_hypothesis \
    --n_sweeps 5
```

**기존 데이터로 할 수 있는 분석**:
- 500샘플을 function label별로 층화
- V01 vs V02 (function scaffold 유/무)의 ROUGE-L gap을 function 균등도별로 비교
- 이미 있는 v01, v02 generation 결과를 활용하므로 GPU 불필요

---

## E2 — Ablation (기존 Table 5 유지 + 보강)

### 현재 상태
Table 5 (V01~V05 ablation)이 이미 완비되어 있음.

### 추가 권장
- **V06 (Domain Axis only, no function scaffold)**: 도메인 축만 있고 기능 축 없는 경우
  → 현재 Tab master_performance가 이에 해당 (function scaffold 없이 평가)
  → Table 4와 Table 5의 프로토콜 차이를 더 명확히 서술 필요

### 해석 강화
- "두 축이 모두 필요함"을 더 직접적으로 표현:
  V04 (no LoRA, -9.43 RL) + V02 (no scaffold, -1.32 RL)의 합이 V05 (no domain axis, -10.93 RL)와 근접
  → Domain axis = LoRA + retrieval partition의 결합 효과

---

## E3 — Routing Behavior 분석 (기존 분석 보강)

### 현재 있는 것
- Routing entropy (Table 7, Section 7.2)
- Confusion matrix (Figure 4)
- Misrouted sample analysis (Table 2)

### 추가 권장
**Expert utilization 히스토그램**:
- 각 도메인 쿼리가 어떤 adapter로 라우팅되는지 분포 시각화
- 예상: CROSS query의 다수가 CROSS adapter로, LAW query의 대다수가 LAW adapter로

**Function axis 라우팅 결정 시각화**:
- Function label 분포 per domain (LAW는 Why가 많은가? MED는 What이 많은가?)
- 이 분포가 function scaffold 효과를 설명하는지 분석

---

## E4 — 최신 강베이스라인 추가

### 현재 상태
- Naive RAG, TF-IDF Router, HyDE, Simplified CRAG, Typed-RAG (미완성)
- Arrow, PHATGOOSE (Table 7에만, CROSS 100샘플)

### [VERIFY] 추가 필요 베이스라인
다음 3개 중 최소 2개 추가 (동일 backbone, 동일 예산):

1. **Self-RAG (Asai 2023)**: 이미 Related Work에서 언급됨; Clean-300으로 평가 추가
2. **HippoRAG (Gutierrez 2024)**: 최신 graph-based RAG; cross-domain query에서의 비교
3. **RAPTOR (Sarthi 2024)**: 계층적 retrieval; CROSS query에서의 성능

> [VERIFY]: GPU 가용 시 위 3개 중 가장 구현 용이한 1~2개 선택. 
> GPU 없을 경우: 기존 논문에서 보고된 ROUGE-L/BERTScore 수치를 직접 비교하고,
> backbone/dataset 불일치를 명시하여 참고값으로 제시.

### 즉시 추가 가능한 것
- Typed-RAG 7B: `scripts/run_typedrag_baseline.py` 실행 필요 [VERIFY: GPU 필요]
- MedQA 72B baseline도 함께 보고하면 강도 높아짐

---

## E5 — 통계적 유의성 (기존 충족, 보강 권장)

### 현재 상태
- 3 seed, std < 0.8 (NexRoute 7B) 보고 ✅
- Paired bootstrap (10,000 resamples), Holm correction ✅
- Cluster bootstrap for NLI contradiction rate ✅

### 보강 권장
**Cohen's d (effect size) 추가**:
```
Clean-300 CROSS ROUGE-L 차이 (28.23 vs 22.17): Cohen's d ≈ 0.30~0.40 (보고됨)
```
- 이미 caption에 언급되어 있으나 본문에서 더 두드러지게 서술 필요

**Bootstrap CI를 main table에 추가** (현재 caption/본문에만 있음):
- Table 4에 95% CI를 별도 행으로 추가하면 심사자 신뢰도 향상

---

## E6 — 비용 보고 (기존 충족, 보강 권장)

### 현재 상태
- VRAM: 14.8GB ✅ (단일 RTX 4090 적합)
- Latency: Oracle-Domain 11.50s, Always-CROSS 8.69s (Table 6에서) ✅
- FLOPs: 미보고 ❌

### 추가 권장
**FLOPs 추정**:
- DeBERTa-large router: ~400M params, 1 forward pass per query
- LoRA adapter switching: 추가 VRAM overhead 없음 (PEFT load_adapter)
- 전체 pipeline FLOPs = retrieval (FAISS) + router (DeBERTa) + generation (Qwen 7B)
- 72B 대비 FLOPs 절약을 수치로 명시하면 efficiency claim 강화

**비용 보고 목적**: "성능 향상이 비용 증가 때문이 아님"을 명시
- NexRoute 7B의 VRAM (14.8GB) vs 72B baselines (36~40GB)
- NexRoute의 CROSS-domain 성능이 72B baselines와 유사함 (Table 4)
  → 10× 파라미터 절약으로 comparable cross-domain performance

---

## 실험 우선순위 (GPU 가용 시)

| 우선순위 | 실험 | 예상 시간 | 필요성 |
|---------|------|---------|--------|
| 1 | LAW adapter v4 재훈련 + 평가 | ~1-2h | \placeholder 제거 필수 |
| 2 | Typed-RAG baseline (Clean-300) | ~2-3h | \placeholder 제거 필수 |
| 3 | External benchmarks (MedQA, MMLU-Law) | ~2h | \placeholder 제거 필수 |
| 4 | E1 hypothesis sweep (기존 데이터 재분석) | ~30min | Proposition 1 검증 |
| 5 | 추가 베이스라인 (Self-RAG 등) | ~4h | E4 강화 |

---

## \placeholder 제거 체크리스트

```bash
# 현재 미완성 값 위치 확인
grep -n "placeholder" paper/main_final.tex | grep -v "newcommand"
```

채워야 할 항목:
1. `tab:law-v4-results`: v4 ROUGE-L, BERTScore, avg sentences, artifact %  → LAW v4 훈련 후
2. `tab:master_performance` Typed-RAG 7B 행: ROUGE-L, BERTScore, BLEU-4 → Typed-RAG 실행 후
3. `tab:external_benchmarks`: MedQA + MMLU-Law 전체 → 외부 벤치마크 실행 후
4. Introduction의 `\placeholder{XX}` (Typed-RAG 대비 gap): → Typed-RAG 결과 확인 후

---

## 결과 CSV 저장 위치

```
results/
├── e1_hypothesis/
│   ├── hypothesis_sweep.csv      # E1: function heterogeneity × routing 비교
│   └── hypothesis_plot.png
├── typedrag/
│   ├── typedrag_results.json
│   └── typedrag_scores.json
├── external_benchmarks/
│   ├── medqa_results.json
│   ├── medqa_scores.json
│   ├── mmlu_law_results.json
│   └── mmlu_law_scores.json
└── law_v4/
    ├── law_v4_gen.json
    └── law_v4_scores.json
```
