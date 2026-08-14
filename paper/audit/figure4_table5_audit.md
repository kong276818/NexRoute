# Figure 4 / Table 5 감사 보고서

생성일: 2026-08-05  
감사 대상: NexRoute 논문 (`paper/main_final.tex`)  
감사 범위: Figure 4 (혼동행렬) / Table 4 (도메인별 정확도) / Table 5 (오분류 분석)

---

## 1. 원본 예측 파일 존재 여부

| 파일 | 존재 여부 | 비고 |
|------|----------|------|
| `eval_results_7b_v3_500.json` | **없음** | `.gitignore`에 의해 제외됨 |
| `nexroute_real_confusion_matrix.json` | **없음** | `.gitignore`에 명시적 등재, 로컬에도 삭제됨 |
| `adaptive_tfidf_results.json` | **없음** | (감사 중 재생성됨) |
| `baseline_results/` | **없음** | 디렉터리 자체 없음 |
| Figure 4 생성 스크립트 | **없음** | 저장소 내 혼동행렬 생성 코드 전무 |

**→ 원본 예측 파일 전부 소실. 두 라우터를 500개 평가셋에 재실행하여 예측을 새로 생성함.**

---

## 2. 기존 Figure 4 생성 방식 판정

저장소에 Figure 4 생성 스크립트가 전혀 없음. 다음 두 가지 단서로 판정:

- `.gitignore`에 `nexroute_real_confusion_matrix.json` 명시 → 한때 실제 예측 기반 혼동행렬이 존재했음
- 그러나 Figure 4에 표시된 NexRoute CROSS 열 값(MED=3, SOC=6, STEM=5, 합계 14)이 실제 예측(MED=9, SOC=15, STEM=9, 합계 33)과 최대 2.5배 불일치

**판정: 원래 Figure 4가 실제 예측 기반으로 생성되었으나, 어느 시점에 수동 편집 또는 정확도만으로 재합성된 값으로 교체된 것으로 보임. 현재 제출된 Figure 4는 실제 예측을 반영하지 않음.**

---

## 3. 재계산 결과 vs 논문 기재값 전체 대조표

### 3-A. NexRoute 도메인별 정확도 (Table 4 오른쪽 열)

| 도메인 | 실제 예측 | 논문 기재 | 일치 여부 |
|--------|----------|----------|----------|
| LAW    | **100**  | 99       | ✗ (논문 오류: +1 낮게 기재) |
| MED    | 85       | 85       | ✓ |
| SOC    | 76       | 76       | ✓ |
| STEM   | **79**   | 80       | ✗ (논문 오류: +1 높게 기재) |
| CROSS  | 97       | 97       | ✓ |
| Total  | 87.4%    | 87.4%    | ✓ |
| 오분류 | **63**   | 63       | ✓ |

> 주: LAW 100 + STEM 79 = 179 = LAW 99 + STEM 80 이므로 총 정확도는 변하지 않음.
> LAW와 STEM 수치가 서로 1씩 뒤바뀐 채 기재된 것으로 판단됨.

### 3-B. NexRoute CROSS 열 off-diagonal (Figure 4 핵심 불일치)

| gold→pred | 실제 예측 | Figure 4 기재 | Table 5 텍스트 | 어느 쪽이 맞나 |
|-----------|----------|--------------|--------------|--------------|
| LAW→CROSS | 0        | 0            | 0            | 모두 일치 |
| MED→CROSS | **9**    | 3            | 9            | **실제=Table5 텍스트 (Figure4 오류)** |
| SOC→CROSS | **15**   | 6            | 15           | **실제=Table5 텍스트 (Figure4 오류)** |
| STEM→CROSS| **9**    | 5            | 9            | **실제=Table5 텍스트 (Figure4 오류)** |
| **합계**  | **33**   | **14**       | **33**       | **실제=33, Figure4는 14로 잘못됨** |

**→ Table 5 텍스트("33개 중 33개가 single→CROSS, MED 9 / SOC 15 / STEM 9")는 실제 예측과 정확히 일치함. Figure 4의 CROSS 열만 잘못됨.**

### 3-C. TF-IDF 라우터 (별도 문제)

| 도메인 | 재훈련 TF-IDF+LR | 논문 기재 |
|--------|----------------|----------|
| LAW    | 88             | 92       |
| MED    | 78             | 84       |
| SOC    | 68             | 68 ✓     |
| STEM   | 87             | 87 ✓     |
| CROSS  | 96             | 95 ✓     |
| Total  | 83.4%          | 85.2%    |

**원인**: 커밋 `2aeb774`(Remove GT-500 benchmark contamination from router dataset)에서 TF-IDF 학습셋에서 GT-500 오염 샘플이 제거됨. 제거 전 학습 결과가 논문에 기재되어 있어 현재 재훈련 결과와 LAW(+4pp), MED(+6pp) 불일치 발생. 이 문제는 Figure 4 / Table 5 오류와 독립적인 데이터 오염 이슈임.

### 3-D. Table 5 생성 지표 검증

| | N | ROUGE-L | BERTScore | BLEU-4 |
|-|---|---------|-----------|--------|
| Correctly routed (논문) | 437 | 24.39 | 87.66 | 7.16 |
| Misrouted (논문) | 63 | 26.87 | 88.78 | 8.99 |
| **검증 결과** | N=437/63 ✓ | 검증 불가 | 검증 불가 | 검증 불가 |

**`eval_results_7b_v3_500.json`(모델 생성 응답 500개)이 존재하지 않아 생성 지표(ROUGE-L, BERTScore, BLEU-4)는 검증 불가.** N 값(437/63)은 실제 예측과 정확히 일치함.

---

## 4. 불일치 항목 목록

| # | 위치 | 논문 기재 | 실제 값 | 어느 쪽이 옳은가 |
|---|------|----------|--------|----------------|
| 1 | Figure 4 (NexRoute CROSS열) MED→CROSS | 3 | **9** | 실제 예측 |
| 2 | Figure 4 (NexRoute CROSS열) SOC→CROSS | 6 | **15** | 실제 예측 |
| 3 | Figure 4 (NexRoute CROSS열) STEM→CROSS | 5 | **9** | 실제 예측 |
| 4 | Figure 4 (NexRoute CROSS열) 합계 | 14 | **33** | 실제 예측 |
| 5 | Table 4 NexRoute LAW | 99 | **100** | 실제 예측 |
| 6 | Table 4 NexRoute STEM | 80 | **79** | 실제 예측 |
| 7 | Table 4 Adaptive RAG Total | 85.2% | ~83.4% | 재훈련 결과 (오염 제거 후); 논문 수치는 오염된 버전 기준 |
| 8 | Table 5 N(correctly routed) | 437 | **437** ✓ | 일치 |
| 9 | Table 5 N(misrouted) | 63 | **63** ✓ | 일치 |
| 10 | Table 5 텍스트 single→CROSS 33 / MED:9 / SOC:15 / STEM:9 | 33,9,15,9 | **33,9,15,9** ✓ | Table5 텍스트가 맞음, Figure4가 틀림 |

---

## 5. 논문 수정이 필요한 문장 (위치 및 수정 전/후)

### [수정 1] Table 4 — NexRoute LAW/STEM 행, Δ 열

```latex
% 수정 전 (line 698)
LAW              & 92.0 & \textbf{99.0} & +7.0\%p \\
% 수정 후
LAW              & 92.0 & \textbf{100.0} & +8.0\%p \\

% 수정 전 (line 701)
STEM             & 87.0 & 80.0           & $-$7.0\%p \\
% 수정 후
STEM             & 87.0 & 79.0           & $-$8.0\%p \\
```

### [수정 2] Figure 4 이미지 파일

```latex
% 수정 전 (line 794)
\includegraphics[...]{figure4.png}
% 수정 후
\includegraphics[...]{figure4_revised_final.png}
```

### [수정 3] Figure 4 캡션 — LAW/STEM 수치

```latex
% 수정 전 (lines 799–802)
off-diagonal misclassification in SOC (+8.0\%p, 68.0\%$\to$76.0\%) and
LAW (+7.0\%p, 92.0\%$\to$99.0\%), where TF-IDF-based routing is most
susceptible to lexical overlap with adjacent domain terminology. STEM accuracy
decreases by $-$7.0\%p (87.0\%$\to$80.0\%), a direct consequence ...

% 수정 후
off-diagonal misclassification in SOC (+8.0\%p, 68.0\%$\to$76.0\%) and
LAW (+8.0\%p, 92.0\%$\to$100.0\%), where TF-IDF-based routing is most
susceptible to lexical overlap with adjacent domain terminology. STEM accuracy
decreases by $-$8.0\%p (87.0\%$\to$79.0\%), a direct consequence ...
```

### [수정 4] §5.2 "Lexical overlap" 단락 — LAW 수치 (line 714-715)

```latex
% 수정 전
The largest gains occur in SOC (+8.0\%p) and LAW
(+7.0\%p), domains where keyword-based routing ...
% 수정 후
The largest gains occur in SOC (+8.0\%p) and LAW
(+8.0\%p), domains where keyword-based routing ...
```

### [수정 5] §5.2 "STEM routing trade-off" 단락 (line 724–729)

```latex
% 수정 전
STEM routing accuracy decreases by 7.0\%p (87.0\%$\to$80.0\%), a pattern ...
The aggregate gains in SOC (+8.0\%p) and LAW (+7.0\%p) outweigh this
reduction at the total level (+2.2\%p; ...
% 수정 후
STEM routing accuracy decreases by 8.0\%p (87.0\%$\to$79.0\%), a pattern ...
The aggregate gains in SOC (+8.0\%p) and LAW (+8.0\%p) outweigh this
reduction at the total level (+2.2\%p; ...
```

---

## 6. 파생 영향 — 다른 표·본문 점검

| 위치 | 내용 | 영향 여부 |
|------|------|----------|
| Abstract (line 62) | `87.4\% routing accuracy (+2.2~pp over strongest baseline)` | **Δ +2.2pp 수치는 논문 TF-IDF 85.2% 기준. 오염 제거 후 재훈련 TF-IDF 83.4% 기준이면 +4.0pp가 맞음. 단, TF-IDF 수치 수정 여부를 별도 결정 필요** |
| Table 2 (threshold sensitivity, line 438) | `87.40\%` | ✓ 변경 없음 |
| §5.2 para "Router Calibration" (line 818) | `NexRoute's 87.4%` | ✓ 변경 없음 |
| Table 5 N 값 | 437/63 | ✓ 변경 없음 |
| Table 5 생성 지표 | ROUGE-L 24.39/26.87 등 | 검증 불가 (원본 응답 파일 없음) |
| Figure 4 캡션 내 CROSS 언급 없음 | — | 추가 수정 불필요 |

---

## 7. 생성 산출물

| 파일 | 설명 |
|------|------|
| `preds_500.csv` | 500개 샘플 NexRoute / TF-IDF 예측 전체 |
| `fig4_counts_nexroute.csv` | NexRoute 5×5 혼동행렬 (count) |
| `fig4_counts_tfidf_lr.csv` | TF-IDF+LR 5×5 혼동행렬 (count, 재훈련) |
| `figure4_revised_final.png` | 수정된 Figure 4 (실제 예측 기반) |
| `audit_summary.json` | assert 결과 및 수치 요약 |

---

## 8. 요약

| 항목 | 판정 |
|------|------|
| Figure 4 NexRoute CROSS 열 | **오류** — 14개로 표시되어 있으나 실제 33개 |
| Table 5 텍스트(33/MED9/SOC15/STEM9) | **정확** — 실제 예측과 일치 |
| Table 4 NexRoute LAW=99 | **오류** — 실제 100 |
| Table 4 NexRoute STEM=80 | **오류** — 실제 79 |
| Table 4 NexRoute 총 정확도 87.4% | **정확** |
| Table 4 TF-IDF 총 정확도 85.2% | **오염된 학습 데이터 기반** — 재현 불가 (83.4%) |
| Table 5 N=437/63 | **정확** |
| Table 5 생성 지표 | **검증 불가** (원본 응답 파일 소실) |
