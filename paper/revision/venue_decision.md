# STEP 6 — 저널 적합성 재판단

## 옵션 비교

| 저널 | IF (2024) | Scope 적합도 | 재투고 전략 | 주의사항 |
|------|----------|------------|------------|--------|
| **ESWA (재투고)** | ~8.5 | ✅ 높음 (데스크에서 scope 적합 인정) | Novelty 재프레이밍 후 직접 투고 | 동일 에디터일 경우 커버레터에서 개선 사항 명시 |
| **Information Fusion** | ~18.6 | ✅ 높음 (fusion of multiple information sources) | 신규 투고 | CROSS-domain fusion 각도를 강조해야 함 |
| **Neurocomputing** | ~6.0 | 중간 (neural architecture 위주) | 신규 투고 | 이론적 분석이 약하면 rejection 가능성 |
| **Knowledge-Based Systems** | ~8.8 | ✅ 높음 (knowledge representation + expert systems) | 신규 투고 | LoRA-based expert knowledge 각도 강화 필요 |
| **ACL/EMNLP (workshop)** | N/A | 중간 (NLP task 중심) | 신규 투고 | conference 타임라인이 저널보다 빠름 |

---

## 권고 결정: **ESWA 재투고 (1순위)** + Information Fusion (backup)

### ESWA 재투고 근거

1. **Scope 적합성 이미 확인**: 데스크 에디터가 "scope is appropriate"라고 명시한 리젝. 이는 scope 문제가 아닌 순수 novelty 프레이밍 문제임을 의미한다. 재투고 시 동일한 scope 강점을 유지하면서 novelty만 강화하면 된다.

2. **ESWA의 "expert and intelligent systems" 요구 사항**: 이 논문은 정확히 전문 도메인 expert system (법률/의료/STEM/사회) 구축 문제를 다룬다. `Expert Systems With Applications`의 핵심 독자층과 완벽히 일치한다.

3. **응용 시나리오 섹션 필요**: ESWA는 "실제 산업/실무 적용 임팩트"를 요구한다. 현재 논문의 Discussion (Section 7.1)에 이미 법률/의료 적용 사례가 언급되어 있으나, 독립 섹션으로 분리하고 구체화해야 한다.

---

## ESWA 재투고를 위한 추가 필수 작업

### 응용 시나리오 섹션 추가 (Section 8: Industrial Application Scenarios)

ESWA 심사자는 "이 시스템을 실제로 어디에 쓸 수 있는가"를 확인한다.
다음 3가지 시나리오를 구체적으로 기술하는 0.5-1페이지 섹션 추가:

**Scenario A: On-Premise Legal Document Review System**
- 문제: 법률 firm에서 외부 API 사용 불가 (데이터 기밀성)
- NexRoute 기여: 14.8GB VRAM으로 단일 GPU에서 운용 가능, LAW expert + 법률 corpus 파티션
- 측정 가능한 임팩트: 100 legal queries/minute throughput (latency 11.5s → batch 처리 시)

**Scenario B: Clinical Decision Support in Data-Restricted Environments**
- 문제: 병원 내 데이터 레지스트리를 외부 서버로 전송 불가
- NexRoute 기여: MED expert + 분리된 임상 corpus, CROSS expert for med-legal scenarios
- 측정 가능한 임팩트: MedQA accuracy + contradiction rate (vs. unified RAG baseline)

**Scenario C: Regulatory Compliance Auditing**
- 문제: 법규/기술표준/사회규범이 동시에 적용되는 cross-domain compliance 쿼리
- NexRoute 기여: LAW-STEM, LAW-SOC cross-domain routing
- 측정 가능한 임팩트: cross-domain contamination 감소 (NLI contradiction rate 3.6% vs 6.7%)

---

## Information Fusion (Backup) 투고 전략

**각도 변경**: "Dual-axis routing"이 아닌 **"Cross-domain evidence fusion architecture"**로 프레이밍.

- 핵심 주장: CROSS query는 두 도메인의 evidence를 융합해야 하는데, 기존 RAG는 이 융합 과정에서 contamination이 발생한다. NexRoute의 domain-partitioned retrieval + dedicated CROSS expert가 이 fusion을 구조적으로 해결한다.
- Figure 추가: evidence fusion 과정을 시각화하는 다이어그램 (두 파티션 → CROSS adapter → synthesized output)
- 강조할 실험: CROSS query의 NLI contradiction rate (3.6% vs 6.7%), MED-LAW case study

---

## 투고 전 체크리스트

- [ ] \placeholder{} 값 전체 채우기 (GPU 실험 완료 후)
- [ ] theory.tex의 Section 3.0 추가
- [ ] Abstract 재작성 (manuscript_v2_abstract_intro.tex 기반)
- [ ] Introduction 재작성 (failure regime 먼저)
- [ ] Related Work 각 문단 끝에 "gap 명시" 문장 추가
- [ ] Experiments 섹션 재배치 (oracle routing 발견 먼저)
- [ ] 응용 시나리오 섹션 추가 (Section 8)
- [ ] LAW adapter v4 결과 채우기
- [ ] Typed-RAG baseline 결과 채우기
- [ ] Cover letter 작성 (self_review.md 기반)
