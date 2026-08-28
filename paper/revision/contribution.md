# STEP 2 — 기여 재정의 (Contribution Redefinition)

## One-Sentence Claim

> **"We show that single-axis routing fails under joint domain-function heterogeneity because it cannot simultaneously bind evidence scope and reasoning format to each query, and resolve this by a jointly trained dual-axis nexus that structurally wires routing decisions to retrieval partitions and prompt scaffolds—demonstrating that this wiring, not routing accuracy, explains quality gains."**

(23단어 이내 영어 버전):
> **"We identify structural wiring—not routing accuracy—as the productive design principle for cross-domain expert systems, and demonstrate it through a dual-axis routing framework that jointly binds evidence scope and reasoning format to each query."**

---

## Contribution 3개 (Problem / Method / Evidence 각 1개)

### C1 — Problem-Level: Failure Regime의 명시적 정의

**기여**: 기존 RAG 기반 expert system이 구조적으로 실패하는 두 가지 조건(failure regime)을 최초로 정의하고 ablation으로 정량화한다.

- **Functional Mismatch** (기능 불일치): 도메인은 올바르게 라우팅되었으나, 해당 도메인 expert가 쿼리가 요구하는 reasoning modality(factual recall, causal reasoning, procedural explanation, comparison)에 맞는 prompt structure를 받지 못해 품질이 저하되는 현상. 단일 축 라우터는 이 차원을 구조적으로 무시한다.

- **Cross-Domain Contamination** (도메인 혼합 오염): 통합 corpus에서 lexical similarity 기반 검색이 쿼리의 실제 도메인과 무관한 passage를 검색하는 현상. Ablation (V03: partitioned retrieval 제거)으로 CROSS-domain NLI contradiction rate가 +3.20pp 증가함을 통계적으로 입증 ($p=0.045$).

**무엇을 새로 알게 되었는가**: 기존에는 "더 좋은 router" = "더 높은 라우팅 정확도 = 더 좋은 결과"로 가정했지만, 이 연구는 oracle-domain routing(정확도 상한)이 predicted routing보다 ROUGE-L -1.72pp 낮음을 보임으로써, **라우팅 정확도와 생성 품질이 단조적으로 연결되지 않는다는 사실**을 처음 실증한다.

---

### C2 — Method-Level: Structural Wiring을 통한 Dual-Axis Nexus

**기여**: 공유 DeBERTa-large encoder에서 jointly trained된 두 개의 독립 분류 헤드(domain axis $R_D$, function axis $R_F$)가 쿼리를 nexus point $(d^*, f^*)$로 매핑하며, 각 결정은 동시에 (1) retrieval partition 선택, (2) LoRA adapter 활성화, (3) function-conditioned prompt scaffold 적용으로 연결된다.

**핵심 구조적 특징**:
- 단일 forward pass에서 두 결정이 동기화되어 context-prompt alignment가 보장됨
- CROSS 탐지는 5번째 클래스 neuron 없이 top-2 domain score threshold로 수행 (soft cross-domain detection)
- CROSS query에 대해 두 contributing domain partition에서 독립 retrieval 후 결합

**무엇을 새로 알게 되었는가**: "어떤 expert를 활성화할 것인가" 보다 **"어떤 evidence scope와 reasoning format을 적용할 것인가"**가 결정적이다. 이 통찰은 기존 routing 연구의 프레임을 재정의한다.

---

### C3 — Evidence-Level: 3가지 수렴 증거로 검증

**기여**: Structural wiring이 routing accuracy보다 중요하다는 주장을 3가지 독립적 증거로 검증한다.

1. **Oracle routing 역전**: Oracle-domain routing (라우팅 정확도 상한) 조건이 NexRoute predicted routing보다 ROUGE-L -1.72pp 저하 ($p<0.001$, 95% CI $[-2.70, -0.78]$). 정확도 최적화가 품질을 보장하지 않음.

2. **LoRA library baseline 비교**: Arrow Routing, PHATGOOSE는 NexRoute와 routing entropy가 유사(91.5%, 100.0% vs 92.5% of $\mathcal{H}_{\max}$)하지만 CROSS-domain ROUGE-L이 ≥18pp 낮음. 라우팅 확률 분포가 아닌 structural design이 품질 차이를 만든다.

3. **Ablation의 division of labor**: LoRA adapter 제거가 가장 큰 단일 성분 품질 저하(-9.43 ROUGE-L), retrieval partitioning 제거가 CROSS-domain factuality 저하(+3.20pp NLI contradiction, $p=0.045$)에 특화된다. 두 축이 서로 보완적이며 독립적으로 필요함.

**무엇을 새로 알게 되었는가**: Expert system 설계에서 productive design choice는 routing probability 최적화가 아니라 **routing decision을 evidence scope + reasoning format에 동시 결합하는 structural wiring**이다.

---

## 주의: 반드시 "성능이 좋다"가 아닌 기여로 서술할 것

| ❌ 기존 (데스크 리젝 유발) | ✅ 수정 (novelty 명확화) |
|--------------------------|------------------------|
| "NexRoute achieves 28.23 ROUGE-L, exceeding baselines by 6pp" | "Oracle routing underperforms predicted routing, showing that routing accuracy is not the mechanism behind quality gains" |
| "We propose a dual-axis routing architecture" | "We identify the failure regime where single-axis routing structurally fails, and show that structural wiring resolves it" |
| "Our method outperforms 7B baselines on all metrics" | "Routing entropy is not the differentiating factor; retrieval partition control and prompt scaffold binding are" |

---

## 수정된 기여 목록 (논문 Introduction 마지막 단락 교체용)

```
Three contributions follow:

1. (Problem) We formally define two failure regimes in single-axis routing systems—
   functional mismatch and cross-domain contamination—and quantify their effects:
   removing retrieval partitioning elevates CROSS-domain NLI contradiction rate
   by +3.20pp (p=0.045), and removing function scaffolding costs 1.32 ROUGE-L (p<0.001).

2. (Method) We introduce the nexus point—a jointly trained dual-axis decision
   that simultaneously binds (i) domain routing to retrieval partition and adapter
   selection, and (ii) function routing to prompt scaffold selection, all in a
   single synchronized forward pass. CROSS-domain detection operates without a
   dedicated fifth class via soft threshold over top-2 domain activations.

3. (Evidence) We demonstrate—through oracle routing experiments, LoRA-library
   routing comparisons, and ablation—that structural wiring, not routing accuracy
   optimization, is the productive design choice: oracle-domain routing degrades
   ROUGE-L by 1.72pp relative to predicted routing (p<0.001), while Arrow and
   PHATGOOSE, despite comparable routing entropy, lag NexRoute by ≥18pp ROUGE-L
   on CROSS queries.
```
