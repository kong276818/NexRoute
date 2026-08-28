# STEP 1 — Novelty Audit: NexRoute vs. Related Work

## 1-1. 관련 연구 계열별 분류 및 비교표

### (a) MoE Routing

| 방법 | 라우팅 신호 | 결정 단위 | 학습 방식 | NexRoute와의 차이 | 이 논문이 반박/보완하는 지점 |
|------|------------|---------|---------|-----------------|--------------------------|
| Switch Transformer (Fedus 2022) | 토큰 임베딩 | 토큰 → top-1 expert | End-to-end joint | 라우팅 결과가 retrieval scope / prompt format에 **전혀 연결되지 않음**; RAG 없음 | NexRoute: 라우팅 결정 자체보다 그 결정이 무엇을 선택하는가(retrieval partition + scaffold)가 품질을 결정함을 실증 |
| Expert Choice (Zhou 2022) | 토큰 임베딩 | expert가 토큰을 선택 | Joint | 동상 | 동상 |
| Soft MoE (Puigcerver 2023) | 슬롯-weighted 임베딩 | soft combination | Joint | 단일 forward에서 soft mixing; 도메인/기능 분리 없음 | 동상 |
| DeepSeek-V3 (DeepSeek 2025) | 토큰 임베딩 + 보조 손실 | top-k token routing | Joint, auxiliary loss | Query-level 라우팅 없음; cross-domain 시나리오 미지원 | 동상 |

**공통 gap**: 모든 MoE routing은 **토큰 수준**에서 작동하며, 쿼리가 어떤 reasoning function을 필요로 하는지를 라우팅 신호로 사용하지 않는다. RAG와 결합 시 retrieval scope를 라우팅으로 제어하는 메커니즘이 없다.

---

### (b) Domain/Task-Adaptive Expert Selection (LoRA-MoE)

| 방법 | 라우팅 신호 | 결정 단위 | 학습 방식 | NexRoute와의 차이 | 이 논문이 반박/보완하는 지점 |
|------|------------|---------|---------|-----------------|--------------------------|
| LoRAMoE (Dou 2024) | 게이팅 네트워크 | LoRA adapter 선택 | Joint gating | retrieval 없음; function axis 없음; cross-domain expert 없음 | 라우팅 메커니즘만 있을 때 cross-domain ROUGE-L이 ≥17pp 낮음 (Table 7) |
| Mixture-of-LoRAs (Wu 2024) | Task embedding | LoRA adapter 선택 | Task-conditional | 동상 | 동상 |
| MedMoE-LoRA (Yang 2025) | 의료 도메인 특화 | Single-domain only | Supervised | Cross-domain 시나리오 구조적으로 미지원 | NexRoute: 단일 도메인 전문화의 한계를 cross-domain expert + 분리 retrieval로 극복 |
| Arrow Routing (2024) | SVD(A-matrix) 주성분 | LoRA library routing | Training-free | retrieval scope 제어 없음; function prompt 없음 | Routing entropy는 NexRoute와 유사(91.5% vs 92.5%)이지만 ROUGE-L은 11.30 vs 29.60 → entropy 최적화가 아닌 structural wiring이 핵심 |
| PHATGOOSE (Muqeeth 2024) | Hidden-state centroids | LoRA library routing | Lightweight fine-tuning | 동상 | 동상 |

**공통 gap**: LoRA-MoE 계열은 **adapter 선택**만 수행하며, (1) 어떤 knowledge partition에서 retrieval할지, (2) 어떤 prompt scaffold를 적용할지를 라우팅 결과로 제어하지 않는다.

---

### (c) LLM Router / Model Routing

| 방법 | 라우팅 신호 | 결정 단위 | 학습 방식 | NexRoute와의 차이 | 이 논문이 반박/보완하는 지점 |
|------|------------|---------|---------|-----------------|--------------------------|
| RouteLLM (Ong 2024) | 쿼리 난이도/비용 | 강모델 vs 약모델 선택 | Preference learning | Domain/function 라우팅 없음; retrieval 없음 | 모델 선택 라우팅 vs 구조적 wiring 라우팅의 대조 |
| Adaptive-RAG (Jeong 2024) | 쿼리 복잡도 | 검색 전략 선택 (0/1/multi-hop) | Classifier 학습 | Unified corpus; domain partitioning 없음; function axis 없음 | NexRoute: complexity-based routing은 domain contamination을 해결하지 못함 |
| Typed-RAG (Min et al.) | Function type (factoid/reasoning 등) | Prompt type 선택 | Classifier 학습 | Unified corpus; domain partitioning 없음; LoRA adapter 없음 | 직접 ablation baseline으로 사용; function axis만으로는 domain axis 기여를 대체할 수 없음 [VERIFY: ≥6pp ROUGE-L gap] |

**공통 gap**: LLM router 계열은 라우팅 신호를 사용하지만, **retrieval scope를 도메인별로 분리하지 않으며** function-conditioned scaffolding을 제공하지 않는다.

---

### (d) Multi-Axis or Hierarchical Gating

| 방법 | 라우팅 신호 | 결정 단위 | 학습 방식 | NexRoute와의 차이 | 이 논문이 반박/보완하는 지점 |
|------|------------|---------|---------|-----------------|--------------------------|
| Hierarchical MoE (일반적) | 계층별 토큰 임베딩 | 계층적 expert tree | Joint | RAG 없음; domain ≠ retrieval partition; function axis 없음 | NexRoute: 두 축이 하나의 공유 encoder에서 jointly trained — disentangled가 아닌 multi-task representation |
| MoE with auxiliary losses | 다중 보조 손실 | Expert load balance | Joint | 모든 결정이 expert activation에만 연결됨 | 동상 |

---

## 1-2. 정리: NexRoute가 각 계열에서 유일하게 채우는 셀

전체 비교에서 **모든 기존 방법**이 비어 있는 셀 조합:

```
라우팅 신호: [도메인 축 + 기능 축] 동시 분류 (공유 encoder, jointly trained)
    ↕
결정 내용: retrieval partition 선택 + adapter 선택 + prompt scaffold 선택
    ↕
Cross-domain: 별도 CROSS expert + 두 파티션으로부터의 분리 retrieval
```

이 3가지를 **단일 synchronized step**에서 수행하는 시스템은 존재하지 않는다.

---

## 1-3. 진짜 빈틈 (1개만 특정)

> **NexRoute만 채우는 유일한 셀:**
>
> **"쿼리-수준(query-level) 이중 축 분류(domain + function)가 공유 encoder에서 jointly trained되고,
> 각 축의 결정이 독립적인 시스템 자원(retrieval partition, adapter, prompt scaffold)에
> 동시 결합(binding)되어, 라우팅 정확도가 아닌 이 structural wiring이
> 생성 품질의 주 기여 요인임을 실증(empirically demonstrated)한 논문."**

구체적으로, 기존 연구들은 다음 셀들 중 일부만 채운다:
- MoE routing: 다중 expert는 있지만 retrieval scope 제어 없음
- LoRA-MoE: adapter 선택은 있지만 retrieval partitioning + function scaffold 없음
- Typed-RAG: function routing은 있지만 domain partitioning + adapter 없음
- Adaptive-RAG: complexity routing은 있지만 domain/function 이중 축 없음

**NexRoute만** 다음을 동시에 충족한다:
1. Domain axis → retrieval partition + adapter 결합
2. Function axis → prompt scaffold 결합
3. 두 축이 공유 encoder에서 jointly trained
4. Cross-domain query를 위한 별도 expert + 이중 파티션 retrieval
5. **핵심 발견**: 라우팅 정확도 최적화(oracle routing)가 생성 품질을 저하시킴을 실증

> ✅ 빈틈 특정 성공. 다음 STEP으로 진행 가능.

---

## 판단 노트

**억지 없음 확인**: 단순히 "두 축을 썼다"는 것 자체는 novelty가 아니다.
진짜 기여는 **"routing decisions가 downstream system resources에 bound되며,
이 structural binding이 routing accuracy optimization보다 품질에 더 중요하다"**
는 empirical finding + 그 finding을 가능하게 한 architecture design이다.

이것이 desk reject 원인과 직결된다: 현재 abstract/intro는 아키텍처 설명에 집중하고
이 핵심 insight(counter-intuitive finding)를 첫 문단에서 드러내지 않는다.
