# TODO(AUTHOR) — 저자 확인 필요 항목
Generated: 2026-08-13

원고 내 % TODO(author) 주석 + 저자만 확인 가능한 사항

1. **Figure 4 confusion matrix off-diagonal 합**  
   본문 Table 5(오분류 63건)와 Figure 4 off-diagonal 합이 일치하는지 재검산.

2. **Cohen's κ = 0.91 이중 등장**  
   §3.4 line 568 (function category validation)과 §3.6 line 630 (annotation audit)에
   동일한 κ=0.91이 등장. 이 두 감사가 동일한 감사인가, 아니면 독립적으로 수행된 두 감사인가?
   → 동일하면: "same audit as above" 또는 cross-reference 추가
   → 다른 감사면: 감사 대상 설명 구분

3. **STEM 라우팅 정확도 하락폭**  
   §4.2 line 782 = −8.0 pp (87.0%→79.0%). Figure 4 혼동행렬과 재검산.

4. **Function axis accuracy 두 지표 정의 구분**  
   - 38.0%: Supplementary S8, strict test accuracy (function routing test set)
   - 50.0%: Table 9 조건 B 캡션, "label accuracy vs gold" (eval set)
   이 두 지표의 정확한 분모(n)와 측정 방법을 §4.2에 명시 필요.
   train/test 라벨링 스킴 불일치의 구체적 내용(어떤 가이드라인이 어떻게 달랐는지)도 2문장으로 서술 필요.

5. **Table 9 CROSS n=88 정의**  
   Table 9 (function axis ablation) CROSS n=88의 정의가 is_cross(predicted) 기반인지
   gold_domain='CROSS' 기반인지 캡션에 명시 필요.

6. **arXiv [30] Yang et al. 2026, arXiv:2601.07935 검증**  
   연도(2026) 및 arXiv ID(2601.07935) 유효성 확인. 존재하지 않으면 인용 삭제.

7. **Retrieval-level evaluation (Recall@K / nDCG) 부재**  
   gold-evidence annotation이 없어 Recall@K를 계산할 수 없음을 7.3절에 명시.
   §4.1 Evaluation metrics 끝에 선제 방어 문단 추가 필요 (Task 6).
