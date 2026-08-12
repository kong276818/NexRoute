# INCONSISTENCY AUDIT — NexRoute main_final.tex
Generated: 2026-08-13

| # | 항목 | 충돌하는 값 | 판정 | 조치 |
|---|------|------------|------|------|
| 1 | 라우터 인코더 | 본문/Fig 2 = DeBERTa-large; **Figure 3 내부 텍스트 = BERT-base-uncased, MLP(768→256→4)** | 본문 정답 (DeBERTa-large, h∈R^1024) | FIGURE_FIXES.md 기록 — 그림 파일 재생성 필요 |
| 2 | VRAM | 본문/Abstract/Highlights = 14.8 GB; **Figure 3 내부 텍스트 = 12.2 GB** | 본문 정답 (14.8 GB, line 994-995) | FIGURE_FIXES.md 기록 |
| 3 | LAW ROUGE-L | §4.2/캡션 = **18.27** (Clean-300 main eval); §4.4 degeneration = **15.09** (full-500 ablation run); §4.5 Exp2 predicted = **19.08** (V01 re-inference clean-300) | 세 값 모두 다른 run/subset — 각각 정합 가능 | TeX에 run·subset 명시 추가 (해결 완료) |
| 4 | CROSS hallucination | Abstract/Fig 6/Table 7 = **3.6%** (full-500); §4.2 line 1001 = **4.47%** (Clean-300); Table 8 V01 = **3.50%** (ablation re-inference) | 세 값 모두 다른 run/subset — Table 7 캡션에 이미 설명. §4.2 line 1001은 맥락 OK | §4.2 line 1001 문장에 "(Clean-300)" 레이블 확인 필요 (현재 있음 ✓) |
| 5 | 라우팅 정확도 | Abstract/§4.2 = **87.4%** (full-500 routing test); §4.5 도입부 = "**81% accuracy**" (Clean-300 exp2) | 두 값 모두 정합 — subset이 다름 | §4.5 문장에 "(on the Clean-300 subset)" 명시 추가 필요 |
| 6 | STEM 라우팅 하락폭 | §4.2 line 782 / Fig 4 = **−8.0 pp** | Fig 4와 대조 불가(그림 파일 없음). 본문 값 유지 | % TODO(author): Figure 4 off-diagonal과 재검산 |
| 7 | Function axis 정확도 | §7.3 Limitations = **38.0% strict test accuracy**; Table 9 B조건 캡션 = **50.0% label accuracy vs gold** | 다른 지표: strict=test set, 50%=eval set gold 대비 | §4.2에서 두 지표 구분 서술 필요 (TODO_AUTHOR 항목) |
| 8 | Overall hallucination | Table 7/Abstract = **10.1%** (main eval); Table 8 V01 = **9.95%** (ablation re-inference) | Table 8 캡션 각주에 이미 설명 ✓ | Abstract/Conclusion에서 10.1%로 통일 (이미 됨 ✓) |
| 9 | LAW citation-format 비율 | §4.4 = **97%** (re-inference run); 각주 = **88%** ablation run / **89%** original ablation | §4.4 line 1259에 이미 "(re-inference run; 88% in the original ablation)" 병기 ✓ | 각 인용마다 run 명시 확인 필요 |
| 10 | Table 5 오분류 63건 ↔ Fig 4 off-diagonal | 본문 63건 | Fig 4 파일 없어 직접 검산 불가 | % TODO(author): Figure 4 confusion matrix off-diagonal 합 재검산 |
| 11 | Cohen's κ = 0.91 | §3.4 line 568 = function category validation; §3.6 line 630 = annotation audit | 같은 κ=0.91 두 곳 — 동일 감사인지 다른 감사인지 불명확 | % TODO(author): 두 κ가 동일 감사인지 명시 필요 |
| 12 | CROSS 샘플 수 | Table 9 (function axis) = **CROSS n=88**; Table 11/§4.1 Clean-300 = **CROSS n=60** | 다른 subset 정의 — Table 9는 is_cross 기반(predicted), Table 11은 gold_domain 기반 | 각 표 캡션에 정의 명시 (Table 9: "CROSS queries as predicted by the router"; Table 11: "gold_domain='CROSS'") |

## 해결 상태 요약
- **즉시 해결 가능 (LaTeX)**: #5, #12, #9(일부)
- **그림 파일 재생성 필요**: #1, #2 → FIGURE_FIXES.md
- **저자 확인 필요**: #6, #10, #11 → TODO_AUTHOR.md
- **이미 올바름**: #4, #8, #9
- **레이블 명시로 해결**: #3, #7
