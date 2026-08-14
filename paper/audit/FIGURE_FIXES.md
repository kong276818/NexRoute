# FIGURE FIXES REQUIRED
Generated: 2026-08-13

## Figure 3 (Architecture Diagram)
두 가지 오류가 내부 텍스트에 있음. 벡터 소스 파일 재생성 필요.

| 위치 | 현재 내부 텍스트 | 수정해야 할 값 | 근거 |
|------|----------------|---------------|------|
| Figure 3 encoder 레이블 | BERT-base-uncased, MLP(768→256→4) | **DeBERTa-large**, h∈R^1024 | main_final.tex line 335, 414: DeBERTa-large, h∈R^{1024} |
| Figure 3 VRAM 레이블 | 12.2 GB | **14.8 GB** | main_final.tex line 994-995: "14.8~GB" |

## Figure 1 (Motivation)
캡션 재작성 불필요. 단, 설명 텍스트에 "Fatal Intelligence-Knowledge Mismatch" 표현이 있으면
**"Intelligence–Knowledge Mismatch"** 로 변경 (과신 표현 완화, Task 10).
→ 그림 파일 내부에 이 텍스트가 있으면 재생성 필요.

## 확인 필요
- Figure 4 (Confusion Matrix): off-diagonal 합이 63인지 재검산 (TODO_AUTHOR)
- Figure 6 (Hallucination bar chart): 3.6% 표기가 full-500 기준임을 캡션에 명시
