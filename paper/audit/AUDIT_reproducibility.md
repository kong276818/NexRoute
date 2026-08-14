# AUDIT_reproducibility.md — 재현성 체크리스트

Generated: 2026-08-13  
Auditor: Claude Code (file existence + package verification)

---

## 1. 체크포인트 현황

| 구성요소 | 경로 | 파일 존재 | 크기 | 날짜 |
|---------|------|----------|------|------|
| 도메인 라우터 (DeBERTa-large) | `router_deberta_best.pt` | ✅ | 1.6 GB | 2026-06-21 |
| LoRA-LAW | `lora_weights_7b_v3/lora_law/adapter_model.safetensors` | ✅ | 155 MB | 2026-05-29 |
| LoRA-MED | `lora_weights_7b_v3/lora_med/adapter_model.safetensors` | ✅ | 155 MB | 2026-05-29 |
| LoRA-SOC | `lora_weights_7b_v3/lora_soc/adapter_model.safetensors` | ✅ | 155 MB | 2026-05-29 |
| LoRA-STEM | `lora_weights_7b_v3/lora_stem/adapter_model.safetensors` | ✅ | 155 MB | 2026-05-29 |
| LoRA-CROSS | `lora_weights_7b_v3/lora_cross/adapter_model.safetensors` | ✅ | 155 MB | 2026-05-29 |

모든 체크포인트 존재 확인. 추가 버전:
- `lora_weights_7b/` — 별도 버전 (law/med/soc/stem/cross 각 155MB)
- `lora_weights_7b_v3/lora_law_v3/` — LAW 재학습 버전 (2026-07-03)

---

## 2. 지식베이스 (KB) 파일 현황

| 파일 | 경로 | 존재 | 크기 |
|------|------|------|------|
| law_rewritten.json | `processed_v2/law_rewritten.json` | ✅ | 5 MB |
| med_rewritten.json | `processed_v2/med_rewritten.json` | ✅ | 4 MB |
| soc_rewritten.json | `processed_v2/soc_rewritten.json` | ✅ | 1 MB |
| stem_rewritten.json | `processed_v2/stem_rewritten.json` | ✅ | 2 MB |
| cross_rewritten.json | `processed_v2/cross_rewritten.json` | ✅ | 1 MB |

**FAISS 인덱스**: 별도 저장 파일 없음. 파이프라인 실행 시 KB JSON으로부터 on-the-fly 구성 (`src/pipeline.py` line 276-281: `faiss.IndexFlatIP`).

---

## 3. 평가 데이터 현황

| 파일 | 경로 | 존재 | n | 비고 |
|------|------|------|---|------|
| gt_dataset_500.json | `processed_v2/gt_dataset_500.json` | ✅ | 500 | 도메인 100×5 균형 |
| preds_500.csv | `preds_500.csv` | ✅ | 500 | DeBERTa 라우팅 점수 |
| v01_gen_500.json | `results/ablation/v01_gen_500.json` | ✅ | 500 | V01 생성물 |

---

## 4. 런타임 환경 확인

| 패키지 | 버전/상태 |
|--------|---------|
| PyTorch | 2.12.1+cu130 ✅ |
| faiss | 설치됨 ✅ |
| rouge_score | 설치됨 ✅ |
| bert_score | 설치됨 ✅ |
| CUDA | **미가용** ⚠ (드라이버 버전 12080 — GPU 추론 불가) |

---

## 5. n=10 스모크 테스트

**상태**: CUDA 미가용으로 GPU 추론 스모크 테스트 실행 불가.

대신 파일 존재 및 코드 구조 검증 수행:
- 모든 5개 LoRA 체크포인트 존재 ✅
- 라우터 체크포인트 존재 (1.6GB) ✅
- `preds_500.csv` 로드 가능: 500행, 올바른 헤더 ✅
- `gt_dataset_500.json` 로드 가능: 500건, gold_domain/instruction/gold_answer 포함 ✅
- KB JSON 5개 모두 존재 ✅

**CUDA 가용 환경에서 스모크 테스트 필요**: `run_ablation_complete.py --phase 3 --ids 0,1,2,...,9`

---

## 6. 추가 발견: 라우터 버전 혼용

학습 스크립트에 여러 라우터 버전 존재:

| 파일 | 크기 | 날짜 | 유형 |
|------|------|------|------|
| `router_multilabel.pt` | 419MB | 2026-05-17 | BERT-base (구버전) |
| `router_deberta_best.pt` | 1.6GB | 2026-06-21 | DeBERTa-large (**현재 사용**) |
| `router_multilabel_v3~v6_best.pt` | 419MB | 2026-06-21 | BERT-base (중간버전) |

- `src/pipeline.py` (line 22): `ROUTER_PATH = router_multilabel.pt` — 구버전 지정
- `train/train_deberta_router.py` (line 16): `BEST_PATH = router_deberta_best.pt` — 신버전
- `src/nexroute_router.py`: DeBERTa-large 구조 정의

**주의**: `src/pipeline.py`의 ROUTER_PATH가 구버전 BERT를 가리킴. 단, `src/inference.py` 주석에 "prototype, NOT used for paper results" 명시. 실제 실험은 `run_reinference_500.py`, `run_ablation_complete.py` 등이 preds_500.csv 사전계산 결과를 사용하므로 실질적 영향 없음.

---

## 7. 재현을 위한 사전 조건

1. CUDA GPU (≥8GB VRAM) 필요
2. `run_reinference_500.py` 실행 전 `preds_500.csv` 존재 확인
3. LORA_DIR = `lora_weights_7b_v3` 경로 확인 (✅ 존재)
4. BASE_MODEL Qwen2.5-7B-Instruct HuggingFace 캐시 경로 확인:
   `/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`

---

## 8. 재현성 판정

| 항목 | 판정 |
|------|------|
| 체크포인트 완전성 | ✅ 완전 |
| KB 파일 완전성 | ✅ 완전 |
| 스크립트 존재 | ✅ 완전 |
| GPU 추론 가능 | ⚠ 현 환경 CUDA 미가용 (드라이버 업데이트 필요) |
| 3-seed 재현성 | ❓ 단일 시드만 확인됨 — 논문의 "3 seeds" 주장 검증 필요 |
