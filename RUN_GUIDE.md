# 실험 실행 가이드 (GPU 필요)

현재 GPU 드라이버 문제: PyTorch 2.12.1+cu130이지만 드라이버가 CUDA 12.0만 지원.
→ 드라이버 업데이트 후 아래 순서로 실행.

---

## Step 1: LAW 어댑터 v4 재훈련 (~30-60분)

```bash
cd /home/sclab/moe_model
python train/train_lora_law_v4.py
```

저장 위치: `lora_weights_7b_v3/lora_law_v4/`
완료되면 `pipeline.py`가 자동으로 v4를 사용함 (`_LAW_ADAPTER` 변수).

---

## Step 2: Typed-RAG 베이스라인 평가 (~2-3시간, 300샘플)

```bash
python scripts/run_typedrag_baseline.py \
    --gt_path processed_v2/gt_dataset_500.json \
    --output_dir results/typedrag \
    --n_samples 300
```

결과 파일:
- `results/typedrag/typedrag_results.json` — 생성 결과
- `results/typedrag/typedrag_scores.json` — ROUGE-L / BERTScore

---

## Step 3: 기존 NexRoute (v4 어댑터) 재평가 (~2시간, 300샘플)

```bash
# 기존 run_reinference_500.py 활용 또는 ablation 스크립트 재실행
python scripts/run_ablation_complete.py
```

---

## Step 4: 외부 벤치마크 평가 (~1시간 x 2)

```bash
# MedQA (200 샘플)
python scripts/run_external_benchmarks.py \
    --benchmark medqa \
    --n_samples 200 \
    --output_dir results/external_benchmarks

# MMLU-Law (200 샘플)
python scripts/run_external_benchmarks.py \
    --benchmark mmlu_law \
    --n_samples 200 \
    --output_dir results/external_benchmarks
```

---

## Step 5: 논문에 결과 채우기

실험 완료 후 `paper/main_final.tex`에서 `\placeholder{...}` 태그를 실제 값으로 교체:

- `tab:law-v4-results` → LAW v4 ROUGE-L, BERTScore, sentence count, artifact %
- `tab:master_performance` → Typed-RAG 7B 결과 행
- `tab:external_benchmarks` → MedQA + MMLU-Law 결과
- `Introduction` / `Related Work` → Typed-RAG 대비 gap 수치

```bash
grep -n "placeholder" paper/main_final.tex | grep -v "newcommand"
```

위 명령으로 채워야 할 위치 전체 확인 가능.

---

## 제출 대상 저널 (추천 순)

1. **Knowledge-Based Systems** (Elsevier) — IF 7.2
2. **Information Processing & Management** (Elsevier) — IF 8.6
3. **Neurocomputing** (Elsevier) — IF 5.5
