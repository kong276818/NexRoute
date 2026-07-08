#!/usr/bin/env bash
# run_ablation_pipeline.sh — 전체 ablation pipeline 오케스트레이션
# Usage: bash run_ablation_pipeline.sh
# Logs: ~/moe_model/logs/ablation_*.log

set -eo pipefail
source /home/sclab/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dualmoe 2>/dev/null || true
cd /home/sclab/moe_model

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
mkdir -p results/full_ablation/{generations,metrics}

# ── Variant 1 확인 (재실행 없음) ──────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Variant 1: reusing eval_results_7b_v3_500.json"
if [ ! -L results/full_ablation/generations/variant_01_full_nexroute.json ]; then
    ln -sf /home/sclab/moe_model/eval_results_7b_v3_500.json \
        results/full_ablation/generations/variant_01_full_nexroute.json
fi

# ── Phase 1: Generation variants 2–9 ─────────────────────────────────────
VARIANTS=(
    "variant_02_wo_function_axis"
    "variant_03_wo_partitioned_retrieval"
    "variant_04_wo_lora"
    "variant_05_wo_domain_axis"
    "variant_07_cross_weight_avg"
    "variant_08_cross_task_arithmetic"
    "variant_09_cross_dominant_single"
)

for VARIANT in "${VARIANTS[@]}"; do
    LOG="${LOG_DIR}/ablation_${VARIANT}.log"
    OUT="results/full_ablation/generations/${VARIANT}.json"

    if [ -f "$OUT" ]; then
        N=$(python3 -c "import json; d=json.load(open('$OUT')); print(len(d))" 2>/dev/null || echo 0)
        echo "[$(date '+%H:%M:%S')] SKIP ${VARIANT} (already ${N} samples)"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] START ${VARIANT}"
    python3 run_ablation_matrix.py --variant "$VARIANT" 2>&1 | tee "$LOG"
    echo "[$(date '+%H:%M:%S')] DONE  ${VARIANT}"
done

echo ""
echo "[$(date '+%H:%M:%S')] ══════ Phase 2: eval_metrics.py ══════"
python3 eval_metrics.py 2>&1 | tee "${LOG_DIR}/eval_metrics.log"

echo ""
echo "[$(date '+%H:%M:%S')] ══════ Phase 3: proxy_citation_alignment.py ══════"
python3 proxy_citation_alignment.py 2>&1 | tee "${LOG_DIR}/citation_alignment.log"

echo ""
echo "[$(date '+%H:%M:%S')] ══════ Phase 4: proxy_calibration_entropy.py ══════"
python3 proxy_calibration_entropy.py 2>&1 | tee "${LOG_DIR}/calibration_entropy.log"

echo ""
echo "[$(date '+%H:%M:%S')] ══════ All phases complete ══════"
echo "  Phase 2 metrics:     results/full_ablation/metrics/"
echo "  Phase 3 citation:    results/full_ablation/citation_alignment.json"
echo "  Phase 4 entropy:     results/full_ablation/calibration_entropy.json"
echo ""
echo "  NOT included in this batch (separate follow-up required):"
echo "    - reasoning chain validity (GPT-4o G-Eval)"
echo "    - domain expert blind rating (human eval)"
echo "    - answer safety evaluation (human eval)"
