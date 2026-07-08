#!/bin/bash
cd /home/sclab/moe_model
DDIR=/home/sclab/moe_model/eval_data
SUFFIX=_mmlu

for DOMAIN in MED LAW STEM SOC; do
    for MODE in base lora_only domain_only dualmoe; do
        python inference.py --eval --domain $DOMAIN --n 100 --mode $MODE \
            --data_dir $DDIR --suffix $SUFFIX \
            > /home/sclab/moe_model/eval_${MODE}_${DOMAIN}_mmlu.txt 2>&1
        echo "$DOMAIN $MODE 완료"
    done
done
