#!/bin/bash
cd /home/sclab/moe_model
DDIR=/home/sclab/moe_model/eval_data

for DOMAIN in MED LAW STEM SOC CROSS; do
    for MODE in base lora_only domain_only dualmoe; do
        echo "====== $DOMAIN $MODE 시작 ======"
        python inference.py --eval --domain $DOMAIN --n 100 --mode $MODE \
            --data_dir $DDIR --suffix _mmlu
        echo "====== $DOMAIN $MODE 완료 ======"
        sleep 5
    done
done

echo ""
echo "===== 전체 결과 ====="
python3 -c "
import json, os, glob
results = {}
for f in sorted(glob.glob('/home/sclab/moe_model/results/eval_*_mmlu.json')):
    with open(f) as fp:
        d = json.load(fp)
    key = f'{d[\"domain\"]}_{d[\"mode\"]}'
    results[key] = round(d['accuracy']*100, 1)

domains = ['MED','LAW','STEM','SOC','CROSS']
modes   = ['base','lora_only','domain_only','dualmoe']

print(f'{'':15s}', end='')
for d in domains:
    print(f'{d:8s}', end='')
print()
print('-'*55)
for m in modes:
    print(f'{m:15s}', end='')
    for d in domains:
        val = results.get(f'{d}_{m}', '-')
        print(f'{str(val):8s}', end='')
    print()
"
