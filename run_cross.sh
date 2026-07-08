#!/bin/bash
python inference.py --eval --domain CROSS --n 100 --mode base > /home/sclab/moe_model/eval_base_cross.txt 2>&1
echo "CROSS base 완료"
python inference.py --eval --domain CROSS --n 100 --mode lora_only > /home/sclab/moe_model/eval_lora_cross.txt 2>&1
echo "CROSS lora_only 완료"
python inference.py --eval --domain CROSS --n 100 --mode domain_only > /home/sclab/moe_model/eval_domain_cross.txt 2>&1
echo "CROSS domain_only 완료"
python inference.py --eval --domain CROSS --n 100 --mode dualmoe > /home/sclab/moe_model/eval_dualmoe_cross.txt 2>&1
echo "CROSS dualmoe 완료"
