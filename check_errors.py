import json

# GT 데이터와 결과 데이터 경로
GT500_PATH = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EVAL_PATH  = "/home/sclab/moe_model/eval_results_7b_v3_500.json"

with open(GT500_PATH) as f:
    gt500 = json.load(f)
with open(EVAL_PATH) as f:
    eval_results = json.load(f)

# 인스트럭션 기반 매핑
text_map = {d['instruction'].strip(): d for d in gt500}

print(f"오분류 샘플 분석:")
print("-" * 50)

error_count = 0
for e in eval_results:
    q = e.get('query', e.get('instruction', '')).strip()
    gt_item = text_map.get(q)
    
    if gt_item:
        true_domain = gt_item.get('expected_domain')
        # 라우팅 예측값 추출 로직 (00_inspect_scores.py와 동일하게)
        preds = e.get('domain_pred', [])
        is_cross = e.get('is_cross', False)
        pred_domain = 'CROSS' if is_cross else (preds[0] if preds else 'UNKNOWN')
        
        if true_domain != pred_domain:
            error_count += 1
            if error_count <= 10:  # 10개만 출력
                print(f"[{error_count}] 정답: {true_domain} / 예측: {pred_domain}")
                print(f"질문: {q[:100]}...")
                print("-" * 50)