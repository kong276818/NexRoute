"""
LAW 도메인 샘플만 재추론하여 eval_results_7b_v3_500.json 업데이트.

변경 이유:
- 기존 LAW pred_answer가 "The correct holding is option X:" MCQ 형식으로
  gold_answer(평균 95단어 서술형)와 형식 불일치 → ROUGE-L=17.52
- 수정된 pipeline.py: max_new_tokens=800, LAW 전용 서술형 프롬프트
"""
import json
import os
from pipeline import DualMoEPipeline

GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
RESULT_PATH = "/home/sclab/moe_model/eval_results_7b_v3_500.json"
BACKUP_PATH = "/home/sclab/moe_model/eval_results_7b_v3_500_before_law_fix.json"

with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

with open(RESULT_PATH) as f:
    results = json.load(f)

# 백업
if not os.path.exists(BACKUP_PATH):
    with open(BACKUP_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[백업] {BACKUP_PATH}")

# LAW GT 쿼리 인덱스 파악
law_indices = []
for i, item in enumerate(results):
    q = item.get("query", "").strip()
    gt = gt_map.get(q) or {}
    if gt.get("expected_domain") == "LAW":
        law_indices.append(i)

print(f"LAW 재추론 대상: {len(law_indices)}개")

# 파이프라인 초기화
pipe = DualMoEPipeline()
pipe.load_router()
pipe.load_embed_model()
pipe.load_base_model()
pipe.build_db()
pipe.load_lora("LAW")

updated = 0
for rank, idx in enumerate(law_indices):
    item  = results[idx]
    query = item["query"].strip()
    print(f"[{rank+1}/{len(law_indices)}] {query[:60]}...")

    try:
        # 도메인 라우팅은 유지(LAW), 생성만 재실행
        routing = pipe.classify_domain(query)
        # LAW GT이므로 is_cross=False, domain=LAW 강제
        context = pipe.retrieve(query, ["LAW"], is_cross=False)
        new_answer = pipe.generate(query, context, is_cross=False, domains=["LAW"])
        results[idx]["pred_answer"] = new_answer
        updated += 1
        print(f"  → {len(new_answer.split())}단어")
    except Exception as e:
        print(f"  [Error] {e}")
        continue

    if (rank + 1) % 20 == 0:
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  [중간저장] {rank+1}개 완료")

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 완료: {updated}개 LAW 샘플 업데이트")
print(f"   저장: {RESULT_PATH}")
print(f"   원본 백업: {BACKUP_PATH}")
print()
print("다음 단계: python recompute_ablation_v2.py")
