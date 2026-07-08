import json
from collections import Counter

GT500_PATH = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EVAL_PATH  = "/home/sclab/moe_model/eval_results_7b_v3_500.json"

with open(GT500_PATH) as f:
    gt500 = json.load(f)
with open(EVAL_PATH) as f:
    eval_results = json.load(f)

gt_old = [d for d in gt500 if 'id' in d and d.get('id') is not None]
gt_new = [d for d in gt500 if 'id' not in d or d.get('id') is None]
print(f"원본 배치: {len(gt_old)}개, 확장 배치: {len(gt_new)}개")

# ── id는 완전히 무시하고, instruction 텍스트로만 통합 매핑 ──
text_map = {}
dup = 0
for d in gt500:
    instr = d['instruction'].strip()
    if instr in text_map:
        dup += 1
    text_map[instr] = d
print(f"전체 500개 텍스트 매핑 구축 (내부 중복 {dup}건)")

def get_pred(e):
    preds = e.get('domain_pred', [])
    is_cross = e.get('is_cross', False)
    return 'CROSS' if is_cross else (preds[0] if preds else 'UNKNOWN')

# id가 매칭에 전혀 관여하지 않도록 query 텍스트로만 룩업
matched, unmatched = 0, 0
old_results, new_results = [], []
old_ids = set(str(d['id']) for d in gt_old)

for e in eval_results:
    q = e.get('query', e.get('instruction', '')).strip()
    gt_item = text_map.get(q)
    if gt_item is None:
        unmatched += 1
        continue
    matched += 1

    true_d = gt_item.get('expected_domain', '')
    pred = get_pred(e)
    is_old_batch = ('id' in gt_item and gt_item.get('id') is not None)

    if is_old_batch:
        old_results.append((true_d, pred))
    else:
        new_results.append((true_d, pred))

print(f"\n텍스트 매칭 성공: {matched}/{len(eval_results)} (실패 {unmatched}개)")
print(f"원본 배치로 분류된 매칭: {len(old_results)}개")
print(f"확장 배치로 분류된 매칭: {len(new_results)}개")

def report(name, results):
    if not results:
        print(f"\n{name}: 매칭된 샘플 없음")
        return
    correct = sum(1 for t, p in results if t == p)
    print(f"\n=== {name} ===")
    print(f"{correct}/{len(results)} = {correct/len(results)*100:.2f}%")
    dom_total, dom_correct = Counter(), Counter()
    for t, p in results:
        dom_total[t] += 1
        if t == p:
            dom_correct[t] += 1
    for d in ['LAW', 'MED', 'SOC', 'STEM', 'CROSS']:
        if dom_total[d] > 0:
            acc = dom_correct[d] / dom_total[d] * 100
            print(f"  {d}: {dom_correct[d]}/{dom_total[d]} = {acc:.1f}%")

report("원본 배치 (텍스트 매칭, id 무시)", old_results)
report("확장 배치 (텍스트 매칭)", new_results)
report("전체 500개 (텍스트 매칭)", old_results + new_results)