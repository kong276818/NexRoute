import json
from collections import Counter

DATA_DIR   = "/home/sclab/moe_model/processed_v2"
GT500_PATH = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EVAL_PATH  = "/home/sclab/moe_model/eval_results_7b_v3_500.json"

domain_files = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

# ── Step 1: 권위 있는 역매핑 구축 (instruction -> 진짜 도메인) ──
print("=== Step 1: 도메인별 원본 파일에서 권위 있는 역매핑 구축 ===")
truth_map = {}
dup_warnings = 0
for domain, fname in domain_files.items():
    path = f"{DATA_DIR}/{fname}"
    with open(path) as f:
        data = json.load(f)
    for d in data:
        instr = d['instruction'].strip()
        if instr in truth_map and truth_map[instr] != domain:
            dup_warnings += 1
        truth_map[instr] = domain
    print(f"  {domain}: {len(data)}개 로드")
print(f"전체 역매핑 항목: {len(truth_map)}개 (중복/충돌 {dup_warnings}건)")

# ── Step 2: GT500 로드, 원본 300 / 확장 200 분리 ──
with open(GT500_PATH) as f:
    gt500 = json.load(f)
gt_old = [d for d in gt500 if 'id' in d and d.get('id') is not None]
gt_new = [d for d in gt500 if 'id' not in d or d.get('id') is None]
print(f"\n원본 배치: {len(gt_old)}개, 확장 배치: {len(gt_new)}개")

# ── Step 3: 원본 300개의 expected_domain이 진짜와 맞는지 검증 ──
print("\n=== Step 3: 원본 300개 expected_domain 진위 검증 ===")
mismatch = 0
not_found = 0
mismatch_samples = []
corrected_old = []  # (instruction, stored_domain, true_domain)

for d in gt_old:
    instr = d['instruction'].strip()
    stored_domain = d.get('expected_domain', '')
    true_domain = truth_map.get(instr)

    if true_domain is None:
        not_found += 1
        continue

    corrected_old.append((instr, stored_domain, true_domain))
    if stored_domain != true_domain:
        mismatch += 1
        if len(mismatch_samples) < 8:
            mismatch_samples.append((instr[:60], stored_domain, true_domain))

print(f"원본 300개 중 소스 파일에서 못 찾은 항목: {not_found}개")
print(f"expected_domain 불일치: {mismatch}/{len(corrected_old)}개 "
      f"({mismatch/len(corrected_old)*100:.1f}%)")
print("\n불일치 샘플:")
for instr, stored, true in mismatch_samples:
    print(f"  저장된 라벨: {stored:6s} | 진짜 도메인: {true:6s} | {instr}...")

# 불일치 패턴 분석 (예: 어떤 라벨이 어떤 라벨로 자주 바뀌었는지)
print("\n--- 라벨 혼동 패턴 (stored -> true) ---")
confusion = Counter()
for instr, stored, true in corrected_old:
    if stored != true:
        confusion[(stored, true)] += 1
for (s, t), cnt in confusion.most_common(15):
    print(f"  저장값={s:6s} -> 진짜={t:6s}: {cnt}건")

# ── Step 4: 교정된 라벨로 GT500 전체 정확도 재계산 ──
print("\n=== Step 4: 교정된 라벨로 전체 500개 라우팅 정확도 재계산 ===")
with open(EVAL_PATH) as f:
    eval_results = json.load(f)

# 매핑: id -> 교정된 true_domain (원본 300), instruction -> true_domain (확장 200, 이미 정상)
id_to_corrected = {}
for d, (instr, stored, true) in zip(
        [x for x in gt_old if x['instruction'].strip() in truth_map], corrected_old):
    id_to_corrected[str(d['id'])] = true

gt_new_map = {d['instruction'].strip(): d.get('expected_domain', '') for d in gt_new}

def get_pred(e):
    preds = e.get('domain_pred', [])
    is_cross = e.get('is_cross', False)
    return 'CROSS' if is_cross else (preds[0] if preds else 'UNKNOWN')

y_true, y_pred = [], []
for e in eval_results:
    eid = str(e.get('id', ''))
    q = e.get('query', e.get('instruction', '')).strip()
    pred = get_pred(e)

    if eid in id_to_corrected:
        y_true.append(id_to_corrected[eid])
        y_pred.append(pred)
    elif q in gt_new_map:
        y_true.append(gt_new_map[q])
        y_pred.append(pred)

correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
print(f"\n교정 후 전체 정확도: {correct}/{len(y_true)} = {correct/len(y_true)*100:.2f}%")

dom_total = Counter()
dom_correct = Counter()
for t, p in zip(y_true, y_pred):
    dom_total[t] += 1
    if t == p:
        dom_correct[t] += 1
print("\n도메인별 정확도 (교정 후):")
for d in ['LAW', 'MED', 'SOC', 'STEM', 'CROSS']:
    if dom_total[d] > 0:
        acc = dom_correct[d] / dom_total[d] * 100
        print(f"  {d}: {dom_correct[d]}/{dom_total[d]} = {acc:.1f}%")

# ── 결과 저장 ──
with open("/home/sclab/moe_model/gt500_corrected_labels.json", "w") as f:
    json.dump({
        "id_to_corrected_domain": id_to_corrected,
        "total_accuracy": correct/len(y_true)*100,
        "n": len(y_true)
    }, f, indent=2)
print("\n저장 완료: gt500_corrected_labels.json")