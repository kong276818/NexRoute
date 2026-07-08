import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

DATA_DIR = "/home/sclab/moe_model/processed_v2"
GT_PATH  = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"

DOMAIN_FILES = {
    'LAW':   'law_rewritten.json',
    'MED':   'med_rewritten.json',
    'SOC':   'soc_rewritten.json',
    'STEM':  'stem_rewritten.json',
    'CROSS': 'cross_rewritten.json',
}
LABELS = ['LAW','MED','SOC','STEM','CROSS']

# GT500 로드 (테스트셋 - 학습에서 제외)
with open(GT_PATH) as f: gt = json.load(f)
gt_instructions = set(d['instruction'].strip() for d in gt)
print(f"GT500: {len(gt_instructions)}개 (학습에서 제외)")

# 학습 데이터 구성 (GT500 제외)
X_train, y_train = [], []
for domain, fname in DOMAIN_FILES.items():
    with open(f"{DATA_DIR}/{fname}") as f:
        data = json.load(f)
    added = 0
    for d in data:
        instr = d['instruction'].strip()
        if instr in gt_instructions:
            continue
        X_train.append(instr)
        y_train.append(domain)
        added += 1
    print(f"  {domain}: {added}개 학습 샘플")

print(f"\n총 학습 샘플: {len(X_train)}개")
print(f"클래스 분포: {Counter(y_train)}")

# TF-IDF + Logistic Regression 학습
print("\n[*] TF-IDF 벡터화...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2,
    sublinear_tf=True,
)
X_tr = tfidf.fit_transform(X_train)

print("[*] Logistic Regression 학습...")
clf = LogisticRegression(
    max_iter=1000,
    C=5.0,
    class_weight='balanced',
    random_state=42,
)
clf.fit(X_tr, y_train)

# GT500 테스트
X_test = [d['instruction'].strip() for d in gt]
y_test = [d.get('expected_domain','') for d in gt]
X_te = tfidf.transform(X_test)
y_pred = clf.predict(X_te)

print("\n=== Adaptive RAG (TF-IDF+LR) 라우팅 정확도 ===")
print(f"\n{'Domain':<8} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
print("-"*38)
total_c, total_t = 0, 0
domain_results = {}
for domain in LABELS:
    mask = [i for i,y in enumerate(y_test) if y==domain]
    if not mask: continue
    correct = sum(1 for i in mask if y_pred[i]==domain)
    total   = len(mask)
    acc     = correct/total*100
    print(f"{domain:<8} {correct:>8} {total:>7} {acc:>9.1f}%")
    domain_results[domain] = {'correct':correct,'total':total,'acc':acc}
    total_c += correct; total_t += total
print("-"*38)
print(f"{'Total':<8} {total_c:>8} {total_t:>7} {total_c/total_t*100:>9.1f}%")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=LABELS))

# 결과 저장
import pickle
with open('/home/sclab/moe_model/adaptive_tfidf_router.pkl','wb') as f:
    pickle.dump({'tfidf':tfidf,'clf':clf}, f)

results = {
    'domain_results': domain_results,
    'total_accuracy': total_c/total_t*100,
    'y_pred': list(y_pred),
    'y_test': y_test,
}
with open('/home/sclab/moe_model/adaptive_tfidf_results.json','w') as f:
    json.dump(results, f, indent=2)
print("\n저장 완료: adaptive_tfidf_router.pkl / adaptive_tfidf_results.json")
