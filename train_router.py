"""
train_router.py — DualMoE 라우터 학습 (키워드 강화 버전)

Domain Router : TF-IDF + 도메인 키워드 특징 + LinearSVC
Func Router   : TF-IDF + LinearSVC

데이터: /home/sclab/moe_model/processed_v2/
저장:   /home/sclab/moe_model/routers/
"""

import os
import json
import pickle
import numpy as np
from collections import Counter
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

DATA_DIR   = "/home/sclab/moe_model/processed_v2"
OUTPUT_DIR = "/home/sclab/moe_model/routers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAIN_FILES = ["med.json","law.json","stem.json","soc.json","cross.json"]

DOMAIN_LABELS = {"MED":0,"LAW":1,"STEM":2,"SOC":3,"CROSS":4}
FUNC_LABELS   = {"WHAT":0,"HOW":1,"WHY":2,"COMPARE":3,"OTHER":4}

# 도메인 특화 키워드
DOMAIN_KEYWORDS = {
    "MED":   ["patient","diagnosis","treatment","symptom","disease","clinical",
              "medical","doctor","hospital","surgery","drug","medication",
              "therapy","infection","cancer","diabetes","blood","pain",
              "pneumonia","fever","vaccine","anatomy","pathology","physician",
              "nurse","dose","prescription","chronic","acute","prognosis"],
    "LAW":   ["court","law","legal","contract","defendant","plaintiff",
              "statute","judge","attorney","criminal","civil","liability",
              "negligence","rights","jurisdiction","ruling","verdict",
              "constitution","regulation","legislation","justice","tort",
              "appeal","damages","prosecution","counsel","testimony"],
    "STEM":  ["physics","chemistry","biology","math","equation","force",
              "energy","molecule","atom","cell","gene","evolution",
              "calculate","formula","reaction","element","experiment",
              "photosynthesis","gravity","velocity","acceleration","circuit",
              "magnetic","electric","quantum","nuclear","organic","compound"],
    "SOC":   ["society","social","culture","psychology","behavior","community",
              "people","group","political","government","economy","education",
              "family","relationship","communication","emotion","cognitive",
              "mental","personality","attitude","belief","norm","socialization",
              "institution","class","identity","gender","race","ethnicity"],
    "CROSS": ["global","world","history","science","technology","environment",
              "research","study","data","analysis","system","theory",
              "international","national","policy","impact","effect","cross",
              "interdisciplinary","multidisciplinary","compare","between"],
}


# ────────────────────────────────────────────
# 키워드 특징 추출기 (top-level class for pickling)
# ────────────────────────────────────────────
class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, keyword_dict):
        self.keyword_dict = keyword_dict
        self.domains = list(keyword_dict.keys())

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            t = text.lower()
            row = [sum(1 for kw in self.keyword_dict[d] if kw in t)
                   for d in self.domains]
            features.append(row)
        return np.array(features, dtype=np.float64)


# ────────────────────────────────────────────
# Hybrid Router (top-level class for pickling)
# ────────────────────────────────────────────
class HybridRouter:
    def __init__(self, tfidf, kw_extractor, clf):
        self.tfidf        = tfidf
        self.kw_extractor = kw_extractor
        self.clf          = clf
        self.classes_     = clf.classes_

    def _features(self, texts):
        t = self.tfidf.transform(texts)
        k = self.kw_extractor.transform(texts)
        return hstack([t, k * 5])

    def predict(self, texts):
        return self.clf.predict(self._features(texts))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._features(texts))


# ────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────
def load_all_data():
    all_data = []
    for fname in DOMAIN_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  ⚠ 없음: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        all_data.extend(data)
        print(f"  {fname}: {len(data)}개")
    print(f"  전체: {len(all_data)}개")
    return all_data


# ────────────────────────────────────────────
# Domain Router 학습
# ────────────────────────────────────────────
def train_domain_router(all_data):
    print(f"\n{'='*55}")
    print("  Domain Router 학습 (TF-IDF + 키워드 특징 + LinearSVC)")
    print(f"{'='*55}")

    save_path = os.path.join(OUTPUT_DIR, "domain_router")
    os.makedirs(save_path, exist_ok=True)

    texts  = [item["instruction"].strip() for item in all_data]
    labels = [item.get("domain", "CROSS") for item in all_data]

    print("\n  라벨 분포:")
    for k, v in sorted(Counter(labels).items()):
        bar = "█" * (v // 200)
        print(f"    {k:10s}: {v:5d}개  {bar}")

    train_X, val_X, train_y, val_y = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"\n  Train: {len(train_X)} / Val: {len(val_X)}")

    # TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )
    kw_extractor = KeywordFeatureExtractor(DOMAIN_KEYWORDS)

    # 특징 결합
    train_feat = hstack([tfidf.fit_transform(train_X), kw_extractor.fit_transform(train_X) * 5])
    val_feat   = hstack([tfidf.transform(val_X),       kw_extractor.transform(val_X) * 5])

    # SVM
    clf = CalibratedClassifierCV(
        LinearSVC(C=2.0, max_iter=5000, class_weight="balanced"), cv=3
    )
    clf.fit(train_feat, train_y)
    val_preds = clf.predict(val_feat)

    val_acc = accuracy_score(val_y, val_preds)
    print(f"\n  Val Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(classification_report(val_y, val_preds, zero_division=0))

    # 저장
    router = HybridRouter(tfidf, kw_extractor, clf)
    with open(os.path.join(save_path, "pipeline.pkl"), "wb") as f:
        pickle.dump(router, f)

    config = {
        "router_type": "domain",
        "val_accuracy": val_acc,
        "label_map": DOMAIN_LABELS,
        "id2label": {v: k for k, v in DOMAIN_LABELS.items()},
        "model": "TF-IDF(1-3gram) + KeywordFeatures(x5) + LinearSVC",
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✅ 저장: {save_path}")
    return router


# ────────────────────────────────────────────
# Function Router 학습
# ────────────────────────────────────────────
def train_func_router(all_data):
    print(f"\n{'='*55}")
    print("  Function Router 학습 (TF-IDF + LinearSVC)")
    print(f"{'='*55}")

    save_path = os.path.join(OUTPUT_DIR, "func_router")
    os.makedirs(save_path, exist_ok=True)

    texts  = [item["instruction"].strip() for item in all_data]
    labels = [item.get("func_type", "OTHER") for item in all_data]

    print("\n  라벨 분포:")
    for k, v in sorted(Counter(labels).items()):
        bar = "█" * (v // 200)
        print(f"    {k:10s}: {v:5d}개  {bar}")

    train_X, val_X, train_y, val_y = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"\n  Train: {len(train_X)} / Val: {len(val_X)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, class_weight="balanced"), cv=3
        )),
    ])

    pipeline.fit(train_X, train_y)
    val_preds = pipeline.predict(val_X)
    val_acc   = accuracy_score(val_y, val_preds)

    print(f"\n  Val Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(classification_report(val_y, val_preds, zero_division=0))

    with open(os.path.join(save_path, "pipeline.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    config = {
        "router_type": "func",
        "val_accuracy": val_acc,
        "label_map": FUNC_LABELS,
        "id2label": {v: k for k, v in FUNC_LABELS.items()},
        "model": "TF-IDF(1-2gram) + LinearSVC",
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✅ 저장: {save_path}")
    return pipeline


# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
if __name__ == "__main__":
    print("DualMoE Router 학습 (키워드 강화)\n")

    all_data = load_all_data()

    domain_router = train_domain_router(all_data)
    func_router   = train_func_router(all_data)

    print("\n" + "="*55)
    print("✅ 전체 라우터 학습 완료!")
    print(f"  Domain: {OUTPUT_DIR}/domain_router/pipeline.pkl")
    print(f"  Func  : {OUTPUT_DIR}/func_router/pipeline.pkl")
    print("="*55)

    # 테스트
    print("\n[라우팅 테스트]")
    queries = [
        "What is the treatment for pneumonia?",
        "What is diabetes?",
        "What is the legal definition of negligence?",
        "How does photosynthesis work?",
        "What causes depression in teenagers?",
        "Compare negligence and malpractice in medical law",
        "What is the formula for calculating force?",
        "Why do societies develop social hierarchies?",
        "What are the legal requirements for a valid contract?",
        "How do vaccines affect public health policy?",
    ]

    print(f"\n  {'쿼리':<52} {'Domain':>8} {'Func':>10}")
    print("  " + "-" * 73)
    for q in queries:
        d      = domain_router.predict([q])[0]
        f      = func_router.predict([q])[0]
        d_prob = max(domain_router.predict_proba([q])[0])
        f_prob = max(func_router.predict_proba([q])[0])
        print(f"  {q[:50]:<52} {d}({d_prob:.2f}) {f}({f_prob:.2f})")