import json, torch, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from collections import defaultdict, Counter
import os, random
random.seed(42)

ROUTER_PATH = "/home/sclab/moe_model/router_multilabel.pt"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
DOMAINS     = ["LAW", "MED", "SOC", "STEM"]
ALL_DOMAINS = ["LAW", "MED", "SOC", "STEM", "CROSS"]
THRESHOLD   = 0.30
GAP         = 0.40

class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, 4)
        )
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

def load_data():
    domain_files = {
        "LAW":   "law.json",
        "MED":   "med.json",
        "SOC":   "soc.json",
        "STEM":  "stem.json",
        "CROSS": "cross_rewritten.json",
    }
    train_data, eval_data = [], []
    for domain, fname in domain_files.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        samples = [{"text": d.get("instruction",""), "domain": domain}
                   for d in data[:200]]
        random.shuffle(samples)
        # ★ 층화 분할: 도메인별로 각각 80/20
        split = int(len(samples) * 0.8)
        train_data.extend(samples[:split])
        eval_data.extend(samples[split:])

    random.shuffle(train_data)
    random.shuffle(eval_data)
    print(f"학습: {len(train_data)}개 / 평가: {len(eval_data)}개")
    print("평가 분포:", dict(Counter(d["domain"] for d in eval_data)))
    return train_data, eval_data

def train_tfidf_svm(train_data):
    print("[*] TF-IDF + SVM 학습...")
    texts  = [d["text"] for d in train_data]
    labels = [d["domain"] for d in train_data]
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    svm = LinearSVC(max_iter=2000)
    le  = LabelEncoder()
    X = vec.fit_transform(texts)
    y = le.fit_transform(labels)
    svm.fit(X, y)
    print("[*] 완료!")
    return vec, svm, le

def predict_tfidf(texts, vec, svm, le):
    X    = vec.transform(texts)
    pred = svm.predict(X)
    return [le.inverse_transform([p])[0] for p in pred]

def load_bert_router():
    print("[*] BERT 라우터 로드...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    router    = DomainRouter()
    router.load_state_dict(torch.load(ROUTER_PATH, map_location="cpu"))
    router.eval().to("cuda")
    print("[*] 완료!")
    return router, tokenizer

def predict_bert(texts, router, tokenizer):
    preds = []
    for text in texts:
        enc = tokenizer(text, max_length=256, truncation=True,
                        padding="max_length", return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = router(enc["input_ids"], enc["attention_mask"])
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()
        domain_scores = dict(zip(DOMAINS, scores))
        sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
        top1, s1 = sorted_d[0]
        top2, s2 = sorted_d[1]
        is_cross = (s2 >= THRESHOLD) and ((s1 - s2) <= GAP)
        preds.append("CROSS" if is_cross else top1)
    return preds

def calc_accuracy(preds, labels):
    total   = defaultdict(int)
    correct = defaultdict(int)
    for pred, label in zip(preds, labels):
        total[label] += 1
        if pred == label:
            correct[label] += 1
    print(f"\n{'도메인':<10} {'정확도':>10} {'맞음/전체':>12}")
    print("-"*35)
    for d in ALL_DOMAINS:
        if total[d] == 0: continue
        acc = correct[d]/total[d]*100
        print(f"{d:<10} {acc:>9.1f}% {correct[d]:>5}/{total[d]:<5}")
    total_acc = sum(correct.values())/sum(total.values())*100
    print("-"*35)
    print(f"{'전체':<10} {total_acc:>9.1f}%")
    return total_acc, {d: correct[d]/total[d]*100 for d in ALL_DOMAINS if total[d]>0}

if __name__ == "__main__":
    train_data, eval_data = load_data()
    eval_texts  = [d["text"] for d in eval_data]
    eval_labels = [d["domain"] for d in eval_data]

    # 실험 A: TF-IDF + SVM
    print("\n"+"="*50)
    print("실험 A: TF-IDF + SVM (Adaptive RAG 라우터)")
    print("="*50)
    vec, svm, le = train_tfidf_svm(train_data)
    preds_a = predict_tfidf(eval_texts, vec, svm, le)
    acc_a, per_a = calc_accuracy(preds_a, eval_labels)

    # 실험 B: BERT
    print("\n"+"="*50)
    print("실험 B: BERT (DualMoE 라우터)")
    print("="*50)
    router, tokenizer = load_bert_router()
    preds_b = predict_bert(eval_texts, router, tokenizer)
    acc_b, per_b = calc_accuracy(preds_b, eval_labels)

    # 최종 비교
    print("\n"+"="*70)
    print("  Ablation: 라우터 기여 vs 구조 기여")
    print("="*70)
    print(f"{'모델':<38} {'전체':>6}  {'SOC':>7}  {'STEM':>7}  {'CROSS':>7}")
    print("-"*70)
    print(f"{'Adaptive RAG (TF-IDF)':<38} {acc_a:>6.1f}%  {per_a.get('SOC',0):>6.1f}%  {per_a.get('STEM',0):>6.1f}%  {per_a.get('CROSS',0):>6.1f}%")
    print(f"{'Adaptive RAG + BERT (라우터만 교체)':<38} {acc_b:>6.1f}%  {per_b.get('SOC',0):>6.1f}%  {per_b.get('STEM',0):>6.1f}%  {per_b.get('CROSS',0):>6.1f}%")
    print(f"{'DualMoE Full (BERT + 구조)':<38} {'95.7':>6}%  {'93.4':>6}%  {'92.5':>6}%  {'95.0':>6}%")
    print("="*70)
    print(f"\n라우터 단독 효과  (TF-IDF → BERT):        {acc_b-acc_a:+.1f}%p")
    print(f"구조 추가 효과   (BERT → DualMoE Full):   {95.7-acc_b:+.1f}%p")
    print(f"전체 향상        (TF-IDF → DualMoE):      {95.7-acc_a:+.1f}%p")

    with open("/home/sclab/moe_model/baseline_results/router_ablation.json","w") as f:
        json.dump({"tfidf":{"total":acc_a,"per":per_a},"bert":{"total":acc_b,"per":per_b}}, f, indent=2)
    print("\n[+] 저장 완료!")