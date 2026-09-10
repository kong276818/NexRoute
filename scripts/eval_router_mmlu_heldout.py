"""
eval_router_mmlu_heldout.py
============================
Approach 1: MMLU held-out subjects (학습에 사용되지 않은 카테고리)
→ 4개 도메인 라우팅 정확도 평가

학습에 사용된 MMLU 카테고리:
  MED:  clinical_knowledge, medical_genetics, anatomy, professional_medicine,
        college_medicine, virology, college_biology, high_school_biology
  LAW:  jurisprudence, international_law, professional_law
  STEM: high_school_physics, college_physics, high_school_chemistry, college_chemistry,
        high_school_mathematics, college_mathematics, electrical_engineering, computer_science,
        abstract_algebra, conceptual_physics, astronomy, college_computer_science
  SOC:  sociology, high_school_psychology, professional_psychology, high_school_geography,
        political_science, econometrics, public_relations, high_school_government_and_politics,
        us_foreign_policy, security_studies, human_sexuality, moral_disputes
  CROSS: global_facts, formal_logic, philosophy, logical_fallacies, miscellaneous, world_religions

held-out 카테고리 (학습 미사용):
  MED:  nutrition, human_aging
  LAW:  business_ethics, professional_accounting  (법적·규범적 추론 포함)
  STEM: machine_learning, high_school_statistics, elementary_mathematics,
        high_school_computer_science
  SOC:  high_school_macroeconomics, high_school_microeconomics, management, marketing,
        moral_scenarios, prehistory, high_school_us_history, high_school_world_history
"""
import json, torch, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from sklearn.metrics import classification_report

ROUTER_PATH = "/home/sclab/moe_model/router_deberta_best.pt"
MODEL_NAME  = "microsoft/deberta-large"
DOMAINS     = ['LAW', 'MED', 'SOC', 'STEM']
TAU, DELTA  = 0.30, 0.40
MAX_LEN     = 128
N_PER_DOMAIN = 150   # 도메인당 최대 샘플 수

HELDOUT = {
    "MED":  ["nutrition", "human_aging"],
    "LAW":  ["business_ethics", "professional_accounting"],
    "STEM": ["machine_learning", "high_school_statistics",
             "elementary_mathematics", "high_school_computer_science"],
    "SOC":  ["high_school_macroeconomics", "high_school_microeconomics",
             "management", "marketing", "moral_scenarios",
             "prehistory", "high_school_us_history", "high_school_world_history"],
}


class DeBERTaRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(MODEL_NAME)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 4)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            kw['token_type_ids'] = token_type_ids
        out    = self.encoder(**kw)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(pooled)


def predict(model, tokenizer, text, device):
    enc = tokenizer(text, max_length=MAX_LEN, truncation=True,
                    padding='max_length', return_tensors='pt')
    kw = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        scores = torch.sigmoid(model(**kw)).squeeze().cpu().tolist()
    ds = dict(zip(DOMAINS, scores))
    sd = sorted(ds.items(), key=lambda x: -x[1])
    s1, s2 = sd[0][1], sd[1][1]
    pred = 'CROSS' if (s2 >= TAU and (s1 - s2) <= DELTA) else sd[0][0]
    return pred, ds


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] 라우터 모델 로드 ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = DeBERTaRouter().to(device)
model.load_state_dict(torch.load(ROUTER_PATH, map_location=device))
model.eval()
print(f"[+] 로드 완료 ({device})")

# ── 데이터 수집 ───────────────────────────────────────────────────────────────
all_items = []   # (question, true_domain)

for domain, subjects in HELDOUT.items():
    items = []
    for subj in subjects:
        try:
            for split in ["test", "validation", "train"]:
                ds = load_dataset("cais/mmlu", subj, split=split)
                for row in ds:
                    q = row.get("question", "").strip()
                    if q and len(q.split()) >= 5:
                        items.append((q, domain))
                if len(items) >= N_PER_DOMAIN:
                    break
        except Exception as e:
            print(f"  ⚠ {subj}: {e}")
        if len(items) >= N_PER_DOMAIN:
            break
    items = items[:N_PER_DOMAIN]
    print(f"  {domain}: {len(items)}개 ({', '.join(subjects)})")
    all_items.extend(items)

print(f"\n총 {len(all_items)}개 샘플")

# ── 라우터 추론 ───────────────────────────────────────────────────────────────
y_true, y_pred = [], []
results = []

for i, (q, true_d) in enumerate(all_items):
    pred, scores = predict(model, tokenizer, q, device)
    y_true.append(true_d)
    y_pred.append(pred)
    results.append({"question": q[:80], "true": true_d, "pred": pred, "scores": scores})
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(all_items)}]")

# ── 결과 계산 ─────────────────────────────────────────────────────────────────
correct = sum(t == p for t, p in zip(y_true, y_pred))
total   = len(y_true)
acc     = correct / total * 100
print(f"\n{'='*60}")
print(f"MMLU Held-out External Routing Accuracy")
print(f"{'='*60}")
print(f"전체: {correct}/{total} = {acc:.1f}%\n")

print("도메인별 정확도:")
for domain in DOMAINS:
    mask = [i for i, t in enumerate(y_true) if t == domain]
    if mask:
        d_correct = sum(1 for i in mask if y_pred[i] == domain)
        d_acc = d_correct / len(mask) * 100
        print(f"  {domain:<6}: {d_correct}/{len(mask)} = {d_acc:.1f}%")

print("\n오분류 분포 (True → Pred):")
from collections import Counter
errors = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
for (t, p), cnt in errors.most_common(10):
    print(f"  {t} → {p}: {cnt}건")

# JSON 저장
out = {
    "total_accuracy": round(acc, 2),
    "n_total": total,
    "n_correct": correct,
    "per_domain": {},
    "confusion": {},
}
for domain in DOMAINS:
    mask = [i for i, t in enumerate(y_true) if t == domain]
    if mask:
        d_correct = sum(1 for i in mask if y_pred[i] == domain)
        out["per_domain"][domain] = {
            "n": len(mask), "correct": d_correct,
            "accuracy": round(d_correct / len(mask) * 100, 1),
        }
    pred_dist = Counter(y_pred[i] for i in mask) if mask else {}
    out["confusion"][domain] = dict(pred_dist)

import json
with open("/home/sclab/moe_model/results/router_eval_mmlu_heldout.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[+] 저장: results/router_eval_mmlu_heldout.json")
print(f"\n[논문용] MMLU held-out {total}개, 정확도 {acc:.1f}%")
