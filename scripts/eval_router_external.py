"""
eval_router_external.py
========================
Approach 2: 완전 외부 벤치마크 (학습 데이터와 전혀 다른 출처)
→ 4개 도메인 라우팅 정확도 평가

데이터 출처 (학습 데이터와 완전 다른 소스):
  MED:  openlifescienceai/medmcqa  (인도 의대 입시, 182k 문제)
  LAW:  nguyen-brat/legal_bench    (contract_nli / sara_entailment 등)
         → 없으면 rcfgroup/legalbench 또는 law_qa 시도
  STEM: allenai/scitail            (과학 텍스트 추론; SciQ와 다른 소스)
         → 없으면 derek-thomas/ScienceQA
  SOC:  allenai/social_i_qa        (사회적 상황 추론)
"""
import json, torch, numpy as np
from datasets import load_dataset
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from collections import Counter

ROUTER_PATH  = "/home/sclab/moe_model/router_deberta_best.pt"
MODEL_NAME   = "microsoft/deberta-large"
DOMAINS      = ['LAW', 'MED', 'SOC', 'STEM']
TAU, DELTA   = 0.30, 0.40
MAX_LEN      = 128
N_PER_DOMAIN = 200
OUT_PATH     = "/home/sclab/moe_model/results/router_eval_external.json"


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


def predict_batch(model, tokenizer, texts, device, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc   = tokenizer(batch, max_length=MAX_LEN, truncation=True,
                          padding='max_length', return_tensors='pt')
        kw = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**kw)
            scores = torch.sigmoid(logits).cpu().tolist()
        for sc in scores:
            ds = dict(zip(DOMAINS, sc))
            sd = sorted(ds.items(), key=lambda x: -x[1])
            s1, s2 = sd[0][1], sd[1][1]
            pred   = 'CROSS' if (s2 >= TAU and (s1 - s2) <= DELTA) else sd[0][0]
            results.append((pred, ds))
    return results


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] 라우터 모델 로드 ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = DeBERTaRouter().to(device)
model.load_state_dict(torch.load(ROUTER_PATH, map_location=device))
model.eval()
print(f"[+] 로드 완료 ({device})")


# ── 데이터 로드 함수 ──────────────────────────────────────────────────────────

def load_med():
    """MedMCQA: 인도 의대 입시 문제 (학습에 사용된 MedQA/MMLU-medical과 완전 다른 소스)"""
    items = []
    try:
        ds = load_dataset("openlifescienceai/medmcqa", split="validation",
                          trust_remote_code=True)
        for row in ds:
            q = row.get("question", "").strip()
            if q and len(q.split()) >= 5:
                items.append(q)
            if len(items) >= N_PER_DOMAIN:
                break
        print(f"  MedMCQA: {len(items)}개")
    except Exception as e:
        print(f"  ⚠ MedMCQA: {e}")
    return items[:N_PER_DOMAIN]


def load_law():
    """LegalBench: 계약 해석 / 법률 추론 태스크 (lex_glue case_hold / ledgar와 다른 소스)"""
    items = []

    # 시도 1: nguyen-brat/legal_bench (contract_nli)
    try:
        ds = load_dataset("nguyen-brat/legal_bench", "contract_nli",
                          split="test", trust_remote_code=True)
        for row in ds:
            q = (row.get("text") or row.get("question") or row.get("premise") or "").strip()
            if q and len(q.split()) >= 5:
                items.append(q[:300])
            if len(items) >= N_PER_DOMAIN:
                break
        print(f"  LegalBench(contract_nli): {len(items)}개")
    except Exception as e:
        print(f"  ⚠ LegalBench contract_nli: {e}")

    # 시도 2: rcfgroup/legalbench
    if len(items) < N_PER_DOMAIN:
        try:
            ds = load_dataset("rcfgroup/legalbench", split="train",
                              trust_remote_code=True)
            for row in ds:
                q = (row.get("input") or row.get("text") or row.get("question") or "").strip()
                if q and len(q.split()) >= 5:
                    items.append(q[:300])
                if len(items) >= N_PER_DOMAIN:
                    break
            print(f"  rcfgroup/legalbench: {len(items)}개 (누적)")
        except Exception as e:
            print(f"  ⚠ rcfgroup/legalbench: {e}")

    # 시도 3: Exploration Labs/legal_qa
    if len(items) < N_PER_DOMAIN:
        try:
            ds = load_dataset("joelito/legal_case_document_summarization",
                              split="train", trust_remote_code=True)
            for row in ds:
                q = (row.get("summary") or "").strip()
                if q and len(q.split()) >= 10:
                    # 첫 문장만 질문처럼 사용
                    q = q.split('.')[0][:300]
                    items.append(q)
                if len(items) >= N_PER_DOMAIN:
                    break
            print(f"  legal_case_document: {len(items)}개 (누적)")
        except Exception as e:
            print(f"  ⚠ legal_case_document: {e}")

    # 시도 4: MMLU professional_law test split (partial; was used in train but test questions differ)
    # 사용 안 함 (in-distribution 위험)

    # 시도 4: coastalcph/lextreme
    if len(items) < N_PER_DOMAIN:
        try:
            # eurlexsum는 EU 법률 텍스트 요약
            ds = load_dataset("coastalcph/lextreme", "swiss_judgment_prediction",
                              split="test", trust_remote_code=True)
            for row in ds:
                q = (row.get("text") or "").strip()
                if q and len(q.split()) >= 10:
                    items.append(q[:300])
                if len(items) >= N_PER_DOMAIN:
                    break
            print(f"  lextreme(swiss_judgment): {len(items)}개 (누적)")
        except Exception as e:
            print(f"  ⚠ lextreme: {e}")

    return items[:N_PER_DOMAIN]


def load_stem():
    """SciTail: 과학 가설 추론 (SciQ, MMLU-STEM과 다른 소스)"""
    items = []

    # 시도 1: allenai/scitail (과학 텍스트 전제-가설 쌍)
    try:
        ds = load_dataset("allenai/scitail", "tsv_format", split="test",
                          trust_remote_code=True)
        for row in ds:
            # premise와 hypothesis를 합쳐 질문처럼 사용
            premise    = (row.get("premise") or "").strip()
            hypothesis = (row.get("hypothesis") or "").strip()
            if premise and len(premise.split()) >= 5:
                q = f"{premise} Is it true that: {hypothesis}" if hypothesis else premise
                items.append(q[:300])
            if len(items) >= N_PER_DOMAIN:
                break
        print(f"  SciTail: {len(items)}개")
    except Exception as e:
        print(f"  ⚠ SciTail: {e}")

    # 시도 2: derek-thomas/ScienceQA (멀티모달이지만 텍스트만 사용)
    if len(items) < N_PER_DOMAIN:
        try:
            ds = load_dataset("derek-thomas/ScienceQA", split="test",
                              trust_remote_code=True)
            for row in ds:
                if row.get("image"):   # 이미지 있는 문제 건너뜀
                    continue
                q = (row.get("question") or "").strip()
                if q and len(q.split()) >= 5:
                    items.append(q[:300])
                if len(items) >= N_PER_DOMAIN:
                    break
            print(f"  ScienceQA(text-only): {len(items)}개 (누적)")
        except Exception as e:
            print(f"  ⚠ ScienceQA: {e}")

    # 시도 3: AI2 ARC (Challenge) - 사용했지만 CROSS로 라벨됨, STEM 추론 포함
    # 사용 안 함 (training에 포함됨)

    # 시도 3: aqua_rat (수학 추론)
    if len(items) < N_PER_DOMAIN:
        try:
            ds = load_dataset("aqua_rat", split="test", trust_remote_code=True)
            for row in ds:
                q = (row.get("question") or "").strip()
                if q and len(q.split()) >= 5:
                    items.append(q[:300])
                if len(items) >= N_PER_DOMAIN:
                    break
            print(f"  AQUA-RAT: {len(items)}개 (누적)")
        except Exception as e:
            print(f"  ⚠ AQUA-RAT: {e}")

    return items[:N_PER_DOMAIN]


def load_soc():
    """Social IQA: 사회적 상황 추론 (CommonsenseQA, MMLU-SOC와 다른 소스)"""
    items = []

    # 시도 1: allenai/social_i_qa
    try:
        ds = load_dataset("allenai/social_i_qa", split="validation",
                          trust_remote_code=True)
        for row in ds:
            ctx = (row.get("context") or "").strip()
            q   = (row.get("question") or "").strip()
            if q and len(q.split()) >= 3:
                full_q = f"{ctx} {q}".strip() if ctx else q
                items.append(full_q[:300])
            if len(items) >= N_PER_DOMAIN:
                break
        print(f"  Social IQA: {len(items)}개")
    except Exception as e:
        print(f"  ⚠ Social IQA: {e}")

    # 시도 2: Rowan/hellaswag
    if len(items) < N_PER_DOMAIN:
        try:
            ds = load_dataset("Rowan/hellaswag", split="validation",
                              trust_remote_code=True)
            for row in ds:
                ctx = (row.get("activity_label") or "") + " " + (row.get("ctx") or "")
                ctx = ctx.strip()
                if ctx and len(ctx.split()) >= 5:
                    items.append(ctx[:300])
                if len(items) >= N_PER_DOMAIN:
                    break
            print(f"  HellaSwag: {len(items)}개 (누적)")
        except Exception as e:
            print(f"  ⚠ HellaSwag: {e}")

    return items[:N_PER_DOMAIN]


# ── 데이터 수집 ───────────────────────────────────────────────────────────────
print("\n[*] 외부 데이터셋 로드...")
loaders = {"MED": load_med, "LAW": load_law, "STEM": load_stem, "SOC": load_soc}

all_texts, all_labels, dataset_info = [], [], {}
for domain, loader_fn in loaders.items():
    print(f"\n--- {domain} ---")
    texts = loader_fn()
    dataset_info[domain] = len(texts)
    all_texts.extend(texts)
    all_labels.extend([domain] * len(texts))

print(f"\n총 {len(all_texts)}개 샘플 (MED={dataset_info.get('MED',0)}, "
      f"LAW={dataset_info.get('LAW',0)}, STEM={dataset_info.get('STEM',0)}, "
      f"SOC={dataset_info.get('SOC',0)})")

# ── 라우터 추론 ───────────────────────────────────────────────────────────────
print("\n[*] 라우터 추론...")
preds_raw = predict_batch(model, tokenizer, all_texts, device, batch_size=32)
y_pred    = [p for p, _ in preds_raw]
y_true    = all_labels

# ── 결과 계산 ─────────────────────────────────────────────────────────────────
correct = sum(t == p for t, p in zip(y_true, y_pred))
total   = len(y_true)
acc     = correct / total * 100

print(f"\n{'='*60}")
print(f"External Cross-Domain Routing Accuracy")
print(f"{'='*60}")
print(f"전체: {correct}/{total} = {acc:.1f}%\n")

print("도메인별 정확도:")
per_domain = {}
for domain in DOMAINS:
    mask = [i for i, t in enumerate(y_true) if t == domain]
    if mask:
        d_correct = sum(1 for i in mask if y_pred[i] == domain)
        d_acc     = d_correct / len(mask) * 100
        per_domain[domain] = {"n": len(mask), "correct": d_correct, "accuracy": round(d_acc, 1)}
        print(f"  {domain:<6}: {d_correct}/{len(mask)} = {d_acc:.1f}%")

print("\n오분류 분포 (True → Pred):")
errors = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
for (t, p), cnt in errors.most_common(10):
    print(f"  {t} → {p}: {cnt}건")

# JSON 저장
out = {
    "total_accuracy": round(acc, 2),
    "n_total": total,
    "n_correct": correct,
    "per_domain": per_domain,
    "dataset_info": {
        "MED": "MedMCQA (openlifescienceai/medmcqa)",
        "LAW": "LegalBench / coastalcph/lextreme",
        "STEM": "SciTail / ScienceQA / AQUA-RAT",
        "SOC": "Social IQA (allenai/social_i_qa)",
    },
    "confusion": {},
}
for domain in DOMAINS:
    mask = [i for i, t in enumerate(y_true) if t == domain]
    pred_dist = Counter(y_pred[i] for i in mask) if mask else {}
    out["confusion"][domain] = dict(pred_dist)

with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[+] 저장: {OUT_PATH}")
print(f"\n[논문용] External cross-domain {total}개, 정확도 {acc:.1f}%")
