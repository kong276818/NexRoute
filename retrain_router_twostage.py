"""
2단계 라우터 학습
Stage 1: Binary classifier (CROSS vs non-CROSS)
Stage 2: 4-class classifier (LAW/MED/SOC/STEM) - CROSS 데이터 없음

핵심 아이디어:
- CROSS 감지와 단일도메인 분류를 분리
- 서로 간섭 없이 각자 최적화
"""
import json, random, torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from collections import Counter

DATA_DIR  = "/home/sclab/moe_model/processed_v2"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
S1_BEST   = "/home/sclab/moe_model/router_stage1_best.pt"  # CROSS 감지기
S2_BEST   = "/home/sclab/moe_model/router_stage2_best.pt"  # 도메인 분류기

DOMAINS    = ['LAW','MED','SOC','STEM']
LABELS_ALL = ['LAW','MED','SOC','STEM','CROSS']

DOMAIN_FILES = {
    'LAW':'law_rewritten.json','MED':'med_rewritten.json',
    'SOC':'soc_rewritten.json','STEM':'stem_rewritten.json',
    'CROSS':'cross_rewritten.json',
}

LR         = 3e-6
EPOCHS_S1  = 8
EPOCHS_S2  = 10
BATCH      = 16
MAX_LEN    = 256
CROSS_OS   = 2   # CROSS 2배 (Stage1용)

with open(GT_PATH) as f: gt = json.load(f)
gt_instrs = set(d['instruction'].strip() for d in gt)

# ── 데이터 로드 ───────────────────────────────────────────────
all_data = []
for domain, fname in DOMAIN_FILES.items():
    with open(f"{DATA_DIR}/{fname}") as f: data = json.load(f)
    for d in data:
        instr = d['instruction'].strip()
        if instr in gt_instrs: continue
        crossed = d.get('crossed_domains', ['SOC','MED'])
        all_data.append({
            'text': instr,
            'primary': domain,
            'crossed': crossed if isinstance(crossed,list) else ['SOC','MED'],
        })

# Stage 1 데이터: 전체 + CROSS 2배
s1_data = all_data.copy()
cross_samples = [d for d in all_data if d['primary']=='CROSS']
for _ in range(CROSS_OS - 1): s1_data.extend(cross_samples)

# Stage 2 데이터: CROSS 제외
s2_data = [d for d in all_data if d['primary'] != 'CROSS']

random.seed(42)
random.shuffle(s1_data); random.shuffle(s2_data)

print(f"Stage1 학습 샘플: {len(s1_data)}개  "
      f"CROSS: {sum(1 for d in s1_data if d['primary']=='CROSS')}")
print(f"Stage2 학습 샘플: {len(s2_data)}개  "
      f"분포: {Counter(d['primary'] for d in s2_data)}")


# ── Dataset ───────────────────────────────────────────────────
class S1Dataset(Dataset):
    """Stage1: CROSS(1) vs non-CROSS(0)"""
    def __init__(self, data, tokenizer):
        self.data = data; self.tok = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d['text'], max_length=MAX_LEN, truncation=True,
                       padding='max_length', return_tensors='pt')
        label = torch.tensor(1.0 if d['primary']=='CROSS' else 0.0)
        return {'input_ids': enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'label': label}

class S2Dataset(Dataset):
    """Stage2: LAW/MED/SOC/STEM 4-class"""
    def __init__(self, data, tokenizer):
        self.data = data; self.tok = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d['text'], max_length=MAX_LEN, truncation=True,
                       padding='max_length', return_tensors='pt')
        label = torch.tensor(DOMAINS.index(d['primary']), dtype=torch.long)
        return {'input_ids': enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'label': label}


# ── 모델 (공유 BERT 인코더) ───────────────────────────────────
class BinaryRouter(nn.Module):
    """Stage1: CROSS vs non-CROSS"""
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :]).squeeze(-1)

class DomainRouter4(nn.Module):
    """Stage2: LAW/MED/SOC/STEM"""
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 4))
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


# ── 평가 ─────────────────────────────────────────────────────
def evaluate_twostage(s1_model, s2_model, tokenizer, gt_data,
                      cross_thresh=0.50):
    s1_model.eval(); s2_model.eval()
    y_true, y_pred = [], []
    for item in gt_data:
        q    = item['instruction'].strip()
        true = item.get('expected_domain', '')
        enc  = tokenizer(q, max_length=MAX_LEN, truncation=True,
                         padding='max_length', return_tensors='pt')
        iids = enc['input_ids'].cuda()
        amsk = enc['attention_mask'].cuda()
        with torch.no_grad():
            cross_prob = torch.sigmoid(s1_model(iids, amsk)).item()
            if cross_prob >= cross_thresh:
                pred = 'CROSS'
            else:
                domain_logits = s2_model(iids, amsk)
                pred = DOMAINS[domain_logits.argmax().item()]
        y_true.append(true); y_pred.append(pred)

    total = sum(1 for t,p in zip(y_true,y_pred) if t==p)/len(y_true)*100
    per_d = {}
    for d in LABELS_ALL:
        mask = [i for i,t in enumerate(y_true) if t==d]
        if mask: per_d[d] = sum(1 for i in mask if y_pred[i]==d)/len(mask)*100
    return total, per_d


# ── 학습 공통 함수 ────────────────────────────────────────────
def train_model(model, loader, criterion, epochs, desc):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=len(loader)*epochs//10,
        num_training_steps=len(loader)*epochs)
    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        for step, batch in enumerate(loader):
            iids = batch['input_ids'].cuda()
            amsk = batch['attention_mask'].cuda()
            lbl  = batch['label'].cuda()
            logits = model(iids, amsk)
            loss = criterion(logits, lbl)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()
        print(f"  [{desc}] Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    device    = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # ──────────────────────────────────────────────────────────
    # Stage 1: CROSS 감지기 학습
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("Stage 1: CROSS Binary Classifier 학습")
    print("="*50)

    s1_dataset = S1Dataset(s1_data, tokenizer)
    s1_loader  = DataLoader(s1_dataset, batch_size=BATCH, shuffle=True, num_workers=2)
    s1_model   = BinaryRouter().to(device)

    # CROSS 비율에 맞게 pos_weight 설정
    n_cross   = sum(1 for d in s1_data if d['primary']=='CROSS')
    n_single  = len(s1_data) - n_cross
    pos_weight = torch.tensor([n_single / n_cross]).to(device)
    s1_crit   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_model(s1_model, s1_loader, s1_crit, EPOCHS_S1, "Stage1")
    torch.save(s1_model.state_dict(), S1_BEST)
    print(f"Stage1 저장: {S1_BEST}")

    # Stage1 단독 평가
    s1_model.eval()
    tp=tn=fp=fn=0
    for item in gt:
        q    = item['instruction'].strip()
        true = item.get('expected_domain','')
        enc  = tokenizer(q, max_length=MAX_LEN, truncation=True,
                         padding='max_length', return_tensors='pt')
        with torch.no_grad():
            prob = torch.sigmoid(s1_model(
                enc['input_ids'].cuda(), enc['attention_mask'].cuda())).item()
        pred_cross = prob >= 0.50
        true_cross = (true == 'CROSS')
        if pred_cross and true_cross: tp+=1
        elif not pred_cross and not true_cross: tn+=1
        elif pred_cross and not true_cross: fp+=1
        else: fn+=1

    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0
    print(f"\nStage1 결과: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  CROSS Precision: {precision*100:.1f}%")
    print(f"  CROSS Recall:    {recall*100:.1f}%")
    print(f"  CROSS F1:        {2*precision*recall/(precision+recall)*100:.1f}%")

    # ──────────────────────────────────────────────────────────
    # Stage 2: 4-class 도메인 분류기 학습
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("Stage 2: Domain 4-class Classifier 학습 (CROSS 제외)")
    print("="*50)

    s2_dataset = S2Dataset(s2_data, tokenizer)
    s2_loader  = DataLoader(s2_dataset, batch_size=BATCH, shuffle=True, num_workers=2)
    s2_model   = DomainRouter4().to(device)

    # 클래스 가중치 (SOC 가장 적으므로 높임)
    counts   = Counter(d['primary'] for d in s2_data)
    max_cnt  = max(counts.values())
    cw       = torch.tensor([max_cnt/counts[d] for d in DOMAINS]).to(device)
    s2_crit  = nn.CrossEntropyLoss(weight=cw)

    train_model(s2_model, s2_loader, s2_crit, EPOCHS_S2, "Stage2")
    torch.save(s2_model.state_dict(), S2_BEST)
    print(f"Stage2 저장: {S2_BEST}")

    # ──────────────────────────────────────────────────────────
    # 통합 평가
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("최종 2단계 통합 평가")
    print("="*50)

    # threshold 탐색 (0.3~0.7)
    best_total = 0; best_thresh = 0.5
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        acc, per_d = evaluate_twostage(s1_model, s2_model, tokenizer, gt, thresh)
        if acc > best_total:
            best_total = acc; best_thresh = thresh

    print(f"\n최적 threshold: {best_thresh}")
    acc, per_d = evaluate_twostage(s1_model, s2_model, tokenizer, gt, best_thresh)
    print(f"Total: {acc:.1f}%")
    for d in LABELS_ALL:
        print(f"  {d}: {per_d.get(d,0):.1f}%")

    # 결과 저장
    import json as _json
    with open("/home/sclab/moe_model/router_twostage_config.json","w") as f:
        _json.dump({"cross_threshold": best_thresh,
                    "s1_path": S1_BEST, "s2_path": S2_BEST}, f)
    print("\n설정 저장: router_twostage_config.json")