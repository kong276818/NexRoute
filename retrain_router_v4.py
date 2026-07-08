import json, random, torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from collections import Counter

DATA_DIR   = "/home/sclab/moe_model/processed_v2"
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
CKPT_OLD   = "/home/sclab/moe_model/router_multilabel.pt"   # 원본에서 fine-tuning
BEST_PATH  = "/home/sclab/moe_model/router_multilabel_v4_best.pt"

DOMAINS    = ['LAW','MED','SOC','STEM']
LABELS_ALL = ['LAW','MED','SOC','STEM','CROSS']
TAU, DELTA = 0.30, 0.40

DOMAIN_FILES = {
    'LAW':   'law_rewritten.json', 'MED':   'med_rewritten.json',
    'SOC':   'soc_rewritten.json', 'STEM':  'stem_rewritten.json',
    'CROSS': 'cross_rewritten.json',
}

LR               = 2e-6
EPOCHS           = 10
BATCH_SIZE       = 16
MAX_LEN          = 256
CROSS_OVERSAMPLE = 3      # 3배 (v3의 5배에서 줄임)
CROSS_LW         = 4.0    # 4배 (v3의 8배에서 줄임)
FOCAL_GAMMA      = 2.0    # 2.0 (v3의 3.0에서 줄임)
CLASS_WEIGHTS    = {'LAW':1.0,'MED':2.5,'SOC':2.5,'STEM':1.2}

with open(GT_PATH) as f: gt = json.load(f)
gt_instrs = set(d['instruction'].strip() for d in gt)

train_data = []
for domain, fname in DOMAIN_FILES.items():
    with open(f"{DATA_DIR}/{fname}") as f: data = json.load(f)
    for d in data:
        instr = d['instruction'].strip()
        if instr in gt_instrs: continue
        if domain == 'CROSS':
            crossed = d.get('crossed_domains', ['SOC','MED'])
            labels  = crossed if isinstance(crossed, list) else ['SOC','MED']
            if len(labels) < 2: labels = ['SOC','MED']
        else:
            labels = [domain]
        train_data.append({'text':instr,'domains':labels,'primary':domain})

cross_samples = [d for d in train_data if d['primary']=='CROSS']
for _ in range(CROSS_OVERSAMPLE - 1):
    train_data.extend(cross_samples)

random.seed(42); random.shuffle(train_data)
print(f"학습 샘플: {len(train_data)}개")
print(f"분포: {Counter(d['primary'] for d in train_data)}")

class RouterDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data; self.tokenizer = tokenizer
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        item = self.data[i]
        enc  = self.tokenizer(item['text'], max_length=MAX_LEN,
                              truncation=True, padding='max_length', return_tensors='pt')
        label = torch.zeros(4)
        for d in item['domains']:
            if d in DOMAINS: label[DOMAINS.index(d)] = 1.0
        return {'input_ids':enc['input_ids'].squeeze(),
                'attention_mask':enc['attention_mask'].squeeze(),
                'labels':label, 'primary':item['primary']}

# 원본 아키텍처 그대로 유지
class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768,256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256,4))
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:,0,:])

def focal_loss(logits, targets, gamma=2.0, weights=None):
    bce   = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    prob  = torch.sigmoid(logits)
    p_t   = prob * targets + (1-prob)*(1-targets)
    focal = (1-p_t)**gamma * bce
    if weights is not None: focal = focal * weights.to(focal.device)
    return focal.mean()

def evaluate(model, tokenizer, gt_data):
    model.eval()
    y_true, y_pred = [], []
    for item in gt_data:
        q    = item['instruction'].strip()
        true = item.get('expected_domain','')
        enc  = tokenizer(q, max_length=MAX_LEN, truncation=True,
                         padding='max_length', return_tensors='pt')
        with torch.no_grad():
            scores = torch.sigmoid(
                model(enc['input_ids'].cuda(), enc['attention_mask'].cuda())
            ).squeeze().cpu().tolist()
        ds = dict(zip(DOMAINS, scores))
        sd = sorted(ds.items(), key=lambda x:-x[1])
        s1, s2 = sd[0][1], sd[1][1]
        pred = 'CROSS' if (s2>=TAU and (s1-s2)<=DELTA) else sd[0][0]
        y_true.append(true); y_pred.append(pred)
    total = sum(1 for t,p in zip(y_true,y_pred) if t==p)/len(y_true)*100
    per_d = {}
    for d in LABELS_ALL:
        mask = [i for i,t in enumerate(y_true) if t==d]
        if mask: per_d[d] = sum(1 for i in mask if y_pred[i]==d)/len(mask)*100
    return total, per_d

if __name__ == "__main__":
    device    = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset   = RouterDataset(train_data, tokenizer)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = DomainRouter().to(device)
    model.load_state_dict(torch.load(CKPT_OLD, map_location=device))
    print(f"[*] 원본 가중치 로드 완료: {CKPT_OLD}")

    cw        = torch.tensor([CLASS_WEIGHTS[d] for d in DOMAINS])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=len(loader)*EPOCHS//10,
        num_training_steps=len(loader)*EPOCHS)

    best_score = 0.0

    # 초기 성능
    acc, per_d = evaluate(model, tokenizer, gt)
    print(f"\n[초기] Total={acc:.1f}%  " +
          "  ".join(f"{d}={per_d.get(d,0):.0f}%" for d in LABELS_ALL))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            lbls  = batch['labels'].to(device)
            prims = batch['primary']

            logits = model(ids, mask)
            loss   = focal_loss(logits, lbls, FOCAL_GAMMA, cw)

            # CROSS 추가 손실
            cross_mask = torch.tensor(
                [1.0 if p=='CROSS' else 0.0 for p in prims]).to(device)
            if cross_mask.sum() > 0:
                cross_focal = focal_loss(logits, lbls, FOCAL_GAMMA, cw)
                # gap penalty (가볍게)
                probs = torch.sigmoid(logits)
                top2, _ = torch.topk(probs, 2, dim=1)
                gap_penalty = (top2[:,0] - top2[:,1]) * cross_mask
                loss = loss + CROSS_LW * cross_focal + 0.3 * gap_penalty.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()

            if (step+1) % 200 == 0:
                print(f"  E{epoch+1} S{step+1}/{len(loader)} loss={total_loss/(step+1):.4f}")

        acc, per_d = evaluate(model, tokenizer, gt)
        cross_acc  = per_d.get('CROSS',0)
        soc_acc    = per_d.get('SOC',0)
        med_acc    = per_d.get('MED',0)

        # 균형 점수: Total + CROSS + SOC + MED 가중
        score = acc*0.25 + cross_acc*0.35 + soc_acc*0.20 + med_acc*0.20

        print(f"\nEpoch {epoch+1}/{EPOCHS} | Total={acc:.1f}%")
        for d in LABELS_ALL:
            marker = " ★" if d in ['CROSS','MED','SOC'] else ""
            print(f"  {d}: {per_d.get(d,0):.1f}%{marker}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  → Best! score={score:.2f}")

    # 최종
    model.load_state_dict(torch.load(BEST_PATH))
    acc, per_d = evaluate(model, tokenizer, gt)
    print(f"\n{'='*40}\n최종 Best:")
    print(f"  Total: {acc:.1f}%")
    for d in LABELS_ALL:
        print(f"  {d}: {per_d.get(d,0):.1f}%")
    print(f"저장: {BEST_PATH}")
# 위 코드에 separator_loss 추가 버전
# 위 코드에 separator_loss 추가 버전
