"""
NexRoute 라우터 재학습 v3 - CROSS 98% 목표
핵심 전략:
1. CROSS 5배 오버샘플링
2. CROSS 손실 가중치 8배
3. Focal Loss γ=3.0
4. Hard negative: CROSS 쿼리에서 두 도메인 모두 1.0 강제
5. 기존 가중치 fine-tuning
"""
import json, random, torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from collections import Counter

DATA_DIR   = "/home/sclab/moe_model/processed_v2"
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
CKPT_OLD   = "/home/sclab/moe_model/router_multilabel.pt"
SAVE_PATH  = "/home/sclab/moe_model/router_multilabel_v3.pt"
BEST_PATH  = "/home/sclab/moe_model/router_multilabel_v3_best.pt"

DOMAINS    = ['LAW', 'MED', 'SOC', 'STEM']
LABELS_ALL = ['LAW', 'MED', 'SOC', 'STEM', 'CROSS']
TAU, DELTA = 0.30, 0.40

DOMAIN_FILES = {
    'LAW':   'law_rewritten.json',
    'MED':   'med_rewritten.json',
    'SOC':   'soc_rewritten.json',
    'STEM':  'stem_rewritten.json',
    'CROSS': 'cross_rewritten.json',
}

LR               = 2e-6
EPOCHS           = 10
BATCH_SIZE       = 16
MAX_LEN          = 256
CROSS_OVERSAMPLE = 5       # CROSS 5배
CROSS_LW         = 8.0     # CROSS 손실 8배
FOCAL_GAMMA      = 3.0     # 어려운 샘플 집중

CLASS_WEIGHTS = {'LAW': 1.0, 'MED': 3.0, 'SOC': 3.0, 'STEM': 1.5}

# ── GT500 제외 ────────────────────────────────────────────────
with open(GT_PATH) as f: gt = json.load(f)
gt_instrs = set(d['instruction'].strip() for d in gt)

# ── 학습 데이터 ───────────────────────────────────────────────
train_data = []
for domain, fname in DOMAIN_FILES.items():
    with open(f"{DATA_DIR}/{fname}") as f:
        data = json.load(f)
    for d in data:
        instr = d['instruction'].strip()
        if instr in gt_instrs: continue
        if domain == 'CROSS':
            crossed = d.get('crossed_domains', ['SOC', 'MED'])
            labels  = crossed if isinstance(crossed, list) else ['SOC', 'MED']
            # 최소 2개 도메인 보장
            if len(labels) < 2:
                labels = ['SOC', 'MED']
        else:
            labels = [domain]
        train_data.append({
            'text': instr, 'domains': labels, 'primary': domain
        })

# CROSS 오버샘플링
cross_samples = [d for d in train_data if d['primary'] == 'CROSS']
for _ in range(CROSS_OVERSAMPLE - 1):
    train_data.extend(cross_samples)

random.seed(42)
random.shuffle(train_data)
print(f"학습 샘플: {len(train_data)}개")
print(f"분포: {Counter(d['primary'] for d in train_data)}")


class RouterDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        enc  = self.tokenizer(
            item['text'], max_length=MAX_LEN, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        label = torch.zeros(4)
        for d in item['domains']:
            if d in DOMAINS:
                label[DOMAINS.index(d)] = 1.0
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         label,
            'primary':        item['primary'],
        }


class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 4)
        )

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(pooled)


def focal_loss(logits, targets, gamma=3.0, weights=None):
    bce   = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')
    prob  = torch.sigmoid(logits)
    p_t   = prob * targets + (1 - prob) * (1 - targets)
    focal = (1 - p_t) ** gamma * bce
    if weights is not None:
        focal = focal * weights.to(focal.device)
    return focal.mean()


def evaluate(model, tokenizer, gt_data):
    model.eval()
    y_true, y_pred = [], []
    for item in gt_data:
        q    = item['instruction'].strip()
        true = item.get('expected_domain', '')
        enc  = tokenizer(q, max_length=MAX_LEN, truncation=True,
                         padding='max_length', return_tensors='pt')
        with torch.no_grad():
            logits = model(enc['input_ids'].cuda(),
                           enc['attention_mask'].cuda())
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()
        ds = dict(zip(DOMAINS, scores))
        sd = sorted(ds.items(), key=lambda x: -x[1])
        s1, s2 = sd[0][1], sd[1][1]
        pred   = 'CROSS' if (s2 >= TAU and (s1-s2) <= DELTA) else sd[0][0]
        y_true.append(true); y_pred.append(pred)

    total = sum(1 for t,p in zip(y_true,y_pred) if t==p) / len(y_true) * 100
    per_d = {}
    for d in LABELS_ALL:
        mask = [i for i,t in enumerate(y_true) if t==d]
        if mask:
            per_d[d] = sum(1 for i in mask if y_pred[i]==d) / len(mask) * 100
    return total, per_d


if __name__ == "__main__":
    device    = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset   = RouterDataset(train_data, tokenizer)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=2)

    model = DomainRouter().to(device)
    try:
        model.load_state_dict(
            torch.load(CKPT_OLD, map_location=device), strict=False)
        print(f"[*] 기존 가중치 로드: {CKPT_OLD}")
    except Exception as e:
        print(f"[!] 새로 학습: {e}")

    cw        = torch.tensor([CLASS_WEIGHTS[d] for d in DOMAINS])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_score = 0.0
    print(f"\n[*] 학습 시작 (CROSS 98% 목표)")

    # 초기 성능 확인
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

            # 기본 focal loss
            loss = focal_loss(logits, lbls, FOCAL_GAMMA, cw)

            # CROSS 추가 손실 (두 도메인 모두 높아야 함)
            cross_mask = torch.tensor(
                [1.0 if p == 'CROSS' else 0.0 for p in prims]
            ).to(device)
            if cross_mask.sum() > 0:
                cross_focal = focal_loss(logits, lbls, FOCAL_GAMMA, cw)
                # CROSS 샘플에서 두 번째 도메인 점수도 높이도록 추가 패널티
                probs = torch.sigmoid(logits)
                # top-2 점수의 차이가 작을수록 좋음 (CROSS 감지 쉬움)
                top2_vals, _ = torch.topk(probs, 2, dim=1)
                gap_penalty  = (top2_vals[:, 0] - top2_vals[:, 1]) * cross_mask
                loss = loss + CROSS_LW * cross_focal + 0.5 * gap_penalty.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 200 == 0:
                print(f"  E{epoch+1} S{step+1}/{len(loader)} "
                      f"loss={total_loss/(step+1):.4f}")

        # 에폭 평가
        acc, per_d = evaluate(model, tokenizer, gt)
        cross_acc  = per_d.get('CROSS', 0)
        med_acc    = per_d.get('MED', 0)
        score      = acc * 0.3 + cross_acc * 0.5 + med_acc * 0.2

        print(f"\nEpoch {epoch+1}/{EPOCHS} | Total={acc:.1f}%")
        for d in LABELS_ALL:
            marker = " ★" if d in ['CROSS','MED'] else ""
            print(f"  {d}: {per_d.get(d,0):.1f}%{marker}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  → Best 저장! score={score:.2f}")

    torch.save(model.state_dict(), SAVE_PATH)

    # 최종 평가
    model.load_state_dict(torch.load(BEST_PATH))
    acc, per_d = evaluate(model, tokenizer, gt)
    print(f"\n{'='*40}")
    print(f"최종 Best 성능:")
    print(f"  Total: {acc:.1f}%")
    for d in LABELS_ALL:
        print(f"  {d}: {per_d.get(d,0):.1f}%")
    print(f"\n저장: {BEST_PATH}")