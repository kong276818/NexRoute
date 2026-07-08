"""
DeBERTa-large 기반 NexRoute 도메인 라우터
- microsoft/deberta-large (400M params)
- 단일 단계 멀티라벨 분류 (5 도메인)
- 목표: LAW 98 / MED 90 / SOC 88 / STEM 93 / CROSS 95 → avg 93%
"""
import json, random, torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from collections import Counter

DATA_DIR   = "/home/sclab/moe_model/processed_v2"
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
BEST_PATH  = "/home/sclab/moe_model/router_deberta_best.pt"
MODEL_NAME = "microsoft/deberta-large"

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

# DeBERTa-large 최적 하이퍼파라미터
LR               = 8e-6
EPOCHS           = 10
BATCH_SIZE       = 8      # 큰 모델이라 배치 줄임
MAX_LEN          = 256
CROSS_OVERSAMPLE = 3
MED_OVERSAMPLE   = 2
SOC_OVERSAMPLE   = 2
FOCAL_GAMMA      = 2.0
# 클래스 가중치: MED/SOC 특별 강조
CLASS_WEIGHTS    = {
    'LAW':  1.0,
    'MED':  3.0,
    'SOC':  3.0,
    'STEM': 1.5,
}

with open(GT_PATH) as f: gt = json.load(f)
gt_instrs = set(d['instruction'].strip() for d in gt)

# ── 학습 데이터 ───────────────────────────────────────────────
train_data = []
for domain, fname in DOMAIN_FILES.items():
    with open(f"{DATA_DIR}/{fname}") as f: data = json.load(f)
    for d in data:
        instr = d['instruction'].strip()
        if instr in gt_instrs: continue
        if domain == 'CROSS':
            crossed = d.get('crossed_domains', ['SOC', 'MED'])
            labels  = crossed if isinstance(crossed, list) else ['SOC', 'MED']
            if len(labels) < 2: labels = ['SOC', 'MED']
        else:
            labels = [domain]
        train_data.append({'text': instr, 'domains': labels, 'primary': domain})

# 오버샘플링 (CROSS + MED + SOC)
for domain, n in [('CROSS', CROSS_OVERSAMPLE-1),
                   ('MED',   MED_OVERSAMPLE-1),
                   ('SOC',   SOC_OVERSAMPLE-1)]:
    samples = [d for d in train_data if d['primary'] == domain]
    for _ in range(n): train_data.extend(samples)

random.seed(42); random.shuffle(train_data)
print(f"학습 샘플: {len(train_data)}개")
print(f"분포: {Counter(d['primary'] for d in train_data)}")


class RouterDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data; self.tok = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        enc  = self.tok(
            item['text'], max_length=MAX_LEN,
            truncation=True, padding='max_length', return_tensors='pt'
        )
        label = torch.zeros(4)
        for d in item['domains']:
            if d in DOMAINS: label[DOMAINS.index(d)] = 1.0
        out = {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         label,
            'primary':        item['primary'],
        }
        # DeBERTa는 token_type_ids 없음
        if 'token_type_ids' in enc:
            out['token_type_ids'] = enc['token_type_ids'].squeeze()
        return out


class DeBERTaRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size  # 1024 for deberta-large
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids
        out    = self.encoder(**kwargs)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(pooled)


def focal_loss(logits, targets, gamma=2.0, weights=None):
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
        enc  = tokenizer(
            q, max_length=MAX_LEN, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        kwargs = {
            'input_ids':      enc['input_ids'].cuda(),
            'attention_mask': enc['attention_mask'].cuda(),
        }
        if 'token_type_ids' in enc:
            kwargs['token_type_ids'] = enc['token_type_ids'].cuda()

        with torch.no_grad():
            logits = model(**kwargs)
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()

        ds = dict(zip(DOMAINS, scores))
        sd = sorted(ds.items(), key=lambda x: -x[1])
        s1, s2 = sd[0][1], sd[1][1]
        pred = 'CROSS' if (s2 >= TAU and (s1 - s2) <= DELTA) else sd[0][0]
        y_true.append(true); y_pred.append(pred)

    total = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) * 100
    per_d = {}
    for d in LABELS_ALL:
        mask = [i for i, t in enumerate(y_true) if t == d]
        if mask:
            per_d[d] = sum(1 for i in mask if y_pred[i] == d) / len(mask) * 100
    return total, per_d


if __name__ == "__main__":
    device = torch.device('cuda')
    print(f"[*] DeBERTa-large 토크나이저 로드...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = RouterDataset(train_data, tokenizer)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )

    print(f"[*] DeBERTa-large 모델 로드 (400M params)...")
    model = DeBERTaRouter().to(device)
    print(f"    파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    cw        = torch.tensor([CLASS_WEIGHTS[d] for d in DOMAINS])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )
    total_steps = len(loader) * EPOCHS
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    best_score = 0.0
    print(f"\n[*] 학습 시작 (EPOCHS={EPOCHS}, BATCH={BATCH_SIZE}, LR={LR})")
    print(f"    총 스텝: {total_steps}")

    for epoch in range(EPOCHS):
        model.train(); total_loss = 0.0

        for step, batch in enumerate(loader):
            ids   = batch['input_ids'].to(device)
            amsk  = batch['attention_mask'].to(device)
            lbls  = batch['labels'].to(device)
            prims = batch['primary']
            ttids = batch.get('token_type_ids')
            if ttids is not None: ttids = ttids.to(device)

            kwargs = {'input_ids': ids, 'attention_mask': amsk}
            if ttids is not None: kwargs['token_type_ids'] = ttids

            logits = model(**kwargs)
            loss   = focal_loss(logits, lbls, FOCAL_GAMMA, cw)

            # CROSS gap penalty
            probs = torch.sigmoid(logits)
            cross_mask = torch.tensor(
                [1.0 if p == 'CROSS' else 0.0 for p in prims]
            ).to(device)
            if cross_mask.sum() > 0:
                top2, _ = torch.topk(probs, 2, dim=1)
                gap_p   = (top2[:, 0] - top2[:, 1]) * cross_mask
                loss    = loss + 2.0 * focal_loss(logits, lbls, FOCAL_GAMMA, cw) + 0.3 * gap_p.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"  E{epoch+1} S{step+1}/{len(loader)} "
                      f"loss={total_loss/(step+1):.4f}")

        acc, per_d = evaluate(model, tokenizer, gt)
        avg_domain = np.mean([per_d.get(d, 0) for d in LABELS_ALL])

        print(f"\nEpoch {epoch+1}/{EPOCHS} | Total={acc:.1f}% | avg={avg_domain:.1f}%")
        for d in LABELS_ALL:
            marker = " ★" if d in ['MED', 'SOC', 'CROSS'] else ""
            print(f"  {d}: {per_d.get(d,0):.1f}%{marker}")

        # 저장 기준: 모든 도메인 평균 + MED/SOC/CROSS 가중
        score = (acc * 0.3
                 + per_d.get('CROSS', 0) * 0.2
                 + per_d.get('MED', 0) * 0.2
                 + per_d.get('SOC', 0) * 0.15
                 + per_d.get('LAW', 0) * 0.075
                 + per_d.get('STEM', 0) * 0.075)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  → Best! score={score:.2f}")

    # 최종 평가
    model.load_state_dict(torch.load(BEST_PATH))
    acc, per_d = evaluate(model, tokenizer, gt)
    avg_domain = np.mean([per_d.get(d, 0) for d in LABELS_ALL])
    print(f"\n{'='*45}")
    print(f"최종 Best 성능:")
    print(f"  Total={acc:.1f}%  도메인평균={avg_domain:.1f}%")
    for d in LABELS_ALL:
        print(f"  {d}: {per_d.get(d,0):.1f}%")
    print(f"\n저장: {BEST_PATH}")