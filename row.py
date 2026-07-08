import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────
BASE       = "/home/sclab/moe_model/processed_v2/"
SAVE_PATH  = "/home/sclab/moe_model/router_multilabel.pt"
DOMAINS    = ["LAW", "MED", "SOC", "STEM"]
BATCH_SIZE = 32
EPOCHS     = 20        # 더 길게, Early Stopping이 잡아줌
LR         = 5e-6      # 2e-5 → 5e-6 (안정화)
MAX_LEN    = 256
THRESHOLD  = 0.35
GAP        = 0.25      # 0.15 → 0.25 (CROSS 감지 범위 확대)
PATIENCE   = 4         # Early Stopping patience

# ── 1. Dataset ───────────────────────────────────
class RouterDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc  = self.tokenizer(
            item["instruction"],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(item["label"], dtype=torch.float)
        }

# ── 2. 모델 ──────────────────────────────────────
class DomainRouter(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

# ── 3. 평가 ──────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss     = 0
    exact_correct  = 0
    domain_correct = 0
    total          = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, labels)
            total_loss += loss.item()

            scores = torch.sigmoid(logits)
            preds  = (scores > THRESHOLD).float()

            exact_correct  += (preds == labels).all(dim=1).sum().item()

            pred_domains   = scores.argmax(dim=1)
            label_domains  = labels.argmax(dim=1)
            domain_correct += (pred_domains == label_domains).sum().item()

            total += labels.size(0)

    return (
        total_loss / len(loader),
        exact_correct / total,
        domain_correct / total
    )

# ── 4. 학습 ──────────────────────────────────────
def train():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(
        RouterDataset(BASE + "router_train.json", tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        RouterDataset(BASE + "router_val.json", tokenizer),
        batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
    )

    model = DomainRouter().to(device)

    # ── 멀티 GPU ──────────────────────────────────
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"멀티 GPU 사용: {num_gpus}개 "
              f"({', '.join([torch.cuda.get_device_name(i) for i in range(num_gpus)])})")
        model = nn.DataParallel(model)
    else:
        print(f"단일 GPU: {torch.cuda.get_device_name(0)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc   = 0.0
    patience_count = 0

    print("=" * 60)
    print(f"학습 시작 | LR: {LR} | GAP: {GAP} | PATIENCE: {PATIENCE}")
    print(f"Train: 17856개 | Val: 2232개 | Max Epoch: {EPOCHS}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        val_loss, exact_acc, domain_acc = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1:02d}")
        print(f"  Train Loss:  {train_loss:.4f}")
        print(f"  Val Loss:    {val_loss:.4f}")
        print(f"  Exact Acc:   {exact_acc:.4f}")
        print(f"  Domain Acc:  {domain_acc:.4f}  ← 논문 기재용")

        if domain_acc > best_val_acc:
            best_val_acc   = domain_acc
            patience_count = 0
            state = (model.module.state_dict()
                     if isinstance(model, nn.DataParallel)
                     else model.state_dict())
            torch.save(state, SAVE_PATH)
            print(f"  ✅ Best 저장 (Domain Acc: {domain_acc:.4f})")
        else:
            patience_count += 1
            print(f"  ⏳ 개선 없음 ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"\n🛑 Early Stopping (Epoch {epoch+1})")
                break

    print(f"\n학습 완료 | Best Domain Acc: {best_val_acc:.4f}")
    print(f"저장: {SAVE_PATH}")

# ── 5. 추론 테스트 ────────────────────────────────
def predict(model, tokenizer, instruction, device):
    enc = tokenizer(
        instruction,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        if isinstance(logits, tuple):
            logits = logits[0]
        scores = torch.sigmoid(logits).squeeze().cpu().tolist()

    domain_scores = dict(zip(DOMAINS, scores))
    sorted_d      = sorted(domain_scores.items(), key=lambda x: -x[1])
    top1, s1      = sorted_d[0]
    top2, s2      = sorted_d[1]

    if s2 >= THRESHOLD and (s1 - s2) <= GAP:
        result = f"CROSS [{top1}+{top2}]"
    else:
        result = top1

    print(f"Q:      {instruction[:70]}")
    print(f"결과:   {result}")
    print(f"scores: { {k: f'{v:.3f}' for k, v in domain_scores.items()} }\n")

# ── 6. 메인 ──────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n=== 추론 테스트 ===")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = DomainRouter().to(device)
    model.load_state_dict(torch.load(SAVE_PATH))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    tests = [
        "What is the mechanism of action of aspirin?",
        "How does contract law apply to digital agreements?",
        "How might CRISPR technology affect healthcare equity and policy?",
        "What causes climate change?",
        "In Islamic finance, how do legal frameworks intersect with economic principles?"
    ]
    for q in tests:
        predict(model, tokenizer, q, device)