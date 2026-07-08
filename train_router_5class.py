import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────
BASE        = "/home/sclab/moe_model/processed_v2/"
SAVE_PATH   = "/home/sclab/moe_model/router_5class.pt"
DOMAINS     = ["LAW", "MED", "SOC", "STEM", "CROSS"]  # ★ 5클래스
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 5e-6
MAX_LEN     = 256
PATIENCE    = 4

# ── 1. Dataset ───────────────────────────────────
class RouterDataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        domain = item["domain"]
        label  = DOMAINS.index(domain)  # ★ 5클래스 인덱스
        enc    = self.tokenizer(
            item["instruction"],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(label, dtype=torch.long)
        }

# ── 2. 모델 (5클래스) ─────────────────────────────
class DomainRouter5(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)   # ★ 5클래스
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

# ── 3. 평가 ──────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    # 도메인별 정확도
    domain_correct = {d: 0 for d in DOMAINS}
    domain_total   = {d: 0 for d in DOMAINS}

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for pred, label in zip(preds.cpu(), labels.cpu()):
                d = DOMAINS[label.item()]
                domain_total[d]   += 1
                if pred.item() == label.item():
                    domain_correct[d] += 1

    acc = correct / total
    domain_acc = {
        d: domain_correct[d] / domain_total[d] * 100
        if domain_total[d] > 0 else 0
        for d in DOMAINS
    }
    return total_loss / len(loader), acc, domain_acc

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

    model = DomainRouter5().to(device)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"멀티 GPU: {num_gpus}개")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()   # ★ 5클래스 CrossEntropy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_val_acc   = 0.0
    patience_count = 0

    print("=" * 60)
    print(f"5클래스 라우터 학습 시작 | LR: {LR}")
    print(f"클래스: {DOMAINS}")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

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
        val_loss, val_acc, domain_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"\nEpoch {epoch+1:02d}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        for d in DOMAINS:
            print(f"    {d:<6}: {domain_acc[d]:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            state = (model.module.state_dict()
                     if isinstance(model, nn.DataParallel)
                     else model.state_dict())
            torch.save(state, SAVE_PATH)
            print(f"  ✅ Best 저장 (Val Acc: {val_acc:.4f})")
        else:
            patience_count += 1
            print(f"  ⏳ 개선 없음 ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"\n🛑 Early Stopping (Epoch {epoch+1})")
                break

    print(f"\n학습 완료 | Best Val Acc: {best_val_acc:.4f}")
    print(f"저장: {SAVE_PATH}")

if __name__ == "__main__":
    train()