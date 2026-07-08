import os
import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR   = "/home/sclab/moe_model/processed_func"
OUTPUT_DIR = "/home/sclab/moe_model/func_classifier"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL2ID = {"WHAT": 0, "WHY": 1, "HOW": 2, "WHAT-IF": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_NAME = "bert-base-uncased"

def load_data():
    all_data = []
    for domain in ["med", "stem", "soc", "law"]:
        path = os.path.join(DATA_DIR, f"{domain}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item.get("func_type") and item.get("instruction"):
                all_data.append({"text": item["instruction"], "label": LABEL2ID.get(item["func_type"], 0)})
    print(f"총 데이터: {len(all_data)}개")
    from collections import Counter
    dist = Counter(d["label"] for d in all_data)
    print(f"분포: { {ID2LABEL[k]: v for k, v in dist.items()} }")
    split = int(len(all_data) * 0.8)
    return all_data[:split], all_data[split:]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}

def train():
    print("[1/3] Loading data")
    train_data, eval_data = load_data()
    train_ds = Dataset.from_list(train_data)
    eval_ds  = Dataset.from_list(eval_data)

    print("[2/3] Loading model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds  = eval_ds.map(tokenize, batched=True)

    print("[3/3] Training")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        logging_steps=50,
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()