import json, os, random, torch, gc, argparse
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
random.seed(42)

BASE_MODEL = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
OUT_DIR    = "/home/sclab/moe_model/lora_weights_7b_v3"

DATA_FILES = {
    "LAW":   "/home/sclab/moe_model/processed_v2/law.json",
    "MED":   "/home/sclab/moe_model/processed_v2/med_rewritten.json",
    "SOC":   "/home/sclab/moe_model/processed_v2/soc_rewritten.json",
    "STEM":  "/home/sclab/moe_model/processed_v2/stem_rewritten.json",
    "CROSS": "/home/sclab/moe_model/processed_v2/cross_rewritten.json",
}

SYSTEM = {
    "LAW":   "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":   "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":   "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM":  "You are a science and engineering expert. Answer accurately based on scientific principles.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains to answer accurately.",
}

BAD_PATTERNS = [
    "Subject:", "The correct answer", "All answers:",
    "Choices:", "Options:", "(A)", "(B)", "(C)", "(D)",
]

MAX_LENGTH = 1024

class DomainDataset(TorchDataset):
    def __init__(self, data, tokenizer, domain):
        self.samples   = data
        self.tokenizer = tokenizer
        self.domain    = domain

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d    = self.samples[idx]
        inst = d["instruction"]
        out  = d["output"]
        sys  = SYSTEM[self.domain]

        prompt = (
            f"<|im_start|>system\n{sys}<|im_end|>\n"
            f"<|im_start|>user\n{inst}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full = prompt + out + "<|im_end|>"

        enc_full   = self.tokenizer(full,   truncation=True, max_length=MAX_LENGTH)
        enc_prompt = self.tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)

        input_ids  = enc_full["input_ids"]
        prompt_len = len(enc_prompt["input_ids"])
        labels     = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids":      torch.tensor(input_ids,   dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(input_ids), dtype=torch.long),
            "labels":         torch.tensor(labels,      dtype=torch.long),
        }

def is_bad(text):
    if not text or len(text.split()) < 50: return True
    for pat in BAD_PATTERNS:
        if pat in text: return True
    return False

def load_domain(domain, path, max_words=400):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    filtered = []
    for d in data:
        inst = d.get("instruction","").strip()
        # context 필드 완전 제거
        out  = d.get("output", d.get("gold_answer","")).strip()
        if not inst or not out: continue
        if is_bad(out): continue
        # 너무 길면 자르기
        words = out.split()
        if len(words) > max_words:
            out = " ".join(words[:max_words])
        filtered.append({"instruction": inst, "output": out})
    print(f"  {domain}: {len(data)}개 -> 필터 후 {len(filtered)}개")
    return filtered

def oversample(data, target):
    if len(data) >= target: return data
    result = data.copy()
    while len(result) < target:
        result.extend(random.choices(data, k=min(len(data), target-len(result))))
    return result[:target]

def train(domain):
    os.makedirs(OUT_DIR, exist_ok=True)
    path     = DATA_FILES[domain]
    out_path = os.path.join(OUT_DIR, f"lora_{domain.lower()}")

    print(f"\n{'='*55}")
    print(f"학습 시작: {domain}")
    print(f"{'='*55}")

    # 데이터 로드
    data = load_domain(domain, path)
    if not data:
        print("[Skip] 데이터 없음"); return

    if domain == "SOC" and len(data) < 2000:
        data = oversample(data, 2000)
        print(f"  SOC 오버샘플링 -> {len(data)}개")

    random.shuffle(data)
    split     = int(len(data) * 0.95)
    train_raw = data[:split]
    eval_raw  = data[split:]
    print(f"  학습: {len(train_raw)}개 / 검증: {len(eval_raw)}개")

    # ★ 매번 새로 모델 로드 (도메인 간 오염 방지)
    print("  [*] 베이스 모델 로드...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    print("  [*] 로드 완료!")

    # LoRA 설정
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()

    train_ds = DomainDataset(train_raw, tokenizer, domain)
    eval_ds  = DomainDataset(eval_raw,  tokenizer, domain)

    args = TrainingArguments(
        output_dir=out_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=peft_model,
        padding=True, pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=peft_model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"  [저장] {out_path}")

    # ★ 완전 언로드 (다음 도메인 위해)
    del trainer, peft_model, model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  [완료] {domain} 학습 끝!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True,
                        help="LAW / MED / SOC / STEM / CROSS")
    args = parser.parse_args()
    train(args.domain.upper())
