import json, os, random, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import gc
random.seed(42)

BASE_MODEL = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
OUT_DIR    = "/home/sclab/moe_model/lora_weights_7b"
os.makedirs(OUT_DIR, exist_ok=True)

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

def load_domain(domain, path, min_words=15, max_words=400):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    filtered = []
    for d in data:
        inst = d.get("instruction","").strip()
        out  = d.get("output", d.get("gold_answer","")).strip()
        if not inst or not out: continue
        words = out.split()
        if len(words) < min_words: continue
        if len(words) > max_words:
            out = " ".join(words[:max_words])
        filtered.append({"instruction": inst, "output": out})
    print(f"  {domain}: {len(data)}개 → 필터 후 {len(filtered)}개")
    return filtered

def oversample(data, target):
    if len(data) >= target: return data
    result = data.copy()
    while len(result) < target:
        result.extend(random.choices(data, k=min(len(data), target-len(result))))
    return result[:target]

def make_prompt(domain, inst, out):
    return (
        f"<|im_start|>system\n{SYSTEM[domain]}<|im_end|>\n"
        f"<|im_start|>user\n{inst}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )

# ── 모델 로드 ─────────────────────────────────────
print("[*] Qwen2.5-7B 로드...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.config.use_cache = False
model.enable_input_require_grads()
print("[*] 로드 완료!")

# ── 도메인별 순차 학습 ────────────────────────────
for domain, path in DATA_FILES.items():
    print(f"\n{'='*55}")
    print(f"학습: {domain}")
    print(f"{'='*55}")

    data = load_domain(domain, path)
    if not data:
        print(f"[Skip] 데이터 없음")
        continue

    # SOC 오버샘플링
    if domain == "SOC" and len(data) < 2500:
        data = oversample(data, 2500)
        print(f"  SOC 오버샘플링 → {len(data)}개")

    random.shuffle(data)
    prompts = [make_prompt(domain, d["instruction"], d["output"]) for d in data]
    dataset = Dataset.from_dict({"text": prompts})
    split   = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"  학습: {len(split['train'])}개 / 검증: {len(split['test'])}개")

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()

    out_path = os.path.join(OUT_DIR, f"lora_{domain.lower()}")
    args = TrainingArguments(
        output_dir=out_path,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
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
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False,
    )

    trainer.train()
    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"  [저장] {out_path}")

    # 다음 도메인 위해 어댑터 초기화
    peft_model.disable_adapter_layers()
    for n, p in peft_model.named_parameters():
        if "lora" in n:
            p.data = torch.zeros_like(p.data)
    torch.cuda.empty_cache()
    gc.collect()

print("\n[+] 전체 학습 완료!")
print(f"저장 위치: {OUT_DIR}")
for domain in DATA_FILES.keys():
    lname = f"lora_{domain.lower()}"
    path  = os.path.join(OUT_DIR, lname)
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"  {exists} {lname}")