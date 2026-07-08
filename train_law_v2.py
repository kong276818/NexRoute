import json, os, random, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

random.seed(42)

BASE_MODEL = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_PATH  = "/home/sclab/moe_model/processed_v2/law_rewritten_v3.json"
OUT_DIR    = "/home/sclab/moe_model/lora_weights_7b_v3/lora_law_v3"
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM = "You are a legal expert. Answer accurately based on legal principles, statutes, and case law."

def make_prompt(inst, out):
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{inst}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )

# ── 데이터 로드 ───────────────────────────────────
print("[*] 데이터 로드...")
raw = json.load(open(DATA_PATH, encoding="utf-8"))
samples = []
for d in raw:
    inst = d.get("instruction", "").strip()
    out  = d.get("output", d.get("gold_answer", "")).strip()
    if not inst or not out:
        continue
    words = out.split()
    if len(words) > 400:
        out = " ".join(words[:400])
    samples.append({"instruction": inst, "output": out})

random.shuffle(samples)
prompts = [make_prompt(d["instruction"], d["output"]) for d in samples]
dataset = Dataset.from_dict({"text": prompts})
split   = dataset.train_test_split(test_size=0.05, seed=42)
print(f"  학습: {len(split['train'])}개 / 검증: {len(split['test'])}개")

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

# ── LoRA 설정 ─────────────────────────────────────
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(model, lora_cfg)
peft_model.print_trainable_parameters()

# ── 학습 ─────────────────────────────────────────
args = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
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
    max_length=1024,
    packing=False,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=peft_model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    processing_class=tokenizer,
)

print("[*] 학습 시작...")
result = trainer.train()
print(f"[*] 학습 완료! 최종 loss: {result.training_loss:.4f}")

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"[*] 저장 완료: {OUT_DIR}")
