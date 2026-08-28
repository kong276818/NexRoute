"""
LAW LoRA 어댑터 v4 재훈련 스크립트

문제: 기존 LAW 어댑터는 MCQ 포맷("The correct holding is option N: ... All holdings:")으로
학습되어 evaluation 쿼리("What is the holding in...")에서 citation-format degeneration 발생.

수정: law_rewritten_v3.json (3,653개, 서술형 단락 출력) + 평가 쿼리 스타일 instruction으로 재훈련.
저장: lora_weights_7b_v3/lora_law_v4
"""
import json, os, random, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import gc

random.seed(42)

BASE_MODEL = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_FILE  = "/home/sclab/moe_model/processed_v2/law_rewritten_v3.json"
OUT_DIR    = "/home/sclab/moe_model/lora_weights_7b_v3/lora_law_v4"
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM = (
    "You are a legal expert. Answer accurately based on legal principles, "
    "statutes, and case law. Provide detailed analytical explanations in "
    "flowing paragraph form. Do NOT use multiple choice format. "
    "Do NOT begin with 'The correct holding is option'."
)

# 평가 쿼리 스타일 instruction prefix (distribution shift 보완)
QUESTION_PREFIXES = [
    "What is the legal holding regarding",
    "How do courts determine",
    "Why might a court rule that",
    "Compare the legal implications of",
    "Explain the legal principle in",
    "What legal doctrine applies to",
    "Analyze the following legal context:",
]

def load_law_data(path, min_words=15, max_words=350):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for d in data:
        inst = d.get("instruction", "").strip()
        out  = d.get("output", "").strip()
        if not inst or not out:
            continue
        # MCQ 잔재 제거
        if "The correct holding" in out or "All holdings" in out:
            continue
        if "The correct answer" in out or "All answers:" in out:
            continue
        words = out.split()
        if len(words) < min_words:
            continue
        if len(words) > max_words:
            out = " ".join(words[:max_words])

        # fragment-style instruction → question-style로 래핑
        inst_stripped = inst.strip()
        if not inst_stripped[0].isupper() or inst_stripped[:3].lower() in ["fro", "whe", "the", "and", "or ", "but", "in ", "of ", "as "]:
            prefix = random.choice(QUESTION_PREFIXES)
            inst = f"{prefix} the following context:\n\n{inst_stripped}"

        filtered.append({"instruction": inst, "output": out})

    print(f"LAW v4: {len(data)}개 → 필터 후 {len(filtered)}개")
    return filtered


def make_prompt(inst, out):
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{inst}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )


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

data = load_law_data(DATA_FILE)
random.shuffle(data)
prompts = [make_prompt(d["instruction"], d["output"]) for d in data]
dataset = Dataset.from_dict({"text": prompts})
split   = dataset.train_test_split(test_size=0.05, seed=42)
print(f"학습: {len(split['train'])}개 / 검증: {len(split['test'])}개")

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(model, lora_cfg)
peft_model.print_trainable_parameters()

args = SFTConfig(
    output_dir=OUT_DIR,
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
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataset_text_field="text",
    max_length=1024,
    packing=False,
)

trainer = SFTTrainer(
    model=peft_model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    processing_class=tokenizer,
)

print("\n[*] LAW v4 훈련 시작...")
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# train_config 저장
with open(os.path.join(OUT_DIR, "train_config.json"), "w") as f:
    json.dump({
        "data_file": DATA_FILE,
        "num_samples": len(data),
        "num_train": len(split["train"]),
        "num_eval": len(split["test"]),
        "note": "Retrained to fix MCQ degeneration. Uses law_rewritten_v3 with paragraph-style outputs.",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "lr": 2e-4,
    }, f, indent=2)

print(f"\n[+] 저장 완료: {OUT_DIR}")
torch.cuda.empty_cache()
gc.collect()
