"""
train_lora_single_v2.py
========================
MED / SOC / STEM 단일 도메인 어댑터 재학습.
개선 포인트:
  - 각 도메인 데이터 + CROSS 데이터 30% mix-in (analytical 스타일 강화)
  - 짧은 출력(<50 words) 및 MCQ 스타일 필터링
  - lora_law_v4와 동일한 설정 (r=16, alpha=32, 3 epochs)
  - 저장: lora_weights_7b_v3/lora_{domain}_v2

실행:
    CUDA_VISIBLE_DEVICES=0 python train/train_lora_single_v2.py
"""
import json, os, random, torch, gc
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

random.seed(42)
torch.manual_seed(42)

BASE_MODEL = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_OUT   = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"

DOMAIN_FILES = {
    "MED":  "med_rewritten.json",
    "SOC":  "soc_rewritten.json",
    "STEM": "stem_rewritten.json",
}
CROSS_FILE = "cross_rewritten.json"
CROSS_MIX_RATIO = 0.30   # 전체 학습 데이터의 30%를 CROSS로 채움

SYSTEM = {
    "MED":  "You are a medical expert. Answer accurately based on medical facts and clinical knowledge. Provide detailed analytical explanations in flowing paragraph form.",
    "SOC":  "You are a social science expert. Answer based on social context, psychology, and sociological research. Provide detailed analytical explanations in flowing paragraph form.",
    "STEM": "You are a science and engineering expert. Answer accurately based on scientific principles and mechanisms. Provide detailed analytical explanations in flowing paragraph form.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains to answer accurately. Provide detailed analytical explanations in flowing paragraph form.",
}

MCQ_MARKERS = [
    "correct answer is", "The answer is", "All of the above", "None of the above",
    "(A)", "(B)", "(C)", "(D)", "\nA.", "\nB.", "\nC.", "\nD.",
    "Option A", "Option B", "Choice A", "Choice B",
]

def is_clean(output):
    if not output or len(output.split()) < 50:
        return False
    for m in MCQ_MARKERS:
        if m in output:
            return False
    return True

def load_domain(domain, path):
    data = json.load(open(path))
    out = []
    for d in data:
        inst = str(d.get("instruction", "")).strip()
        text = str(d.get("output", "")).strip()
        if not inst or not is_clean(text):
            continue
        # 최대 400 words
        words = text.split()
        if len(words) > 400:
            text = " ".join(words[:400])
        out.append({"instruction": inst, "output": text, "domain": domain})
    print(f"  {domain}: {len(data)}개 → 필터 후 {len(out)}개")
    return out

def load_cross():
    data = json.load(open(os.path.join(DATA_DIR, CROSS_FILE)))
    out = []
    for d in data:
        inst = str(d.get("instruction", "")).strip()
        text = str(d.get("output", "")).strip()
        if not inst or not is_clean(text):
            continue
        words = text.split()
        if len(words) > 400:
            text = " ".join(words[:400])
        out.append({"instruction": inst, "output": text, "domain": "CROSS"})
    print(f"  CROSS: {len(data)}개 → 필터 후 {len(out)}개")
    return out

def make_prompt(item):
    domain = item["domain"]
    sys = SYSTEM[domain]
    inst = item["instruction"]
    out = item["output"]
    return (
        f"<|im_start|>system\n{sys}\n\n"
        f"Do NOT use multiple choice format. Do NOT list choices. "
        f"Write in flowing paragraph form only.<|im_end|>\n"
        f"<|im_start|>user\n{inst}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
print("[*] Qwen2.5-7B-Instruct 4-bit 로드...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="cuda:0",
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="eager",
)
model.config.use_cache = False
model.enable_input_require_grads()
print("[*] 로드 완료!")

cross_data = load_cross()

# ── 도메인별 순차 학습 ─────────────────────────────────────────────────────────
for domain, fname in DOMAIN_FILES.items():
    print(f"\n{'='*60}")
    print(f"학습: {domain}")
    print(f"{'='*60}")

    domain_data = load_domain(domain, os.path.join(DATA_DIR, fname))
    if not domain_data:
        print(f"[Skip] 데이터 없음")
        continue

    # CROSS mix-in: domain 데이터의 30%에 해당하는 CROSS 샘플 추가
    n_cross = int(len(domain_data) * CROSS_MIX_RATIO / (1 - CROSS_MIX_RATIO))
    n_cross = min(n_cross, len(cross_data))
    cross_sample = random.sample(cross_data, n_cross)

    combined = domain_data + cross_sample
    random.shuffle(combined)
    print(f"  domain={len(domain_data)}개 + CROSS mix={n_cross}개 → 합계={len(combined)}개")

    prompts = [make_prompt(item) for item in combined]
    dataset = Dataset.from_dict({"text": prompts})
    split   = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"  학습: {len(split['train'])}개 / 검증: {len(split['test'])}개")

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()

    out_path = os.path.join(LORA_OUT, f"lora_{domain.lower()}_v2")
    args = SFTConfig(
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
        save_total_limit=1,
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

    print(f"[*] {domain} 학습 시작...")
    trainer.train()
    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"[+] 저장: {out_path}")

    # 다음 도메인을 위해 LoRA 가중치 초기화
    for n, p in peft_model.named_parameters():
        if "lora_" in n:
            torch.nn.init.zeros_(p.data)
    del peft_model
    torch.cuda.empty_cache()
    gc.collect()

print("\n[+] 전체 학습 완료!")
for domain in DOMAIN_FILES:
    path = os.path.join(LORA_OUT, f"lora_{domain.lower()}_v2")
    ok = "OK" if os.path.exists(os.path.join(path, "adapter_model.safetensors")) else "MISSING"
    print(f"  {domain}: {ok}  → {path}")
