"""
train_lora.py v2 — DualMoE 도메인 LoRA 학습

데이터: /home/sclab/moe_model/processed_v2/
저장:   /home/sclab/moe_model/lora_weights_v2/

GPU: 4090 24GB + 3090 24GB x2
모델: Qwen2.5-72B-Instruct (4bit QLoRA)
"""

import os
import gc
import json
import random
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
OUTPUT_DIR = "/home/sclab/moe_model/lora_weights_v2"

MAX_SAMPLES = {
    "MED":   3772,
    "LAW":   5000,
    "STEM":  3456,
    "SOC":   3399,
    "CROSS": 1321,   # cross_rewritten.json 실제 샘플 수
}

# ★ CROSS는 cross_rewritten.json 사용
DATA_FILES = {
    "MED":   "med.json",
    "LAW":   "law.json",
    "STEM":  "stem.json",
    "SOC":   "soc.json",
    "CROSS": "cross_rewritten.json",   # ← 변경
}

DOMAIN_SYSTEM = {
    "MED":   "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "LAW":   "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "STEM":  "You are a science and engineering expert. Answer accurately based on scientific principles.",
    "SOC":   "You are a social science expert. Answer based on social context, psychology, and research.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains to answer accurately.",
}

# ────────────────────────────────────────────
# GPU 메모리 해제
# ────────────────────────────────────────────
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {mem:.2f} GB 잔여")

# ────────────────────────────────────────────
# 프롬프트 생성
# ────────────────────────────────────────────
def make_prompt(item: dict, tokenizer) -> str:
    domain     = item.get("domain", "CROSS")
    system_msg = DOMAIN_SYSTEM.get(domain, DOMAIN_SYSTEM["CROSS"])

    # ★ CROSS 도메인: crossed_domains 필드 사용 (cross_rewritten.json 구조)
    if domain == "CROSS":
        crossed = item.get("crossed_domains", [])
        if crossed:
            cross_pair = " + ".join(crossed)
            system_msg += f" This question spans: {cross_pair}."

    ctx = f"\n\n[Reference]\n{item['context']}" if item.get("context", "").strip() else ""
    user_content = f"{item['instruction']}{ctx}"

    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": item["output"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

# ────────────────────────────────────────────
# 품질 필터 + 샘플링
# ────────────────────────────────────────────
def filter_and_sample(data: list, domain: str) -> list:
    filtered = []
    for item in data:
        out  = item.get("output", "")
        inst = item.get("instruction", "")
        if len(out.split()) < 10:
            continue
        if len(inst.strip()) < 20:
            continue
        if out.strip() in ["Yes", "No", "True", "False"]:
            continue
        # CROSS 전용: crossed_domains 없으면 스킵
        if domain == "CROSS" and not item.get("crossed_domains"):
            continue
        filtered.append(item)

    print(f"  품질 필터: {len(data)} → {len(filtered)}개")

    max_n = MAX_SAMPLES.get(domain, 4000)
    if len(filtered) > max_n:
        random.seed(42)
        random.shuffle(filtered)
        filtered = filtered[:max_n]
        print(f"  샘플 제한: {max_n}개")

    return filtered

# ────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────
def load_data(domain: str, tokenizer):
    # ★ 도메인별 파일명 분기
    fname = DATA_FILES.get(domain, f"{domain.lower()}.json")
    path  = os.path.join(DATA_DIR, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")

    print(f"  데이터 파일: {fname}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = filter_and_sample(data, domain)

    for item in data:
        item["text"] = make_prompt(item, tokenizer)

    split      = int(len(data) * 0.9)
    train_data = Dataset.from_list(data[:split])
    eval_data  = Dataset.from_list(data[split:])

    steps   = (len(train_data) // 16) * 2
    est_hrs = steps * 40 / 3600
    print(f"  Train: {len(train_data)} / Eval: {len(eval_data)}")
    print(f"  예상 스텝: {steps} / 예상 시간: {est_hrs:.1f}시간")
    return train_data, eval_data

# ────────────────────────────────────────────
# 학습
# ────────────────────────────────────────────
def train(domain: str):
    print(f"\n{'='*60}")
    print(f"  Domain LoRA Training: {domain}")
    print(f"  Data: {DATA_FILES.get(domain, domain.lower() + '.json')}")
    print(f"{'='*60}")

    save_path = os.path.join(OUTPUT_DIR, f"lora_{domain.lower()}")
    os.makedirs(save_path, exist_ok=True)

    print("\n  GPU 메모리 정리...")
    clear_gpu()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[2/5] Loading model (4bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    print("[3/5] Applying LoRA (r=8)...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[4/5] Loading dataset...")
    train_data, eval_data = load_data(domain, tokenizer)

    print("[5/5] Training...")
    sft_config = SFTConfig(
        output_dir=save_path,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        max_length=384,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"\n  저장 중: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    config = {
        "domain":         domain,
        "base_model":     BASE_MODEL,
        "data_file":      DATA_FILES.get(domain, f"{domain.lower()}.json"),
        "lora_r":         8,
        "lora_alpha":     16,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
        "max_length":     384,
        "max_samples":    MAX_SAMPLES.get(domain, 4000),
    }
    with open(os.path.join(save_path, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("  메모리 해제...")
    clear_gpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  ✅ [{domain}] 완료 → {save_path}")

# ────────────────────────────────────────────
# 전체 순차 학습
# ────────────────────────────────────────────
def train_all():
    domains = ["MED", "LAW", "STEM", "SOC", "CROSS"]
    print(f"순차 학습: {domains}")
    print(f"예상 총 시간: 20~25시간\n")

    for domain in domains:
        try:
            train(domain)
        except Exception as e:
            print(f"\n⚠ [{domain}] 실패: {e}")
            clear_gpu()
            continue

    print("\n" + "="*60)
    print("✅ 전체 학습 완료!")
    print(f"저장: {OUTPUT_DIR}")
    print("="*60)

# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
        choices=["MED", "LAW", "STEM", "SOC", "CROSS", "ALL"],
        required=True,
    )
    args = parser.parse_args()

    if args.domain == "ALL":
        train_all()
    else:
        train(args.domain)