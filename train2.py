import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DATA_DIR   = "/home/sclab/moe_model/processed_func"
OUTPUT_DIR = "/home/sclab/moe_model/lora_weights"

FUNC_INSTRUCTIONS = {
    "WHAT":    "Represent the definition query: ",
    "WHY":     "Represent the causal query: ",
    "HOW":     "Represent the procedural query: ",
    "WHAT-IF": "Represent the hypothetical query: ",
}

def make_prompt(item, func_type):
    prefix = FUNC_INSTRUCTIONS[func_type]
    ctx = f"\nContext: {item['context']}" if item.get("context") else ""
    return (
        f"### Function: {func_type}\n"
        f"### Question: {prefix}{item['instruction']}"
        f"{ctx}\n"
        f"### Answer: {item['output']}"
    )

def load_data(func_type):
    filename = f"func_{func_type.lower().replace('-', '_')}.json"
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["text"] = make_prompt(item, func_type)
    split = int(len(data) * 0.8)
    train_data = Dataset.from_list(data[:split])
    eval_data  = Dataset.from_list(data[split:])
    print(f"  Train: {len(train_data)} / Eval: {len(eval_data)}")
    return train_data, eval_data

def train(func_type):
    print(f"Functional LoRA Training: {func_type}")
    save_name = f"lora_func_{func_type.lower().replace('-', '_')}"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    os.makedirs(save_path, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print("[1/4] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[2/4] Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager"
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    print("[3/4] Applying LoRA")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[4/4] Loading dataset")
    train_data, eval_data = load_data(func_type)

    print("[5/5] Starting training")
    sft_config = SFTConfig(
        output_dir=save_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        bf16=True,
        fp16=False,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
        dataset_text_field="text",
        max_length=256
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer
    )
    trainer.train()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Done! Saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, required=True,
                        choices=["WHAT", "WHY", "HOW", "WHAT-IF"])
    args = parser.parse_args()
    train(args.func)