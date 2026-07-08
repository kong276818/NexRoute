import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL   = "Qwen/Qwen2.5-72B-Instruct"
QUERYSET     = "/home/sclab/moe_model/queryset/queryset.json"
OUTPUT_PATH  = "/home/sclab/moe_model/queryset/groundtruth.json"

def load_model():
    print("[*] Loading Qwen 72B...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    print("[*] Model loaded!")
    return model, tok

def generate_answer(model, tok, query):
    prompt = f"Answer the following question accurately and concisely in 3-5 sentences.\n\nQuestion: {query}\nAnswer:"
    msg = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True, pad_token_id=tok.eos_token_id)
    return tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def main():
    with open(QUERYSET, "r") as f:
        queries = json.load(f)

    model, tok = load_model()
    results = []

    for i, item in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {item['query'][:60]}...")
        answer = generate_answer(model, tok, item["query"])
        results.append({
            "query": item["query"],
            "domain_type": item["domain_type"],
            "domains": item["domains"],
            "ground_truth": answer
        })

        # 10개마다 중간 저장
        if (i+1) % 10 == 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  중간 저장: {i+1}개")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n완료! {len(results)}개 저장: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
