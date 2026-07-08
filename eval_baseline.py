"""
eval_baseline.py — 베이스라인 평가 (LoRA 없이)
Base 모델 (Qwen2.5-72B) 단독 성능 측정
"""
import json, re, random, torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"

def evaluate_base(domain, n=100):
    print(f"[BASE] {domain} 평가 ({n}개)")

    # 모델 로드
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        local_files_only=True, attn_implementation="eager"
    )
    model.config.use_cache = True

    # 데이터 로드
    with open(f"{DATA_DIR}/{domain.lower()}.json") as f:
        data = json.load(f)
    random.seed(42)
    samples = random.sample(data, min(n, len(data)))

    correct = 0
    for i, item in enumerate(samples):
        q = item["instruction"]
        expected = item["output"]

        messages = [
            {"role": "system", "content": "You are a helpful expert assistant."},
            {"role": "user",   "content": q},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256,
                                     temperature=0.1, do_sample=True,
                                     pad_token_id=tokenizer.eos_token_id)

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response  = tokenizer.decode(generated, skip_special_tokens=True)

        # MCQ 정답 추출
        exp_label = re.search(r"correct answer is ([A-D])", expected)
        pred_label = re.search(r"\b([A-D])\b", response[:50])

        if exp_label and pred_label:
            if exp_label.group(1) == pred_label.group(1):
                correct += 1

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{n}] 정확도: {correct/(i+1):.3f}")

    acc = correct / len(samples)
    print(f"\n[BASE {domain}] 최종 정확도: {acc:.4f} ({acc*100:.1f}%)")

    result = {"domain": domain, "model": "base", "accuracy": acc, "n": n}
    with open(f"/home/sclab/moe_model/eval_base_{domain.lower()}.json", "w") as f:
        json.dump(result, f)
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True,
                        choices=["MED","LAW","STEM","SOC","CROSS","ALL"])
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    if args.domain == "ALL":
        scores = {}
        for d in ["MED","LAW","STEM","SOC","CROSS"]:
            scores[d] = evaluate_base(d, args.n)
        print("\n전체 결과:")
        for d, s in scores.items():
            print(f"  {d}: {s:.4f}")
    else:
        evaluate_base(args.domain, args.n)
