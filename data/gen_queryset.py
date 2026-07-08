import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
OUTPUT_DIR = "/home/sclab/moe_model/queryset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SINGLE_PROMPTS = {
    "MED": "Generate {n} diverse medical/health questions in English. Include WHAT, WHY, HOW, WHAT-IF types. Topics: diseases, treatments, medications, anatomy. Return ONLY a JSON array of strings. Generate exactly {n} questions:",
    "LAW": "Generate {n} diverse legal questions in English. Include WHAT, WHY, HOW, WHAT-IF types. Topics: contracts, criminal law, civil rights, liability. Return ONLY a JSON array of strings. Generate exactly {n} questions:",
    "STEM": "Generate {n} diverse science and technology questions in English. Include WHAT, WHY, HOW, WHAT-IF types. Topics: biology, chemistry, physics, computer science. Return ONLY a JSON array of strings. Generate exactly {n} questions:",
    "SOC": "Generate {n} diverse social science questions in English. Include WHAT, WHY, HOW, WHAT-IF types. Topics: psychology, sociology, economics, culture. Return ONLY a JSON array of strings. Generate exactly {n} questions:",
}

CROSS_PROMPTS = {
    "MED+LAW": "Generate {n} questions requiring BOTH medical AND legal knowledge in English. Topics: malpractice, patient rights, drug regulations, bioethics. Return ONLY a JSON array of strings. Generate exactly {n} questions:",
    "STEM+SOC": "Generate {n} questions requiring BOTH science AND social science knowledge in English. Topics: climate change, AI ethics, technology and inequality. Return ONLY a JSON array of strings. Generate exactly {n} questions:",
    "MED+STEM": "Generate {n} questions requiring BOTH medical AND scientific knowledge in English. Topics: gene therapy, biotechnology, pharmacology, neuroscience. Return ONLY a JSON array of strings. Generate exactly {n} questions:",
}

def load_model():
    print("[*] Loading Qwen 72B...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    print("[*] Model loaded!")
    return model, tok

def generate_questions(model, tok, prompt, n):
    msg = [{"role": "user", "content": prompt.format(n=n)}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.8, do_sample=True, pad_token_id=tok.eos_token_id)
    result = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        s, e = result.find("["), result.rfind("]") + 1
        if s != -1 and e > s:
            qs = json.loads(result[s:e])
            return [q for q in qs if isinstance(q, str)]
    except:
        pass
    return [l.strip().strip('"') for l in result.split("\n") if len(l.strip()) > 10 and "?" in l]

def main():
    model, tok = load_model()
    all_queries = []

    for domain, n in [("MED",25),("LAW",25),("STEM",25),("SOC",25)]:
        print(f"\n[{domain}] {n}개 생성 중...")
        qs = generate_questions(model, tok, SINGLE_PROMPTS[domain], n)[:n]
        print(f"  완료: {len(qs)}개")
        for q in qs:
            all_queries.append({"query": q, "domain_type": "single", "domains": [domain], "func_type": None})

    for pair, n in [("MED+LAW",40),("STEM+SOC",30),("MED+STEM",30)]:
        print(f"\n[{pair}] {n}개 생성 중...")
        qs = generate_questions(model, tok, CROSS_PROMPTS[pair], n)[:n]
        print(f"  완료: {len(qs)}개")
        for q in qs:
            all_queries.append({"query": q, "domain_type": "cross", "domains": pair.split("+"), "func_type": None})

    out = os.path.join(OUTPUT_DIR, "queryset.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)
    print(f"\n총 {len(all_queries)}개 저장: {out}")

if __name__ == "__main__":
    main()
