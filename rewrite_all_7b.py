import json, re, os, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR = "/home/sclab/moe_model/processed_v2"

DOMAIN_FILES = {
    "LAW":   ("law.json",            "law_rewritten.json"),
    "MED":   ("med.json",            "med_rewritten.json"),
    "SOC":   ("soc.json",            "soc_rewritten.json"),
    "STEM":  ("stem.json",           "stem_rewritten.json"),
    "CROSS": ("cross_rewritten.json","cross_rewritten.json"),
}

SYSTEM = {
    "LAW":   "You are a legal expert. Provide accurate explanations based on legal principles and case law.",
    "MED":   "You are a medical expert. Provide accurate explanations based on medical facts.",
    "SOC":   "You are a social science expert. Provide explanations based on social science research.",
    "STEM":  "You are a science expert. Provide accurate scientific explanations.",
    "CROSS": "You are a multidisciplinary expert. Provide comprehensive explanations.",
}

BAD_PATTERNS = [
    "The correct holding", "All holdings:",
    "The correct answer", "All answers:",
    "Choices:", "Options:", "(A)", "(B)", "Subject:",
    "\nA:", "\nB:", "\nC:", "\nD:",
]

def is_bad(text):
    return any(p in text for p in BAD_PATTERNS) or len(text.split()) < 15

def parse_answer(text):
    # holding 패턴 (LAW)
    m = re.search(r"The correct holding is option \d+:\s*(.+?)(?:\nAll|\Z)", text, re.DOTALL)
    if m: return m.group(1).strip()
    # MCQ 패턴
    m = re.search(r"The correct answer is [A-D][:\s]+(.+?)(?:\n|$)", text)
    if m: return m.group(1).strip()
    # 보기에서 정답 추출
    m = re.search(r"correct[^\n]*?:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m: return m.group(1).strip()
    return None

def rewrite(q, answer, domain, model, tokenizer):
    sys = SYSTEM[domain]
    prompt = (
        f"{sys}\n\n"
        f"Question: {q}\n\n"
        f"The correct answer is: {answer}\n\n"
        f"Write a comprehensive explanatory paragraph (3-5 sentences) that clearly "
        f"explains this answer with key reasoning and implications. "
        f"Write in flowing paragraph form only. No bullet points, no choices.\n\nAnswer:"
    )
    msgs   = [{"role":"user","content":prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # 후처리
    for kw in BAD_PATTERNS + ["\n0:", "\n1:", "\n2:", "\n3:"]:
        if kw in result:
            result = result[:result.index(kw)].strip()
    return result if len(result.split()) >= 15 else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="ALL")
    args = parser.parse_args()

    # 7B 로드
    print("[*] Qwen2.5-7B 로드...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
            device_map="auto", trust_remote_code=True, local_files_only=True,
            attn_implementation="eager")
    print("[*] 로드 완료!")

    domains = list(DOMAIN_FILES.keys()) if args.domain == "ALL" else [args.domain.upper()]

    for domain in domains:
        in_file, out_file = DOMAIN_FILES[domain]
        in_path  = os.path.join(DATA_DIR, in_file)
        out_path = os.path.join(DATA_DIR, out_file)

        # CROSS는 이미 좋음 → 스킵
        if domain == "CROSS":
            print(f"[Skip] CROSS: 이미 정상 데이터")
            continue

        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n{'='*50}\n{domain}: {len(data)}개\n{'='*50}")
        results, rw, keep, skip = [], 0, 0, 0

        for i, d in enumerate(data):
            q   = d.get("instruction","").strip()
            out = d.get("output", d.get("gold_answer","")).strip()
            if not q or not out: skip+=1; continue

            if is_bad(out):
                answer = parse_answer(out)
                if not answer: skip+=1; continue
                new_out = rewrite(q, answer, domain, model, tokenizer)
                if new_out:
                    results.append({"domain":domain,"instruction":q,"output":new_out,"rewritten":True})
                    rw+=1
                else:
                    skip+=1
            else:
                results.append({"domain":domain,"instruction":q,"output":out,"rewritten":False})
                keep+=1

            if (i+1) % 200 == 0:
                with open(out_path,"w") as f: json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"[{i+1}/{len(data)}] 변환:{rw} 유지:{keep} 스킵:{skip}")

        with open(out_path,"w") as f: json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n{domain} 완료: 변환:{rw} 유지:{keep} 스킵:{skip} 최종:{len(results)}개")

    print("\n[+] 전체 리라이팅 완료!")
