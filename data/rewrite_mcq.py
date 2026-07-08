"""
MCQ → 서술형 변환 스크립트
"The correct answer is B: Disorganization and despair"
→ "Disorganization and despair is the longest phase of grieving because..."
"""
import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

BASE_MODEL = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
OUT_DIR    = "/home/sclab/moe_model/processed_v2"

DOMAIN_FILES = {
    "LAW":  "law.json",
    "MED":  "med.json",
    "SOC":  "soc.json",
    "STEM": "stem.json",
}

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert. Provide accurate explanations based on legal principles, case law, and statutes.",
    "MED":  "You are a medical expert. Provide accurate, detailed medical explanations.",
    "SOC":  "You are a social science expert. Provide detailed explanations based on social science research.",
    "STEM": "You are a science and engineering expert. Provide accurate scientific explanations.",
}

# ── MCQ 파싱 ─────────────────────────────────────────
def parse_mcq(output_text):
    """
    "The correct answer is B: Disorganization and despair\n\nChoices:\nA:...\nB:...\nC:...\nD:..."
    → {"letter": "B", "answer": "Disorganization and despair", "choices": {...}}
    """
    result = {"letter": None, "answer": None, "choices": {}}

    # 정답 추출
    # holding 패턴 (LAW 전용)
    m = re.search(r"The correct holding is option \d+:\s*(.+?)(?:\n|All)", output_text, re.IGNORECASE|re.DOTALL)
    if m:
        result["answer"] = m.group(1).strip()
        return result
    # holding 패턴 (LAW 전용)
    m = re.search(r"The correct holding is option \d+:\s*(.+?)(?:\n|All)", output_text, re.IGNORECASE|re.DOTALL)
    if m:
        result["answer"] = m.group(1).strip()
        return result
    patterns = [
        r"The correct answer is ([A-D]):\s*(.+?)(?:\n|$)",
        r"The correct answer is ([A-D])[.)]\s*(.+?)(?:\n|$)",
        r"Answer:\s*([A-D]):\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, output_text, re.IGNORECASE)
        if m:
            result["letter"] = m.group(1).strip()
            result["answer"] = m.group(2).strip()
            break

    # 보기 추출
    choice_pattern = r"([A-D])[:.]\s*(.+?)(?=\n[A-D][:.] |\Z)"
    matches = re.findall(choice_pattern, output_text, re.DOTALL)
    for letter, text in matches:
        result["choices"][letter] = text.strip()

    return result

def is_mcq(text):
    mcq_patterns = [
        r"The correct answer is",
        r"All answers:",
        r"^\s*[A-D]\.",
        r"\(A\)", r"\(B\)",
        r"Choices:\n",
        r"Options:\n",
    ]
    for pat in mcq_patterns:
        if re.search(pat, text, re.MULTILINE | re.IGNORECASE):
            return True
    return False

# ── 모델 로드 ────────────────────────────────────────
def load_model():
    print("[*] 72B 모델 로드 중...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    print("[*] 로드 완료!")
    return model, tokenizer

# ── 서술형 변환 ──────────────────────────────────────
def rewrite_mcq(question, parsed, context, domain, model, tokenizer):
    answer_text = parsed["answer"]
    if not answer_text:
        return None

    system = DOMAIN_SYSTEM[domain]

    # 컨텍스트가 있으면 포함
    ctx_part = ""
    if context and context.strip() and context.strip() != "[]":
        ctx_part = f"\nContext: {context.strip()}"

    prompt = (
        f"{system}\n\n"
        f"Question: {question}{ctx_part}\n\n"
        f"The correct answer is: {answer_text}\n\n"
        f"Write a comprehensive explanatory paragraph (3-5 sentences) that:\n"
        f"1. States the answer clearly\n"
        f"2. Explains WHY this is correct with key mechanisms/reasons\n"
        f"3. Provides relevant context or implications\n"
        f"Do NOT use bullet points. Write in flowing paragraph form only.\n\n"
        f"Answer:"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1500
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # 후처리: MCQ 잔여 제거
    result = re.sub(r'The correct answer is[^\n]*\n?', '', result).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in result:
            result = result[:result.index(kw)].strip()

    return result if len(result.split()) >= 15 else None

# ── 메인 ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="ALL")
    parser.add_argument("--max_samples", type=int, default=99999,
                        help="도메인당 최대 변환 수 (테스트용)")
    args = parser.parse_args()

    model, tokenizer = load_model()

    domains = list(DOMAIN_FILES.keys()) if args.domain == "ALL" else [args.domain.upper()]

    for domain in domains:
        fname    = DOMAIN_FILES[domain]
        in_path  = os.path.join(DATA_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname.replace(".json", "_rewritten.json"))

        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n{'='*50}")
        print(f"{domain}: {len(data)}개 처리 시작")
        print(f"{'='*50}")

        results  = []
        rewrite_count = 0
        keep_count    = 0
        skip_count    = 0

        for i, d in enumerate(data[:args.max_samples]):
            question = d.get("instruction", "")
            output   = d.get("output", d.get("gold_answer", ""))
            context  = d.get("context", "")

            if not question or not output:
                skip_count += 1
                continue

            if is_mcq(output):
                # MCQ → 서술형 변환
                parsed = parse_mcq(output)
                if not parsed["answer"]:
                    skip_count += 1
                    continue

                new_output = rewrite_mcq(
                    question, parsed, context, domain, model, tokenizer
                )
                if new_output:
                    results.append({
                        "domain":      domain,
                        "func_type":   d.get("func_type", ""),
                        "instruction": question,
                        "context":     context,
                        "output":      new_output,
                        "rewritten":   True,
                    })
                    rewrite_count += 1
                    if (rewrite_count) % 10 == 0:
                        print(f"  [{i+1}/{len(data)}] 변환 {rewrite_count}개 완료")
                else:
                    skip_count += 1
            else:
                # 이미 서술형 → 그대로 유지
                if len(output.split()) >= 15:
                    results.append({
                        "domain":      domain,
                        "func_type":   d.get("func_type", ""),
                        "instruction": question,
                        "context":     context,
                        "output":      output,
                        "rewritten":   False,
                    })
                    keep_count += 1

            # 중간 저장
            if (i + 1) % 100 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"  [중간저장] {i+1}개 처리 / 변환:{rewrite_count} 유지:{keep_count} 스킵:{skip_count}")

        # 최종 저장
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n{domain} 완료:")
        print(f"  변환(MCQ→서술형): {rewrite_count}개")
        print(f"  유지(이미 서술형): {keep_count}개")
        print(f"  스킵:             {skip_count}개")
        print(f"  최종 데이터:      {len(results)}개")
        print(f"  저장: {out_path}")

        torch.cuda.empty_cache()
        import gc; gc.collect()

    print("\n[+] 전체 변환 완료!")