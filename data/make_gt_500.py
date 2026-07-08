"""
GT 데이터 500개로 확장
각 도메인별 100개 (현재 60개 → 40개 추가)
"""
import json, os, random, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
random.seed(42)

BASE_72B = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
DATA_DIR = "/home/sclab/moe_model/processed_v2"
GT_OLD   = "/home/sclab/moe_model/processed_v2/gt_dataset_rewritten.json"
GT_NEW   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

SYSTEM = {
    "LAW":   "You are a legal expert. Answer accurately based on legal principles and case law.",
    "MED":   "You are a medical expert. Answer accurately based on medical facts.",
    "SOC":   "You are a social science expert. Answer based on social science research.",
    "STEM":  "You are a science expert. Answer accurately based on scientific principles.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains.",
}

TARGET_PER_DOMAIN = 100
ADD_PER_DOMAIN    = 40  # 60 → 100

def generate_answer(q, domain, model, tokenizer):
    sys = SYSTEM[domain]
    prompt = (
        f"{sys}\n\n"
        f"Question: {q}\n\n"
        f"Provide a comprehensive, detailed analytical answer in 4-6 sentences. "
        f"Write in flowing paragraph form only.\n\nAnswer:"
    )
    msgs   = [{"role":"user","content":prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # 후처리
    for kw in ["The correct","Choices:","Options:","\nA:","\nB:","\nC:","\nD:"]:
        if kw in result: result = result[:result.index(kw)].strip()
    return result if len(result.split()) >= 20 else None

# 기존 GT 로드
with open(GT_OLD) as f: gt_old = json.load(f)

# 기존에 사용된 질문 수집 (중복 방지)
used_questions = {d["instruction"] for d in gt_old}
print(f"기존 GT: {len(gt_old)}개")

# 72B 로드
print("\n[*] 72B 로드...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_72B, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_72B, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, local_files_only=True,
        attn_implementation="eager")
print("[*] 로드 완료!")

new_gt = gt_old.copy()

for domain, fname in DOMAIN_FILES.items():
    print(f"\n{'='*45}\n{domain}: {ADD_PER_DOMAIN}개 추가 생성\n{'='*45}")

    # 소스 데이터 로드
    with open(os.path.join(DATA_DIR, fname)) as f:
        src = json.load(f)

    # 기존에 안 쓴 질문만 후보로
    candidates = [d for d in src
                  if d.get("instruction","") not in used_questions
                  and len(d.get("instruction","").split()) >= 5]
    random.shuffle(candidates)
    print(f"  후보 질문: {len(candidates)}개")

    added = 0
    for d in candidates:
        if added >= ADD_PER_DOMAIN: break
        q = d["instruction"].strip()

        answer = generate_answer(q, domain, model, tokenizer)
        if not answer: continue

        new_gt.append({
            "instruction":     q,
            "gold_answer":     answer,
            "expected_domain": domain,
            "source":          "generated_500",
        })
        used_questions.add(q)
        added += 1

        if added % 10 == 0:
            with open(GT_NEW, "w") as f:
                json.dump(new_gt, f, ensure_ascii=False, indent=2)
            print(f"  [{added}/{ADD_PER_DOMAIN}] 저장 완료")

    print(f"  {domain} 추가: {added}개")

with open(GT_NEW, "w") as f:
    json.dump(new_gt, f, ensure_ascii=False, indent=2)

# 최종 분포 확인
from collections import Counter
dist = Counter(d.get("expected_domain","") for d in new_gt)
print(f"\n최종 GT 분포:")
for k, v in sorted(dist.items()):
    print(f"  {k}: {v}개")
print(f"  합계: {len(new_gt)}개")
print(f"\n[+] 저장: {GT_NEW}")
