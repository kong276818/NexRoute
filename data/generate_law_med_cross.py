"""
LAW×MED 크로스 도메인 QA 생성
cross_rewritten.json 과 cross.json 에 추가할 LAW×MED 아이템 생성
func_type: WHAT / WHY / HOW / COMPARE 각 25개 = 총 100개 목표
"""
import json, os, re, sys, torch, random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

random.seed(42)

BASE_7B  = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR = "/home/sclab/moe_model/lora_weights_7b_v3"
OUT_REWRITTEN = "/home/sclab/moe_model/processed_v2/cross_rewritten.json"
OUT_CROSS     = "/home/sclab/moe_model/processed_v2/cross.json"
OUT_NEW       = "/home/sclab/moe_model/law_med_cross_generated.json"

# ── 원본 데이터 로드 ──────────────────────────────────────────────────────
with open("/home/sclab/moe_model/processed_v2/law.json") as f:
    law_data = json.load(f)
with open("/home/sclab/moe_model/processed_v2/med.json") as f:
    med_data = json.load(f)

# 의료-법률 겹치는 항목 필터
med_kw  = ['medical','patient','physician','hospital','drug','treatment','health',
           'clinical','malpractice','informed consent','pharmaceutical','mental',
           'surgery','diagnosis','negligence','injury']
law_kw  = ['medical','patient','physician','hospital','malpractice','health',
           'pharmaceutical','drug','clinical','treatment','consent','injury']

law_pool = [d for d in law_data
            if any(k in (d['instruction']+d.get('context','')).lower() for k in law_kw)
            and 'The correct holding is option' not in d.get('output','')]
med_pool = [d for d in med_data
            if any(k in (d['instruction']+d.get('context','')).lower() for k in med_kw)
            and 'The correct answer is' not in d.get('output','')]

print(f"LAW pool: {len(law_pool)}, MED pool: {len(med_pool)}")
random.shuffle(law_pool)
random.shuffle(med_pool)

# ── 모델 로드 ─────────────────────────────────────────────────────────────
print("\n모델 로드 중...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="cuda:0",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager")
model = PeftModel.from_pretrained(model, os.path.join(LORA_DIR,"lora_cross"), adapter_name="CROSS")
model.set_adapter("CROSS")
model.eval()
print("모델 로드 완료")

def generate(prompt, max_new_tokens=400):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=0.8, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()

# ── 프롬프트 템플릿 ───────────────────────────────────────────────────────
FUNC_PROMPTS = {
    "WHAT": (
        "You are an expert in both medical science and law.\n\n"
        "Using the following two sources, create ONE cross-domain question "
        "that requires integrated knowledge of BOTH law AND medicine to answer. "
        "The question should ask 'what' (definition, concept, or finding).\n\n"
        "LAW context:\n{law_ctx}\n\n"
        "MED context:\n{med_ctx}\n\n"
        "First write the question on a line starting with 'QUESTION:', "
        "then write a comprehensive integrated answer on a line starting with 'ANSWER:'. "
        "The answer must draw on both legal and medical knowledge. "
        "Do NOT use multiple choice format. Write in flowing paragraph form."
    ),
    "WHY": (
        "You are an expert in both medical science and law.\n\n"
        "Using the following two sources, create ONE cross-domain question "
        "that requires integrated knowledge of BOTH law AND medicine to answer. "
        "The question should ask 'why' (reason, cause, or justification).\n\n"
        "LAW context:\n{law_ctx}\n\n"
        "MED context:\n{med_ctx}\n\n"
        "First write the question on a line starting with 'QUESTION:', "
        "then write a comprehensive integrated answer on a line starting with 'ANSWER:'. "
        "The answer must draw on both legal and medical knowledge. "
        "Do NOT use multiple choice format. Write in flowing paragraph form."
    ),
    "HOW": (
        "You are an expert in both medical science and law.\n\n"
        "Using the following two sources, create ONE cross-domain question "
        "that requires integrated knowledge of BOTH law AND medicine to answer. "
        "The question should ask 'how' (mechanism, process, or procedure).\n\n"
        "LAW context:\n{law_ctx}\n\n"
        "MED context:\n{med_ctx}\n\n"
        "First write the question on a line starting with 'QUESTION:', "
        "then write a comprehensive integrated answer on a line starting with 'ANSWER:'. "
        "The answer must draw on both legal and medical knowledge. "
        "Do NOT use multiple choice format. Write in flowing paragraph form."
    ),
    "COMPARE": (
        "You are an expert in both medical science and law.\n\n"
        "Using the following two sources, create ONE cross-domain question "
        "that requires integrated knowledge of BOTH law AND medicine to answer. "
        "The question should compare or contrast two legal/medical concepts, standards, or approaches.\n\n"
        "LAW context:\n{law_ctx}\n\n"
        "MED context:\n{med_ctx}\n\n"
        "First write the question on a line starting with 'QUESTION:', "
        "then write a comprehensive integrated answer on a line starting with 'ANSWER:'. "
        "The answer must draw on both legal and medical knowledge. "
        "Do NOT use multiple choice format. Write in flowing paragraph form."
    ),
}

def parse_qa(text):
    q, a = "", ""
    lines = text.split("\n")
    mode = None
    for line in lines:
        if line.startswith("QUESTION:"):
            q = line.replace("QUESTION:", "").strip()
            mode = "q"
        elif line.startswith("ANSWER:"):
            a = line.replace("ANSWER:", "").strip()
            mode = "a"
        elif mode == "q" and line.strip():
            q += " " + line.strip()
        elif mode == "a" and line.strip():
            a += " " + line.strip()
    return q.strip(), a.strip()

# ── 생성 루프 ─────────────────────────────────────────────────────────────
results = []
N_PER_FUNC = 25
TARGET_TOTAL = N_PER_FUNC * 4

pair_idx = 0
func_types = ["WHAT", "WHY", "HOW", "COMPARE"]
func_counts = {f: 0 for f in func_types}

print(f"\nLAW×MED 크로스 QA 생성 시작 (목표: {TARGET_TOTAL}개)...")

while len(results) < TARGET_TOTAL and pair_idx < min(len(law_pool), len(med_pool)):
    law_item = law_pool[pair_idx % len(law_pool)]
    med_item = med_pool[pair_idx % len(med_pool)]
    pair_idx += 1

    # 가장 적게 생성된 func_type 선택
    func = min(func_counts, key=func_counts.get)
    if func_counts[func] >= N_PER_FUNC:
        remaining = [f for f in func_types if func_counts[f] < N_PER_FUNC]
        if not remaining:
            break
        func = remaining[0]

    law_ctx = (law_item['instruction'] + " " + law_item.get('context', ''))[:400]
    med_ctx = (med_item['instruction'] + " " + med_item.get('context', ''))[:400]

    prompt = FUNC_PROMPTS[func].format(law_ctx=law_ctx, med_ctx=med_ctx)

    try:
        raw = generate(prompt, max_new_tokens=500)
        question, answer = parse_qa(raw)

        if len(question) < 20 or len(answer) < 50:
            print(f"  [{len(results)+1}] 파싱 실패 (Q={len(question)}, A={len(answer)}) — 스킵")
            continue

        item = {
            "domain": "CROSS",
            "crossed_domains": ["LAW", "MED"],
            "func_type": func,
            "instruction": question,
            "output": answer,
        }
        results.append(item)
        func_counts[func] += 1

        print(f"  [{len(results):>3}/{TARGET_TOTAL}] {func}  Q: {question[:70]}...")
    except Exception as e:
        print(f"  [{pair_idx}] 오류: {e}")
        continue

print(f"\n생성 완료: {len(results)}개")
print("func_type 분포:", func_counts)

# ── 저장 ─────────────────────────────────────────────────────────────────
with open(OUT_NEW, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"신규 생성 저장: {OUT_NEW}")

# cross_rewritten.json 에 추가
with open(OUT_REWRITTEN) as f:
    existing = json.load(f)
# func_type 필드 정리 (cross_rewritten은 func_type 없는 경우도 있음)
for item in results:
    new_item = {k: v for k, v in item.items()}
    existing.append(new_item)
with open(OUT_REWRITTEN, "w") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
print(f"cross_rewritten.json 업데이트: {len(existing)}개 (기존+{len(results)})")

# cross.json 에도 추가 (cross_pair 필드 포함)
with open(OUT_CROSS) as f:
    existing_cross = json.load(f)
for item in results:
    new_item = {
        "domain": "CROSS",
        "cross_pair": "LAW+MED",
        "func_type": item["func_type"],
        "instruction": item["instruction"],
        "context": "",
        "output": item["output"],
    }
    existing_cross.append(new_item)
with open(OUT_CROSS, "w") as f:
    json.dump(existing_cross, f, indent=2, ensure_ascii=False)
print(f"cross.json 업데이트: {len(existing_cross)}개 (기존+{len(results)})")
