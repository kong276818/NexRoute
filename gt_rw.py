import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── 1. 설정 ──
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
INPUT_FILE = "/home/sclab/moe_model/processed_v2/gt_dataset.json"
OUTPUT_FILE = "/home/sclab/moe_model/processed_v2/gt_dataset_rewritten.json"

# ── 2. 프롬프트 정의 (CROSS용 / 단일 도메인용 분류) ──
CROSS_PROMPT = """You are an expert dataset creator for training AI routing models.
Rewrite the following multiple-choice question into a complex, open-ended dilemma that explicitly fuses exactly TWO distinct domains from this list: [MED, LAW, STEM, SOC].
(MED = Medical, LAW = Law/Policy, STEM = Science/Tech, SOC = Sociology/Business)

CRITICAL INSTRUCTION:
You MUST create a question that perfectly matches the question type: '{target_func}'.
- If WHAT: Ask about definitions, concepts, or phenomena.
- If WHY: Ask about reasons, causes, or justifications.
- If HOW: Ask about mechanisms, processes, or methods.
- If COMPARE: Ask to evaluate, contrast, or weigh two different aspects.

Do NOT include multiple choices. 

Original Data:
Query/Instruction: {query}
Gold Answer/Choices: {gold_answer}

Respond ONLY with a raw, valid JSON object:
{{
    "id": "{id}",
    "expected_domain": "CROSS",
    "crossed_domains": ["<Domain 1>", "<Domain 2>"],
    "instruction": "<Your {target_func} question>",
    "gold_answer": "<The comprehensive gold answer>",
    "expected_func": "{target_func}"
}}"""

SINGLE_PROMPT = """You are an expert dataset creator for training AI routing models.
Rewrite the following multiple-choice question into a complex, open-ended query strictly within the [{target_domain}] domain.
Do NOT include multiple choices. Make it sound like a real-world professional or academic question.

CRITICAL INSTRUCTION:
You MUST create a question that perfectly matches the question type: '{target_func}'.
- If WHAT: Ask about definitions, concepts, or phenomena.
- If WHY: Ask about reasons, causes, or justifications.
- If HOW: Ask about mechanisms, processes, or methods.
- If COMPARE: Ask to evaluate, contrast, or weigh two different aspects.

Original Data:
Query/Instruction: {query}
Gold Answer/Choices: {gold_answer}

Respond ONLY with a raw, valid JSON object:
{{
    "id": "{id}",
    "expected_domain": "{target_domain}",
    "instruction": "<Your {target_func} question>",
    "gold_answer": "<The comprehensive gold answer>",
    "expected_func": "{target_func}"
}}"""

def main():
    print(f"[*] '{BASE_MODEL}' 모델을 준비합니다...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
    print("[*] 모델 로드 완료! GT 데이터셋 변환을 시작합니다. 🚀")

    if not os.path.exists(INPUT_FILE):
        print(f"[-] 에러: {INPUT_FILE} 파일이 존재하지 않습니다.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        old_gt_list = json.load(f)

    new_gt_list = []
    total_count = len(old_gt_list)
    func_types = ["WHAT", "WHY", "HOW", "COMPARE"]
    
    print(f"[*] 총 {total_count}개의 GT 데이터를 리라이팅합니다.")

    for idx, item in enumerate(old_gt_list):
        # 순서대로 타입을 강제 배분하여 황금 비율(25%) 달성
        current_func = func_types[idx % 4]
        
        orig_id = item.get("id", idx + 1)
        domain = item.get("expected_domain", "UNKNOWN").upper()
        query = item.get("query", item.get("instruction", ""))
        gold_answer = item.get("gold_answer", "")

        print(f"[{idx+1}/{total_count}] ID: {orig_id} | 도메인: {domain} ➔ 목표 타입: {current_func}")

        # CROSS와 단일 도메인 분기 처리
        if domain == "CROSS":
            prompt = CROSS_PROMPT.format(
                id=orig_id,
                target_func=current_func,
                query=query,
                gold_answer=gold_answer
            )
        else:
            prompt = SINGLE_PROMPT.format(
                id=orig_id,
                target_domain=domain,
                target_func=current_func,
                query=query,
                gold_answer=gold_answer
            )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.7, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        result_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # JSON 블록 정규식 파싱
        try:
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                new_item = json.loads(json_match.group(0))
                new_gt_list.append(new_item)
                print(f"  └ 성공! 파싱 완료.")
            else:
                raise ValueError("JSON not found")
        except Exception as e:
            print(f"  └ [에러] 파싱 실패. 안전 백업 데이터 생성.")
            # 예외 처리용 백업 포맷
            fallback_item = {
                "id": orig_id,
                "expected_domain": domain,
                "instruction": query,
                "gold_answer": result_text,
                "expected_func": current_func
            }
            if domain == "CROSS":
                fallback_item["crossed_domains"] = ["UNKNOWN", "UNKNOWN"]
            new_gt_list.append(fallback_item)

        # 10개마다 자동 중간 저장
        if (idx + 1) % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(new_gt_list, f, ensure_ascii=False, indent=2)

    # 최종 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(new_gt_list, f, ensure_ascii=False, indent=2)
        
    print(f"\n[+] 🎉 GT 데이터셋 환골탈태 완료: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()