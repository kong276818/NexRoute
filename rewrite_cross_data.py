import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── 1. 설정 (인증이 필요 없는 완전 오픈형 7B 모델) ──
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
INPUT_FILE = "/home/sclab/moe_model/processed_v2/cross.json"
OUTPUT_FILE = "/home/sclab/moe_model/processed_v2/cross_rewritten.json"

# ── 2. 프롬프트 (정확히 2개 도메인 추출 및 융합 강제) ──
REWRITE_PROMPT = """You are an expert dataset creator for training AI routing models.
Rewrite the following multiple-choice question into a complex, open-ended dilemma that explicitly fuses exactly TWO distinct domains from this list: [MED, LAW, STEM, SOC].
(MED = Medical, LAW = Law/Policy, STEM = Science/Tech, SOC = Sociology/Business)

Do NOT include multiple choices. 
Choose ONE func_type: WHAT, WHY, HOW, COMPARE.

Original Data:
Instruction: {instruction}
Output: {output}
Context: {context}

Respond ONLY with a raw, valid JSON object. No markdown, no intro text.
{{
    "domain": "CROSS",
    "crossed_domains": ["<Domain 1>", "<Domain 2>"],
    "instruction": "<The newly written open-ended query>",
    "output": "<The comprehensive gold answer>",
    "func_type": "<WHAT, WHY, HOW, or COMPARE>"
}}"""

def main():
    print(f"[*] '{BASE_MODEL}' 모델을 준비합니다. (권한 불필요, 바로 다운로드)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    print("[*] Qwen 7B 로딩 완료! 엄청난 속도로 변환을 시작합니다 🚀")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    new_data = []
    total_count = len(old_data)
    print(f"[*] 총 변환할 데이터 개수: {total_count}개")

    for idx, item in enumerate(old_data):
        print(f"[{idx+1}/{total_count}] 생성 중...")
        
        prompt = REWRITE_PROMPT.format(
            instruction=item.get("instruction", ""), 
            output=item.get("output", ""), 
            context=item.get("context", "")
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
        
        # ── 3. 초강력 JSON 파싱 ──
        try:
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                new_item = json.loads(json_match.group(0))
                new_data.append(new_item)
                
                domains = new_item.get('crossed_domains', [])
                func = new_item.get('func_type', '')
                print(f"  └ 성공! 태그: {domains} | 타입: {func}")
            else:
                 raise ValueError("JSON block not found")
        except Exception as e:
            print(f"  └ [에러] 모델이 딴소리를 했습니다. 일단 스킵합니다.")
            new_data.append({
                "domain": "CROSS",
                "crossed_domains": ["UNKNOWN", "UNKNOWN"],
                "instruction": item.get("instruction", ""),
                "output": result_text,
                "func_type": "OTHER"
            })

        # 20개마다 자동 저장 (안전장치)
        if (idx + 1) % 20 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

    # 최종 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n[+] 🎉 데이터 변환 완료! 저장 위치: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()