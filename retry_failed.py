import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch.nn as nn
from transformers import AutoModel

# pipeline.py 에서 클래스 임포트
import sys
sys.path.append("/home/sclab/moe_model")
from pipeline import DualMoEPipeline

RESULT_PATH = "/home/sclab/moe_model/eval_results.json"

def retry_failed():
    # 결과 로드
    with open(RESULT_PATH) as f:
        results = json.load(f)

    failed_idx = [
        i for i, x in enumerate(results)
        if "[GENERATION_FAILED]" in x.get("pred_answer", "")
    ]
    print(f"재생성 대상: {len(failed_idx)}개")

    # 파이프라인 로드
    pipe = DualMoEPipeline()
    pipe.load_router()
    pipe.load_embed_model()
    pipe.load_base_model()
    pipe.build_db()

    for i, idx in enumerate(failed_idx):
        item  = results[idx]
        query = item["query"]
        print(f"[{i+1}/{len(failed_idx)}] ID:{item['id']} | {query[:50]}...")

        try:
            # ★ 더 강한 프롬프트로 직접 생성
            routing  = pipe.classify_domain(query)
            is_cross = routing["is_cross"]
            domains  = routing["domains"]
            lora_target = "CROSS" if is_cross else domains[0]
            pipe.load_lora(lora_target)
            context  = pipe.retrieve(query, domains)

            # 강제 서술형 프롬프트
            ctx_text = "\n".join(
                [f"[{r['domain']}] {r['text'][:200]}" for r in context]
            )
            prompt = (
                f"Answer the following question with a detailed explanation "
                f"in 3-5 sentences. You MUST write a paragraph. "
                f"Do NOT use A/B/C/D choices. Do NOT say 'correct answer'.\n\n"
                f"Context:\n{ctx_text}\n\n"
                f"Question: {query}\n\n"
                f"Paragraph answer:"
            )
            messages = [{"role": "user", "content": prompt}]
            text = pipe.base_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = pipe.base_tokenizer(
                text, return_tensors="pt"
            ).to(pipe.base_model.device)

            with torch.no_grad():
                outputs = pipe.base_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,      # ★ 더 낮게 (안정적 생성)
                    do_sample=True,
                    pad_token_id=pipe.base_tokenizer.eos_token_id
                )

            answer = pipe.base_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # 후처리
            answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
            answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
            for kw in ["Choices:", "All answers:", "\nA:", "\nB:", "\nC:", "\nD:"]:
                if kw in answer:
                    answer = answer[:answer.index(kw)].strip()
            answer = re.sub(r'\n\d+:.*', '', answer).strip()

            if not answer.strip():
                answer = "[GENERATION_FAILED]"

            results[idx]["pred_answer"]   = answer
            results[idx]["domain_pred"]   = domains
            results[idx]["is_cross"]      = is_cross
            results[idx]["domain_scores"] = routing["scores"]

            print(f"  → {answer[:80]}...")

        except Exception as e:
            print(f"  [Error] {e}")
            continue

        # 10개마다 중간 저장
        if (i + 1) % 10 == 0:
            with open(RESULT_PATH, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [저장] {i+1}개 중간 저장")

    # 최종 저장
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 결과 확인
    still_failed = [
        x for x in results
        if "[GENERATION_FAILED]" in x.get("pred_answer", "")
    ]
    print(f"\n완료!")
    print(f"재생성 성공: {len(failed_idx) - len(still_failed)}개")
    print(f"여전히 실패: {len(still_failed)}개")

if __name__ == "__main__":
    retry_failed()