"""
CROSS 합성 소규모 재현 실험 (논문 Appendix A 통계용)
- 기존 cross_rewritten.json 에서 80개 샘플을 무작위 추출해
  동일 조건(Qwen2.5-7B, NF4, same prompt)으로 재합성
- 이후 Qwen2.5-7B LLM-as-a-judge 필터 적용:
    PASS / DIALOGUE_DRIFT / TOPIC_CONCAT / SINGLE_DOMAIN_COLLAPSE
- 결과 JSON + 대표 실패 사례 저장

출력:
  cross_repro_raw.json       — 생성 직후 원본 (전처리 없음)
  cross_repro_filtered.json  — PASS 판정 통과 샘플만
  cross_repro_stats.json     — 통계 및 실패 사례 원문
"""

import json
import os
import re
import random
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

random.seed(42)

# ── 설정 ──────────────────────────────────────────────────────────────────
LOCAL_MODEL = (
    "/home/sclab/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
INPUT_FILE   = "/home/sclab/moe_model/processed_v2/cross.json"
OUT_RAW      = "/home/sclab/moe_model/cross_repro_raw.json"
OUT_FILTERED = "/home/sclab/moe_model/cross_repro_filtered.json"
OUT_STATS    = "/home/sclab/moe_model/cross_repro_stats.json"
N_SAMPLES    = 80   # 재현 샘플 수

# ── 합성 프롬프트 (원본 rewrite_cross_data.py 동일) ───────────────────────
SYNTH_PROMPT = """\
You are an expert dataset creator for training AI routing models.
Rewrite the following multiple-choice question into a complex, open-ended \
dilemma that explicitly fuses exactly TWO distinct domains from this list: \
[MED, LAW, STEM, SOC].
(MED = Medical, LAW = Law/Policy, STEM = Science/Tech, SOC = Sociology/Business)

Do NOT include multiple choices.
Choose ONE func_type: WHAT, WHY, HOW, COMPARE.

The fused query MUST require causal reasoning that spans BOTH domains \
simultaneously — not merely topic-adjacent information from each.

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

# ── LLM-as-a-judge 프롬프트 ───────────────────────────────────────────────
# 판정: PASS / DIALOGUE_DRIFT / TOPIC_CONCAT / SINGLE_DOMAIN_COLLAPSE
JUDGE_PROMPT = """\
You are a strict quality judge for interdisciplinary AI training data.
Evaluate the following synthesized instruction-answer pair.

A high-quality CROSS-domain sample MUST satisfy ALL of the following:
1. The instruction explicitly invokes TWO distinct professional domains \
(e.g., MED + LAW).
2. Answering it REQUIRES joint causal reasoning across BOTH domains — \
not just recalling facts from each independently.
3. The answer demonstrates genuine cross-domain synthesis \
(not just sequential paragraphs from each domain).

Classify the sample into EXACTLY ONE category:
- PASS: Satisfies all three criteria.
- DIALOGUE_DRIFT: The model drifted from the cross-domain prompt and \
generated a response that is primarily single-domain or discusses \
general principles without domain fusion.
- TOPIC_CONCAT: The instruction or answer merely concatenates information \
from two domains side-by-side without causal linkage between them.
- SINGLE_DOMAIN_COLLAPSE: Despite being labeled CROSS, the content \
effectively addresses only one of the stated domains.

Sample to evaluate:
Instruction: {instruction}
Answer: {output}
Stated domains: {domains}

Respond ONLY with a JSON object:
{{
    "verdict": "<PASS|DIALOGUE_DRIFT|TOPIC_CONCAT|SINGLE_DOMAIN_COLLAPSE>",
    "reason": "<one sentence explaining the verdict>"
}}"""

# ── 모델 로드 ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  CROSS 합성 소규모 재현 실험 (N={})".format(N_SAMPLES))
print("=" * 65)

print("\n[1/3] 모델 로드 중 (Qwen2.5-7B NF4)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL, trust_remote_code=True, local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="eager",
)
model.eval()
print("  모델 로드 완료")


def generate(prompt_text: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=3072
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ── 데이터 로드 & 샘플링 ──────────────────────────────────────────────────
print("\n[2/3] 입력 데이터 로드 & 샘플링...")
with open(INPUT_FILE, encoding="utf-8") as f:
    pool = json.load(f)

# 이미 CROSS 변환된 원본 MCQ에서 샘플링 (원본 pipeline과 동일 소스)
sample = random.sample(pool, min(N_SAMPLES, len(pool)))
print(f"  {len(sample)}개 샘플 선택 (총 풀: {len(pool)}개)")

# ── Step 1: 합성 ─────────────────────────────────────────────────────────
print("\n[3a/3] 합성 단계 시작...")
raw_results = []
t0 = time.time()

for i, item in enumerate(sample, 1):
    prompt = SYNTH_PROMPT.format(
        instruction=item.get("instruction", ""),
        output=item.get("output", ""),
        context=item.get("context", ""),
    )
    raw_text = generate(prompt, max_new_tokens=512)
    parsed = parse_json(raw_text)

    if parsed and parsed.get("instruction") and parsed.get("output"):
        result = {
            "idx": i,
            "source": item,
            "raw_generation": raw_text,
            "synthesized": parsed,
            "parse_ok": True,
        }
    else:
        result = {
            "idx": i,
            "source": item,
            "raw_generation": raw_text,
            "synthesized": None,
            "parse_ok": False,
        }
    raw_results.append(result)

    elapsed = time.time() - t0
    eta = elapsed / i * (len(sample) - i)
    print(
        f"  [{i:3d}/{len(sample)}] parse={'OK' if result['parse_ok'] else 'FAIL'}"
        f"  ETA {eta/60:.1f}min"
    )

    # 중간 저장
    if i % 20 == 0:
        with open(OUT_RAW, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2)

with open(OUT_RAW, "w", encoding="utf-8") as f:
    json.dump(raw_results, f, ensure_ascii=False, indent=2)

n_parse_ok = sum(1 for r in raw_results if r["parse_ok"])
print(f"\n  합성 완료: {n_parse_ok}/{len(sample)} 파싱 성공")

# ── Step 2: LLM-as-a-judge 필터 ─────────────────────────────────────────
print("\n[3b/3] LLM-as-a-judge 필터링 시작...")

# 판정 카운터
verdict_counts = {
    "PASS": 0,
    "DIALOGUE_DRIFT": 0,
    "TOPIC_CONCAT": 0,
    "SINGLE_DOMAIN_COLLAPSE": 0,
    "PARSE_FAIL_SYNTH": 0,   # 합성 단계에서 JSON 파싱 실패
    "JUDGE_FAIL": 0,          # judge 응답 파싱 실패
}

# 실패 사례 보관 (각 카테고리별 최대 3개)
failure_examples = {
    "DIALOGUE_DRIFT": [],
    "TOPIC_CONCAT": [],
    "SINGLE_DOMAIN_COLLAPSE": [],
}
MAX_EXAMPLES = 3

filtered = []
judge_details = []

for r in raw_results:
    if not r["parse_ok"]:
        verdict_counts["PARSE_FAIL_SYNTH"] += 1
        judge_details.append({
            "idx": r["idx"],
            "verdict": "PARSE_FAIL_SYNTH",
            "reason": "Synthesis step produced invalid JSON",
            "instruction": r["source"].get("instruction", ""),
            "raw_generation": r["raw_generation"],
        })
        continue

    syn = r["synthesized"]
    instr = syn.get("instruction", "")
    output = syn.get("output", "")
    domains = syn.get("crossed_domains", [])

    judge_text = generate(
        JUDGE_PROMPT.format(
            instruction=instr,
            output=output,
            domains=domains,
        ),
        max_new_tokens=200,
    )
    judge_parsed = parse_json(judge_text)

    if not judge_parsed or "verdict" not in judge_parsed:
        verdict_counts["JUDGE_FAIL"] += 1
        verdict = "JUDGE_FAIL"
        reason = judge_text[:300]
    else:
        verdict = judge_parsed["verdict"].strip().upper()
        reason = judge_parsed.get("reason", "")
        if verdict not in verdict_counts:
            verdict = "JUDGE_FAIL"
        verdict_counts[verdict] += 1

    detail = {
        "idx": r["idx"],
        "verdict": verdict,
        "reason": reason,
        "instruction": instr,
        "output": output,
        "crossed_domains": domains,
        "func_type": syn.get("func_type", ""),
        "judge_raw": judge_text,
    }
    judge_details.append(detail)

    if verdict == "PASS":
        filtered.append(syn)
    elif verdict in failure_examples and len(failure_examples[verdict]) < MAX_EXAMPLES:
        failure_examples[verdict].append(detail)

    print(
        f"  idx={r['idx']:3d}  verdict={verdict:<28s}  "
        f"domains={domains}"
    )

# ── 통계 집계 ────────────────────────────────────────────────────────────
n_raw = len(sample)
n_parse_fail = verdict_counts["PARSE_FAIL_SYNTH"]
n_judged = n_raw - n_parse_fail
n_pass = verdict_counts["PASS"]
n_excluded = n_raw - n_pass

stats = {
    "n_raw_attempts": n_raw,
    "n_parse_fail_synthesis": n_parse_fail,
    "n_submitted_to_judge": n_judged,
    "verdict_counts": verdict_counts,
    "n_pass": n_pass,
    "n_excluded_total": n_excluded,
    "exclusion_rate_pct": round(n_excluded / n_raw * 100, 1),
    "failure_examples": failure_examples,
    "judge_details": judge_details,
}

with open(OUT_STATS, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

with open(OUT_FILTERED, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

# ── 최종 보고 ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  재현 실험 결과 요약")
print("=" * 65)
print(f"  Raw 시도:           {n_raw}")
print(f"  합성 파싱 실패:     {n_parse_fail}")
print(f"  Judge 제출:         {n_judged}")
print(f"  PASS:               {n_pass}")
print(f"  DIALOGUE_DRIFT:     {verdict_counts['DIALOGUE_DRIFT']}")
print(f"  TOPIC_CONCAT:       {verdict_counts['TOPIC_CONCAT']}")
print(f"  SINGLE_DOMAIN_COL:  {verdict_counts['SINGLE_DOMAIN_COLLAPSE']}")
print(f"  JUDGE_FAIL:         {verdict_counts['JUDGE_FAIL']}")
print(f"  제외율:             {stats['exclusion_rate_pct']}%")
print()

for cat, examples in failure_examples.items():
    if examples:
        print(f"\n[대표 실패 사례: {cat}]")
        ex = examples[0]
        print(f"  Instruction: {ex['instruction'][:200]}")
        print(f"  Reason: {ex['reason']}")

print(f"\n✅ 결과 저장:")
print(f"   Raw     → {OUT_RAW}")
print(f"   Filtered→ {OUT_FILTERED}")
print(f"   Stats   → {OUT_STATS}")
