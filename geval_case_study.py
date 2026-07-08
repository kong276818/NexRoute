"""
G-Eval: Case Study 케이스 스터디 응답 평가
Judge: Qwen2.5-72B-Instruct (4-bit quantized)
"""
import json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import glob

CASE_PATH = "/home/sclab/moe_model/case_study_outputs.json"
OUT_PATH  = "/home/sclab/moe_model/geval_case_study_results.json"

_cands = glob.glob(
    "/home/sclab/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-72B-Instruct/snapshots/*"
)
BASE_72B = _cands[0] if _cands else "Qwen/Qwen2.5-72B-Instruct"

MODELS = ["Naive RAG 7B", "Adaptive RAG 7B", "HyDE 7B", "CRAG 7B", "NexRoute 7B"]

# gold answer (cross-domain LAW+MED 참조 답변)
GOLD = (
    "The physician may be held liable despite the consent form because informed "
    "consent requires specific disclosure of known risks, especially for off-label "
    "use. A general consent form does not cover risks unique to non-approved "
    "indications. Since hepatotoxicity is a known risk of Drug X and the off-label "
    "use for weight loss was the physician's decision, the duty to warn about "
    "liver damage specifically falls on the prescribing doctor. Failure to disclose "
    "this risk breaches the standard of care and establishes proximate causation "
    "between the omission and the patient's injury."
)

EVAL_PROMPT = """You are an expert evaluator assessing AI-generated responses to a medical-legal question.

Question: {question}

Reference Answer: {reference}

Model Response: {response}

Evaluate the model response on these criteria (score 1-5):
1. Relevance: Does it directly address the legal liability question? (1=off-topic, 5=perfectly on-topic)
2. Coherence: Is it logically structured and easy to follow? (1=incoherent, 5=excellent)
3. Factuality: Is it accurate and consistent with the reference? (1=many errors, 5=fully accurate)
4. Domain Expertise: Does it integrate both medical and legal knowledge appropriately? (1=superficial, 5=expert-level)

Respond ONLY in this exact JSON format:
{{"relevance": X, "coherence": X, "factuality": X, "domain_expertise": X}}"""


def parse_scores(text):
    try:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            scores = json.loads(match.group())
            keys = ['relevance', 'coherence', 'factuality', 'domain_expertise']
            if all(k in scores for k in keys):
                return {k: float(scores[k]) for k in keys}
    except: pass
    nums = re.findall(r'\b[1-5]\b', text)
    if len(nums) >= 4:
        keys = ['relevance', 'coherence', 'factuality', 'domain_expertise']
        return {k: float(n) for k, n in zip(keys, nums[:4])}
    return None


# ── 72B 모델 로드 ─────────────────────────────────────────────
print("[*] Judge 모델 로드 중 (Qwen2.5-72B 4-bit)...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_72B)
judge = AutoModelForCausalLM.from_pretrained(
    BASE_72B, quantization_config=bnb_cfg, device_map="auto"
)
judge.eval()
print("[*] 완료\n")

# ── 케이스 스터디 로드 ─────────────────────────────────────────
with open(CASE_PATH) as f:
    case_data = json.load(f)

query = case_data["query"]


def evaluate(question, reference, response):
    prompt = EVAL_PROMPT.format(
        question=question[:600],
        reference=reference[:600],
        response=response[:600],
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(judge.device)
    with torch.no_grad():
        out = judge.generate(
            **inputs, max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    return parse_scores(raw), raw


# ── 평가 실행 ─────────────────────────────────────────────────
print(f"{'Model':<22}  {'Relev':>5}  {'Coher':>5}  {'Fact':>5}  {'Expert':>6}  {'Avg':>5}")
print("─" * 58)

results = {}
for name in MODELS:
    response = case_data.get(name, "")
    if not response:
        print(f"  {name:<22}  [SKIP]")
        continue
    scores, raw = evaluate(query, GOLD, response)
    if scores:
        avg = sum(scores.values()) / len(scores)
        results[name] = {**scores, "average": round(avg, 2)}
        print(f"  {name:<22}  "
              f"{scores['relevance']:>5.1f}  "
              f"{scores['coherence']:>5.1f}  "
              f"{scores['factuality']:>5.1f}  "
              f"{scores['domain_expertise']:>6.1f}  "
              f"{avg:>5.2f}")
    else:
        results[name] = {"error": raw[:100]}
        print(f"  {name:<22}  [PARSE ERROR] {raw[:60]}")

with open(OUT_PATH, "w") as f:
    json.dump({"query": query, "gold": GOLD, "results": results}, f,
              indent=2, ensure_ascii=False)

print(f"\n✅ 저장: {OUT_PATH}")
print("\n" + "="*58)
print(f"{'Model':<22}  {'Average G-Eval':>14}")
print("─"*40)
for name, v in results.items():
    if "average" in v:
        marker = " ◀" if "NexRoute" in name else ""
        print(f"  {name:<22}  {v['average']:>14.2f}{marker}")
