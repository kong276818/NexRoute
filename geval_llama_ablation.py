"""
Ablation G-Eval: Llama-3.1-70B-Instruct judge
- V01~V05: CROSS 100개 subset / V07~V09: 전체 100개 (all CROSS)
- 기준: Relevance / Coherence / Factuality / Domain Expertise (1-5)
- 체크포인트: ~/moe_model/llama_geval_ablation_checkpoint.json
- 출력: ~/moe_model/llama_geval_ablation_results.json

Footnote: ablation-level G-Eval judge: Llama-3.1-70B-Instruct —
identical to main held-out G-Eval judge, enabling direct cross-comparison.
"""
import os, json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# ── 인증 ─────────────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_TOKEN 환경변수가 필요합니다: export HF_TOKEN=<token>")
login(token=hf_token, add_to_git_credential=False)

# ── 경로 ─────────────────────────────────────────────────────────────────
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
GEN_DIR    = "/home/sclab/moe_model/results/full_ablation/generations"
CKPT_PATH  = "/home/sclab/moe_model/llama_geval_ablation_checkpoint.json"
OUT_PATH   = "/home/sclab/moe_model/llama_geval_ablation_results.json"
LLAMA_PATH = (
    "/home/sclab/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-70B-Instruct/snapshots/"
    "1605565b47bb9346c5515c34102e054115b4f98b"
)

# ── Variant 정의 ──────────────────────────────────────────────────────────
# V01~V05: CROSS 100개만 평가 / V07~V09: 전체 (all CROSS)
VARIANTS = {
    "V01_full_nexroute":            {"file": "variant_01_full_nexroute.json",            "mode": "gt_lookup"},
    "V02_wo_function_axis":         {"file": "variant_02_wo_function_axis.json",         "mode": "expected_domain"},
    "V03_wo_partitioned_retrieval": {"file": "variant_03_wo_partitioned_retrieval.json", "mode": "expected_domain"},
    "V04_wo_lora":                  {"file": "variant_04_wo_lora.json",                  "mode": "expected_domain"},
    "V05_wo_domain_axis":           {"file": "variant_05_wo_domain_axis.json",           "mode": "expected_domain"},
    "V07_cross_weight_avg":         {"file": "variant_07_cross_weight_avg.json",         "mode": "all"},
    "V08_cross_task_arithmetic":    {"file": "variant_08_cross_task_arithmetic.json",    "mode": "all"},
    "V09_cross_dominant_single":    {"file": "variant_09_cross_dominant_single.json",    "mode": "all"},
}

# ── G-Eval 프롬프트 (llama_geval.py와 동일) ──────────────────────────────
EVAL_PROMPT = """You are an expert evaluator assessing AI-generated responses.

Question: {question}

Reference Answer: {reference}

Model Response: {response}

Evaluate the response on these criteria (score 1-5 each):
1. Relevance: Does the response directly address the question? (1=off-topic, 5=perfectly on-topic)
2. Coherence: Is the response logically structured and easy to follow? (1=incoherent, 5=excellent)
3. Factuality: Is the content accurate and consistent with the reference? (1=many errors, 5=fully accurate)
4. Domain Expertise: Does the response show appropriate domain knowledge? (1=superficial, 5=expert-level)

Respond ONLY in this exact JSON format:
{{"relevance": X, "coherence": X, "factuality": X, "domain_expertise": X}}

where X is an integer 1-5."""

SCORE_KEYS = ["relevance", "coherence", "factuality", "domain_expertise"]

def parse_scores(text):
    try:
        m = re.search(r"\{[^}]+\}", text)
        if m:
            scores = json.loads(m.group())
            if all(k in scores for k in SCORE_KEYS):
                return {k: float(scores[k]) for k in SCORE_KEYS}
    except Exception:
        pass
    nums = re.findall(r"\b[1-5]\b", text)
    if len(nums) >= 4:
        return {k: float(n) for k, n in zip(SCORE_KEYS, nums[:4])}
    return None

def run_judge(question, reference, response, model, tokenizer):
    prompt = EVAL_PROMPT.format(
        question=question[:3000],
        reference=reference[:3000],
        response=response[:3000],
    )
    msgs   = [{"role": "user", "content": prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.1,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return parse_scores(result)

def summarize(score_list):
    if not score_list:
        return {}
    return {k: round(float(np.mean([s[k] for s in score_list])), 3)
            for k in SCORE_KEYS + ["avg"]}

# ── GT500 로드 (V01 도메인 판별용) ────────────────────────────────────────
with open(GT_PATH) as f:
    gt500 = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt500}

# ── Llama 로드 ────────────────────────────────────────────────────────────
print(f"[*] Llama-3.1-70B-Instruct 4-bit 로드...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
judge = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, local_files_only=True)
print("[*] Llama 로드 완료!\n")

# ── 체크포인트 로드 ───────────────────────────────────────────────────────
all_results = {}
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH) as f:
        all_results = json.load(f)
    print(f"체크포인트 로드: {list(all_results.keys())}")

def save_ckpt():
    with open(CKPT_PATH, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

# ── 평가 루프 ─────────────────────────────────────────────────────────────
for variant_name, cfg in VARIANTS.items():
    if variant_name in all_results:
        print(f"[SKIP] {variant_name}")
        continue

    fpath = os.path.join(GEN_DIR, cfg["file"])
    if not os.path.exists(fpath):
        print(f"[SKIP] {variant_name}: 파일 없음")
        continue

    with open(fpath) as f:
        data = json.load(f)

    # ── CROSS 항목 추출 ───────────────────────────────────────────────────
    mode = cfg["mode"]
    if mode == "all":
        eval_items = data
    elif mode == "gt_lookup":
        # V01: GT500 룩업으로 CROSS 판별
        eval_items = [
            d for d in data
            if gt_map.get(d.get("query", "").strip(), {}).get("expected_domain") == "CROSS"
        ]
    else:
        # V02~V05: expected_domain 필드 사용
        eval_items = [d for d in data if d.get("expected_domain") == "CROSS"]

    print(f"\n{'='*55}")
    print(f"{variant_name}  (n={len(eval_items)}, mode={mode})")
    print(f"{'='*55}")

    scores_list = []
    law_flagged = 0

    for i, item in enumerate(eval_items):
        q    = item.get("query", item.get("instruction", "")).strip()
        gold = item.get("gold_answer", "")
        pred = item.get("pred_answer", "")

        if not pred:
            continue

        # GT에서 gold 보완
        if not gold:
            gold = gt_map.get(q, {}).get("gold_answer", "")

        # LAW 도메인 플래그 (MCQ artifact 위험 표시)
        domain = item.get("expected_domain", gt_map.get(q, {}).get("expected_domain", ""))
        is_law = (domain == "LAW")
        if is_law:
            law_flagged += 1

        scores = run_judge(q, gold, pred, judge, tokenizer)
        if scores:
            scores["query"]    = q[:80]
            scores["domain"]   = domain
            scores["is_law"]   = is_law
            scores["avg"]      = float(np.mean([scores[k] for k in SCORE_KEYS]))
            scores_list.append(scores)

        if (i + 1) % 10 == 0:
            all_results[variant_name] = {"scores": scores_list}
            save_ckpt()
            avg = np.mean([s["avg"] for s in scores_list]) if scores_list else 0
            print(f"  [{i+1}/{len(eval_items)}] avg={avg:.2f}  law_flagged={law_flagged}")

    # ── 요약 ──────────────────────────────────────────────────────────────
    non_law = [s for s in scores_list if not s.get("is_law")]
    all_results[variant_name] = {
        "scores":         scores_list,
        "summary":        summarize(scores_list),
        "summary_no_law": summarize(non_law),
        "n_total":        len(scores_list),
        "n_law_flagged":  law_flagged,
        "mode":           mode,
    }
    save_ckpt()

    sm  = all_results[variant_name]["summary"]
    smn = all_results[variant_name]["summary_no_law"]
    print(f"\n  ALL (n={len(scores_list)}):       avg={sm.get('avg',0):.3f}")
    print(f"  Non-LAW (n={len(non_law)}): avg={smn.get('avg',0):.3f}")

# ── V01 vs V02 핵심 비교 ─────────────────────────────────────────────────
print("\n\n" + "="*60)
print("V01 vs V02 Function Axis 기여도 비교")
print("="*60)
for vname in ["V01_full_nexroute", "V02_wo_function_axis"]:
    if vname in all_results:
        sm = all_results[vname].get("summary", {})
        smn = all_results[vname].get("summary_no_law", {})
        print(f"  {vname}:")
        for k in SCORE_KEYS + ["avg"]:
            print(f"    {k:20s}: {sm.get(k, 0):.3f}  (non-LAW: {smn.get(k, 0):.3f})")

# ── 전체 요약 테이블 ──────────────────────────────────────────────────────
print("\n\n" + "="*70)
print(f"{'Variant':<32}  {'Rel':>5}  {'Coh':>5}  {'Fac':>5}  {'Dom':>5}  {'Avg':>5}  {'n':>4}")
print("-"*70)
for vname, res in all_results.items():
    sm = res.get("summary", {})
    n  = res.get("n_total", 0)
    print(f"  {vname:<32}  "
          f"{sm.get('relevance',0):>5.2f}  "
          f"{sm.get('coherence',0):>5.2f}  "
          f"{sm.get('factuality',0):>5.2f}  "
          f"{sm.get('domain_expertise',0):>5.2f}  "
          f"{sm.get('avg',0):>5.2f}  {n:>4}")

# ── 최종 저장 ─────────────────────────────────────────────────────────────
final = {
    "llama_geval_ablation": all_results,
    "meta": {
        "judge":    "meta-llama/Llama-3.1-70B-Instruct",
        "criteria": SCORE_KEYS,
        "subset":   "CROSS only (V01-V05: 100 items, V07-V09: 100 items)",
        "footnote": (
            "ablation-level G-Eval judge: Llama-3.1-70B-Instruct — "
            "identical to main held-out G-Eval judge, enabling direct cross-comparison."
        ),
        "law_note": (
            "LAW-domain items flagged (is_law=True) due to known MCQ artifact risk. "
            "summary_no_law excludes these items."
        ),
    }
}
with open(OUT_PATH, "w") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(f"\n저장 완료: {OUT_PATH}")
