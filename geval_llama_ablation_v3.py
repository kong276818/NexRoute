"""
Ablation G-Eval v3: Llama-3.1-70B-Instruct, 3-criteria (Domain Accuracy / Logic / Utility)
  - 1-10 integer scale, 3× repeats (seeds 42/43/44), temperature=0.3
  - V01~V09: CROSS 항목만, 100개 per variant
  - Main G-Eval v3 완료 후 실행

보안: HF_TOKEN은 os.environ["HF_TOKEN"]으로만 읽음.
"""
import os, sys, json, re, time, torch, numpy as np
from collections import Counter
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
SEEDS          = [42, 43, 44]
TEMPERATURE    = 0.3
MAX_NEW_TOKENS = 200

# ══════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_TOKEN 환경변수가 필요합니다: export HF_TOKEN=<token>")
login(token=hf_token, add_to_git_credential=False)

# ══════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════
BASE     = "/home/sclab/moe_model"
GT_PATH  = f"{BASE}/processed_v2/gt_dataset_500.json"
GEN_DIR  = f"{BASE}/results/full_ablation/generations"
CKPT_PATH = f"{BASE}/llama_geval_ablation_v3_checkpoint.json"
OUT_PATH  = f"{BASE}/llama_geval_ablation_v3_results.json"
LLAMA_PATH = (
    "/home/sclab/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-70B-Instruct/snapshots/"
    "1605565b47bb9346c5515c34102e054115b4f98b"
)

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

SCORE_KEYS = ["domain_accuracy", "logic", "utility"]

# ══════════════════════════════════════════════════════════════════════════
#  PROMPT (llama_geval_v3.py와 동일)
# ══════════════════════════════════════════════════════════════════════════
EVAL_PROMPT = """\
You are evaluating a response to a cross-domain professional query.

Question: {question}

Reference Answer: {reference}

Model Response: {response}

Evaluate the response on THREE criteria using a 1-10 integer scale (higher = better):

1. **Domain Accuracy** – Is the response factually correct and consistent with established domain knowledge? Are technical terms and concepts used correctly?
   - 1-3: Multiple factual errors or misuse of domain concepts
   - 4-6: Mostly correct but some inaccuracies or vague usage
   - 7-8: Accurate with minor imprecisions
   - 9-10: Fully accurate, precise domain expertise demonstrated

2. **Logic** – Does the response construct a coherent causal or explanatory chain connecting premises to conclusions? Is the reasoning well-structured?
   - 1-3: Disjointed facts without logical connection
   - 4-6: Some structure but gaps in reasoning
   - 7-8: Clear logical flow with minor gaps
   - 9-10: Tight causal chain, every step justified

3. **Utility** – Does the response directly address the query intent and provide actionable, relevant information?
   NOTE: verbose or lengthy responses are NOT inherently better. A concise, precise answer is superior to a long, repetitive one.
   - 1-3: Off-topic or padded with irrelevant content
   - 4-6: Addresses query but with unnecessary tangents
   - 7-8: Focused and relevant with minor digressions
   - 9-10: Precisely addresses the query, no wasted content

**IMPORTANT**:
- Judge based on content quality, NOT response length.
- A 100-word precise answer can score higher than a 500-word verbose one.

Before assigning scores, briefly justify each rating in 1-2 sentences.

Respond in this EXACT JSON format:
{{
  "domain_accuracy_reasoning": "...",
  "domain_accuracy": X,
  "logic_reasoning": "...",
  "logic": X,
  "utility_reasoning": "...",
  "utility": X
}}
where X is an integer 1-10."""


def parse_scores(text):
    try:
        m = re.search(r'\{[\s\S]*?"domain_accuracy"[\s\S]*?\}', text)
        if not m:
            m = re.search(r'\{[\s\S]*?\}', text)
        if m:
            obj = json.loads(m.group())
            scores = {}
            for k in SCORE_KEYS:
                if k in obj:
                    v = obj[k]
                    v_f = float(v) if isinstance(v, (int, float)) else None
                    if v_f is not None and 1 <= v_f <= 10:
                        scores[k] = v_f
            reasoning = {f"{k}_reasoning": str(obj.get(f"{k}_reasoning", ""))[:400]
                         for k in SCORE_KEYS}
            if all(k in scores for k in SCORE_KEYS):
                return {**scores, **reasoning}
    except Exception:
        pass
    scores = {}
    for k in SCORE_KEYS:
        m = re.search(rf'"{k}"\s*:\s*(10|[1-9])', text)
        if m:
            scores[k] = float(m.group(1))
    if all(k in scores for k in SCORE_KEYS):
        return scores
    nums = [int(n) for n in re.findall(r'\b(10|[1-9])\b', text) if 1 <= int(n) <= 10]
    if len(nums) >= 3:
        return {k: float(nums[i]) for i, k in enumerate(SCORE_KEYS)}
    return None


def run_once(question, reference, response, model, tokenizer, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    raw    = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    scores = parse_scores(raw)
    if scores is None:
        print(f"    [파싱 실패] seed={seed} | raw: {raw[:120]}")
    return scores, raw


def aggregate_runs(run_list):
    if not run_list:
        return {}
    result = {}
    for k in SCORE_KEYS:
        vals = [r[k] for r in run_list if k in r]
        if vals:
            ddof = 1 if len(vals) > 1 else 0
            result[f"{k}_mean"] = round(float(np.mean(vals)), 3)
            result[f"{k}_std"]  = round(float(np.std(vals, ddof=ddof)), 3)
    avgs = [np.mean([r[k] for k in SCORE_KEYS if k in r]) for r in run_list]
    ddof = 1 if len(avgs) > 1 else 0
    result["avg_mean"] = round(float(np.mean(avgs)), 3)
    result["avg_std"]  = round(float(np.std(avgs, ddof=ddof)), 3)
    return result


def model_summary(sample_list):
    avgs = [s["aggregate"]["avg_mean"] for s in sample_list
            if "avg_mean" in s.get("aggregate", {})]
    if not avgs:
        return {}
    n        = len(avgs)
    mean_val = float(np.mean(avgs))
    std_val  = float(np.std(avgs, ddof=1)) if n > 1 else 0.0
    se       = std_val / np.sqrt(n)
    t_crit   = (float(scipy_stats.t.ppf(0.975, df=n-1)) if HAS_SCIPY and n > 1
                else 1.96)
    ci       = t_crit * se
    per = {}
    for k in SCORE_KEYS:
        vals = [s["aggregate"].get(f"{k}_mean") for s in sample_list
                if f"{k}_mean" in s.get("aggregate", {})]
        if vals:
            per[f"{k}_mean"] = round(float(np.mean(vals)), 3)
            per[f"{k}_std"]  = round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0), 3)
    return {
        "n_samples":     n,
        "avg_mean":      round(mean_val, 3),
        "avg_std":       round(std_val, 3),
        "avg_ci95_half": round(ci, 3),
        "avg_ci95":      f"[{round(mean_val - ci, 3):.3f}, {round(mean_val + ci, 3):.3f}]",
        **per,
    }


# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
with open(GT_PATH) as f:
    gt500 = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt500}

print(f"[*] Llama-3.1-70B-Instruct 4-bit 로드 중...")
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

# 체크포인트 로드
all_results = {}
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH) as f:
        ckpt_data = json.load(f)
    if ckpt_data:
        all_results = ckpt_data
        print(f"체크포인트 로드: {list(all_results.keys())}")

def save_ckpt():
    with open(CKPT_PATH, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

# ══════════════════════════════════════════════════════════════════════════
#  평가 루프
# ══════════════════════════════════════════════════════════════════════════
for variant_name, cfg in VARIANTS.items():
    existing = all_results.get(variant_name, {})
    if existing.get("n_total", 0) >= 100:
        print(f"[SKIP] {variant_name}")
        continue

    fpath = os.path.join(GEN_DIR, cfg["file"])
    if not os.path.exists(fpath):
        print(f"[SKIP] {variant_name}: 파일 없음")
        continue

    with open(fpath) as f:
        data = json.load(f)

    mode = cfg["mode"]
    if mode == "all":
        eval_items = data
    elif mode == "gt_lookup":
        eval_items = [d for d in data
                      if gt_map.get(d.get("query", "").strip(), {}).get("expected_domain") == "CROSS"]
    else:
        eval_items = [d for d in data if d.get("expected_domain") == "CROSS"]

    print(f"\n{'='*55}\n{variant_name}  (n={len(eval_items)}, mode={mode})\n{'='*55}")

    done_queries = {s["query"] for s in existing.get("samples", [])}
    samples      = existing.get("samples", [])

    for i, item in enumerate(eval_items):
        q    = item.get("query", item.get("instruction", "")).strip()
        gold = item.get("gold_answer", "")
        pred = item.get("pred_answer", "")

        if q[:80] in done_queries or not pred:
            continue
        if not gold:
            gold = gt_map.get(q, {}).get("gold_answer", "")

        runs = []
        for seed in SEEDS:
            sc, _ = run_once(q, gold, pred, judge, tokenizer, seed)
            if sc:
                sc["seed"] = seed
                runs.append(sc)

        agg = aggregate_runs(runs)
        samples.append({
            "query":        q[:80],
            "domain":       item.get("expected_domain", gt_map.get(q, {}).get("expected_domain", "")),
            "response_len": len(pred),
            "runs":         runs,
            "aggregate":    agg,
        })
        done_queries.add(q[:80])

        n = len(samples)
        if n % 10 == 0:
            all_results[variant_name] = {
                "samples": samples,
                "n_total": n,
            }
            save_ckpt()
            avg = np.mean([s["aggregate"].get("avg_mean", 0) for s in samples])
            print(f"  [{n}/{len(eval_items)}] avg={avg:.3f}")

    sm = model_summary(samples)
    all_results[variant_name] = {
        "samples": samples,
        "summary": sm,
        "n_total": len(samples),
        "mode":    mode,
    }
    save_ckpt()
    print(f"\n  {variant_name}: DA={sm.get('domain_accuracy_mean',0):.3f}  "
          f"Logic={sm.get('logic_mean',0):.3f}  Util={sm.get('utility_mean',0):.3f}  "
          f"Avg={sm.get('avg_mean',0):.3f}  CI95={sm.get('avg_ci95','?')}")

# ── 최종 테이블 ─────────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print(f"{'Variant':<32}  {'DA':>5}  {'Logic':>5}  {'Util':>5}  {'Avg':>6}  {'CI95':>22}  {'n':>4}")
print("-" * 80)
for vname, res in all_results.items():
    sm = res.get("summary", {})
    print(f"  {vname:<32}  "
          f"{sm.get('domain_accuracy_mean',0):>5.2f}  "
          f"{sm.get('logic_mean',0):>5.2f}  "
          f"{sm.get('utility_mean',0):>5.2f}  "
          f"{sm.get('avg_mean',0):>6.3f}  "
          f"{sm.get('avg_ci95','[?,?]'):>22}  "
          f"{res.get('n_total',0):>4}")

final = {
    "llama_geval_ablation_v3": all_results,
    "meta": {
        "version":        "v3",
        "judge":          "meta-llama/Llama-3.1-70B-Instruct",
        "criteria":       SCORE_KEYS,
        "scale":          "1-10 integer",
        "seeds":          SEEDS,
        "temperature":    TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "subset":         "CROSS only (100 items per variant)",
    }
}
with open(OUT_PATH, "w") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(f"\n저장 완료: {OUT_PATH}")
