"""
Phase 2: Llama-3.1-70B-Instruct G-Eval
- 5개 모델 × 200샘플 (GT500 CROSS 100 + held-out CROSS 100)
- 기준: Relevance / Coherence / Factuality / Domain Expertise (1-5 각)
- 매 10개마다 체크포인트: ~/moe_model/llama_geval_checkpoint.json
- 기존 Qwen-72B 결과(geval_results.json)와 Spearman correlation 계산
"""
import os, sys, json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy.stats import spearmanr
from huggingface_hub import login

# ── HF 토큰 인증 ─────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise EnvironmentError(
        "HF_TOKEN 환경변수가 설정되지 않았습니다.\n"
        "터미널에서: export HF_TOKEN=<your_token>"
    )
login(token=hf_token, add_to_git_credential=False)

# ── 경로 설정 ────────────────────────────────────────────────────────────
GT_PATH       = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
HELDOUT_PATH  = "/home/sclab/moe_model/heldout_cross_predictions.json"
PREV_GEVAL    = "/home/sclab/moe_model/geval_results.json"
CKPT_PATH     = "/home/sclab/moe_model/llama_geval_checkpoint.json"
OUT_PATH      = "/home/sclab/moe_model/llama_geval_results.json"

LLAMA_PATH = (
    "/home/sclab/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-70B-Instruct/snapshots/"
    "1605565b47bb9346c5515c34102e054115b4f98b"
)

# 평가 대상 7B 모델 (파일 경로)
EVAL_FILES = {
    "Naive RAG 7B":    "/home/sclab/moe_model/baseline_results/naive_rag_7b_500.json",
    "Adaptive RAG 7B": "/home/sclab/moe_model/baseline_results/adaptive_rag_7b_500.json",
    "HyDE 7B":         "/home/sclab/moe_model/baseline_results/hyde_7b_500.json",
    "CRAG 7B":         "/home/sclab/moe_model/baseline_results/crag_7b_500_v2.json",
    "NexRoute 7B":     "/home/sclab/moe_model/eval_results_7b_v3_500.json",
}

# ── G-Eval 프롬프트 ───────────────────────────────────────────────────────
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

# ── 데이터 준비 ───────────────────────────────────────────────────────────
with open(GT_PATH) as f:
    gt500 = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt500}

# held-out 예측 로드
if not os.path.exists(HELDOUT_PATH):
    raise FileNotFoundError(
        f"{HELDOUT_PATH} 없음. llama_geval_heldout_infer.py를 먼저 실행하세요.")
with open(HELDOUT_PATH) as f:
    heldout_preds = json.load(f)

# GT500 CROSS 항목 맵 (query → gt 정보)
cross_gt_set = {d["instruction"].strip() for d in gt500 if d.get("expected_domain") == "CROSS"}
print(f"GT500 CROSS 항목: {len(cross_gt_set)}개")

# 기존 Qwen-72B G-Eval 결과 로드 (cross-judge agreement용)
prev_geval = {}
if os.path.exists(PREV_GEVAL):
    with open(PREV_GEVAL) as f:
        prev_geval = json.load(f)
    print(f"기존 Qwen-72B 결과 로드: {list(prev_geval.keys())}")

# ── Llama 로드 ─────────────────────────────────────────────────────────────
print(f"\n[*] Llama-3.1-70B-Instruct 4-bit 로드 중: {LLAMA_PATH}")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_PATH, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
judge = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, local_files_only=True)
print("[*] Llama 로드 완료!")

# ── 체크포인트 로드 ────────────────────────────────────────────────────────
all_results = {}
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH) as f:
        all_results = json.load(f)
    print(f"체크포인트 로드: {list(all_results.keys())}")

def save_ckpt():
    with open(CKPT_PATH, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

# ── 평가 루프 ───────────────────────────────────────────────────────────────
for model_name, pred_path in EVAL_FILES.items():
    if model_name in all_results:
        print(f"[SKIP] {model_name}")
        continue
    if not os.path.exists(pred_path):
        print(f"[SKIP] {model_name}: 파일 없음 ({pred_path})")
        continue

    print(f"\n{'='*55}\n{model_name}\n{'='*55}")
    with open(pred_path) as f:
        model_preds = json.load(f)

    # query → pred_answer 맵
    pred_map = {}
    for item in model_preds:
        q = item.get("query", item.get("instruction", "")).strip()
        pred_map[q] = item.get("pred_answer", "").strip()

    # ── Part A: GT500 CROSS 100개 ─────────────────────────────────────────
    gt_cross_scores = []
    print(f"[Part A] GT500 CROSS 100개...")
    for j, d in enumerate(gt500):
        if d.get("expected_domain") != "CROSS":
            continue
        q    = d["instruction"].strip()
        gold = d.get("gold_answer", "")
        pred = pred_map.get(q, "")
        if not pred:
            continue
        scores = run_judge(q, gold, pred, judge, tokenizer)
        if scores:
            scores["query"]  = q[:80]
            scores["source"] = "gt500_cross"
            scores["avg"]    = float(np.mean([scores[k] for k in SCORE_KEYS]))
            gt_cross_scores.append(scores)

        done = len(gt_cross_scores)
        if done % 10 == 0 and done > 0:
            all_results[model_name] = {
                "gt500_cross": gt_cross_scores,
                "heldout_cross": all_results.get(model_name, {}).get("heldout_cross", []),
            }
            save_ckpt()
            avg = np.mean([s["avg"] for s in gt_cross_scores])
            print(f"  [A {done}/100] avg={avg:.2f}")

    # ── Part B: Held-out CROSS 100개 ─────────────────────────────────────
    heldout_scores = []
    heldout_model  = heldout_preds.get(model_name, [])
    print(f"[Part B] Held-out CROSS {len(heldout_model)}개...")
    for j, item in enumerate(heldout_model):
        q    = item.get("query", "").strip()
        gold = item.get("gold_answer", "")
        pred = item.get("pred_answer", "")
        if not pred:
            continue
        scores = run_judge(q, gold, pred, judge, tokenizer)
        if scores:
            scores["query"]     = q[:80]
            scores["source"]    = "heldout_cross"
            scores["cross_pair"] = item.get("cross_pair", "")
            scores["avg"]       = float(np.mean([scores[k] for k in SCORE_KEYS]))
            heldout_scores.append(scores)

        done = len(heldout_scores)
        if done % 10 == 0 and done > 0:
            all_results[model_name] = {
                "gt500_cross":  gt_cross_scores,
                "heldout_cross": heldout_scores,
            }
            save_ckpt()
            avg = np.mean([s["avg"] for s in heldout_scores])
            print(f"  [B {done}/{len(heldout_model)}] avg={avg:.2f}")

    # ── 요약 ──────────────────────────────────────────────────────────────
    def summarize(score_list):
        if not score_list:
            return {}
        return {k: round(float(np.mean([s[k] for s in score_list])), 3)
                for k in SCORE_KEYS + ["avg"]}

    all_results[model_name] = {
        "gt500_cross":          gt_cross_scores,
        "heldout_cross":        heldout_scores,
        "gt500_cross_summary":  summarize(gt_cross_scores),
        "heldout_cross_summary": summarize(heldout_scores),
        "combined_summary":     summarize(gt_cross_scores + heldout_scores),
        "n_gt500":   len(gt_cross_scores),
        "n_heldout": len(heldout_scores),
        "n_total":   len(gt_cross_scores) + len(heldout_scores),
    }
    save_ckpt()

    c  = all_results[model_name]["gt500_cross_summary"]
    h  = all_results[model_name]["heldout_cross_summary"]
    cm = all_results[model_name]["combined_summary"]
    print(f"\n  GT500 CROSS  (n={len(gt_cross_scores)}): avg={c.get('avg',0):.2f}")
    print(f"  Held-out     (n={len(heldout_scores)}): avg={h.get('avg',0):.2f}")
    print(f"  Combined     (n={len(gt_cross_scores)+len(heldout_scores)}): avg={cm.get('avg',0):.2f}")

# ── Cross-Judge Agreement (Spearman) ────────────────────────────────────
print("\n\n" + "="*55)
print("Cross-Judge Agreement: Llama vs Qwen-72B (Spearman ρ)")
print("="*55)

agreement_results = {}
target_models = ["Naive RAG 7B", "Adaptive RAG 7B", "HyDE 7B", "CRAG 7B", "NexRoute 7B"]

for model_name in target_models:
    if model_name not in all_results or model_name not in prev_geval:
        print(f"  {model_name}: 데이터 없음 (skip)")
        continue

    # 기존 Qwen-72B 결과: query → avg score
    qwen_cross = prev_geval[model_name].get("cross", [])
    qwen_map   = {s["query"][:80]: s["avg"] for s in qwen_cross if "query" in s and "avg" in s}

    # Llama 결과: gt500_cross의 query → avg score
    llama_cross = all_results[model_name].get("gt500_cross", [])
    llama_map   = {s["query"][:80]: s["avg"] for s in llama_cross if "query" in s and "avg" in s}

    # 교집합 쿼리
    common_queries = sorted(set(qwen_map.keys()) & set(llama_map.keys()))
    if len(common_queries) < 5:
        print(f"  {model_name}: 공통 쿼리 {len(common_queries)}개 (too few)")
        continue

    qwen_scores  = [qwen_map[q]  for q in common_queries]
    llama_scores = [llama_map[q] for q in common_queries]
    rho, pval    = spearmanr(qwen_scores, llama_scores)
    agreement_results[model_name] = {
        "n_common": len(common_queries),
        "spearman_rho": round(float(rho), 4),
        "p_value":      round(float(pval), 4),
    }
    print(f"  {model_name:<22}: n={len(common_queries):3d}  ρ={rho:.3f}  p={pval:.3f}")

# ── 최종 테이블 ──────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print(f"{'Model':<22}  {'GT500 CROSS':>11}  {'Held-out':>8}  {'Combined':>8}  {'n_total':>7}")
print("-"*70)
for name, res in all_results.items():
    c  = res.get("gt500_cross_summary",  {}).get("avg", 0)
    h  = res.get("heldout_cross_summary", {}).get("avg", 0)
    cm = res.get("combined_summary",      {}).get("avg", 0)
    n  = res.get("n_total", 0)
    print(f"  {name:<22}  {c:>11.2f}  {h:>8.2f}  {cm:>8.2f}  {n:>7}")

# ── 최종 저장 ────────────────────────────────────────────────────────────
final = {
    "llama_geval":         all_results,
    "cross_judge_agreement": agreement_results,
    "meta": {
        "judge":    "meta-llama/Llama-3.1-70B-Instruct",
        "criteria": SCORE_KEYS,
        "n_per_model": "100 (gt500_cross) + 100 (heldout_cross)",
        "existing_judge": "Qwen/Qwen2.5-72B-Instruct (n=50 CROSS per model)",
    }
}
with open(OUT_PATH, "w") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(f"\n저장 완료: {OUT_PATH}")
