"""
G-Eval v3: Llama-3.1-70B-Instruct, 3-criteria (Domain Accuracy / Logic / Utility)
  - 1-10 integer scale with rubric anchors
  - 3× repeated scoring per sample (seeds 42/43/44), temperature=0.3
  - verbosity-agnostic rubric (length NOT a proxy for quality)
  - PILOT_ONLY=True  → 5 samples/model (25 total) → llama_geval_v3_pilot_results.json
  - PILOT_ONLY=False → 200 samples/model (GT500 CROSS 100 + held-out 100) → llama_geval_v3_results.json

보안: HF_TOKEN은 os.environ["HF_TOKEN"]으로만 읽음, 코드/로그에 노출 금지.
"""
import os, sys, json, re, time, torch, numpy as np
from collections import Counter
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
try:
    from scipy import stats as scipy_stats
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG  ← 파일럿 확인 후 PILOT_ONLY=False 로 변경
# ══════════════════════════════════════════════════════════════════════════
PILOT_ONLY     = True
SEEDS          = [42, 43, 44]
TEMPERATURE    = 0.3
MAX_NEW_TOKENS = 200   # reasoning 포함 ~100-150 토큰 예상, 여유 50 확보
PILOT_N        = 5     # 파일럿 시 모델당 GT500 CROSS 샘플 수

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
BASE         = "/home/sclab/moe_model"
GT_PATH      = f"{BASE}/processed_v2/gt_dataset_500.json"
HELDOUT_PATH = f"{BASE}/heldout_cross_predictions.json"
CKPT_PATH    = f"{BASE}/llama_geval_v3_checkpoint.json"
OUT_PATH     = f"{BASE}/llama_geval_v3_results.json"
PILOT_OUT    = f"{BASE}/llama_geval_v3_pilot_results.json"
LLAMA_PATH   = (
    "/home/sclab/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-70B-Instruct/snapshots/"
    "1605565b47bb9346c5515c34102e054115b4f98b"
)

EVAL_FILES = {
    "Naive RAG 7B":    f"{BASE}/baseline_results/naive_rag_7b_500.json",
    "Adaptive RAG 7B": f"{BASE}/baseline_results/adaptive_rag_7b_500.json",
    "HyDE 7B":         f"{BASE}/baseline_results/hyde_7b_500.json",
    "CRAG 7B":         f"{BASE}/baseline_results/crag_7b_500_v2.json",
    "NexRoute 7B":     f"{BASE}/eval_results_7b_v3_500.json",
}

SCORE_KEYS = ["domain_accuracy", "logic", "utility"]

# ══════════════════════════════════════════════════════════════════════════
#  PROMPT  (원 논문 §4.4 기준과 일치)
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
- Structured responses are not inherently better than well-written prose.

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


# ══════════════════════════════════════════════════════════════════════════
#  PARSE
# ══════════════════════════════════════════════════════════════════════════
def parse_scores(text):
    # 1) JSON parse (가장 신뢰할 수 있는 방법)
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

    # 2) 키워드 기반 정규식
    scores = {}
    for k in SCORE_KEYS:
        m = re.search(rf'"{k}"\s*:\s*(10|[1-9])', text)
        if m:
            scores[k] = float(m.group(1))
    if all(k in scores for k in SCORE_KEYS):
        return scores

    # 3) 최후 수단: 텍스트에서 1-10 정수 3개 추출
    nums = [int(n) for n in re.findall(r'\b(10|[1-9])\b', text) if 1 <= int(n) <= 10]
    if len(nums) >= 3:
        return {k: float(nums[i]) for i, k in enumerate(SCORE_KEYS)}

    return None


# ══════════════════════════════════════════════════════════════════════════
#  SINGLE JUDGE CALL
# ══════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════
#  AGGREGATE 3 RUNS → mean / std per criterion
# ══════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════
#  MODEL-LEVEL SUMMARY with 95% CI
# ══════════════════════════════════════════════════════════════════════════
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
        "n_samples":      n,
        "avg_mean":       round(mean_val, 3),
        "avg_std":        round(std_val, 3),
        "avg_ci95_half":  round(ci, 3),
        "avg_ci95":       f"[{round(mean_val - ci, 3):.3f}, {round(mean_val + ci, 3):.3f}]",
        **per,
    }


# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
with open(GT_PATH) as f:
    gt500 = json.load(f)
gt_map         = {d["instruction"].strip(): d for d in gt500}
cross_gt500    = [d for d in gt500 if d.get("expected_domain") == "CROSS"]
print(f"GT500 CROSS: {len(cross_gt500)}개")

heldout_preds = {}
if not PILOT_ONLY and os.path.exists(HELDOUT_PATH):
    with open(HELDOUT_PATH) as f:
        heldout_preds = json.load(f)
    print(f"Held-out 로드: {list(heldout_preds.keys())}")

# ══════════════════════════════════════════════════════════════════════════
#  LOAD LLAMA
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[*] Llama-3.1-70B-Instruct 4-bit 로드 중...")
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


# ══════════════════════════════════════════════════════════════════════════
#  ① PILOT MODE
# ══════════════════════════════════════════════════════════════════════════
if PILOT_ONLY:
    pilot_samples_gt = cross_gt500[:PILOT_N]
    print("=" * 65)
    print(f"PILOT: {PILOT_N} samples × {len(EVAL_FILES)} models × {len(SEEDS)} seeds "
          f"= {PILOT_N * len(EVAL_FILES) * len(SEEDS)} judge calls")
    print("=" * 65)

    pilot_results = {}
    t_pilot_start = time.time()

    for model_name, pred_path in EVAL_FILES.items():
        if not os.path.exists(pred_path):
            print(f"[SKIP] {model_name}: 파일 없음 ({pred_path})")
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        pred_map = {d.get("query", d.get("instruction", "")).strip():
                    d.get("pred_answer", "") for d in preds}

        print(f"\n{'─'*55}\n  {model_name}\n{'─'*55}")
        samples = []

        for i, gt_item in enumerate(pilot_samples_gt):
            q    = gt_item["instruction"].strip()
            gold = gt_item.get("gold_answer", "")
            pred = pred_map.get(q, "")
            if not pred:
                print(f"  [{i+1}] 예측 없음 (skip)")
                continue

            t0   = time.time()
            runs = []
            for seed in SEEDS:
                sc, raw = run_once(q, gold, pred, judge, tokenizer, seed)
                if sc:
                    sc["seed"]        = seed
                    sc["raw_preview"] = raw[:250]
                    runs.append(sc)

            elapsed = time.time() - t0
            agg     = aggregate_runs(runs)
            samples.append({
                "query":        q[:80],
                "response_len": len(pred),
                "runs":         runs,
                "aggregate":    agg,
            })

            da = agg.get("domain_accuracy_mean", "?")
            lo = agg.get("logic_mean", "?")
            ut = agg.get("utility_mean", "?")
            av = agg.get("avg_mean", "?")
            print(f"  [{i+1}/{PILOT_N}] DA={da}  Lo={lo}  Ut={ut}  Avg={av}"
                  f"  (len={len(pred)}ch, {elapsed:.0f}s/3runs)")

        pilot_results[model_name] = {
            "samples": samples,
            "summary": model_summary(samples),
        }

    total_pilot_time = time.time() - t_pilot_start

    # ── 파일럿 리포트 ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("PILOT REPORT")
    print("=" * 65)

    # [1] 점수 분포
    all_scores_flat = []
    for res in pilot_results.values():
        for s in res["samples"]:
            for run in s.get("runs", []):
                for k in SCORE_KEYS:
                    if k in run:
                        all_scores_flat.append(int(run[k]))
    cnt = Counter(all_scores_flat)
    total_scores = sum(cnt.values())
    print(f"\n[1] 점수 분포 (1-10, 전체 {total_scores}개 점수):")
    for sc in range(1, 11):
        n_sc = cnt.get(sc, 0)
        bar  = "█" * n_sc
        pct  = 100 * n_sc / total_scores if total_scores else 0
        print(f"    {sc:2d}: {bar:<25} {n_sc:3d}개 ({pct:4.1f}%)")
    hi = sum(cnt.get(sc, 0) for sc in [9, 10])
    print(f"    9-10 집중도: {100*hi/total_scores:.1f}% (30% 이상이면 ceiling 주의)")

    # [2] 모델 간 비교
    print(f"\n[2] 모델별 평균 (n={PILOT_N}):")
    print(f"  {'Model':<22}  {'DA':>5}  {'Logic':>5}  {'Util':>5}  {'Avg':>6}  {'AvgLen':>7}")
    print(f"  {'─'*22}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*7}")
    model_avgs = []
    for mn, res in pilot_results.items():
        sm      = res["summary"]
        avg_len = np.mean([s["response_len"] for s in res["samples"]]) if res["samples"] else 0
        da_m    = sm.get("domain_accuracy_mean", 0)
        lo_m    = sm.get("logic_mean", 0)
        ut_m    = sm.get("utility_mean", 0)
        av_m    = sm.get("avg_mean", 0)
        model_avgs.append(av_m)
        print(f"  {mn:<22}  {da_m:>5.2f}  {lo_m:>5.2f}  {ut_m:>5.2f}  {av_m:>6.3f}  {avg_len:>7.0f}")

    if model_avgs:
        spread = max(model_avgs) - min(model_avgs)
        print(f"\n  모델 간 최대 차이: {spread:.3f}점 (기준: ≥0.3점이면 유의미)")

    # [3] 반복 채점 안정성
    all_stds = [s["aggregate"].get("avg_std", 0)
                for res in pilot_results.values()
                for s in res["samples"]
                if "avg_std" in s.get("aggregate", {})]
    if all_stds:
        print(f"\n[3] 반복 채점 안정성 (3-run avg_std):")
        print(f"    평균 std = {np.mean(all_stds):.3f}")
        print(f"    최대 std = {max(all_stds):.3f}")
        print(f"    (std < 0.5이면 안정적)")

    # [4] Verbosity bias
    lens_v, avgs_v = [], []
    for res in pilot_results.values():
        for s in res["samples"]:
            if "response_len" in s and "avg_mean" in s.get("aggregate", {}):
                lens_v.append(s["response_len"])
                avgs_v.append(s["aggregate"]["avg_mean"])
    if len(lens_v) >= 5 and HAS_SCIPY:
        rho, pval = spearmanr(lens_v, avgs_v)
        print(f"\n[4] Verbosity bias (Spearman ρ 길이 vs 점수):")
        print(f"    ρ = {rho:.3f}, p = {pval:.3f}")
        if rho > 0.3 and pval < 0.1:
            print("    ⚠️  양의 상관 → verbosity bias 가능성")
        else:
            print("    ✅  길이-점수 상관 낮음 → verbosity bias 완화 확인")

    # [5] 소요 시간 + 전체 예상
    calls_done     = PILOT_N * len(pilot_results) * len(SEEDS)
    sec_per_call   = total_pilot_time / calls_done if calls_done > 0 else 0
    full_calls     = len(EVAL_FILES) * 200 * len(SEEDS)
    est_full_hours = full_calls * sec_per_call / 3600
    print(f"\n[5] 소요 시간:")
    print(f"    파일럿 실측: {total_pilot_time:.0f}초 ({total_pilot_time/60:.1f}분)")
    print(f"    judge 호출당: {sec_per_call:.1f}초")
    print(f"    전체 예상 ({full_calls} calls): {est_full_hours:.1f}시간")

    # 저장
    with open(PILOT_OUT, "w") as f:
        json.dump({
            "pilot_results": pilot_results,
            "meta": {
                "mode":         "pilot",
                "n_per_model":  PILOT_N,
                "seeds":        SEEDS,
                "temperature":  TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "criteria":     SCORE_KEYS,
                "scale":        "1-10 integer",
                "sec_per_call": round(sec_per_call, 2),
                "est_full_run_hours": round(est_full_hours, 1),
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n파일럿 결과 저장: {PILOT_OUT}")
    print("\n── 전체 실행 전 위 결과를 확인하고 PILOT_ONLY=False로 변경해주세요 ──")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════
#  ② FULL RUN
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(f"FULL RUN: {len(EVAL_FILES)} models × 200 samples × {len(SEEDS)} seeds")
print("=" * 65)

all_results = {}
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH) as f:
        existing_ckpt = json.load(f)
    if existing_ckpt:
        all_results = existing_ckpt
        print(f"체크포인트 로드: {list(all_results.keys())}")

def save_ckpt():
    with open(CKPT_PATH, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

for model_name, pred_path in EVAL_FILES.items():
    existing_model = all_results.get(model_name, {})
    n_done = existing_model.get("n_total", 0)
    if n_done >= 200:
        print(f"[SKIP] {model_name} (완료 {n_done}개)")
        continue
    if not os.path.exists(pred_path):
        print(f"[SKIP] {model_name}: 파일 없음")
        continue

    print(f"\n{'='*55}\n{model_name}\n{'='*55}")
    with open(pred_path) as f:
        preds = json.load(f)
    pred_map = {d.get("query", d.get("instruction", "")).strip():
                d.get("pred_answer", "") for d in preds}

    gt_cross_scores  = existing_model.get("gt500_cross",   [])
    heldout_scores   = existing_model.get("heldout_cross", [])
    gt_done_set      = {s["query"] for s in gt_cross_scores}
    heldout_done_set = {s["query"] for s in heldout_scores}

    # ── Part A: GT500 CROSS 100개 ──────────────────────────────────────
    n_already = len(gt_cross_scores)
    print(f"[Part A] GT500 CROSS 100개 (이미 {n_already}개 완료)")
    for gt_item in cross_gt500:
        q = gt_item["instruction"].strip()
        if q[:80] in gt_done_set:
            continue
        gold = gt_item.get("gold_answer", "")
        pred = pred_map.get(q, "")
        if not pred:
            continue

        runs = []
        for seed in SEEDS:
            sc, _ = run_once(q, gold, pred, judge, tokenizer, seed)
            if sc:
                sc["seed"] = seed
                runs.append(sc)

        agg = aggregate_runs(runs)
        gt_cross_scores.append({
            "query":        q[:80],
            "source":       "gt500_cross",
            "response_len": len(pred),
            "runs":         runs,
            "aggregate":    agg,
        })
        gt_done_set.add(q[:80])

        n = len(gt_cross_scores)
        if n % 10 == 0:
            all_results[model_name] = {
                "gt500_cross":   gt_cross_scores,
                "heldout_cross": heldout_scores,
                "n_total": n + len(heldout_scores),
            }
            save_ckpt()
            avg = np.mean([s["aggregate"].get("avg_mean", 0) for s in gt_cross_scores])
            print(f"  [A {n}/100] avg={avg:.3f}")

    # ── Part B: Held-out CROSS 100개 ──────────────────────────────────
    heldout_model = heldout_preds.get(model_name, [])
    n_already_h   = len(heldout_scores)
    print(f"[Part B] Held-out {len(heldout_model)}개 (이미 {n_already_h}개 완료)")
    for item in heldout_model:
        q = item.get("query", "").strip()
        if q[:80] in heldout_done_set:
            continue
        gold = item.get("gold_answer", "")
        pred = item.get("pred_answer", "")
        if not pred:
            continue

        runs = []
        for seed in SEEDS:
            sc, _ = run_once(q, gold, pred, judge, tokenizer, seed)
            if sc:
                sc["seed"] = seed
                runs.append(sc)

        agg = aggregate_runs(runs)
        heldout_scores.append({
            "query":        q[:80],
            "source":       "heldout_cross",
            "response_len": len(pred),
            "cross_pair":   item.get("cross_pair", ""),
            "runs":         runs,
            "aggregate":    agg,
        })
        heldout_done_set.add(q[:80])

        n = len(heldout_scores)
        if n % 10 == 0:
            all_results[model_name] = {
                "gt500_cross":   gt_cross_scores,
                "heldout_cross": heldout_scores,
                "n_total": len(gt_cross_scores) + n,
            }
            save_ckpt()
            avg = np.mean([s["aggregate"].get("avg_mean", 0) for s in heldout_scores])
            print(f"  [B {n}/{len(heldout_model)}] avg={avg:.3f}")

    # ── 요약 ───────────────────────────────────────────────────────────
    combined = gt_cross_scores + heldout_scores
    all_results[model_name] = {
        "gt500_cross":           gt_cross_scores,
        "heldout_cross":         heldout_scores,
        "gt500_cross_summary":   model_summary(gt_cross_scores),
        "heldout_cross_summary": model_summary(heldout_scores),
        "combined_summary":      model_summary(combined),
        "n_gt500":   len(gt_cross_scores),
        "n_heldout": len(heldout_scores),
        "n_total":   len(combined),
    }
    save_ckpt()

    sm = all_results[model_name]["combined_summary"]
    print(f"\n  Combined (n={len(combined)}): "
          f"DA={sm.get('domain_accuracy_mean',0):.3f}  "
          f"Logic={sm.get('logic_mean',0):.3f}  "
          f"Util={sm.get('utility_mean',0):.3f}  "
          f"Avg={sm.get('avg_mean',0):.3f}  CI95={sm.get('avg_ci95','?')}")

# ── 최종 테이블 ─────────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print(f"{'Model':<22}  {'DA':>5}  {'Logic':>5}  {'Util':>5}  {'Avg':>6}  "
      f"{'CI95':>22}  {'n':>4}")
print("-" * 80)
for mn, res in all_results.items():
    sm = res.get("combined_summary", {})
    print(f"  {mn:<22}  "
          f"{sm.get('domain_accuracy_mean', 0):>5.2f}  "
          f"{sm.get('logic_mean', 0):>5.2f}  "
          f"{sm.get('utility_mean', 0):>5.2f}  "
          f"{sm.get('avg_mean', 0):>6.3f}  "
          f"{sm.get('avg_ci95', '[?,?]'):>22}  "
          f"{res.get('n_total', 0):>4}")

# ── 최종 저장 ────────────────────────────────────────────────────────────
final = {
    "llama_geval_v3": all_results,
    "meta": {
        "version":        "v3",
        "judge":          "meta-llama/Llama-3.1-70B-Instruct",
        "criteria":       SCORE_KEYS,
        "scale":          "1-10 integer",
        "seeds":          SEEDS,
        "temperature":    TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_per_model":    "100 (gt500_cross) + 100 (heldout_cross)",
        "changes_from_v2": [
            "criteria: Relevance/Coherence/Factuality/DomainExpertise → DomainAccuracy/Logic/Utility",
            "scale: 1-5 → 1-10",
            "repeats: 1 → 3 (seeds 42,43,44)",
            "temperature: 0.1 → 0.3",
            "verbosity-agnostic rubric: length NOT a proxy for quality",
        ],
    }
}
with open(OUT_PATH, "w") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print(f"\n저장 완료: {OUT_PATH}")
