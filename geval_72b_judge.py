"""
G-Eval: LLM-as-Judge evaluation using Qwen2.5-72B-Instruct
Criteria: Relevance, Coherence, Factuality, Domain Expertise (1-5 each)
Models evaluated: Naive/Adaptive/HyDE/CRAG 7B, NexRoute 7B,
                  Naive/Adaptive/HyDE/CRAG 72B, Adaptive TFIDF 7B
"""
import json, torch, glob, re, os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

GT_PATH  = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_PATH = "/home/sclab/moe_model/geval_results.json"

_cands = glob.glob(
    "/home/sclab/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-72B-Instruct/snapshots/*"
)
BASE_72B = _cands[0] if _cands else "Qwen/Qwen2.5-72B-Instruct"

EVAL_FILES = {
    "Naive RAG 7B":         "/home/sclab/moe_model/baseline_results/naive_rag_7b_500.json",
    "Adaptive RAG 7B":      "/home/sclab/moe_model/baseline_results/adaptive_rag_7b_500.json",
    "Adaptive RAG TFIDF 7B":"/home/sclab/moe_model/baseline_results/adaptive_rag_tfidf_7b_500.json",
    "HyDE 7B":              "/home/sclab/moe_model/baseline_results/hyde_7b_500.json",
    "CRAG 7B":              "/home/sclab/moe_model/baseline_results/crag_7b_500.json",
    "NexRoute 7B":          "/home/sclab/moe_model/eval_results_7b_v3_500.json",
    "Naive RAG 72B":        "/home/sclab/moe_model/baseline_results/naive_rag_72b_500_v2.json",
    "Adaptive RAG 72B":     "/home/sclab/moe_model/baseline_results/adaptive_rag_72b_500_v2.json",
    "HyDE 72B":             "/home/sclab/moe_model/baseline_results/hyde_72b_500_v2.json",
    "CRAG 72B":             "/home/sclab/moe_model/baseline_results/crag_72b_500_v2.json",
    "NexRoute 72B":         "/home/sclab/moe_model/baseline_results/nexroute_72b_500.json",
}

EVAL_PROMPT = """You are an expert evaluator assessing the quality of AI-generated responses.

Question: {question}

Reference Answer: {reference}

Model Response: {response}

Evaluate the model response on the following criteria (score 1-5):
1. Relevance: Does the response directly address the question? (1=off-topic, 5=perfectly on-topic)
2. Coherence: Is the response logically structured and easy to follow? (1=incoherent, 5=excellent structure)
3. Factuality: Is the content accurate and consistent with the reference? (1=many errors, 5=fully accurate)
4. Domain Expertise: Does the response demonstrate appropriate domain knowledge? (1=superficial, 5=expert-level)

Respond ONLY in this exact JSON format:
{{"relevance": X, "coherence": X, "factuality": X, "domain_expertise": X}}

where X is an integer 1-5."""


def parse_scores(text):
    """JSON에서 점수 파싱"""
    try:
        # JSON 블록 찾기
        match = re.search(r'\{[^}]+\}', text)
        if match:
            scores = json.loads(match.group())
            keys = ['relevance', 'coherence', 'factuality', 'domain_expertise']
            if all(k in scores for k in keys):
                return {k: float(scores[k]) for k in keys}
    except:
        pass
    # fallback: 숫자 추출
    nums = re.findall(r'\b[1-5]\b', text)
    if len(nums) >= 4:
        keys = ['relevance', 'coherence', 'factuality', 'domain_expertise']
        return {k: float(n) for k, n in zip(keys, nums[:4])}
    return None


def evaluate_response(question, reference, response, model, tokenizer):
    prompt = EVAL_PROMPT.format(
        question=question[:500],
        reference=reference[:500],
        response=response[:500]
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    first_device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(first_device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=100,
            temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True).strip()
    return parse_scores(result)


# ── GT500 로드 ────────────────────────────────────────────────
with open(GT_PATH) as f:
    gt_data = json.load(f)
text_map = {d["instruction"].strip(): d for d in gt_data}

# ── 72B Judge 로드 ────────────────────────────────────────────
print(f"[*] Judge 모델 로드: {BASE_72B}")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_72B, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
judge = AutoModelForCausalLM.from_pretrained(
    BASE_72B, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager"
)
print("[*] Judge 로드 완료!\n")

# ── 기존 결과 로드 (재시작 대비) ─────────────────────────────
all_results = {}
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        all_results = json.load(f)
    print(f"[*] 기존 결과 로드: {list(all_results.keys())}")

# ── 평가 실행 ─────────────────────────────────────────────────
N_SAMPLE = 100  # 모델당 100개 (CROSS 50 + Single 50)

for model_name, path in EVAL_FILES.items():
    if model_name in all_results:
        print(f"[SKIP] {model_name} (이미 완료)")
        continue
    if not os.path.exists(path):
        print(f"[SKIP] {model_name}: 파일 없음")
        continue

    print(f"\n{'='*50}\n{model_name}\n{'='*50}")
    with open(path) as f:
        data = json.load(f)

    # CROSS 50개 + Single 50개 샘플링
    cross_samples, single_samples = [], []
    for item in data:
        q = item.get("query", item.get("instruction", "")).strip()
        gt_item = text_map.get(q)
        if not gt_item: continue
        domain = gt_item.get("expected_domain", "")
        if domain == "CROSS" and len(cross_samples) < 50:
            cross_samples.append(item)
        elif domain != "CROSS" and len(single_samples) < 50:
            single_samples.append(item)
        if len(cross_samples) == 50 and len(single_samples) == 50:
            break

    samples = cross_samples + single_samples
    model_results = {"cross": [], "single": []}

    for i, item in enumerate(samples):
        q    = item.get("query", item.get("instruction", "")).strip()
        pred = item.get("pred_answer", "").strip()
        gt_item = text_map.get(q)
        if not gt_item or not pred: continue
        gold   = gt_item.get("gold_answer", "")
        domain = gt_item.get("expected_domain", "")
        split  = "cross" if domain == "CROSS" else "single"

        scores = evaluate_response(q, gold, pred, judge, tokenizer)
        if scores:
            scores["query"]  = q[:80]
            scores["domain"] = domain
            scores["avg"]    = np.mean(list(scores.values())[:4])
            model_results[split].append(scores)

        if (i + 1) % 10 == 0:
            c_avg = np.mean([s["avg"] for s in model_results["cross"]]) if model_results["cross"] else 0
            s_avg = np.mean([s["avg"] for s in model_results["single"]]) if model_results["single"] else 0
            print(f"  [{i+1}/{len(samples)}] CROSS avg={c_avg:.2f}, Single avg={s_avg:.2f}")

    # 요약
    def summarize(scores_list):
        if not scores_list: return {}
        keys = ['relevance','coherence','factuality','domain_expertise','avg']
        return {k: round(np.mean([s[k] for s in scores_list]), 3) for k in keys if k in scores_list[0]}

    model_results["cross_summary"]  = summarize(model_results["cross"])
    model_results["single_summary"] = summarize(model_results["single"])
    all_results[model_name] = model_results

    c = model_results["cross_summary"]
    s = model_results["single_summary"]
    print(f"\n  CROSS  (N={len(model_results['cross'])}):  avg={c.get('avg',0):.2f}")
    print(f"  Single (N={len(model_results['single'])}): avg={s.get('avg',0):.2f}")

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  [저장] {OUT_PATH}")

# ── 최종 테이블 출력 ──────────────────────────────────────────
print("\n\n" + "="*70)
print(f"{'Model':<25} {'CROSS G-Eval':>13} {'Single G-Eval':>14}")
print("-"*70)
for name, res in all_results.items():
    c = res.get("cross_summary", {}).get("avg", 0)
    s = res.get("single_summary", {}).get("avg", 0)
    print(f"{name:<25} {c:>13.2f} {s:>14.2f}")

with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print("\n최종 저장 완료!")