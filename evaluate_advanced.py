"""
evaluate_advanced.py  ·  Dual-MoE RAG 고급 평가 지표

포함 지표:
  1. MoE Routing  : Routing Accuracy / Expert Load Balance / Routing Entropy
  2. RAG Quality  : Context Precision / Context Recall / Faithfulness / Answer Relevance
  3. LLM-as-Judge : Coherence / Fluency / Factualness / Relevance  (Qwen 72B 로컬)
  4. Efficiency   : Inference Latency (ms/query) — 모델 로딩 제외, warm-up 포함

사전 조건:
  - dual_moe_predictions.json 항목 구조:
      {
        "answer":       "...",
        "pred_domains": ["MED", "LAW"],        # 활성 도메인 목록
        "domain_scores": {"MED":0.6, "LAW":0.3, "STEM":0.07, "SOC":0.03},
        "contexts":     ["retrieved chunk 1", "retrieved chunk 2", ...]
      }
  - pipeline.py run() 반환값에 위 필드들이 있어야 함
  - pip install scipy numpy bert_score rouge_score sentence-transformers
"""

import os, json, time, math, statistics, random, re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
RESULTS_DIR     = "/home/sclab/moe_model/results"
QUERYSET_PATH   = "/home/sclab/moe_model/queryset/groundtruth.json"
BASE_MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"

ALL_DOMAINS = ["MED", "LAW", "STEM", "SOC"]   # 실제 도메인 목록에 맞게 수정


# ════════════════════════════════════════════════════════════════════════════════
# 0. 데이터 로드
# ════════════════════════════════════════════════════════════════════════════════

def load_data():
    with open(QUERYSET_PATH) as f:
        raw = json.load(f)
    queries       = [d["query"]        for d in raw]
    ground_truths = [d["ground_truth"] for d in raw]
    domain_types  = [d["domain_type"]  for d in raw]
    gt_domains    = [d["domains"]      for d in raw]   # e.g. ["MED"] or ["MED","LAW"]
    return queries, ground_truths, domain_types, gt_domains


def load_predictions(model_name: str) -> list:
    path = os.path.join(RESULTS_DIR, f"{model_name}_predictions.json")
    with open(path) as f:
        data = json.load(f)
    # 구형 포맷(str 리스트)을 dict 리스트로 통일
    return [{"answer": p} if isinstance(p, str) else p for p in data]


def _answer(pred: dict) -> str:
    return pred.get("answer", "")


# ════════════════════════════════════════════════════════════════════════════════
# 1. MoE Routing 지표
# ════════════════════════════════════════════════════════════════════════════════

# ── 1-A. Routing Accuracy ────────────────────────────────────────────────────
def compute_routing_accuracy(preds: list, gt_domains: list) -> dict:
    """
    GT 도메인이 pred_domains 에 포함되었는지를 도메인별로 집계.
    - overall  : 전체 (도메인 단위) 적중률
    - per_domain : 도메인별 적중률
    """
    correct, total = 0, 0
    per = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, gts in zip(preds, gt_domains):
        pd = set(pred.get("pred_domains", []))
        for gt in gts:
            per[gt]["total"] += 1
            total += 1
            if gt in pd:
                per[gt]["correct"] += 1
                correct += 1

    overall = correct / total if total else 0
    per_acc = {d: (v["correct"] / v["total"] if v["total"] else 0)
               for d, v in per.items()}
    return {
        "overall":    round(overall, 4),
        "per_domain": {d: round(v, 4) for d, v in per_acc.items()},
        "n_evaluated": total,
    }


# ── 1-B. Expert Load Balance ─────────────────────────────────────────────────
def compute_load_balance(preds: list) -> dict:
    """
    활성 도메인 선택 빈도 분포 측정.
    - CV (Coefficient of Variation) : 낮을수록 균형
    - Load Imbalance (max − min)    : 낮을수록 균형
    - KL Divergence from Uniform    : 낮을수록 균형
    """
    counts = defaultdict(int)
    total  = 0
    for pred in preds:
        for d in pred.get("pred_domains", []):
            counts[d] += 1
            total += 1

    if total == 0:
        return {"note": "pred_domains 필드 없음"}

    freqs  = [counts.get(d, 0) / total for d in ALL_DOMAINS]
    ideal  = 1.0 / len(ALL_DOMAINS)

    # CV
    mean_f = statistics.mean(freqs)
    cv     = statistics.stdev(freqs) / (mean_f + 1e-9)

    # KL divergence: KL(actual || uniform)
    kl = sum(p * math.log((p + 1e-9) / ideal) for p in freqs)

    return {
        "distribution":             {d: round(f, 4) for d, f in zip(ALL_DOMAINS, freqs)},
        "ideal_uniform":            round(ideal, 4),
        "coefficient_of_variation": round(cv, 4),
        "load_imbalance":           round(max(freqs) - min(freqs), 4),
        "kl_from_uniform":          round(kl, 4),
        "note": "CV·load_imbalance·KL 모두 낮을수록 균형 잡힌 라우팅",
    }


# ── 1-C. Routing Entropy ─────────────────────────────────────────────────────
def compute_routing_entropy(preds: list, domain_types: list) -> dict:
    """
    Gating 확률 분포의 Shannon Entropy.
    H = -Σ p_i log(p_i),  max = log(|domains|) ≈ 1.386 (4개 균등)

    cross-domain 쿼리는 높은 엔트로피가, single-domain 은 낮은 엔트로피가 정상.
    두 그룹을 분리해 평균 비교.
    """
    def entropy(scores: dict) -> float:
        vals  = list(scores.values())
        total = sum(vals)
        if total == 0:
            return 0.0
        probs = [v / total for v in vals]
        return -sum(p * math.log(p + 1e-9) for p in probs)

    single_ent, cross_ent, all_ent = [], [], []
    for pred, dtype in zip(preds, domain_types):
        ds = pred.get("domain_scores", {})
        if not ds:
            continue
        h = entropy(ds)
        all_ent.append(h)
        (cross_ent if dtype == "cross" else single_ent).append(h)

    def stats(lst):
        if not lst:
            return {}
        return {
            "mean": round(statistics.mean(lst), 4),
            "std":  round(statistics.stdev(lst) if len(lst) > 1 else 0, 4),
            "min":  round(min(lst), 4),
            "max":  round(max(lst), 4),
        }

    return {
        "overall":        stats(all_ent),
        "single_domain":  stats(single_ent),
        "cross_domain":   stats(cross_ent),
        "max_theoretical": round(math.log(len(ALL_DOMAINS)), 4),
        "note": "cross 엔트로피 > single 엔트로피면 라우터가 도메인 특성을 잘 반영",
    }


def moe_routing_report(preds, gt_domains, domain_types):
    print("\n" + "=" * 60)
    print("1. MoE Routing 지표")
    print("=" * 60)

    acc = compute_routing_accuracy(preds, gt_domains)
    print(f"\n[Routing Accuracy]  (n={acc['n_evaluated']})")
    print(f"  Overall : {acc['overall']:.4f}")
    for d, v in acc["per_domain"].items():
        print(f"  {d:8s}: {v:.4f}")

    lb = compute_load_balance(preds)
    print(f"\n[Expert Load Balance]")
    print(f"  Distribution : {lb.get('distribution', {})}")
    print(f"  CV           : {lb.get('coefficient_of_variation', 'N/A')}")
    print(f"  Load Imbal.  : {lb.get('load_imbalance', 'N/A')}")
    print(f"  KL(uniform)  : {lb.get('kl_from_uniform', 'N/A')}")

    ent = compute_routing_entropy(preds, domain_types)
    print(f"\n[Routing Entropy]  (max={ent['max_theoretical']})")
    for grp in ("overall", "single_domain", "cross_domain"):
        s = ent.get(grp, {})
        if s:
            print(f"  {grp:15s}: mean={s['mean']:.4f}  std={s['std']:.4f}")

    return {"routing_accuracy": acc, "load_balance": lb, "routing_entropy": ent}


# ════════════════════════════════════════════════════════════════════════════════
# 2. RAG 특화 지표
# ════════════════════════════════════════════════════════════════════════════════

def _token_set(text: str) -> set:
    return set(text.lower().split())


# ── 2-A. Context Precision & Recall ─────────────────────────────────────────
def compute_context_precision_recall(preds: list, ground_truths: list,
                                     relevance_threshold: float = 0.15) -> dict:
    """
    Precision : retrieved contexts 중 GT와 겹치는 비율
    Recall    : GT 핵심 토큰 중 contexts에서 커버된 비율

    relevance_threshold : context-GT 사이 token overlap 비율이 이 값 이상이면 '관련'으로 판정
    """
    precisions, recalls, f1s = [], [], []

    for pred, gt in zip(preds, ground_truths):
        contexts = pred.get("contexts", [])
        if not contexts:
            continue

        gt_tokens = _token_set(gt)
        if not gt_tokens:
            continue

        # context별 관련성 판정
        relevant_flags = []
        covered_tokens = set()
        for ctx in contexts:
            ctx_tokens = _token_set(ctx)
            overlap    = len(gt_tokens & ctx_tokens) / len(gt_tokens)
            is_rel     = overlap >= relevance_threshold
            relevant_flags.append(is_rel)
            if is_rel:
                covered_tokens |= (gt_tokens & ctx_tokens)

        n_rel     = sum(relevant_flags)
        precision = n_rel / len(contexts)
        recall    = len(covered_tokens) / len(gt_tokens)
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    if not precisions:
        return {"note": "contexts 필드 없음 — pipeline.py 에 contexts 저장 필요"}

    return {
        "context_precision": round(statistics.mean(precisions), 4),
        "context_recall":    round(statistics.mean(recalls),    4),
        "context_f1":        round(statistics.mean(f1s),        4),
        "n_evaluated":       len(precisions),
        "relevance_threshold": relevance_threshold,
    }


# ── 2-B. Faithfulness ────────────────────────────────────────────────────────
def compute_faithfulness(preds: list) -> dict:
    """
    답변 토큰 중 retrieved contexts 에서 찾을 수 있는 비율.
    1.0 에 가까울수록 Hallucination 이 적음 (context-grounded 생성).

    개선 포인트:
      - stopword 제거로 의미 있는 토큰만 비교
      - 단순 vocab 포함 여부 → context 전체 concat 에서 n-gram 검색으로 강화
    """
    # 간단한 영어 stopword (필요 시 nltk 사용)
    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","of","in","to","for",
        "on","at","by","with","from","as","or","and","but","not","it",
        "that","this","these","those","i","we","you","he","she","they",
    }

    scores = []
    for pred in preds:
        answer   = _answer(pred)
        contexts = pred.get("contexts", [])
        if not contexts or not answer:
            continue

        context_vocab = set()
        for ctx in contexts:
            context_vocab |= _token_set(ctx)

        answer_tokens = [t for t in answer.lower().split() if t not in STOPWORDS]
        if not answer_tokens:
            continue

        covered = sum(1 for t in answer_tokens if t in context_vocab)
        scores.append(covered / len(answer_tokens))

    if not scores:
        return {"note": "contexts 필드 없음 — pipeline.py 수정 필요"}

    return {
        "faithfulness":  round(statistics.mean(scores), 4),
        "std":           round(statistics.stdev(scores) if len(scores) > 1 else 0, 4),
        "n_evaluated":   len(scores),
        "note": "1.0 에 가까울수록 context 기반 생성 (hallucination 낮음)",
    }


# ── 2-C. Answer Relevance (query-answer cosine similarity) ───────────────────
def compute_answer_relevance(preds: list, queries: list) -> dict:
    """
    Query 와 Answer 사이 임베딩 코사인 유사도.
    높을수록 질문 의도에 부합하는 답변.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return {"note": "sentence-transformers / scikit-learn 미설치"}

    model = SentenceTransformer("all-MiniLM-L6-v2")

    q_texts = []
    a_texts = []
    for pred, q in zip(preds, queries):
        ans = _answer(pred)
        if ans:
            q_texts.append(q)
            a_texts.append(ans)

    if not q_texts:
        return {"note": "답변이 없음"}

    q_embs = model.encode(q_texts, batch_size=64, show_progress_bar=False)
    a_embs = model.encode(a_texts, batch_size=64, show_progress_bar=False)

    sims = [float(cosine_similarity([q], [a])[0][0])
            for q, a in zip(q_embs, a_embs)]

    return {
        "answer_relevance": round(statistics.mean(sims), 4),
        "std":              round(statistics.stdev(sims) if len(sims) > 1 else 0, 4),
        "n_evaluated":      len(sims),
        "note": "query-answer cosine similarity (higher = more relevant)",
    }


def rag_metrics_report(preds, ground_truths, queries):
    print("\n" + "=" * 60)
    print("2. RAG 특화 지표")
    print("=" * 60)

    cp = compute_context_precision_recall(preds, ground_truths)
    print(f"\n[Context Precision / Recall / F1]")
    for k, v in cp.items():
        print(f"  {k}: {v}")

    faith = compute_faithfulness(preds)
    print(f"\n[Faithfulness]")
    for k, v in faith.items():
        print(f"  {k}: {v}")

    ar = compute_answer_relevance(preds, queries)
    print(f"\n[Answer Relevance]")
    for k, v in ar.items():
        print(f"  {k}: {v}")

    return {"context_metrics": cp, "faithfulness": faith, "answer_relevance": ar}


# ════════════════════════════════════════════════════════════════════════════════
# 3. LLM-as-Judge (로컬 Qwen 72B)
# ════════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """You are an expert evaluator for RAG system outputs.
Score the model answer on the four criteria below using a 1–5 integer scale.
Respond ONLY with valid JSON — no explanation, no markdown, no extra text.

Criteria:
- coherence   : Is the answer logically consistent and well-structured?
- fluency     : Is the answer natural and easy to read?
- factualness : Does the answer match the reference answer factually?
- relevance   : Does the answer address the intent of the question?

Question     : {query}
Reference    : {ground_truth}
Model Answer : {answer}

JSON format (integers 1-5 only):
{{"coherence": <score>, "fluency": <score>, "factualness": <score>, "relevance": <score>}}"""

JUDGE_CRITERIA = ["coherence", "fluency", "factualness", "relevance"]


def _parse_judge_output(text: str) -> Optional[Dict]:
    """JSON 파싱 — 마크다운 fence 등 노이즈 제거 후 시도."""
    text = re.sub(r"```[a-z]*", "", text).strip()
    m = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group())
        # 값이 모두 1~5 정수인지 확인
        if all(isinstance(parsed.get(k), (int, float)) and 1 <= parsed[k] <= 5
               for k in JUDGE_CRITERIA):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def llm_judge_batch(queries, ground_truths, preds,
                    sample_n: int = 30, seed: int = 42) -> dict:
    """
    Qwen 72B (4-bit) 로 LLM-as-Judge.
    - 전체가 아닌 sample_n 개만 평가 (비용/시간 절감)
    - 파싱 실패 시 최대 1회 재시도
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    random.seed(seed)
    indices = random.sample(range(len(queries)), min(sample_n, len(queries)))

    print(f"\n  Qwen 72B (4-bit) 로드 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model     = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print(f"  로드 완료. {len(indices)}개 쿼리 평가 시작...")

    def infer(prompt_text: str) -> str:
        messages = [{"role": "user", "content": prompt_text}]
        text     = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=False,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    per_query = []
    for idx in indices:
        q   = queries[idx]
        gt  = ground_truths[idx]
        ans = _answer(preds[idx])

        prompt  = JUDGE_PROMPT.format(query=q, ground_truth=gt, answer=ans)
        decoded = infer(prompt)
        scores  = _parse_judge_output(decoded)

        # 파싱 실패 시 1회 재시도 (프롬프트 앞에 힌트 추가)
        if scores is None:
            retry_prompt = prompt + "\nRemember: reply ONLY with the JSON object."
            decoded      = infer(retry_prompt)
            scores       = _parse_judge_output(decoded)

        if scores is None:
            scores = {k: None for k in JUDGE_CRITERIA}

        scores["query_idx"] = idx
        per_query.append(scores)
        print(f"    [{idx:3d}] {scores}")

    # 모델 해제 (다음 latency 측정 시 GPU 확보)
    del model
    torch.cuda.empty_cache()

    valid = [r for r in per_query if r.get("coherence") is not None]
    if not valid:
        return {"error": "모든 샘플 파싱 실패", "per_query": per_query}

    aggregated = {}
    for k in JUDGE_CRITERIA:
        vals = [r[k] for r in valid if isinstance(r.get(k), (int, float))]
        aggregated[k] = round(statistics.mean(vals), 3) if vals else None

    aggregated["sample_n"]      = len(valid)
    aggregated["parse_fail_n"]  = len(per_query) - len(valid)
    aggregated["scale"]         = "1-5"

    return {"per_query": per_query, "aggregated": aggregated}


def llm_judge_report(queries, ground_truths, preds, sample_n=30):
    print("\n" + "=" * 60)
    print("3. LLM-as-Judge (Qwen 72B)")
    print("=" * 60)
    result = llm_judge_batch(queries, ground_truths, preds, sample_n)
    agg    = result.get("aggregated", {})
    print(f"\n  평가 샘플 : {agg.get('sample_n', 0)}  "
          f"(파싱 실패 {agg.get('parse_fail_n', 0)}개)")
    for k in JUDGE_CRITERIA:
        print(f"  {k:12s}: {agg.get(k, 'N/A')} / 5")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# 4. Efficiency (Inference Latency)
# ════════════════════════════════════════════════════════════════════════════════

def _load_model_instance(model_name: str):
    """모델 인스턴스 반환 (로딩 시간은 latency 에서 제외)."""
    if model_name == "naive_rag":
        from naive_rag import NaiveRAG
        m = NaiveRAG(); m.load_embed_model(); m.load_model(); m.build_db()
    elif model_name == "self_rag":
        from self_rag import SelfRAG
        m = SelfRAG(); m.load_embed_model(); m.load_model(); m.build_db()
    elif model_name == "adaptive_rag":
        from adaptive_rag import AdaptiveRAG
        m = AdaptiveRAG(); m.load_embed_model(); m.load_model(); m.build_db()
    elif model_name == "single_lora_rag":
        from single_lora_rag import SingleLoRARAG
        m = SingleLoRARAG(); m.load_embed_model(); m.load_model(); m.build_db()
    elif model_name == "dual_moe":
        from pipeline import DualMoEPipeline
        m = DualMoEPipeline()
        m.load_func_classifier(); m.load_embed_model()
        m.load_base_model(); m.build_db()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m


def benchmark_latency(model_name: str, queries: list,
                      n_warmup: int = 3, n_measure: int = 20,
                      seed: int = 42) -> dict:
    """
    추론 레이턴시 측정.
    - 모델 로딩 시간 제외 (별도 측정)
    - warm-up n_warmup 회 실행 후 n_measure 회 측정
    - 단위: ms / query
    """
    random.seed(seed)
    sample = random.sample(queries, min(n_warmup + n_measure, len(queries)))
    warmup_qs  = sample[:n_warmup]
    measure_qs = sample[n_warmup:n_warmup + n_measure]

    # 로딩 시간 별도 측정
    t_load_start = time.perf_counter()
    m = _load_model_instance(model_name)
    load_ms = (time.perf_counter() - t_load_start) * 1000

    # Warm-up (GPU cache / JIT 안정화)
    print(f"    warm-up {n_warmup}회...")
    for q in warmup_qs:
        m.run(q)

    # 실제 측정
    latencies = []
    for q in measure_qs:
        t0 = time.perf_counter()
        m.run(q)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p90_idx = max(0, int(len(latencies) * 0.9) - 1)

    return {
        "model":              model_name,
        "n_warmup":           n_warmup,
        "n_measured":         len(latencies),
        "model_load_ms":      round(load_ms, 1),
        "mean_ms":            round(statistics.mean(latencies), 1),
        "median_ms":          round(statistics.median(latencies), 1),
        "p90_ms":             round(latencies[p90_idx], 1),
        "std_ms":             round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 1),
        "min_ms":             round(min(latencies), 1),
        "max_ms":             round(max(latencies), 1),
        "note": "모델 로딩 제외, warm-up 후 순수 추론 latency",
    }


def efficiency_report(queries, models=None, n_warmup=3, n_measure=20):
    if models is None:
        models = ["naive_rag", "adaptive_rag", "dual_moe"]

    print("\n" + "=" * 60)
    print("4. Efficiency 지표 (Inference Latency)")
    print("=" * 60)

    results = {}
    for model_name in models:
        print(f"\n  [{model_name}]  warm-up={n_warmup}  measure={n_measure}")
        try:
            r = benchmark_latency(model_name, queries, n_warmup, n_measure)
            results[model_name] = r
            print(f"    load   : {r['model_load_ms']:8.1f} ms")
            print(f"    mean   : {r['mean_ms']:8.1f} ms/query")
            print(f"    median : {r['median_ms']:8.1f} ms/query")
            print(f"    p90    : {r['p90_ms']:8.1f} ms/query")
            print(f"    std    : {r['std_ms']:8.1f} ms")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[model_name] = {"error": str(e)}

    # 비교 출력
    if len(results) > 1:
        print(f"\n  {'Model':<20} {'Mean(ms)':>10} {'P90(ms)':>10} {'Load(ms)':>10}")
        print("  " + "-" * 54)
        for name, r in results.items():
            if "error" not in r:
                print(f"  {name:<20} {r['mean_ms']:>10.1f} "
                      f"{r['p90_ms']:>10.1f} {r['model_load_ms']:>10.1f}")

    return results


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_llm_judge",  action="store_true")
    parser.add_argument("--skip_latency",    action="store_true")
    parser.add_argument("--judge_samples",   type=int, default=30)
    parser.add_argument("--warmup",          type=int, default=3)
    parser.add_argument("--measure",         type=int, default=20)
    args = parser.parse_args()

    queries, ground_truths, domain_types, gt_domains = load_data()
    preds = load_predictions("dual_moe")

    # 1. MoE Routing
    routing_result = moe_routing_report(preds, gt_domains, domain_types)

    # 2. RAG 특화 지표
    rag_result = rag_metrics_report(preds, ground_truths, queries)

    # 3. LLM-as-Judge
    judge_result = {}
    if not args.skip_llm_judge:
        judge_result = llm_judge_report(
            queries, ground_truths, preds, args.judge_samples
        )
    else:
        print("\n[LLM-as-Judge] SKIP (--skip_llm_judge)")

    # 4. Efficiency
    efficiency_result = {}
    if not args.skip_latency:
        efficiency_result = efficiency_report(
            queries,
            models=["naive_rag", "adaptive_rag", "dual_moe"],
            n_warmup=args.warmup,
            n_measure=args.measure,
        )
    else:
        print("\n[Efficiency] SKIP (--skip_latency)")

    # 저장
    final = {
        "moe_routing":  routing_result,
        "rag_metrics":  rag_result,
        "llm_judge":    judge_result,
        "efficiency":   efficiency_result,
    }
    out_path = os.path.join(RESULTS_DIR, "advanced_results.json")
    with open(out_path, "w") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 최종 요약
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    ra   = routing_result.get("routing_accuracy", {})
    lb   = routing_result.get("load_balance", {})
    ent  = routing_result.get("routing_entropy", {})
    faith = rag_result.get("faithfulness", {})
    ar   = rag_result.get("answer_relevance", {})
    agg  = judge_result.get("aggregated", {})

    print(f"  Routing Accuracy       : {ra.get('overall', 'N/A')}")
    print(f"  Load Imbalance (CV)    : {lb.get('coefficient_of_variation', 'N/A')}")
    print(f"  KL from Uniform        : {lb.get('kl_from_uniform', 'N/A')}")
    ov_ent = ent.get("overall", {})
    si_ent = ent.get("single_domain", {})
    cr_ent = ent.get("cross_domain",  {})
    print(f"  Entropy (single/cross) : {si_ent.get('mean','N/A')} / {cr_ent.get('mean','N/A')}")
    print(f"  Faithfulness           : {faith.get('faithfulness', 'N/A')}")
    print(f"  Answer Relevance       : {ar.get('answer_relevance', 'N/A')}")
    if agg:
        for k in JUDGE_CRITERIA:
            print(f"  Judge-{k:12s}: {agg.get(k, 'N/A')} / 5")

    if efficiency_result:
        for name, r in efficiency_result.items():
            if "error" not in r:
                print(f"  {name} latency (mean)  : {r['mean_ms']} ms/query")

    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()