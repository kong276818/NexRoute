"""
Latency + Hallucination Rate 측정 스크립트
- Latency: 각 모델로 샘플 50개 추론 타이밍 측정 (평균 ± std)
- Hallucination Rate: NLI 모델(DeBERTa-large-mnli)로 자동 측정
  premise=retrieved_context, hypothesis=pred_answer
  → entailment 비율이 낮으면 hallucination
"""
import json, os, time, torch
import numpy as np
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForSequenceClassification,
                          BitsAndBytesConfig, pipeline)
from sentence_transformers import SentenceTransformer
import faiss, glob

# ── 경로 설정 ────────────────────────────────────────────────
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"
NLI_MODEL   = "cross-encoder/nli-deberta-v3-large"
OUT_PATH    = "/home/sclab/moe_model/latency_hallucination_results.json"
N_LATENCY   = 50   # 타이밍 측정용 샘플 수 (모델당)

_cands = glob.glob(
    "/home/sclab/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-72B-Instruct/snapshots/*"
)
BASE_72B = _cands[0] if _cands else "Qwen/Qwen2.5-72B-Instruct"
BASE_7B  = "/home/sclab/.cache/huggingface/hub/" \
           "models--Qwen--Qwen2.5-7B-Instruct/snapshots"
_cands7 = glob.glob(BASE_7B + "/*")
BASE_7B  = _cands7[0] if _cands7 else "Qwen/Qwen2.5-7B-Instruct"

RESULT_FILES = {
    "Naive RAG 7B":     "/home/sclab/moe_model/baseline_results/naive_rag_7b_500.json",
    "Adaptive RAG 7B":  "/home/sclab/moe_model/baseline_results/adaptive_rag_7b_500.json",
    "HyDE 7B":          "/home/sclab/moe_model/baseline_results/hyde_7b_500.json",
    "CRAG 7B":          "/home/sclab/moe_model/baseline_results/crag_7b_500.json",
    "NexRoute 7B":      "/home/sclab/moe_model/eval_results_7b_v3_500.json",
    "Naive RAG 72B":    "/home/sclab/moe_model/baseline_results/naive_rag_72b_500.json",
    "Adaptive RAG 72B": "/home/sclab/moe_model/baseline_results/adaptive_rag_72b_500.json",
    "HyDE 72B":         "/home/sclab/moe_model/baseline_results/hyde_72b_500.json",
    "CRAG 72B":         "/home/sclab/moe_model/baseline_results/crag_72b_500.json",
}

results = {}


# ════════════════════════════════════════════════════════════
# STEP 1: Hallucination Rate (NLI 기반, 생성된 답변 사용)
# ════════════════════════════════════════════════════════════
def compute_hallucination_rate(model_name, data, gt_map, nli_pipe):
    """
    Hallucination = NLI contradiction 비율
    premise:    gold_answer (검증된 사실)
    hypothesis: pred_answer (모델 생성)
    → contradiction 비율 = hallucination rate
    """
    labels_count = {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0}
    n_valid = 0

    for item in data:
        query = item.get("query", item.get("instruction", "")).strip()
        pred  = item.get("pred_answer", "").strip()
        gt_item = gt_map.get(query)
        if not gt_item or not pred:
            continue
        gold = gt_item.get("gold_answer", "").strip()
        if not gold:
            continue

        # NLI 입력: premise=gold, hypothesis=pred (각 512 토큰 제한)
        try:
            out = nli_pipe(
                {"text": gold[:1000], "text_pair": pred[:1000]},
                truncation=True
            )
            label = out[0]["label"].upper()
            if label in labels_count:
                labels_count[label] += 1
                n_valid += 1
        except Exception as e:
            continue

    if n_valid == 0:
        return None, labels_count
    halluc_rate = labels_count["CONTRADICTION"] / n_valid * 100
    print(f"  [{model_name}] NLI 분류 완료 (N={n_valid}): "
          f"ENT={labels_count['ENTAILMENT']} "
          f"NEU={labels_count['NEUTRAL']} "
          f"CON={labels_count['CONTRADICTION']} "
          f"→ Halluc={halluc_rate:.1f}%")
    return halluc_rate, labels_count


# ════════════════════════════════════════════════════════════
# STEP 2: Latency 측정 (실제 추론 타이밍)
# ════════════════════════════════════════════════════════════
def build_unified_db(embed_model):
    domain_files = {
        "LAW": "law_rewritten.json", "MED": "med_rewritten.json",
        "SOC": "soc_rewritten.json", "STEM": "stem_rewritten.json",
        "CROSS": "cross_rewritten.json",
    }
    all_texts = []
    for domain, fname in domain_files.items():
        with open(f"{DATA_DIR}/{fname}") as f:
            data = json.load(f)
        for d in data:
            out = d.get("output", d.get("gold_answer", ""))
            all_texts.append(f"{d['instruction']} {out}")
    embs = embed_model.encode(all_texts, show_progress_bar=False, batch_size=64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    return all_texts, index


def measure_latency(model, tokenizer, embed_model, texts, index,
                    sample_queries, model_type="naive"):
    """N_LATENCY개 샘플에 대해 end-to-end 추론 시간 측정"""
    latencies = []
    for query in sample_queries[:N_LATENCY]:
        t0 = time.perf_counter()

        # retrieve
        emb = embed_model.encode([query])
        emb = emb / np.linalg.norm(emb)
        _, I = index.search(emb.astype(np.float32), 10)
        ctx = [texts[i] for i in I[0]]

        # generate
        ctx_text = "\n".join([c[:200] for c in ctx])
        prompt = (f"You are a helpful expert.\n\nContext:\n{ctx_text}\n\n"
                  f"Question: {query}\n\nAnswer:")
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=300,
                           do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    mean_lat = np.mean(latencies)
    std_lat  = np.std(latencies)
    print(f"  Latency: {mean_lat:.2f}s ± {std_lat:.2f}s (N={len(latencies)})")
    return round(mean_lat, 2), round(std_lat, 2)


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # GT 로드
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_map = {d["instruction"].strip(): d for d in gt}
    sample_queries = [d["instruction"] for d in gt[:N_LATENCY]]

    # ── NLI 모델 로드 (CPU도 가능, GPU면 더 빠름) ──
    print("[*] NLI 모델 로드...")
    nli_pipe = pipeline(
        "text-classification",
        model=NLI_MODEL,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True, max_length=512
    )
    print("[*] NLI 로드 완료")

    # ── Hallucination Rate 전체 모델 ──
    print("\n=== STEP 1: Hallucination Rate ===")
    for model_name, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {model_name}: 파일 없음")
            continue
        with open(path) as f:
            data = json.load(f)
        hr, counts = compute_hallucination_rate(
            model_name, data, gt_map, nli_pipe)
        results.setdefault(model_name, {})["hallucination_rate"] = hr
        results[model_name]["nli_counts"] = counts

    # 중간 저장
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Hallucination Rate 저장 완료: {OUT_PATH}")

    del nli_pipe
    torch.cuda.empty_cache()

    # ── Latency 측정 ──
    print("\n=== STEP 2: Latency 측정 ===")
    embed_model = SentenceTransformer(EMBED_MODEL)
    texts, index = build_unified_db(embed_model)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 7B 모델 타이밍
    print(f"\n[*] 7B 모델 로드: {BASE_7B}")
    tok7 = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True)
    tok7.pad_token = tok7.eos_token
    mdl7 = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager")

    for name in ["Naive RAG 7B", "Adaptive RAG 7B", "HyDE 7B",
                 "CRAG 7B", "NexRoute 7B"]:
        print(f"\n  [{name}]")
        mean_l, std_l = measure_latency(
            mdl7, tok7, embed_model, texts, index, sample_queries)
        results.setdefault(name, {})["latency_mean"] = mean_l
        results[name]["latency_std"] = std_l

    del mdl7, tok7
    torch.cuda.empty_cache()

    # 72B 모델 타이밍
    print(f"\n[*] 72B 모델 로드: {BASE_72B}")
    tok72 = AutoTokenizer.from_pretrained(
        BASE_72B, trust_remote_code=True, local_files_only=True)
    tok72.pad_token = tok72.eos_token
    mdl72 = AutoModelForCausalLM.from_pretrained(
        BASE_72B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager")

    for name in ["Naive RAG 72B", "Adaptive RAG 72B",
                 "HyDE 72B", "CRAG 72B"]:
        print(f"\n  [{name}]")
        mean_l, std_l = measure_latency(
            mdl72, tok72, embed_model, texts, index, sample_queries)
        results.setdefault(name, {})["latency_mean"] = mean_l
        results[name]["latency_std"] = std_l

    del mdl72, tok72
    torch.cuda.empty_cache()

    # 최종 저장
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] 전체 완료! 저장: {OUT_PATH}")

    # 요약 출력
    print("\n=== 최종 요약 ===")
    print(f"{'Model':<22} {'Latency(s)':>12} {'Halluc.Rate':>12}")
    print("-" * 50)
    for name, res in results.items():
        lat  = res.get("latency_mean", "N/A")
        hr   = res.get("hallucination_rate", "N/A")
        lat_s  = f"{lat:.2f}" if isinstance(lat, float) else str(lat)
        hr_s   = f"{hr:.1f}%" if isinstance(hr, float) else str(hr)
        print(f"{name:<22} {lat_s:>12} {hr_s:>12}")