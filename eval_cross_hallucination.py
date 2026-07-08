"""
CROSS 도메인 샘플만 NLI 할루시네이션 측정
GT의 expected_domain == "CROSS" 기준으로 필터링
"""
import json, torch
from transformers import pipeline

GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
OUT_PATH  = "/home/sclab/moe_model/cross_hallucination_results.json"

RESULT_FILES = {
    "Naive RAG 7B":     "/home/sclab/moe_model/baseline_results/naive_rag_7b_500.json",
    "Adaptive RAG 7B":  "/home/sclab/moe_model/baseline_results/adaptive_rag_7b_500.json",
    "HyDE 7B":          "/home/sclab/moe_model/baseline_results/hyde_7b_500.json",
    "CRAG 7B":          "/home/sclab/moe_model/baseline_results/crag_7b_500.json",
    "NexRoute 7B":      "/home/sclab/moe_model/eval_results_7b_v3_500.json",
    "Naive RAG 72B":    "/home/sclab/moe_model/baseline_results/naive_rag_72b_500_v2.json",
    "Adaptive RAG 72B": "/home/sclab/moe_model/baseline_results/adaptive_rag_72b_500_v2.json",
    "HyDE 72B":         "/home/sclab/moe_model/baseline_results/hyde_72b_500_v2.json",
    "CRAG 72B":         "/home/sclab/moe_model/baseline_results/crag_72b_500_v2.json",
    "NexRoute 72B":     "/home/sclab/moe_model/baseline_results/nexroute_72b_500.json",
}

# GT → query: expected_domain 맵
with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}
cross_queries = {d["instruction"].strip() for d in gt_list if d.get("expected_domain") == "CROSS"}
print(f"CROSS 샘플 수: {len(cross_queries)}")

# NLI 로드
print("[*] NLI 모델 로드...")
device   = 0 if torch.cuda.is_available() else -1
nli_pipe = pipeline("text-classification", model=NLI_MODEL, device=device,
                    truncation=True, max_length=512)
print("[*] 완료\n")


def run_hallucination(pred_path, model_name):
    with open(pred_path) as f:
        data = json.load(f)

    n_valid = 0
    counts  = {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0}

    for item in data:
        q    = item.get("query", item.get("instruction", "")).strip()
        pred = item.get("pred_answer", "").strip()
        gold = item.get("gold_answer", "").strip()

        # CROSS 도메인 필터
        if q not in cross_queries:
            continue
        if not pred or not gold:
            # gold 없으면 gt_map에서 가져오기
            gt_item = gt_map.get(q)
            if not gt_item:
                continue
            gold = gt_item.get("gold_answer", "").strip()
        if not pred or not gold:
            continue

        try:
            out   = nli_pipe({"text": gold[:1000], "text_pair": pred[:1000]}, truncation=True)
            label = (out["label"] if isinstance(out, dict) else out[0]["label"]).upper()
            if label in counts:
                counts[label] += 1
                n_valid += 1
        except Exception:
            continue

    rate = counts["CONTRADICTION"] / n_valid * 100 if n_valid else 0
    print(f"  {model_name:<22}  N={n_valid:>3}  "
          f"ENT={counts['ENTAILMENT']:>3}  NEU={counts['NEUTRAL']:>3}  "
          f"CON={counts['CONTRADICTION']:>3}  →  Halluc={rate:.1f}%")
    return {"hallucination_rate": round(rate, 1), "n": n_valid, "counts": counts}


all_results = {}
for name, path in RESULT_FILES.items():
    import os
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: 파일 없음")
        continue
    all_results[name] = run_hallucination(path, name)

with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ 저장: {OUT_PATH}")
print("\n=== CROSS 도메인 할루시네이션 최종 요약 ===")
print(f"{'Model':<22}  {'Halluc Rate':>12}")
print("-" * 38)
for name, v in all_results.items():
    print(f"  {name:<22}  {v['hallucination_rate']:>10.1f}%")
