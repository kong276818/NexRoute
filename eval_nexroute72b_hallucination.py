"""
NexRoute 72B — NLI 기반 Hallucination Rate 계산
기존 latency_hallucination_results.json 에 NexRoute 72B 항목 추가
(latency 재측정 없음 — 기존 예측 파일 사용)
"""
import json, torch
from transformers import pipeline

GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PRED_PATH = "/home/sclab/moe_model/baseline_results/nexroute_72b_500.json"
OUT_PATH  = "/home/sclab/moe_model/latency_hallucination_results.json"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"

with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

with open(PRED_PATH) as f:
    preds = json.load(f)

print("[*] NLI 모델 로드...")
device = 0 if torch.cuda.is_available() else -1
nli_pipe = pipeline(
    "text-classification",
    model=NLI_MODEL,
    device=device,
    truncation=True, max_length=512,
)
print("[*] NLI 로드 완료")

labels_count = {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0}
n_valid = 0

for i, item in enumerate(preds):
    pred = item.get("pred_answer", "").strip()
    # gold_answer가 파일 내에 직접 있으면 사용, 없으면 gt_map 룩업
    gold = item.get("gold_answer", "").strip()
    if not gold:
        q    = item.get("query", item.get("instruction", "")).strip()
        gt_item = gt_map.get(q)
        if not gt_item:
            continue
        gold = gt_item.get("gold_answer", "").strip()
    if not pred or not gold:
        continue
    try:
        out = nli_pipe({"text": gold[:1000], "text_pair": pred[:1000]}, truncation=True)
        # 파이프라인이 dict 또는 list 둘 다 반환할 수 있음
        if isinstance(out, list):
            label = out[0]["label"].upper()
        else:
            label = out["label"].upper()
        if label in labels_count:
            labels_count[label] += 1
            n_valid += 1
    except Exception as e:
        continue

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/500  ENT={labels_count['ENTAILMENT']}  NEU={labels_count['NEUTRAL']}  CON={labels_count['CONTRADICTION']}")

halluc_rate = labels_count["CONTRADICTION"] / n_valid * 100 if n_valid else 0
print(f"\n[NexRoute 72B] N={n_valid}  ENT={labels_count['ENTAILMENT']}  "
      f"NEU={labels_count['NEUTRAL']}  CON={labels_count['CONTRADICTION']}  "
      f"→ Halluc={halluc_rate:.1f}%")

# 기존 결과 파일 업데이트
with open(OUT_PATH) as f:
    results = json.load(f)

results["NexRoute 72B"] = {
    "hallucination_rate": round(halluc_rate, 1),
    "nli_counts": {
        "entailment":    labels_count["ENTAILMENT"],
        "neutral":       labels_count["NEUTRAL"],
        "contradiction": labels_count["CONTRADICTION"],
    },
    "latency_mean": None,
    "latency_std":  None,
}

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ 저장 완료: {OUT_PATH}")
print("\n=== Hallucination Rate 비교 ===")
for name, v in results.items():
    hr = v.get("hallucination_rate")
    print(f"  {name:<22}  {hr:.1f}%" if hr is not None else f"  {name:<22}  N/A")
