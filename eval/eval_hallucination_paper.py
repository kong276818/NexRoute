"""
논문용 Hallucination Rate — Sentence-level NLI
pred_answer를 문장 단위로 분리 후 각 문장을 gold_answer에 대해 NLI 검증.
CONTRADICTION 문장 비율 = hallucination rate.

전체 / CROSS / Single 세 구분으로 결과 출력.
"""
import json, os, torch
from transformers import pipeline
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
OUT_PATH  = "/home/sclab/moe_model/hallucination_paper_results.json"

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

with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

print("[*] NLI 모델 로드...")
device   = 0 if torch.cuda.is_available() else -1
nli_pipe = pipeline("text-classification", model=NLI_MODEL, device=device,
                    truncation=True, max_length=512)
print("[*] 완료\n")


def sentence_halluc(gold: str, pred: str) -> dict:
    """
    pred를 문장으로 분리 후 각 문장을 gold에 대해 NLI.
    returns: {"contra": k, "total": n, "rate": k/n*100}
    """
    sents = [s.strip() for s in sent_tokenize(pred) if len(s.strip()) > 15]
    if not sents:
        return {"contra": 0, "total": 0, "rate": 0.0}

    pairs  = [{"text": gold[:800], "text_pair": s[:400]} for s in sents]
    contra = 0
    for pair in pairs:
        try:
            out   = nli_pipe(pair, truncation=True)
            label = (out["label"] if isinstance(out, dict) else out[0]["label"]).upper()
            if label == "CONTRADICTION":
                contra += 1
        except Exception:
            continue
    return {"contra": contra, "total": len(sents), "rate": contra / len(sents) * 100}


def eval_model(pred_path: str, model_name: str) -> dict:
    with open(pred_path) as f:
        data = json.load(f)

    cross_rates, single_rates, all_rates = [], [], []

    for i, item in enumerate(data):
        q    = item.get("query", item.get("instruction", "")).strip()
        pred = item.get("pred_answer", "").strip()
        gold = item.get("gold_answer", "").strip()

        if not gold:
            gt_item = gt_map.get(q)
            if not gt_item:
                continue
            gold = gt_item.get("gold_answer", "").strip()
        if not pred or not gold:
            continue

        gt_item = gt_map.get(q) or {}
        # is_cross 필드 우선 → 없으면 domain 필드 → 없으면 GT 룩업
        if "is_cross" in item:
            domain = "CROSS" if item["is_cross"] else "SINGLE"
        else:
            domain = (item.get("domain") or
                      gt_item.get("expected_domain", "SINGLE"))
            if domain == "CROSS":
                domain = "CROSS"

        res  = sentence_halluc(gold, pred)
        rate = res["rate"]
        all_rates.append(rate)
        if domain == "CROSS":
            cross_rates.append(rate)
        else:
            single_rates.append(rate)

        if (i + 1) % 100 == 0:
            def _avg(lst): return f"{sum(lst)/len(lst):.1f}" if lst else "-"
            print(f"    [{i+1}/{len(data)}] all={_avg(all_rates)}%  "
                  f"cross={_avg(cross_rates)}%  single={_avg(single_rates)}%")

    def avg(lst): return round(sum(lst) / len(lst), 1) if lst else None

    result = {
        "all":    {"rate": avg(all_rates),    "n": len(all_rates)},
        "cross":  {"rate": avg(cross_rates),  "n": len(cross_rates)},
        "single": {"rate": avg(single_rates), "n": len(single_rates)},
    }
    fmt = lambda v: f"{v:>5}" if v is not None else "  N/A"
    print(f"  {model_name:<22}  ALL={fmt(result['all']['rate'])}%  "
          f"CROSS={fmt(result['cross']['rate'])}%  "
          f"Single={fmt(result['single']['rate'])}%")
    return result


print(f"{'Model':<22}  {'ALL':>7}  {'CROSS':>7}  {'Single':>8}")
print("─" * 55)

all_results = {}
for name, path in RESULT_FILES.items():
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: 파일 없음")
        continue
    print(f"\n[▶] {name}")
    all_results[name] = eval_model(path, name)

with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n\n✅ 저장: {OUT_PATH}")
print("\n" + "="*60)
print(f"{'Model':<22}  {'ALL(%)':>8}  {'CROSS(%)':>10}  {'Single(%)':>11}")
print("─"*60)
for name, v in all_results.items():
    a = v["all"]["rate"]
    c = v["cross"]["rate"]
    s = v["single"]["rate"]
    marker = " ◀" if "NexRoute" in name else ""
    print(f"  {name:<22}  {str(a):>8}  {str(c):>10}  {str(s):>11}{marker}")
