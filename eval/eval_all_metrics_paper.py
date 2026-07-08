"""
논문 Table 3 재측정 — raw BERTScore (rescale_with_baseline=False)
ROUGE-L, BERTScore(raw), BLEU-4 for all models
"""
import json, numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk, torch
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_PATH  = "/home/sclab/moe_model/final_table3_paper.json"

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
    "NexRoute 72B":     "/home/sclab/moe_model/baseline_results/nexroute_72b_500_v2.json",
}

with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

device = "cuda" if torch.cuda.is_available() else "cpu"


def split_domain(data):
    cross, single = [], []
    for item in data:
        q    = item.get("query", item.get("instruction", "")).strip()
        pred = item.get("pred_answer", "").strip()
        gold = item.get("gold_answer", "").strip()

        # gold 없으면 GT 룩업
        if not gold:
            gt_item = gt_map.get(q)
            if not gt_item:
                continue
            gold = gt_item.get("gold_answer", "").strip()
        if not pred or not gold:
            continue

        # 도메인 판별: is_cross 우선, 없으면 GT 룩업
        if "is_cross" in item:
            is_cross = item["is_cross"]
        else:
            gt_item = gt_map.get(q) or {}
            domain  = item.get("domain", gt_item.get("expected_domain", ""))
            is_cross = (domain == "CROSS")

        entry = {"gold": gold, "pred": pred}
        if is_cross:
            cross.append(entry)
        else:
            single.append(entry)
    return cross, single


def compute_metrics(items, tag, model_name):
    if not items:
        return {"rouge_l": None, "bertscore": None, "bleu4": None, "n": 0}

    golds = [x["gold"] for x in items]
    preds = [x["pred"] for x in items]
    n     = len(items)

    # ROUGE-L
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = np.mean([rscorer.score(g, p)["rougeL"].fmeasure
                  for g, p in zip(golds, preds)]) * 100

    # BERTScore — raw (rescale_with_baseline=False)
    _, _, F1 = bert_score_fn(
        preds, golds,
        lang="en",
        model_type="roberta-large",    # ← 논문과 동일 (88~90 범위)
        rescale_with_baseline=False,
        verbose=False,
        device=device,
    )
    bs = F1.mean().item() * 100

    # BLEU-4
    smooth = SmoothingFunction().method1
    bleu = np.mean([
        sentence_bleu(
            [nltk.word_tokenize(g.lower())],
            nltk.word_tokenize(p.lower()),
            smoothing_function=smooth
        ) for g, p in zip(golds, preds)
    ]) * 100

    print(f"    [{tag}] n={n}  ROUGE-L={rl:.2f}  BS={bs:.2f}  BLEU4={bleu:.2f}")
    return {"rouge_l": round(float(rl),2), "bertscore": round(float(bs),2),
            "bleu4": round(float(bleu),2), "n": n}


results = {}
for model_name, path in RESULT_FILES.items():
    import os
    if not os.path.exists(path):
        print(f"[SKIP] {model_name}: 파일 없음")
        continue
    print(f"\n[▶] {model_name}")
    with open(path) as f:
        data = json.load(f)

    cross_items, single_items = split_domain(data)
    print(f"  CROSS={len(cross_items)}  Single={len(single_items)}")

    results[model_name] = {
        "cross":  compute_metrics(cross_items,  "CROSS",  model_name),
        "single": compute_metrics(single_items, "Single", model_name),
    }

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

# ── 최종 테이블 출력 ──────────────────────────────────────────────────
print("\n\n" + "="*95)
print(f"{'Model':<22}  {'CROSS ROUGE-L':>13} {'CROSS BS':>9} {'CROSS BLEU4':>11}"
      f"  {'Single ROUGE-L':>14} {'Single BS':>9} {'Single BLEU4':>12}")
print("─"*95)
for name, v in results.items():
    c = v["cross"]; s = v["single"]
    def f(x): return f"{x:.2f}" if x is not None else " N/A"
    marker = " ◀" if "NexRoute" in name else ""
    print(f"  {name:<22}  {f(c['rouge_l']):>13} {f(c['bertscore']):>9} {f(c['bleu4']):>11}"
          f"  {f(s['rouge_l']):>14} {f(s['bertscore']):>9} {f(s['bleu4']):>12}{marker}")

print(f"\n✅ 저장: {OUT_PATH}")
