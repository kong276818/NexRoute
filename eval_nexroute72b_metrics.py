"""
NexRoute 72B — BERTScore / ROUGE-L / BLEU-4 계산
기존 final_table3_scores_v3.json 에 NexRoute 72B 항목 추가
"""
import json, numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PRED_PATH  = "/home/sclab/moe_model/baseline_results/nexroute_72b_500_v2.json"
TABLE_PATH = "/home/sclab/moe_model/final_table3_scores_v3.json"

with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

with open(PRED_PATH) as f:
    preds = json.load(f)

cross_items, single_items = [], []
for item in preds:
    q    = item.get("query", item.get("instruction", "")).strip()
    pred = item.get("pred_answer", "").strip()
    gt_item = gt_map.get(q)
    if not gt_item or not pred:
        continue
    gold   = gt_item.get("gold_answer", "").strip()
    domain = gt_item.get("expected_domain", item.get("domain_gt", ""))
    entry  = {"gold": gold, "pred": pred}
    if domain == "CROSS":
        cross_items.append(entry)
    else:
        single_items.append(entry)

print(f"Cross: {len(cross_items)}개 / Single: {len(single_items)}개")


def compute_metrics(items, tag):
    golds = [x["gold"] for x in items]
    preds = [x["pred"] for x in items]
    n     = len(items)

    # ROUGE-L
    print(f"  [{tag}] ROUGE-L 계산 중...")
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = np.mean([rscorer.score(g, p)["rougeL"].fmeasure
                  for g, p in zip(golds, preds)]) * 100

    # BERTScore (rescale_with_baseline=True → 기존 json 수치와 동일 스케일)
    print(f"  [{tag}] BERTScore 계산 중...")
    _, _, F1 = bert_score_fn(
        preds, golds,
        lang="en",
        model_type="bert-base-uncased",
        rescale_with_baseline=True,
        verbose=False,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    bs = F1.mean().item() * 100

    # BLEU-4
    print(f"  [{tag}] BLEU-4 계산 중...")
    smooth = SmoothingFunction().method1
    bleu = np.mean([
        sentence_bleu(
            [nltk.word_tokenize(g.lower())],
            nltk.word_tokenize(p.lower()),
            smoothing_function=smooth
        ) for g, p in zip(golds, preds)
    ]) * 100

    result = {
        "rouge_l":    round(float(rl),   2),
        "bertscore":  round(float(bs),   2),
        "bleu4":      round(float(bleu), 2),
        "n":          n,
    }
    print(f"  [{tag}] ROUGE-L={rl:.2f}  BERTScore={bs:.2f}  BLEU4={bleu:.2f}  n={n}")
    return result


cross_metrics  = compute_metrics(cross_items,  "CROSS")
single_metrics = compute_metrics(single_items, "Single")

# final_table3_scores_v3.json 업데이트
with open(TABLE_PATH) as f:
    table = json.load(f)

table["NexRoute 72B"] = {
    "cross":  cross_metrics,
    "single": single_metrics,
}

with open(TABLE_PATH, "w") as f:
    json.dump(table, f, indent=2, ensure_ascii=False)

print(f"\n✅ 저장 완료: {TABLE_PATH}")
print("\n=== 현재 전체 모델 요약 ===")
for name, v in table.items():
    c = v.get("cross", {})
    s = v.get("single", {})
    print(f"  {name:<25} CROSS BS={c.get('bertscore','?'):>5}  Single BS={s.get('bertscore','?'):>5}")
