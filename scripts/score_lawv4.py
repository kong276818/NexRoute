"""
score_lawv4.py
==============
lora_law_v4 결과 채점 + 기존 NexRoute와 비교.

출력:
  reinference_results/scores_500_nexroute_lawv4.csv
  reinference_results/lawv4_comparison.json
"""
import os, json, csv
import numpy as np
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

GT_PATH  = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_DIR  = "/home/sclab/moe_model/reinference_results"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

CLEAN_INDICES = set(range(0, 300))

with open(GT_PATH) as f:
    gt_list = json.load(f)

rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
sf = SmoothingFunction()


def compute_rouge(gold, pred):
    return rscorer.score(gold, pred)["rougeL"].fmeasure * 100


def compute_bleu(gold, pred):
    return sentence_bleu(
        [nltk.word_tokenize(gold.lower())],
        nltk.word_tokenize(pred.lower()),
        smoothing_function=sf.method1,
    ) * 100


def score_file(pred_path, score_path):
    with open(pred_path) as f:
        preds = json.load(f)

    print(f"\n[▶] {os.path.basename(pred_path)}  ({len(preds)} samples)")

    valid = []
    for item in preds:
        idx    = item["idx"]
        gold   = item.get("gold_answer", gt_list[idx].get("gold_answer", "")).strip()
        pred   = item.get("pred_answer", "").strip()
        domain = item.get("expected_domain", gt_list[idx].get("expected_domain", ""))
        is_cross = (domain == "CROSS")
        split    = "clean" if idx in CLEAN_INDICES else "exposed"

        if not gold or not pred or pred in ("[FAILED]", "[GENERATION_FAILED]"):
            valid.append({"idx": idx, "domain": domain, "is_cross": is_cross,
                          "split": split, "failed": True})
            continue
        valid.append({
            "idx": idx, "domain": domain, "is_cross": is_cross,
            "split": split, "gold": gold, "pred": pred, "failed": False,
        })

    ok    = [r for r in valid if not r["failed"]]
    golds = [r["gold"] for r in ok]
    preds_text = [r["pred"] for r in ok]

    print(f"  BERTScore 계산 ({len(ok)}개)...")
    _, _, F1 = bert_score_fn(preds_text, golds, lang="en",
                             model_type="roberta-large",
                             rescale_with_baseline=False,
                             verbose=False, device=DEVICE)
    bs_map = {r["idx"]: round(float(v)*100, 4) for r, v in zip(ok, F1)}

    rows = []
    for r in valid:
        if r["failed"]:
            rows.append({**r, "rouge_l": None, "bertscore": None, "bleu4": None})
            continue
        rl   = round(compute_rouge(r["gold"], r["pred"]), 4)
        bleu = round(compute_bleu(r["gold"], r["pred"]), 4)
        bs   = bs_map.get(r["idx"])
        rows.append({**r, "rouge_l": rl, "bertscore": bs, "bleu4": bleu})

    with open(score_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx","domain","is_cross","split","rouge_l","bertscore","bleu4","failed"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    def mean(vals):
        v = [x for x in vals if x is not None]
        return round(float(np.mean(v)), 4) if v else None

    def group(rows, fn):
        sub = [r for r in rows if fn(r) and not r["failed"]]
        return {"rouge_l": mean([r["rouge_l"] for r in sub]),
                "bertscore": mean([r["bertscore"] for r in sub]),
                "bleu4": mean([r["bleu4"] for r in sub]),
                "n": len(sub)}

    summary = {
        "overall":      group(rows, lambda r: True),
        "clean":        group(rows, lambda r: r["split"]=="clean"),
        "cross_clean":  group(rows, lambda r: r["is_cross"] and r["split"]=="clean"),
        "single_clean": group(rows, lambda r: not r["is_cross"] and r["split"]=="clean"),
        "law_clean":    group(rows, lambda r: r["domain"]=="LAW" and r["split"]=="clean"),
        "law_all":      group(rows, lambda r: r["domain"]=="LAW"),
    }

    for k, v in summary.items():
        print(f"    {k:<16}  n={v['n']:<4}  RL={v['rouge_l']}  BS={v['bertscore']}")

    return rows, summary


# ── 채점 실행 ──────────────────────────────────────────────────────────────────
orig_path  = os.path.join(OUT_DIR, "preds_500_nexroute.json")
v4_path    = os.path.join(OUT_DIR, "preds_500_nexroute_lawv4.json")

orig_score_path = os.path.join(OUT_DIR, "scores_500_nexroute.csv")
v4_score_path   = os.path.join(OUT_DIR, "scores_500_nexroute_lawv4.csv")

print("=" * 60)
print("기존 NexRoute (lora_law)")
orig_rows, orig_sum = score_file(orig_path, orig_score_path)

print("=" * 60)
print("NexRoute v4 (lora_law_v4)")
v4_rows, v4_sum = score_file(v4_path, v4_score_path)

# ── 비교 ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON: v4 - orig")
print("=" * 60)
compare = {}
for k in orig_sum:
    o = orig_sum[k]
    v = v4_sum[k]
    delta_rl = round((v["rouge_l"] or 0) - (o["rouge_l"] or 0), 4) if o["rouge_l"] and v["rouge_l"] else None
    delta_bs = round((v["bertscore"] or 0) - (o["bertscore"] or 0), 4) if o["bertscore"] and v["bertscore"] else None
    compare[k] = {"orig": o, "v4": v, "delta_rouge_l": delta_rl, "delta_bertscore": delta_bs}
    print(f"  {k:<16}  RL: {o['rouge_l']} → {v['rouge_l']} (Δ={delta_rl:+.4f})  "
          f"BS: {o['bertscore']} → {v['bertscore']} (Δ={delta_bs:+.4f})")

# ── 저장 ──────────────────────────────────────────────────────────────────────
cmp_path = os.path.join(OUT_DIR, "lawv4_comparison.json")
with open(cmp_path, "w") as f:
    json.dump({"orig": orig_sum, "v4": v4_sum, "delta": compare}, f, indent=2)

print(f"\n[+] 저장: {cmp_path}")
print(f"[+] 저장: {v4_score_path}")
