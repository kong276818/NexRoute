"""
STEP 2 + 3: OOD 추론 및 집계
사용법:
  python ood_eval/run_ood_inference.py            # 전체 실행 (MED 100 + LAW 100)
  python ood_eval/run_ood_inference.py --limit 10 # smoke test (MED 10개만)
"""

import argparse
import csv
import json
import os
import sys

import nltk
import numpy as np
import torch
from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer as rs

# 기존 모듈 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import DualMoEPipeline, BASE_MODEL
from eval_baselines_500 import (
    BASE_7B,
    build_db,
    retrieve,
    generate,
    KEYWORDS,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR)

# ── 지표 계산 ──────────────────────────────────────────────────────────────
print("[*] BERTScorer 로드 (bert-large-uncased)...")
_bs = BERTScorer(
    model_type="bert-large-uncased",
    lang="en",
    rescale_with_baseline=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print("[*] BERTScorer 준비 완료")

_rouge  = rs.RougeScorer(["rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def calc_rouge_l(ref: str, hyp: str) -> float:
    return _rouge.score(ref, hyp)["rougeL"].fmeasure * 100


def calc_bleu4(ref: str, hyp: str) -> float:
    ref_toks = [nltk.word_tokenize(ref.lower())]
    hyp_toks = nltk.word_tokenize(hyp.lower())
    return sentence_bleu(ref_toks, hyp_toks, smoothing_function=_smooth) * 100


def calc_bertscore_batch(preds: list, refs: list) -> list:
    _, _, F = _bs.score(preds, refs)
    return [float(f) * 100 for f in F]


# ── 데이터 로드 ────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── NexRoute 추론 ──────────────────────────────────────────────────────────

def run_nexroute(items: list, limit: int | None = None) -> tuple[list, object]:
    """
    NexRoute 7B 추론.
    embed_model은 Naive RAG에서 재사용하기 위해 함께 반환.
    LLM (base_model)은 추론 완료 후 해제.
    """
    pipe = DualMoEPipeline()
    pipe.load_router()
    pipe.load_embed_model()
    pipe.load_base_model()
    pipe.build_db()

    subset = items[:limit] if limit else items
    results = []

    for i, item in enumerate(subset):
        q = item["question"]
        print(f"  [NexRoute {i+1}/{len(subset)}] {q[:60]}...")
        try:
            out = pipe.run(q)
            pred_domain = out["domains"][0]   # primary 예측 도메인
            results.append({
                "id":               item["id"],
                "benchmark":        item["gold_domain"],   # MED / LAW
                "gold_domain":      item["gold_domain"],
                "system":           "NexRoute-7B",
                "question":         q,
                "gold_answer":      item["gold_answer"],
                "generated_answer": out["answer"],
                "pred_domain":      pred_domain,
            })
        except Exception as e:
            print(f"    [Error] {e}")
            results.append({
                "id":               item["id"],
                "benchmark":        item["gold_domain"],
                "gold_domain":      item["gold_domain"],
                "system":           "NexRoute-7B",
                "question":         q,
                "gold_answer":      item["gold_answer"],
                "generated_answer": "[FAILED]",
                "pred_domain":      "UNKNOWN",
            })

    # LLM 해제 (embed_model은 보존)
    embed_model = pipe.embed_model
    del pipe.base_model
    pipe.base_model = None
    torch.cuda.empty_cache()
    print("[*] NexRoute LLM 해제 완료")

    return results, embed_model


# ── Naive RAG 7B 추론 ──────────────────────────────────────────────────────

def run_naive_rag(items: list, embed_model, texts: list, index,
                  limit: int | None = None) -> list:
    """
    Naive RAG 7B 추론.
    eval_baselines_500.py의 retrieve / generate 함수를 그대로 사용.
    """
    print("[*] Naive RAG 7B 모델 로드...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_7B,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    print("[*] Naive RAG 7B 로드 완료")

    subset = items[:limit] if limit else items
    results = []

    for i, item in enumerate(subset):
        q = item["question"]
        print(f"  [NaiveRAG {i+1}/{len(subset)}] {q[:60]}...")
        try:
            ctx = retrieve(q, embed_model, texts, index)
            ans = generate(q, ctx, model, tokenizer)   # system="You are a helpful expert."
            results.append({
                "id":               item["id"],
                "benchmark":        item["gold_domain"],
                "gold_domain":      item["gold_domain"],
                "system":           "NaiveRAG-7B",
                "question":         q,
                "gold_answer":      item["gold_answer"],
                "generated_answer": ans,
                "pred_domain":      "",   # Naive RAG는 라우팅 없음
            })
        except Exception as e:
            print(f"    [Error] {e}")
            results.append({
                "id":               item["id"],
                "benchmark":        item["gold_domain"],
                "gold_domain":      item["gold_domain"],
                "system":           "NaiveRAG-7B",
                "question":         q,
                "gold_answer":      item["gold_answer"],
                "generated_answer": "[FAILED]",
                "pred_domain":      "",
            })

    del model
    torch.cuda.empty_cache()
    print("[*] Naive RAG 7B 해제 완료")
    return results


# ── 지표 붙이기 ────────────────────────────────────────────────────────────

def attach_metrics(records: list) -> list:
    valid = [r for r in records if r["generated_answer"] not in ("[FAILED]", "")]
    if not valid:
        return records

    preds = [r["generated_answer"] for r in valid]
    refs  = [r["gold_answer"]      for r in valid]

    print(f"  BERTScore 배치 계산 ({len(valid)}개)...")
    bs_scores = calc_bertscore_batch(preds, refs)

    for r, bss in zip(valid, bs_scores):
        r["rougeL"]    = round(calc_rouge_l(r["gold_answer"], r["generated_answer"]), 4)
        r["bleu4"]     = round(calc_bleu4(r["gold_answer"], r["generated_answer"]),   4)
        r["bertscore"] = round(bss, 4)

    # 실패 레코드에는 0 채우기
    for r in records:
        for m in ("rougeL", "bleu4", "bertscore"):
            if m not in r:
                r[m] = 0.0

    return records


# ── CSV 저장 ──────────────────────────────────────────────────────────────

RESULT_COLS = ["id", "benchmark", "gold_domain", "system", "question",
               "gold_answer", "generated_answer", "pred_domain",
               "rougeL", "bertscore", "bleu4"]

def save_results_csv(records: list, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()
        writer.writerows(records)
    print(f"[*] 결과 저장: {path}  ({len(records)}행)")


# ── STEP 3: 집계 ──────────────────────────────────────────────────────────

def aggregate(records: list, out_path: str):
    from collections import defaultdict

    valid = [r for r in records if r["generated_answer"] not in ("[FAILED]", "")]

    # (benchmark, system) → [metrics]
    groups = defaultdict(lambda: {"rougeL": [], "bertscore": [], "bleu4": [],
                                  "routing_correct": [], "n": 0})

    for r in valid:
        key = (r["benchmark"], r["system"])
        g   = groups[key]
        g["rougeL"].append(r["rougeL"])
        g["bertscore"].append(r["bertscore"])
        g["bleu4"].append(r["bleu4"])
        g["n"] += 1
        if r["system"] == "NexRoute-7B":
            g["routing_correct"].append(
                1 if r["pred_domain"] == r["gold_domain"] else 0
            )

    print("\n" + "=" * 75)
    print(f"  {'Benchmark':<12} {'System':<14} {'N':>4}  "
          f"{'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8} {'RouteAcc':>10}")
    print("-" * 75)

    summary_rows = []
    for (bench, system), g in sorted(groups.items()):
        avg_r  = round(float(np.mean(g["rougeL"])),    2)
        avg_b  = round(float(np.mean(g["bertscore"])), 2)
        avg_bl = round(float(np.mean(g["bleu4"])),     2)
        acc    = (round(float(np.mean(g["routing_correct"])) * 100, 1)
                  if g["routing_correct"] else "—")

        print(f"  {bench:<12} {system:<14} {g['n']:>4}  "
              f"{avg_r:>8.2f} {avg_b:>10.2f} {avg_bl:>8.2f} {str(acc):>10}")

        summary_rows.append({
            "benchmark":   bench,
            "system":      system,
            "n":           g["n"],
            "rouge_l":     avg_r,
            "bertscore":   avg_b,
            "bleu4":       avg_bl,
            "routing_acc": acc if isinstance(acc, float) else "",
        })

    print("=" * 75)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["benchmark", "system", "n",
                           "rouge_l", "bertscore", "bleu4", "routing_acc"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[*] 집계 저장: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="각 benchmark당 처리할 최대 샘플 수 (smoke test용)")
    args = parser.parse_args()

    # 데이터 로드
    med_path = os.path.join(DATA_DIR, "medquad_med.jsonl")
    law_path = os.path.join(DATA_DIR, "lawse_law.jsonl")

    med_items = load_jsonl(med_path)
    law_items = load_jsonl(law_path)
    print(f"[*] MedQuAD: {len(med_items)}개  /  LawSE: {len(law_items)}개")

    if args.limit:
        print(f"[!] --limit {args.limit}: MedQuAD만 {args.limit}개로 smoke test")
        all_items = med_items  # smoke test는 MED만
    else:
        all_items = med_items + law_items

    # ── NexRoute 추론 ──
    print("\n" + "=" * 50)
    print("NexRoute-7B 추론")
    print("=" * 50)
    nr_records, embed_model = run_nexroute(all_items, limit=args.limit)

    # ── Naive RAG DB 구축 (embed_model 재사용) ──
    print("\n[*] Naive RAG 단일 DB 구축...")
    texts, index = build_db(embed_model)

    # ── Naive RAG 추론 ──
    print("\n" + "=" * 50)
    print("NaiveRAG-7B 추론")
    print("=" * 50)
    naive_records = run_naive_rag(all_items, embed_model, texts, index,
                                  limit=args.limit)

    # ── 지표 계산 ──
    print("\n[*] 지표 계산 중...")
    print("  NexRoute...")
    nr_records = attach_metrics(nr_records)
    print("  NaiveRAG...")
    naive_records = attach_metrics(naive_records)

    # ── CSV 저장 ──
    all_records = nr_records + naive_records
    result_path  = os.path.join(OUT_DIR, "ood_results.csv")
    summary_path = os.path.join(OUT_DIR, "ood_summary.csv")

    save_results_csv(all_records, result_path)

    # ── STEP 3: 집계 출력 ──
    print("\n[STEP 3] 집계 결과")
    aggregate(all_records, summary_path)

    print("\n[완료]")


if __name__ == "__main__":
    main()
