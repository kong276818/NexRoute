"""
BioASQ Cross-Domain Evaluation (MED+STEM)
rag-datasets/rag-mini-bioasq 기반, 5개 모델 비교
"""

import csv
import json
import os
import random
import sys

import nltk
import numpy as np
import torch
from bert_score import BERTScorer
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer as rs
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import DualMoEPipeline
from eval_baselines_500 import BASE_7B, build_db, retrieve, generate, KEYWORDS

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
SEED      = 42
N_SAMPLE  = 100
MIN_ANS_WORDS = 20

SYSTEMS = {
    'LAW':  "You are a legal expert.",
    'MED':  "You are a medical expert.",
    'SOC':  "You are a social science expert.",
    'STEM': "You are a science expert.",
}

# ── 지표 ──────────────────────────────────────────────────────────────────
print("[*] BERTScorer 로드 (roberta-large, rescale=False)...")
_bs = BERTScorer(
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
_rouge  = rs.RougeScorer(["rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def calc_rouge_l(ref, hyp):
    return _rouge.score(ref, hyp)["rougeL"].fmeasure * 100

def calc_bleu4(ref, hyp):
    ref_toks = [nltk.word_tokenize(ref.lower())]
    hyp_toks =  nltk.word_tokenize(hyp.lower())
    return sentence_bleu(ref_toks, hyp_toks, smoothing_function=_smooth) * 100

def calc_bertscore_batch(preds, refs):
    _, _, F = _bs.score(preds, refs)
    return [float(f) * 100 for f in F]


# ── 데이터 로드 ───────────���──────────────────────────────────────────────────
def load_bioasq(n=N_SAMPLE):
    print(f"\n[BioASQ] rag-mini-bioasq 로드 중...")
    ds = load_dataset(
        "rag-datasets/rag-mini-bioasq",
        "question-answer-passages",
        cache_dir=os.path.join(OUT_DIR, "datasets"),
    )
    rows = ds["test"]
    print(f"  전체: {len(rows)}개")

    filtered = []
    for row in rows:
        q = (row.get("question") or "").strip()
        a = (row.get("answer")   or "").strip()
        if not q or not a:
            continue
        if len(a.split()) < MIN_ANS_WORDS:
            continue
        filtered.append({"question": q, "gold_answer": a})

    print(f"  answer ≥ {MIN_ANS_WORDS}단어 필터 후: {len(filtered)}개")

    random.seed(SEED)
    sample = random.sample(filtered, min(n, len(filtered)))

    result = [
        {"id": i, "question": r["question"], "gold_answer": r["gold_answer"],
         "gold_domain": "MED+STEM"}
        for i, r in enumerate(sample)
    ]
    print(f"  최종 샘플: {len(result)}개")
    for r in result[:3]:
        print(f"    Q: {r['question'][:90]}...")
        print(f"    A: {r['gold_answer'][:100]}...")
    return result


# ── NexRoute 추론 ──────────────────────────────────────���──────────────────────
def run_nexroute(items):
    pipe = DualMoEPipeline()
    pipe.load_router()
    pipe.load_embed_model()
    pipe.load_base_model()
    pipe.build_db()

    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [NexRoute {i+1}/{len(items)}] {q[:60]}...")
        try:
            out = pipe.run(q)
            results.append({
                "id":               item["id"],
                "system":           "NexRoute-7B",
                "question":         q,
                "gold_answer":      item["gold_answer"],
                "generated_answer": out["answer"],
                "pred_domains":     "+".join(out["domains"]),
                "is_cross":         out["is_cross"],
            })
        except Exception as e:
            print(f"    [Error] {e}")
            results.append({
                "id":               item["id"],
                "system":           "NexRoute-7B",
                "question":         q,
                "gold_answer":      item["gold_answer"],
                "generated_answer": "[FAILED]",
                "pred_domains":     "UNKNOWN",
                "is_cross":         False,
            })

    embed_model = pipe.embed_model
    del pipe.base_model
    pipe.base_model = None
    torch.cuda.empty_cache()
    print("[*] NexRoute LLM 해제 완료")
    return results, embed_model


# ── 7B 모델 로드 (공통) ──────────────────────────────────────────────────
def load_7b():
    print("[*] 7B 모델 로드...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    print("[*] 7B 로드 완료")
    return model, tokenizer


# ── NaiveRAG 추론 ────────────────────────────────────────────────────────
def run_naive_rag(items, embed_model, texts, index, model, tokenizer):
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [NaiveRAG {i+1}/{len(items)}] {q[:60]}...")
        try:
            ctx = retrieve(q, embed_model, texts, index)
            ans = generate(q, ctx, model, tokenizer)
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
        results.append({
            "id": item["id"], "system": "NaiveRAG-7B",
            "question": q, "gold_answer": item["gold_answer"],
            "generated_answer": ans, "pred_domains": "", "is_cross": False,
        })
    return results


# ── AdaptiveRAG 추론 ──────────────────────────────────────────────────────
def run_adaptive_rag(items, embed_model, texts, index, model, tokenizer):
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [AdaptiveRAG {i+1}/{len(items)}] {q[:60]}...")
        try:
            ctx = retrieve(q, embed_model, texts, index)
            ql = q.lower()
            scores = {d: sum(1 for k in kws if k in ql) for d, kws in KEYWORDS.items()}
            domain = max(scores, key=scores.get)
            system = SYSTEMS.get(domain, "You are a helpful expert.")
            ans = generate(q, ctx, model, tokenizer, system=system)
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
        results.append({"id": item["id"], "system": "AdaptiveRAG-7B",
                        "question": q, "gold_answer": item["gold_answer"],
                        "generated_answer": ans, "pred_domains": "", "is_cross": False})
    return results


# ── HyDE 추론 ────────────────────────────────────────────────────────────
def run_hyde(items, embed_model, texts, index, model, tokenizer):
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [HyDE {i+1}/{len(items)}] {q[:60]}...")
        try:
            hypo = generate(q, [], model, tokenizer,
                            system="Write a brief hypothetical answer (2-3 sentences) to help retrieve relevant documents.")
            hypo_emb = embed_model.encode([hypo])
            hypo_emb = hypo_emb / np.linalg.norm(hypo_emb)
            _, I2 = index.search(hypo_emb.astype(np.float32), 10)
            ctx = [texts[i] for i in I2[0]]
            ans = generate(q, ctx, model, tokenizer)
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
        results.append({"id": item["id"], "system": "HyDE-7B",
                        "question": q, "gold_answer": item["gold_answer"],
                        "generated_answer": ans, "pred_domains": "", "is_cross": False})
    return results


# ── CRAG 추론 ────────────────────────────────────────────────────────────
def run_crag(items, embed_model, texts, index, model, tokenizer):
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [CRAG {i+1}/{len(items)}] {q[:60]}...")
        try:
            emb = embed_model.encode([q])
            emb = emb / np.linalg.norm(emb)
            D, I = index.search(emb.astype(np.float32), 10)
            ctx = [texts[j] for j in I[0]]
            top_score = float(D[0][0])
            if top_score >= 0.85:
                final_ctx = ctx
            elif top_score >= 0.60:
                exp = f"{q} explanation details mechanism"
                emb2 = embed_model.encode([exp])
                emb2 = emb2 / np.linalg.norm(emb2)
                _, I2 = index.search(emb2.astype(np.float32), 5)
                final_ctx = ctx[:5] + [texts[j] for j in I2[0]]
            else:
                final_ctx = []
            ans = generate(q, final_ctx, model, tokenizer)
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
        results.append({"id": item["id"], "system": "CRAG-7B",
                        "question": q, "gold_answer": item["gold_answer"],
                        "generated_answer": ans, "pred_domains": "", "is_cross": False})
    return results


# ── 지표 계산 ──────────────────────────────────────────────────────────────
def attach_metrics(records):
    valid = [r for r in records if r["generated_answer"] not in ("[FAILED]", "")]
    if not valid:
        return records
    preds = [r["generated_answer"] for r in valid]
    refs  = [r["gold_answer"]      for r in valid]
    print(f"  BERTScore 배치 계산 ({len(valid)}개)...")
    bs_scores = calc_bertscore_batch(preds, refs)
    for r, bss in zip(valid, bs_scores):
        r["rougeL"]    = round(calc_rouge_l(r["gold_answer"], r["generated_answer"]), 4)
        r["bleu4"]     = round(calc_bleu4(  r["gold_answer"], r["generated_answer"]), 4)
        r["bertscore"] = round(bss, 4)
    for r in records:
        for m in ("rougeL", "bleu4", "bertscore"):
            if m not in r:
                r[m] = 0.0
    return records


# ── 집계 & 출력 ──────────────────────────────────────────────────────────
def aggregate(all_records):
    from collections import Counter

    def stats(records):
        valid = [r for r in records if r["generated_answer"] not in ("[FAILED]", "")]
        if not valid:
            return {"n": 0, "rouge_l": 0, "bertscore": 0, "bleu4": 0}
        return {
            "n":         len(valid),
            "rouge_l":   round(float(np.mean([r["rougeL"]    for r in valid])), 2),
            "bertscore": round(float(np.mean([r["bertscore"] for r in valid])), 2),
            "bleu4":     round(float(np.mean([r["bleu4"]     for r in valid])), 2),
        }

    order = ["NaiveRAG-7B", "AdaptiveRAG-7B", "HyDE-7B", "CRAG-7B", "NexRoute-7B"]
    by_system = {s: [r for r in all_records if r["system"] == s] for s in order}

    print("\n" + "=" * 75)
    print("  BioASQ (MED+STEM Cross-Domain) 평가 결과")
    print("=" * 75)
    print(f"  {'System':<16} {'N':>4}  {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8}")
    print("-" * 75)

    summary = {}
    for system in order:
        s = stats(by_system[system])
        print(f"  {system:<16} {s['n']:>4}  {s['rouge_l']:>8.2f} {s['bertscore']:>10.2f} {s['bleu4']:>8.2f}")
        summary[system] = s

    print("=" * 75)

    # NexRoute 라우팅 분포
    nr_valid = [r for r in by_system["NexRoute-7B"] if r["generated_answer"] not in ("[FAILED]", "")]
    domain_dist = Counter(r["pred_domains"] for r in nr_valid)
    cross_rate  = sum(1 for r in nr_valid if r["is_cross"]) / max(len(nr_valid), 1) * 100
    print(f"\n  [NexRoute 라우팅 분포]")
    for dom, cnt in sorted(domain_dist.items(), key=lambda x: -x[1]):
        print(f"    {dom:<14}: {cnt:>3}개 ({cnt/max(len(nr_valid),1)*100:.1f}%)")
    print(f"  Cross-domain 판정 비율: {cross_rate:.1f}%")

    summary["routing_dist"] = dict(domain_dist)
    out_path = os.path.join(OUT_DIR, "bioasq_cross_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 저장: {out_path}")

    # per-sample CSV (t-test 및 품질 확인용)
    COLS = ["id", "system", "question", "gold_answer", "generated_answer",
            "pred_domains", "is_cross", "rougeL", "bertscore", "bleu4"]
    csv_path = os.path.join(OUT_DIR, "bioasq_cross_peritem.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[*] per-sample 저장: {csv_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    items = load_bioasq(N_SAMPLE)

    # NexRoute 먼저 (embed_model 재사용)
    print("\n" + "=" * 50 + "\nNexRoute-7B\n" + "=" * 50)
    nr_records, embed_model = run_nexroute(items)

    print("\n[*] Vector DB 구축...")
    texts, index = build_db(embed_model)

    # 7B 공통 모델 로드
    model, tokenizer = load_7b()

    print("\n" + "=" * 50 + "\nNaiveRAG-7B\n" + "=" * 50)
    naive_records = run_naive_rag(items, embed_model, texts, index, model, tokenizer)

    print("\n" + "=" * 50 + "\nAdaptiveRAG-7B\n" + "=" * 50)
    adaptive_records = run_adaptive_rag(items, embed_model, texts, index, model, tokenizer)

    print("\n" + "=" * 50 + "\nHyDE-7B\n" + "=" * 50)
    hyde_records = run_hyde(items, embed_model, texts, index, model, tokenizer)

    print("\n" + "=" * 50 + "\nCRAG-7B\n" + "=" * 50)
    crag_records = run_crag(items, embed_model, texts, index, model, tokenizer)

    del model
    torch.cuda.empty_cache()

    print("\n[*] 지표 계산 중...")
    all_records = []
    for name, records in [("NexRoute", nr_records), ("NaiveRAG", naive_records),
                           ("AdaptiveRAG", adaptive_records), ("HyDE", hyde_records),
                           ("CRAG", crag_records)]:
        print(f"  {name}...")
        all_records.extend(attach_metrics(records))

    aggregate(all_records)
    print("\n[완료]")


if __name__ == "__main__":
    main()
