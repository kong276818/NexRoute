"""
OOD Cross-domain 평가: law_med_cross_generated.json
모델: NaiveRAG, AdaptiveRAG, HyDE, CRAG, NexRoute-7B
BERTScore: roberta-large (rescale 없음)
"""

import csv
import json
import os
import sys

import faiss
import nltk
import numpy as np
import torch
from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer as rs
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import DualMoEPipeline

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
OOD_DIR     = os.path.dirname(os.path.abspath(__file__))
CROSS_PATH  = "/home/sclab/moe_model/law_med_cross_generated.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':  ['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':  ['social','society','culture','economic','political','community','psychology'],
    'STEM': ['science','physics','chemical','biology','math','engineering','technology'],
}

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
print("[*] BERTScorer 준비 완료")

_rouge  = rs.RougeScorer(["rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def calc_rouge_l(ref, hyp):
    return _rouge.score(ref, hyp)["rougeL"].fmeasure * 100


def calc_bleu4(ref, hyp):
    return sentence_bleu([nltk.word_tokenize(ref.lower())],
                         nltk.word_tokenize(hyp.lower()),
                         smoothing_function=_smooth) * 100


def calc_bertscore_batch(preds, refs):
    _, _, F = _bs.score(preds, refs)
    return [round(float(f) * 100, 4) for f in F]


# ── 데이터 ────────────────────────────────────────────────────────────────

def load_cross_data():
    with open(CROSS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for i, d in enumerate(raw):
        items.append({
            "id":          i,
            "question":    d["instruction"],
            "gold_answer": d["output"],
            "gold_domain": "CROSS",
        })
    print(f"[*] Cross OOD 데이터: {len(items)}개")
    return items


# ── DB / 유틸 ─────────────────────────────────────────────────────────────

def build_db(embed_model):
    print("[*] Vector DB 구축...")
    all_texts = []
    for fname in DOMAIN_FILES.values():
        with open(os.path.join(DATA_DIR, fname)) as f:
            data = json.load(f)
        for d in data:
            out = d.get("output", d.get("gold_answer", ""))
            all_texts.append(f"{d['instruction']} {out}")
    embs = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"  DB: {len(all_texts)}개")
    return all_texts, index


def retrieve(query_or_emb, embed_model, texts, index, top_k=10, is_emb=False):
    if is_emb:
        emb = query_or_emb
    else:
        emb = embed_model.encode([query_or_emb])
        emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]], D[0]


def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    return max(scores, key=scores.get)


def load_7b():
    print("[*] 7B 모델 로드...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    print("[*] 7B 로드 완료")
    return model, tokenizer


def generate_text(prompt, model, tokenizer, max_tokens=300):
    msgs   = [{"role": "user", "content": prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in result:
            result = result[:result.index(kw)].strip()
    return result or "[FAILED]"


# ── 모델별 추론 ───────────────────────────────────────────────────────────

def run_naive_rag(items, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*50}\nNaiveRAG-7B\n{'='*50}")
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:60]}...")
        try:
            ctx, _ = retrieve(q, embed_model, texts, index)
            ctx_text = "\n".join([c[:200] for c in ctx])
            prompt = (f"You are a helpful expert.\n\nContext:\n{ctx_text}\n\n"
                      f"Question: {q}\n\nProvide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
            ans = generate_text(prompt, model, tokenizer)
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
        results.append({**item, "system": "NaiveRAG-7B", "generated_answer": ans, "pred_domain": ""})
    return results


def run_adaptive_rag(items, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*50}\nAdaptiveRAG-7B\n{'='*50}")
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:60]}...")
        try:
            ctx, _ = retrieve(q, embed_model, texts, index)
            domain = keyword_route(q)
            system = SYSTEMS.get(domain, "You are a helpful expert.")
            ctx_text = "\n".join([c[:200] for c in ctx])
            prompt = (f"{system}\n\nContext:\n{ctx_text}\n\n"
                      f"Question: {q}\n\nProvide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
            ans = generate_text(prompt, model, tokenizer)
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
        results.append({**item, "system": "AdaptiveRAG-7B", "generated_answer": ans, "pred_domain": ""})
    return results


def run_hyde(items, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*50}\nHyDE-7B\n{'='*50}")
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:60]}...")
        try:
            hypo_prompt = (f"You are a helpful expert.\n\nQuestion: {q}\n\n"
                           f"Write a brief hypothetical answer (2-3 sentences) to help retrieve relevant documents:\n\nAnswer:")
            hypo_answer = generate_text(hypo_prompt, model, tokenizer, max_tokens=100)
            hypo_emb = embed_model.encode([hypo_answer])
            hypo_emb = hypo_emb / np.linalg.norm(hypo_emb)
            ctx, _ = retrieve(hypo_emb, embed_model, texts, index, is_emb=True)
            ctx_text = "\n".join([c[:200] for c in ctx])
            prompt = (f"You are a helpful expert.\n\nContext:\n{ctx_text}\n\n"
                      f"Question: {q}\n\nProvide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
            ans = generate_text(prompt, model, tokenizer)
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
        results.append({**item, "system": "HyDE-7B", "generated_answer": ans, "pred_domain": ""})
    return results


def run_crag(items, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*50}\nCRAG-7B\n{'='*50}")
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:60]}...")
        try:
            ctx, scores = retrieve(q, embed_model, texts, index)
            top_score = float(scores[0])
            if top_score >= 0.85:
                final_ctx = ctx
            elif top_score >= 0.60:
                expanded = f"{q} explanation details mechanism"
                extra_ctx, _ = retrieve(expanded, embed_model, texts, index, top_k=5)
                final_ctx = ctx[:5] + extra_ctx[:5]
            else:
                final_ctx = []

            if final_ctx:
                ctx_text = "\n".join([c[:200] for c in final_ctx])
                prompt = (f"You are a helpful expert.\n\nContext:\n{ctx_text}\n\n"
                          f"Question: {q}\n\nProvide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
            else:
                prompt = (f"You are a helpful expert.\n\nQuestion: {q}\n\n"
                          f"Based on your knowledge, provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
            ans = generate_text(prompt, model, tokenizer)
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
        results.append({**item, "system": "CRAG-7B", "generated_answer": ans, "pred_domain": ""})
    return results


def run_nexroute(items, embed_model):
    print(f"\n{'='*50}\nNexRoute-7B\n{'='*50}")
    pipe = DualMoEPipeline()
    pipe.load_router()
    pipe.embed_model = embed_model
    pipe.load_base_model()
    pipe.build_db()

    results = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:60]}...")
        try:
            out = pipe.run(q)
            pred_domain = out["domains"][0]
            ans = out["answer"]
        except Exception as e:
            print(f"    [Error] {e}")
            ans = "[FAILED]"
            pred_domain = "UNKNOWN"
        results.append({**item, "system": "NexRoute-7B", "generated_answer": ans, "pred_domain": pred_domain})

    del pipe.base_model
    pipe.base_model = None
    torch.cuda.empty_cache()
    print("[*] NexRoute LLM 해제")
    return results


# ── 지표 붙이기 ───────────────────────────────────────────────────────────

def attach_metrics(records):
    valid = [r for r in records if r["generated_answer"] not in ("[FAILED]", "")]
    if not valid:
        return records
    preds = [r["generated_answer"] for r in valid]
    refs  = [r["gold_answer"]      for r in valid]
    print(f"  BERTScore 계산 ({len(valid)}개)...")
    bs_scores = calc_bertscore_batch(preds, refs)
    for r, bss in zip(valid, bs_scores):
        r["rougeL"]    = round(calc_rouge_l(r["gold_answer"], r["generated_answer"]), 4)
        r["bleu4"]     = round(calc_bleu4(r["gold_answer"],   r["generated_answer"]), 4)
        r["bertscore"] = bss
    for r in records:
        for m in ("rougeL", "bleu4", "bertscore"):
            if m not in r:
                r[m] = 0.0
    return records


# ── 집계 ─────────────────────────────────────────────────────────────────

def aggregate(all_records):
    from collections import defaultdict
    valid = [r for r in all_records if r["generated_answer"] not in ("[FAILED]", "")]
    groups = defaultdict(lambda: {"rougeL": [], "bertscore": [], "bleu4": [],
                                  "routing_correct": [], "n": 0})
    for r in valid:
        key = r["system"]
        g   = groups[key]
        g["rougeL"].append(float(r["rougeL"]))
        g["bertscore"].append(float(r["bertscore"]))
        g["bleu4"].append(float(r["bleu4"]))
        g["n"] += 1
        if r["system"] == "NexRoute-7B":
            g["routing_correct"].append(
                1 if r["pred_domain"] == r["gold_domain"] else 0
            )

    order = ["NaiveRAG-7B", "AdaptiveRAG-7B", "HyDE-7B", "CRAG-7B", "NexRoute-7B"]

    print("\n" + "=" * 75)
    print(f"  {'System':<16} {'N':>4}  "
          f"{'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8} {'RouteAcc':>10}")
    print("-" * 75)

    summary_rows = []
    for system in order:
        g = groups.get(system)
        if not g or g["n"] == 0:
            continue
        avg_r  = round(float(np.mean(g["rougeL"])),    2)
        avg_b  = round(float(np.mean(g["bertscore"])), 2)
        avg_bl = round(float(np.mean(g["bleu4"])),     2)
        acc    = (round(float(np.mean(g["routing_correct"])) * 100, 1)
                  if g["routing_correct"] else "—")
        print(f"  {system:<16} {g['n']:>4}  "
              f"{avg_r:>8.2f} {avg_b:>10.2f} {avg_bl:>8.2f} {str(acc):>10}")
        summary_rows.append({
            "system": system, "n": g["n"],
            "rouge_l": avg_r, "bertscore": avg_b, "bleu4": avg_bl,
            "routing_acc": acc if isinstance(acc, float) else "",
        })
    print("=" * 75)

    summary_path = os.path.join(OOD_DIR, "ood_cross_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "n", "rouge_l", "bertscore", "bleu4", "routing_acc"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[*] 집계 저장: {summary_path}")

    result_path = os.path.join(OOD_DIR, "ood_cross_results.csv")
    COLS = ["id", "system", "question", "gold_answer", "generated_answer",
            "pred_domain", "gold_domain", "rougeL", "bertscore", "bleu4"]
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[*] 상세 결과 저장: {result_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────

def main():
    items = load_cross_data()

    embed_model = SentenceTransformer(EMBED_MODEL)
    texts, index = build_db(embed_model)

    # NexRoute 먼저 (LLM 해제 후 재사용)
    nr_records = run_nexroute(items, embed_model)

    # 7B 공통 모델 로드
    model, tokenizer = load_7b()

    naive_records    = run_naive_rag(items, embed_model, texts, index, model, tokenizer)
    adaptive_records = run_adaptive_rag(items, embed_model, texts, index, model, tokenizer)
    hyde_records     = run_hyde(items, embed_model, texts, index, model, tokenizer)
    crag_records     = run_crag(items, embed_model, texts, index, model, tokenizer)

    del model
    torch.cuda.empty_cache()

    print("\n[*] 지표 계산 중...")
    all_records = []
    for name, records in [("NexRoute",   nr_records),
                           ("NaiveRAG",   naive_records),
                           ("AdaptiveRAG",adaptive_records),
                           ("HyDE",       hyde_records),
                           ("CRAG",       crag_records)]:
        print(f"  {name}...")
        all_records.extend(attach_metrics(records))

    aggregate(all_records)
    print("\n[완료]")


if __name__ == "__main__":
    main()
