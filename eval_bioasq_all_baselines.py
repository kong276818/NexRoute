"""
BioASQ Cross-Domain Evaluation — All Baselines
rag-datasets/rag-mini-bioasq 기반, NexRoute vs 6개 베이스라인 비교

Systems:
  1. NexRoute-7B       (our system)
  2. NaiveRAG-7B       (vanilla retrieve-then-generate)
  3. AdaptiveKW-7B     (keyword-based domain routing)
  4. AdaptiveTFIDF-7B  (TF-IDF + LR routing)
  5. HyDE-7B           (Hypothetical Document Embeddings)
  6. CRAG-7B           (Corrective RAG)
  7. SingleLoRA-7B     (single MED LoRA, no MoE)
"""

import json
import os
import pickle
import random
import sys

import faiss
import nltk
import numpy as np
import torch
from bert_score import BERTScorer
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from rouge_score import rouge_scorer as rs
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import DualMoEPipeline

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_7B      = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR     = "/home/sclab/moe_model/processed_v2"
LORA_MED_7B  = "/home/sclab/moe_model/lora_weights_7b_v3/lora_med"
ROUTER_PATH  = "/home/sclab/moe_model/adaptive_tfidf_router.pkl"
EMBED_MODEL  = "intfloat/multilingual-e5-large"
OUT_DIR      = os.path.dirname(os.path.abspath(__file__))
OUT_PATH     = os.path.join(OUT_DIR, "bioasq_all_baselines_results.json")
CKPT_PATH    = os.path.join(OUT_DIR, "bioasq_all_baselines_ckpt.json")

SEED          = 42
N_SAMPLE      = 100
MIN_ANS_WORDS = 20
TOP_K         = 10

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}
DOMAIN_SYSTEMS = {
    "LAW":   "You are a legal expert.",
    "MED":   "You are a medical expert.",
    "SOC":   "You are a social science expert.",
    "STEM":  "You are a science expert.",
    "CROSS": "You are a multidisciplinary expert.",
}
KW_MAP = {
    "LAW":  ["law","legal","court","statute","plaintiff","contract","judge","attorney"],
    "MED":  ["medical","patient","disease","treatment","clinical","diagnosis","symptom","drug"],
    "SOC":  ["social","society","culture","economic","political","community","psychology"],
    "STEM": ["science","physics","chemical","biology","math","engineering","technology"],
}

# ── 지표 ─────────────────────────────────────────────────────────────────────
print("[*] BERTScorer 로드 (roberta-large)...")
_bs = BERTScorer(
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
_rouge  = rs.RougeScorer(["rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def calc_rouge_l(ref, hyp):
    return _rouge.score(ref, hyp)["rougeL"].fmeasure * 100

def calc_bleu4(ref, hyp):
    return sentence_bleu(
        [nltk.word_tokenize(ref.lower())],
        nltk.word_tokenize(hyp.lower()),
        smoothing_function=_smooth,
    ) * 100

def calc_bertscore_batch(preds, refs):
    _, _, F = _bs.score(preds, refs)
    return [float(f) * 100 for f in F]

def attach_metrics(records):
    valid = [r for r in records if r.get("generated_answer","") not in ("", "[FAILED]")]
    if not valid:
        return records
    bs_scores = calc_bertscore_batch(
        [r["generated_answer"] for r in valid],
        [r["gold_answer"]      for r in valid],
    )
    for r, bss in zip(valid, bs_scores):
        r["rougeL"]    = round(calc_rouge_l(r["gold_answer"], r["generated_answer"]), 4)
        r["bleu4"]     = round(calc_bleu4(  r["gold_answer"], r["generated_answer"]), 4)
        r["bertscore"] = round(bss, 4)
    for r in records:
        for m in ("rougeL", "bleu4", "bertscore"):
            r.setdefault(m, 0.0)
    return records


# ── BioASQ 데이터 로드 ────────────────────────────────────────────────────────
def load_bioasq(n=N_SAMPLE):
    print(f"\n[BioASQ] rag-mini-bioasq 로드...")
    ds = load_dataset(
        "rag-datasets/rag-mini-bioasq",
        "question-answer-passages",
        cache_dir=os.path.join(OUT_DIR, "datasets"),
    )
    rows = ds["test"]
    filtered = [
        {"question": (r.get("question") or "").strip(),
         "gold_answer": (r.get("answer") or "").strip()}
        for r in rows
        if (r.get("question") or "").strip()
        and len((r.get("answer") or "").split()) >= MIN_ANS_WORDS
    ]
    print(f"  answer ≥ {MIN_ANS_WORDS}단어 필터 후: {len(filtered)}개")
    random.seed(SEED)
    sample = random.sample(filtered, min(n, len(filtered)))
    items  = [{"id": i, **r, "gold_domain": "MED+STEM"} for i, r in enumerate(sample)]
    print(f"  최종 샘플: {len(items)}개")
    return items


# ── Vector DB ─────────────────────────────────────────────────────────────────
def build_db(embed_model):
    print("[*] Vector DB 구축 (processed_v2)...")
    all_texts = []
    for domain, fname in DOMAIN_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        for d in data:
            out = d.get("output", d.get("gold_answer", ""))
            all_texts.append(f"{d['instruction']} {out}")
    embs  = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"  전체 DB: {len(all_texts)}개")
    return all_texts, index


# ── 공용 유틸 ─────────────────────────────────────────────────────────────────
def _retrieve(query, embed_model, texts, index, top_k=TOP_K):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]], D[0]

def _retrieve_emb(emb, texts, index, top_k=TOP_K):
    emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]], D[0]

def _load_7b():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    return mdl, tok

def _unload(model):
    del model
    torch.cuda.empty_cache()

def _generate(query, context, model, tokenizer, system="You are a helpful expert.", max_tokens=300):
    ctx_text = "\n".join([c[:200] for c in context])
    prompt = (
        f"{system}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
    )
    msgs   = [{"role": "user", "content": prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in ans:
            ans = ans[:ans.index(kw)].strip()
    return ans if ans else "[FAILED]"

def _save_ckpt(results_dict):
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)


# ── 1. NexRoute-7B ────────────────────────────────────────────────────────────
def run_nexroute(items):
    print(f"\n{'='*60}\n[1/7] NexRoute-7B\n{'='*60}")
    pipe = DualMoEPipeline()
    pipe.load_router()
    pipe.load_embed_model()
    pipe.load_base_model()
    pipe.build_db()

    records = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:65]}...")
        try:
            out = pipe.run(q)
            records.append({
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
            records.append({
                "id": item["id"], "system": "NexRoute-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": "[FAILED]", "pred_domains": "UNKNOWN", "is_cross": False,
            })

    embed_model = pipe.embed_model
    _unload(pipe.base_model)
    pipe.base_model = None
    print("[*] NexRoute LLM 해제")
    return records, embed_model


# ── 2. NaiveRAG-7B ────────────────────────────────────────────────────────────
def run_naive_rag(items, embed_model, texts, index):
    print(f"\n{'='*60}\n[2/7] NaiveRAG-7B\n{'='*60}")
    model, tok = _load_7b()
    records = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:65]}...")
        try:
            ctx, _ = _retrieve(q, embed_model, texts, index)
            ans     = _generate(q, ctx, model, tok)
            records.append({
                "id": item["id"], "system": "NaiveRAG-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": ans,
            })
        except Exception as e:
            print(f"    [Error] {e}")
            records.append({
                "id": item["id"], "system": "NaiveRAG-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": "[FAILED]",
            })
    _unload(model)
    print("[*] NaiveRAG 해제")
    return records


# ── 3. AdaptiveKeyword-7B ─────────────────────────────────────────────────────
def _kw_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KW_MAP.items()}
    return max(scores, key=scores.get)

def run_adaptive_kw(items, embed_model, texts, index):
    print(f"\n{'='*60}\n[3/7] AdaptiveKeyword-7B\n{'='*60}")
    model, tok = _load_7b()
    records = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:65]}...")
        try:
            ctx, _  = _retrieve(q, embed_model, texts, index)
            domain  = _kw_route(q)
            system  = DOMAIN_SYSTEMS.get(domain, "You are a helpful expert.")
            ans     = _generate(q, ctx, model, tok, system=system)
            records.append({
                "id": item["id"], "system": "AdaptiveKW-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": ans, "pred_domain": domain,
            })
        except Exception as e:
            print(f"    [Error] {e}")
            records.append({
                "id": item["id"], "system": "AdaptiveKW-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": "[FAILED]", "pred_domain": "UNKNOWN",
            })
    _unload(model)
    print("[*] AdaptiveKW 해제")
    return records


# ── 4. AdaptiveTFIDF-7B ───────────────────────────────────────────────────────
def run_adaptive_tfidf(items, embed_model, texts, index):
    print(f"\n{'='*60}\n[4/7] AdaptiveTFIDF-7B\n{'='*60}")
    print("[*] TF-IDF+LR 라우터 로드...")
    with open(ROUTER_PATH, "rb") as f:
        rd = pickle.load(f)
    tfidf_vec, clf = rd["tfidf"], rd["clf"]

    model, tok = _load_7b()
    records = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:65]}...")
        try:
            ctx, _  = _retrieve(q, embed_model, texts, index)
            domain  = clf.predict(tfidf_vec.transform([q]))[0]
            system  = DOMAIN_SYSTEMS.get(domain, "You are a helpful expert.")
            ans     = _generate(q, ctx, model, tok, system=system)
            records.append({
                "id": item["id"], "system": "AdaptiveTFIDF-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": ans, "pred_domain": domain,
            })
        except Exception as e:
            print(f"    [Error] {e}")
            records.append({
                "id": item["id"], "system": "AdaptiveTFIDF-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": "[FAILED]", "pred_domain": "UNKNOWN",
            })
    _unload(model)
    print("[*] AdaptiveTFIDF 해제")
    return records


# ── 5. HyDE-7B ────────────────────────────────────────────────────────────────
def run_hyde(items, embed_model, texts, index):
    print(f"\n{'='*60}\n[5/7] HyDE-7B\n{'='*60}")
    model, tok = _load_7b()
    records = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:65]}...")
        try:
            hypo_prompt = (
                "You are a helpful expert.\n\n"
                f"Question: {q}\n\n"
                "Write a brief hypothetical answer (2-3 sentences) "
                "to help retrieve relevant documents:\n\nAnswer:"
            )
            hypo_ans = _generate(q, [], model, tok,
                                  system="You are a helpful expert.", max_tokens=100)
            hypo_emb = embed_model.encode([hypo_ans]).astype(np.float32)
            ctx, _   = _retrieve_emb(hypo_emb, texts, index)
            ans      = _generate(q, ctx, model, tok)
            records.append({
                "id": item["id"], "system": "HyDE-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": ans,
            })
        except Exception as e:
            print(f"    [Error] {e}")
            records.append({
                "id": item["id"], "system": "HyDE-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": "[FAILED]",
            })
    _unload(model)
    print("[*] HyDE 해제")
    return records


# ── 6. CRAG-7B ────────────────────────────────────────────────────────────────
def run_crag(items, embed_model, texts, index):
    print(f"\n{'='*60}\n[6/7] CRAG-7B\n{'='*60}")
    model, tok = _load_7b()
    stats  = {"correct": 0, "ambiguous": 0, "incorrect": 0}
    records = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:65]}...")
        try:
            ctx, scores = _retrieve(q, embed_model, texts, index)
            top_score   = float(scores[0])

            if top_score >= 0.85:
                quality   = "correct"
                final_ctx = ctx
            elif top_score >= 0.60:
                quality   = "ambiguous"
                extra_ctx, _ = _retrieve(
                    f"{q} explanation details mechanism", embed_model, texts, index, top_k=5)
                final_ctx = ctx[:5] + extra_ctx[:5]
            else:
                quality   = "incorrect"
                final_ctx = []

            stats[quality] += 1
            if final_ctx:
                ans = _generate(q, final_ctx, model, tok)
            else:
                no_ctx_prompt = (
                    "You are a helpful expert.\n\n"
                    f"Question: {q}\n\n"
                    "Based on your knowledge, provide a comprehensive analytical answer "
                    "in flowing paragraph form.\n\nAnswer:"
                )
                msgs   = [{"role": "user", "content": no_ctx_prompt}]
                text   = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=300, temperature=0.7,
                        do_sample=True, pad_token_id=tok.eos_token_id,
                    )
                ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            records.append({
                "id": item["id"], "system": "CRAG-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": ans or "[FAILED]",
                "retrieval_quality": quality,
            })
        except Exception as e:
            print(f"    [Error] {e}")
            records.append({
                "id": item["id"], "system": "CRAG-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": "[FAILED]", "retrieval_quality": "error",
            })
    _unload(model)
    print(f"[*] CRAG 해제 | 품질 분포: {stats}")
    return records


# ── 7. SingleLoRA-7B ──────────────────────────────────────────────────────────
def run_single_lora(items, embed_model, texts, index):
    print(f"\n{'='*60}\n[7/7] SingleLoRA-7B (MED LoRA)\n{'='*60}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True)
    tok.pad_token = tok.eos_token
    base  = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, LORA_MED_7B)
    model.eval()
    print(f"[*] SingleLoRA 로드 완료 ({LORA_MED_7B})")

    records = []
    for i, item in enumerate(items):
        q = item["question"]
        print(f"  [{i+1}/{len(items)}] {q[:65]}...")
        try:
            ctx, _ = _retrieve(q, embed_model, texts, index)
            ans    = _generate(q, ctx, model, tok, system="You are a medical expert.")
            records.append({
                "id": item["id"], "system": "SingleLoRA-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": ans,
            })
        except Exception as e:
            print(f"    [Error] {e}")
            records.append({
                "id": item["id"], "system": "SingleLoRA-7B",
                "question": q, "gold_answer": item["gold_answer"],
                "generated_answer": "[FAILED]",
            })
    _unload(model)
    del base
    torch.cuda.empty_cache()
    print("[*] SingleLoRA 해제")
    return records


# ── 집계 & 출력 ────────────────────────────────────────────────────────────────
def aggregate(all_results):
    from collections import Counter

    def stats(records):
        valid = [r for r in records if r.get("generated_answer","") not in ("", "[FAILED]")]
        if not valid:
            return {"n": 0, "rouge_l": 0.0, "bertscore": 0.0, "bleu4": 0.0}
        return {
            "n":         len(valid),
            "rouge_l":   round(float(np.mean([r["rougeL"]    for r in valid])), 2),
            "bertscore": round(float(np.mean([r["bertscore"] for r in valid])), 2),
            "bleu4":     round(float(np.mean([r["bleu4"]     for r in valid])), 2),
        }

    systems = [
        "NexRoute-7B", "NaiveRAG-7B", "AdaptiveKW-7B", "AdaptiveTFIDF-7B",
        "HyDE-7B", "CRAG-7B", "SingleLoRA-7B",
    ]

    print("\n" + "=" * 70)
    print("  BioASQ (MED+STEM Cross-Domain) — All Baselines")
    print("=" * 70)
    print(f"  {'System':<20} {'N':>4}  {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8}")
    print("-" * 70)

    summary = {}
    for sys_name in systems:
        recs = all_results.get(sys_name, [])
        s    = stats(recs)
        summary[sys_name] = s
        marker = " ◀" if sys_name == "NexRoute-7B" else ""
        print(f"  {sys_name:<20} {s['n']:>4}  {s['rouge_l']:>8.2f} {s['bertscore']:>10.2f} {s['bleu4']:>8.2f}{marker}")

    print("=" * 70)

    # NexRoute 라우팅 분포
    nr_recs = all_results.get("NexRoute-7B", [])
    nr_valid = [r for r in nr_recs if r.get("generated_answer","") not in ("","[FAILED]")]
    if nr_valid:
        dist = Counter(r.get("pred_domains","") for r in nr_valid)
        cross_rate = sum(1 for r in nr_valid if r.get("is_cross")) / len(nr_valid) * 100
        print(f"\n  [NexRoute 라우팅 분포]")
        for dom, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {dom:<14}: {cnt:>3}개 ({cnt/len(nr_valid)*100:.1f}%)")
        print(f"  Cross-domain 판정 비율: {cross_rate:.1f}%")
        summary["NexRoute-7B"]["cross_rate"] = round(cross_rate, 1)

    # CRAG 품질 분포
    crag_recs = all_results.get("CRAG-7B", [])
    crag_valid = [r for r in crag_recs if r.get("generated_answer","") not in ("","[FAILED]")]
    if crag_valid:
        q_dist = Counter(r.get("retrieval_quality","") for r in crag_valid)
        print(f"\n  [CRAG 검색 품질 분포]")
        for q, cnt in q_dist.items():
            print(f"    {q:<12}: {cnt}개")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 저장: {OUT_PATH}")
    return summary


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    # 체크포인트 로드
    ckpt = {}
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            ckpt = json.load(f)
        print(f"[*] 체크포인트 로드: {list(ckpt.keys())}")

    items = load_bioasq(N_SAMPLE)

    # 1. NexRoute (embed_model 공유)
    if "NexRoute-7B" not in ckpt:
        nr_records, embed_model = run_nexroute(items)
        nr_records = attach_metrics(nr_records)
        ckpt["NexRoute-7B"] = nr_records
        _save_ckpt(ckpt)
        # embed_model 유지
    else:
        print("[*] NexRoute 체크포인트 재사용")
        print("[*] embed_model 로드...")
        embed_model = SentenceTransformer(EMBED_MODEL)

    texts, index = build_db(embed_model)

    # 2~7. 나머지 베이스라인
    runners = [
        ("NaiveRAG-7B",      lambda: run_naive_rag(items, embed_model, texts, index)),
        ("AdaptiveKW-7B",    lambda: run_adaptive_kw(items, embed_model, texts, index)),
        ("AdaptiveTFIDF-7B", lambda: run_adaptive_tfidf(items, embed_model, texts, index)),
        ("HyDE-7B",          lambda: run_hyde(items, embed_model, texts, index)),
        ("CRAG-7B",          lambda: run_crag(items, embed_model, texts, index)),
        ("SingleLoRA-7B",    lambda: run_single_lora(items, embed_model, texts, index)),
    ]

    for sys_name, runner in runners:
        if sys_name in ckpt:
            print(f"[*] {sys_name} 체크포인트 재사용")
            continue
        records = runner()
        print(f"[*] {sys_name} 지표 계산 중...")
        records = attach_metrics(records)
        ckpt[sys_name] = records
        _save_ckpt(ckpt)

    aggregate(ckpt)
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    print("\n[완료]")


if __name__ == "__main__":
    main()
