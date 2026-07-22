"""
eval_medquad_hyde_crag_7b.py
============================
MedQuAD 비교군 추가 평가: HyDE & CRAG (7B 모델)

평가 대상:
  1. HyDE 7B   — Hypothetical Document Embeddings
                 LLM이 가상 답변을 생성하고, 그 임베딩으로 검색
  2. CRAG 7B   — Corrective RAG
                 유사도 기반 문서 품질 판정 후 보정/정제

기존 eval_medquad_7b.py 와 동일한 조건:
  - 200 샘플 (seed=42)
  - Qwen2.5-7B-Instruct 4-bit NF4
  - intfloat/multilingual-e5-large
  - ROUGE-L, BERTScore(roberta-large), BLEU-4

출력:
  /home/sclab/moe_model/medquad_results/
    medquad_hyde_7b.json
    medquad_crag_7b.json
    medquad_scores_hyde_crag_7b.json
"""

import os, json, re, torch, numpy as np, random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── 경로 (eval_medquad_7b.py 와 동일) ─────────────────────────────────────────
BASE_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL = "intfloat/multilingual-e5-large"
OUT_DIR     = "/home/sclab/moe_model/medquad_results"
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES   = 200
TOP_K       = 10
CTX_LEN     = 200       # eval_medquad_7b.py 기준 ctx[:200]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# CRAG 임계값
CRAG_HIGH   = 0.70      # 이상: 문서 신뢰 → 그대로 사용
CRAG_LOW    = 0.45      # 미만: 문서 불신 → LLM 보조 지식 생성

SYSTEM_MED  = "You are a medical expert. Answer accurately based on medical facts and clinical knowledge."

FEW_SHOT_MED = (
    "Q: How does the renin-angiotensin system regulate blood pressure?\n"
    "A: The renin-angiotensin system regulates blood pressure through a cascade "
    "of enzymatic reactions initiated when the kidneys detect low blood pressure "
    "or reduced sodium levels. Renin converts angiotensinogen to angiotensin I, "
    "which is subsequently converted to angiotensin II by ACE in the lungs. "
    "Angiotensin II causes vasoconstriction and stimulates aldosterone secretion, "
    "increasing sodium retention and blood volume, ultimately elevating blood pressure."
)


# ── MedQuAD 로드 (동일 샘플) ──────────────────────────────────────────────────
def load_medquad(n: int, seed: int = 42) -> list:
    print("\n[*] MedQuAD 로드...")
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    items = []
    for row in ds:
        q = (row.get("Question") or "").strip()
        a = (row.get("Answer")   or "").strip()
        if not q or not a or len(a.split()) < 10:
            continue
        items.append({"instruction": q, "gold_answer": a,
                      "qtype": row.get("qtype", "")})
    random.seed(seed)
    sampled = random.sample(items, min(n, len(items)))
    print(f"    샘플링: {len(sampled)}개")
    return sampled


# ── 통합 벡터 DB 구축 ─────────────────────────────────────────────────────────
def build_unified_db(embed_model) -> tuple:
    print("\n[*] 통합 벡터 DB 구축...")
    files = {
        "LAW":   "law_rewritten.json",
        "MED":   "med_rewritten.json",
        "SOC":   "soc_rewritten.json",
        "STEM":  "stem_rewritten.json",
        "CROSS": "cross_rewritten.json",
    }
    all_texts = []
    for domain, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"    [Skip] {fname}")
            continue
        with open(path) as f:
            data = json.load(f)
        for d in data:
            out = d.get("output", d.get("gold_answer", ""))
            if any(k in out for k in ["The correct answer", "All answers:"]):
                continue
            all_texts.append(f"{d['instruction']} {out}")

    embs  = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"    통합 DB: {len(all_texts)}개")
    return all_texts, index


# ── 기본 검색 ─────────────────────────────────────────────────────────────────
def retrieve(query_or_emb, embed_model, texts, index, top_k=TOP_K,
             precomputed: bool = False):
    """precomputed=True면 query_or_emb가 이미 정규화된 임베딩 벡터."""
    if precomputed:
        emb = query_or_emb
    else:
        emb = embed_model.encode([query_or_emb])
        emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]], D[0]   # (texts, scores)


# ── LLM 생성 헬퍼 ─────────────────────────────────────────────────────────────
def _generate(prompt: str, model, tokenizer,
              max_new_tokens: int = 400, temperature: float = 0.7) -> str:
    msgs   = [{"role": "user", "content": prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    # MCQ 잔재 제거 (기존 스크립트와 동일)
    answer = re.sub(r"The correct answer is[^\n]*\n?", "", answer).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[FAILED]"


def generate_final(query: str, ctx_strs: list, model, tokenizer) -> str:
    ctx = "\n".join([c[:CTX_LEN] for c in ctx_strs])
    prompt = (
        f"{SYSTEM_MED}\n\n"
        f"Here is an example of a high-quality answer:\n{FEW_SHOT_MED}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form. "
        f"Do NOT use multiple choice format.\n\nAnswer:"
    )
    return _generate(prompt, model, tokenizer)


# ══════════════════════════════════════════════════════════════════════════════
#  HyDE — Hypothetical Document Embeddings
# ══════════════════════════════════════════════════════════════════════════════
def hyde_hypothetical(query: str, model, tokenizer) -> str:
    """컨텍스트 없이 가상 답변 생성."""
    prompt = (
        f"{SYSTEM_MED}\n\n"
        f"Question: {query}\n\n"
        f"Write a short, medically accurate hypothetical answer (2-3 sentences) "
        f"that could appear in a medical textbook. "
        f"Do NOT use multiple choice format.\n\nAnswer:"
    )
    return _generate(prompt, model, tokenizer, max_new_tokens=150, temperature=0.5)


def run_hyde(medquad_data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*55}\n[HyDE 7B] MedQuAD 평가\n{'='*55}")
    results  = []
    out_path = os.path.join(OUT_DIR, "medquad_hyde_7b.json")

    for i, item in enumerate(medquad_data):
        query = item["instruction"]
        gold  = item["gold_answer"]

        # 1. 가상 답변 생성
        hypo_doc = hyde_hypothetical(query, model, tokenizer)

        # 2. 가상 답변 임베딩으로 검색
        hypo_emb = embed_model.encode([hypo_doc])
        hypo_emb = hypo_emb / np.linalg.norm(hypo_emb)
        ctx_strs, _ = retrieve(hypo_emb, embed_model, texts, index,
                               precomputed=True)

        # 3. 최종 답변 생성
        answer = generate_final(query, ctx_strs, model, tokenizer)

        results.append({
            "query": query, "gold_answer": gold,
            "hypo_doc": hypo_doc, "pred_answer": answer
        })

        if (i + 1) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(medquad_data)}] 중간 저장")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CRAG — Corrective RAG
# ══════════════════════════════════════════════════════════════════════════════
def crag_evaluate_relevance(query: str, docs: list, scores: np.ndarray) -> str:
    """
    임베딩 코사인 유사도로 문서 품질 판정.
    'correct'   : max_score >= CRAG_HIGH
    'incorrect' : max_score <  CRAG_LOW
    'ambiguous' : 그 사이
    """
    max_score = float(scores[0]) if len(scores) > 0 else 0.0
    if max_score >= CRAG_HIGH:
        return "correct"
    elif max_score < CRAG_LOW:
        return "incorrect"
    else:
        return "ambiguous"


def crag_refine_strips(docs: list, query: str) -> list:
    """
    Ambiguous: 문서를 문장 단위로 분해하고 쿼리 키워드 포함 문장만 선택.
    (경량 지식 정제 — 웹검색 없이)
    """
    q_words = set(query.lower().split())
    strips  = []
    for doc in docs:
        for sent in re.split(r'(?<=[.!?])\s+', doc):
            sent = sent.strip()
            if not sent:
                continue
            s_words = set(sent.lower().split())
            if len(q_words & s_words) >= 2:
                strips.append(sent)
    return strips[:TOP_K] if strips else docs[:3]


def crag_generate_fallback(query: str, model, tokenizer) -> str:
    """Incorrect 판정: LLM 파라메트릭 지식으로 보조 컨텍스트 생성."""
    prompt = (
        f"{SYSTEM_MED}\n\n"
        f"The retrieved documents are not relevant. "
        f"Generate factual background knowledge about the following medical question "
        f"based on your training knowledge (3-4 sentences, no multiple choice).\n\n"
        f"Question: {query}\n\nKnowledge:"
    )
    return _generate(prompt, model, tokenizer, max_new_tokens=200, temperature=0.5)


def run_crag(medquad_data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*55}\n[CRAG 7B] MedQuAD 평가\n{'='*55}")
    results   = []
    out_path  = os.path.join(OUT_DIR, "medquad_crag_7b.json")
    judgments = {"correct": 0, "ambiguous": 0, "incorrect": 0}

    for i, item in enumerate(medquad_data):
        query = item["instruction"]
        gold  = item["gold_answer"]

        # 1. 초기 검색
        ctx_strs, scores = retrieve(query, embed_model, texts, index)

        # 2. 관련성 판정
        judgment = crag_evaluate_relevance(query, ctx_strs, scores)
        judgments[judgment] += 1

        # 3. 보정
        if judgment == "correct":
            final_ctx = ctx_strs

        elif judgment == "ambiguous":
            refined   = crag_refine_strips(ctx_strs, query)
            final_ctx = refined if refined else ctx_strs

        else:  # incorrect
            fallback  = crag_generate_fallback(query, model, tokenizer)
            # fallback 지식 + 상위 문서 혼합
            final_ctx = [fallback] + ctx_strs[:3]

        # 4. 최종 답변 생성
        answer = generate_final(query, final_ctx, model, tokenizer)

        results.append({
            "query": query, "gold_answer": gold,
            "pred_answer": answer,
            "crag_judgment": judgment,
            "max_retrieval_score": float(scores[0]) if len(scores) > 0 else 0.0
        })

        if (i + 1) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(medquad_data)}] 중간 저장  "
                  f"판정: {judgments}")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_path}")
    print(f"  [CRAG] 판정 분포: {judgments}")
    return results, judgments


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────
def compute_metrics(items: list, tag: str) -> dict:
    if not items:
        return {"rouge_l": None, "bertscore": None, "bleu4": None, "n": 0}

    golds = [x["gold"] for x in items]
    preds = [x["pred"] for x in items]
    n     = len(items)

    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = np.mean([rscorer.score(g, p)["rougeL"].fmeasure
                  for g, p in zip(golds, preds)]) * 100

    _, _, F1 = bert_score_fn(
        preds, golds, lang="en", model_type="roberta-large",
        rescale_with_baseline=False, verbose=False, device=DEVICE
    )
    bs = F1.mean().item() * 100

    smooth = SmoothingFunction().method1
    bleu   = np.mean([
        sentence_bleu(
            [nltk.word_tokenize(g.lower())],
            nltk.word_tokenize(p.lower()),
            smoothing_function=smooth
        ) for g, p in zip(golds, preds)
    ]) * 100

    print(f"  [{tag}] n={n}  ROUGE-L={rl:.2f}  BERTScore={bs:.2f}  BLEU-4={bleu:.2f}")
    return {"rouge_l": round(float(rl), 2),
            "bertscore": round(float(bs), 2),
            "bleu4":    round(float(bleu), 2),
            "n": n}


# ── 최종 테이블 출력 ──────────────────────────────────────────────────────────
def print_table(score_table: dict):
    print("\n" + "=" * 70)
    print(f"  MedQuAD HyDE & CRAG 7B 평가 결과 (N={N_SAMPLES})")
    print("=" * 70)
    hdr = f"  {'Model':<20}  {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8}"
    print(hdr)
    print("-" * 70)
    for name, v in score_table.items():
        m = v["metrics"]
        def f(x): return f"{x:.2f}" if x is not None else "  N/A"
        print(f"  {name:<20}  {f(m['rouge_l']):>8} {f(m['bertscore']):>10} {f(m['bleu4']):>8}")
    print("=" * 70)

    # 기존 결과 (참고용)
    existing_path = os.path.join(OUT_DIR, "medquad_scores_7b.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            prev = json.load(f)
        print("\n  ── 기존 결과 (참고) ──")
        print(f"  {'Model':<20}  {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8}")
        print("-" * 70)
        for name, v in prev.get("models", {}).items():
            m = v["metrics"]
            def f(x): return f"{x:.2f}" if x is not None else "  N/A"
            print(f"  {name:<20}  {f(m['rouge_l']):>8} {f(m['bertscore']):>10} {f(m['bleu4']):>8}")
        print("=" * 70)


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. 데이터 & 임베딩 모델
    medquad_data = load_medquad(N_SAMPLES)
    print("\n[*] 임베딩 모델 로드...")
    embed_model  = SentenceTransformer(EMBED_MODEL)

    # 2. 통합 벡터 DB
    texts, unified_index = build_unified_db(embed_model)

    # 3. 7B 베이스 모델 (4-bit NF4)
    print("\n[*] Qwen2.5-7B-Instruct 로드 (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager"
    )
    print("[*] 7B 로드 완료!")

    score_table = {}

    # ── HyDE 7B ──────────────────────────────────────────────────────────────
    hyde_results = run_hyde(medquad_data, embed_model, texts, unified_index,
                            model, tokenizer)
    items_hyde = [{"gold": r["gold_answer"], "pred": r["pred_answer"]}
                  for r in hyde_results if r["pred_answer"] not in ["", "[FAILED]"]]
    score_table["HyDE 7B"] = {
        "metrics": compute_metrics(items_hyde, "HyDE 7B"),
        "routing_acc": None
    }

    # ── CRAG 7B ──────────────────────────────────────────────────────────────
    crag_results, crag_judgments = run_crag(medquad_data, embed_model, texts,
                                            unified_index, model, tokenizer)
    items_crag = [{"gold": r["gold_answer"], "pred": r["pred_answer"]}
                  for r in crag_results if r["pred_answer"] not in ["", "[FAILED]"]]
    score_table["CRAG 7B"] = {
        "metrics": compute_metrics(items_crag, "CRAG 7B"),
        "routing_acc": None,
        "crag_judgments": crag_judgments
    }

    # ── 저장 & 출력 ──────────────────────────────────────────────────────────
    final_path = os.path.join(OUT_DIR, "medquad_scores_hyde_crag_7b.json")
    with open(final_path, "w") as f:
        json.dump({
            "dataset": "MedQuAD (keivalya/MedQuad-MedicalQnADataset)",
            "n_samples": N_SAMPLES,
            "models": score_table,
            "crag_judgment_distribution": crag_judgments,
        }, f, indent=2, ensure_ascii=False)

    print_table(score_table)
    print(f"\n최종 저장: {final_path}")
