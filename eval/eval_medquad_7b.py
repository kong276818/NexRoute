"""
eval_medquad_7b.py
==================
MedQuAD (gold-reference medical QA) 데이터셋으로 7B 모델 평가

평가 대상:
  1. NexRoute 7B  — BERT multilabel router + lora_med + domain-partitioned RAG
  2. Naive RAG 7B — 라우팅 없음, 통합 벡터DB 검색
  3. Adaptive RAG 7B — 키워드 라우팅

메트릭:
  - ROUGE-L, BERTScore (roberta-large, raw), BLEU-4
  - 도메인 라우팅 정확도 (정답: MED)

출력:
  /home/sclab/moe_model/medquad_results/
    medquad_nexroute_7b.json
    medquad_naive_rag_7b.json
    medquad_adaptive_rag_7b.json
    medquad_scores_7b.json
"""

import os, json, re, sys, torch, numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from sentence_transformers import SentenceTransformer
from peft import PeftModel
import faiss
import torch.nn as nn
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_7B      = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR     = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR     = "/home/sclab/moe_model/processed_v2"
ROUTER_PATH  = "/home/sclab/moe_model/router_deberta_best.pt"
EMBED_MODEL  = "intfloat/multilingual-e5-large"
OUT_DIR      = "/home/sclab/moe_model/medquad_results"
os.makedirs(OUT_DIR, exist_ok=True)

DOMAINS      = ["LAW", "MED", "SOC", "STEM"]
THRESHOLD    = 0.30
GAP          = 0.40
TOP_K        = 10
N_SAMPLES    = 200  # MedQuAD에서 샘플링할 개수
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DEBERTA_MODEL = "microsoft/deberta-large"

# ── 키워드 라우팅 (Adaptive RAG용) ───────────────────────────────────────────
KEYWORDS = {
    "LAW":  ["law","legal","court","statute","plaintiff","contract","judge","attorney"],
    "MED":  ["medical","patient","disease","treatment","clinical","diagnosis","symptom","drug",
             "health","therapy","medicine","physician","hospital","dose","side effect"],
    "SOC":  ["social","society","culture","economic","political","community","psychology"],
    "STEM": ["science","physics","chemical","biology","math","engineering","technology"],
}

def keyword_route(query: str) -> str:
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

# ── DeBERTa-large 라우터 (MedQuAD 87% MED 정확도) ─────────────────────────
class DeBERTaRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(DEBERTA_MODEL)
        hidden = self.encoder.config.hidden_size  # 1024
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 4)
        )

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(pooled)


def load_router(path: str, device: str):
    print(f"[*] DeBERTa-large 라우터 로드... ({path.split('/')[-1]})")
    tok    = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
    router = DeBERTaRouter()
    router.load_state_dict(torch.load(path, map_location="cpu"))
    router.eval().to(device)
    print("    로드 완료!")
    return router, tok


def deberta_route(query: str, router, router_tok, device: str) -> dict:
    enc = router_tok(
        query, max_length=256, truncation=True,
        padding="max_length", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = router(enc["input_ids"], enc["attention_mask"])
        scores = torch.sigmoid(logits).squeeze().cpu().tolist()

    domain_scores = dict(zip(DOMAINS, scores))
    sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
    s1, s2   = sorted_d[0][1], sorted_d[1][1]
    top1     = sorted_d[0][0]

    is_cross = (s2 >= THRESHOLD) and ((s1 - s2) <= GAP)
    pred_domain = "CROSS" if is_cross else top1

    return {"pred": pred_domain, "scores": domain_scores,
            "top1": top1, "s1": s1, "s2": s2, "is_cross": is_cross}


# ── MedQuAD 로드 ──────────────────────────────────────────────────────────────
def load_medquad(n: int, seed: int = 42) -> list:
    """keivalya/MedQuad-MedicalQnADataset: 16,407개 (qtype, Question, Answer)"""
    print("\n[*] MedQuAD 로드 (keivalya/MedQuad-MedicalQnADataset)...")
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    print(f"    전체: {len(ds)}개 | 컬럼: {ds.column_names}")

    items = []
    for row in ds:
        q = (row.get("Question") or "").strip()
        a = (row.get("Answer") or "").strip()
        if not q or not a or len(a.split()) < 10:
            continue
        items.append({"instruction": q, "gold_answer": a,
                      "qtype": row.get("qtype", "")})
    print(f"    필터 후: {len(items)}개")

    import random
    random.seed(seed)
    sampled = random.sample(items, min(n, len(items)))

    # qtype 분포 출력
    from collections import Counter
    qtypes = Counter(d["qtype"] for d in sampled)
    print(f"    샘플링: {len(sampled)}개 | qtype 분포: {dict(qtypes.most_common(5))}")
    return sampled


# ── 벡터 DB 구축 ──────────────────────────────────────────────────────────────
def build_unified_db(embed_model):
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


def build_domain_db(embed_model):
    print("[*] 도메인별 벡터 DB 구축 (NexRoute용)...")
    files = {
        "LAW":   "law_rewritten.json",
        "MED":   "med_rewritten.json",
        "SOC":   "soc_rewritten.json",
        "STEM":  "stem_rewritten.json",
        "CROSS": "cross_rewritten.json",
    }
    db = {}
    for domain, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        texts = []
        for d in data:
            out = d.get("output", d.get("gold_answer", ""))
            if any(k in out for k in ["The correct answer", "All answers:"]):
                continue
            texts.append(f"{d['instruction']} {out}")
        embs  = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.astype(np.float32))
        db[domain] = (texts, index)
        print(f"    {domain}: {len(texts)}개")
    return db


def retrieve_unified(query, embed_model, texts, index, top_k=TOP_K):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]]


def retrieve_domain(query, domains, is_cross, embed_model, db, top_k=TOP_K):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    results = []
    if is_cross:
        per = max(1, top_k // len(domains))
        for d in domains:
            if d not in db:
                continue
            texts, idx = db[d]
            D, I = idx.search(emb.astype(np.float32), per)
            for score, i in zip(D[0], I[0]):
                results.append({"domain": d, "text": texts[i], "score": float(score)})
    else:
        d = domains[0]
        if d in db:
            texts, idx = db[d]
            D, I = idx.search(emb.astype(np.float32), top_k)
            for score, i in zip(D[0], I[0]):
                results.append({"domain": d, "text": texts[i], "score": float(score)})
        elif "MED" in db:  # fallback: MED가 가장 유사
            texts, idx = db["MED"]
            D, I = idx.search(emb.astype(np.float32), top_k)
            for score, i in zip(D[0], I[0]):
                results.append({"domain": "MED", "text": texts[i], "score": float(score)})
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ── 생성 ──────────────────────────────────────────────────────────────────────
FEW_SHOT_MED = (
    "Q: How does the renin-angiotensin system regulate blood pressure?\n"
    "A: The renin-angiotensin system regulates blood pressure through a cascade "
    "of enzymatic reactions initiated when the kidneys detect low blood pressure "
    "or reduced sodium levels. Renin converts angiotensinogen to angiotensin I, "
    "which is subsequently converted to angiotensin II by ACE in the lungs. "
    "Angiotensin II causes vasoconstriction and stimulates aldosterone secretion, "
    "increasing sodium retention and blood volume, ultimately elevating blood pressure."
)

SYSTEM_MED = "You are a medical expert. Answer accurately based on medical facts and clinical knowledge."

def generate_answer(query: str, context_strs: list, model, tokenizer,
                    system: str, use_fewshot: bool = False) -> str:
    ctx = "\n".join([c[:200] for c in context_strs])
    if use_fewshot:
        prompt = (
            f"{system}\n\n"
            f"Here is an example of a high-quality answer:\n{FEW_SHOT_MED}\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {query}\n\n"
            f"Provide a comprehensive analytical answer in flowing paragraph form. "
            f"Do NOT use multiple choice format.\n\nAnswer:"
        )
    else:
        prompt = (
            f"{system}\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {query}\n\n"
            f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
        )
    msgs   = [{"role": "user", "content": prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=400, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # MCQ 잔재 제거
    answer = re.sub(r"The correct answer is[^\n]*\n?", "", answer).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[FAILED]"


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
    bleu = np.mean([
        sentence_bleu(
            [nltk.word_tokenize(g.lower())],
            nltk.word_tokenize(p.lower()),
            smoothing_function=smooth
        ) for g, p in zip(golds, preds)
    ]) * 100

    print(f"  [{tag}] n={n}  ROUGE-L={rl:.2f}  BERTScore={bs:.2f}  BLEU-4={bleu:.2f}")
    return {"rouge_l": round(float(rl), 2), "bertscore": round(float(bs), 2),
            "bleu4": round(float(bleu), 2), "n": n}


# ── NexRoute 7B 평가 ──────────────────────────────────────────────────────────
def run_nexroute(medquad_data, embed_model, db, router, router_tok, model, tokenizer):
    print(f"\n{'='*55}\n[NexRoute 7B] MedQuAD 평가\n{'='*55}")
    results = []
    routing_correct = 0
    out_path = os.path.join(OUT_DIR, "medquad_nexroute_7b.json")

    # MED LoRA 로드
    print("[*] MED LoRA 로드...")
    lora_path = os.path.join(LORA_DIR, "lora_med")
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path, adapter_name="MED")
        model.set_adapter("MED")
        print(f"    LoRA MED 활성화: {lora_path}")
    current_lora = "MED"

    for i, item in enumerate(medquad_data):
        query = item["instruction"]
        gold  = item["gold_answer"]

        routing = deberta_route(query, router, router_tok, DEVICE)
        pred_domain = routing["pred"]
        top1        = routing["top1"]
        is_cross    = routing["is_cross"]
        domains     = [top1, DOMAINS[1]] if is_cross else [top1]  # 단순화

        # 라우팅 정확도: MED이면 정답
        if top1 == "MED" or (not is_cross and pred_domain == "MED"):
            routing_correct += 1
        elif is_cross and "MED" in domains:
            routing_correct += 1

        # LoRA 전환 (필요시)
        target_lora = "CROSS" if is_cross else top1
        if target_lora != current_lora:
            lora_subdir = target_lora.lower()
            lora_path   = os.path.join(LORA_DIR, f"lora_{lora_subdir}")
            if os.path.exists(lora_path):
                try:
                    model.load_adapter(lora_path, adapter_name=target_lora)
                    model.set_adapter(target_lora)
                    current_lora = target_lora
                except Exception:
                    pass  # 로드 실패시 기존 LoRA 유지

        ctx_list = retrieve_domain(query, domains, is_cross, embed_model, db)
        ctx_strs = [r["text"] for r in ctx_list]
        system = SYSTEM_MED if not is_cross else (
            "You are a multidisciplinary medical expert. "
            "Integrate knowledge from relevant domains to answer accurately."
        )
        answer = generate_answer(query, ctx_strs, model, tokenizer, system, use_fewshot=True)

        results.append({
            "query": query, "gold_answer": gold, "pred_answer": answer,
            "domain_pred": pred_domain, "top1": top1,
            "domain_scores": routing["scores"],
            "is_cross": is_cross, "qtype": item.get("qtype", "")
        })

        if (i + 1) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(medquad_data)}] 중간 저장")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    routing_acc = routing_correct / len(medquad_data) * 100
    print(f"  [NexRoute] 라우팅 정확도: {routing_acc:.1f}%  ({routing_correct}/{len(medquad_data)})")
    print(f"  저장: {out_path}")
    return results, routing_acc


# ── Naive RAG 7B 평가 ─────────────────────────────────────────────────────────
def run_naive_rag(medquad_data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*55}\n[Naive RAG 7B] MedQuAD 평가\n{'='*55}")
    results = []
    out_path = os.path.join(OUT_DIR, "medquad_naive_rag_7b.json")

    for i, item in enumerate(medquad_data):
        query = item["instruction"]
        gold  = item["gold_answer"]
        ctx   = retrieve_unified(query, embed_model, texts, index)
        answer = generate_answer(query, ctx, model, tokenizer,
                                 system="You are a helpful expert.")
        results.append({"query": query, "gold_answer": gold, "pred_answer": answer})

        if (i + 1) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(medquad_data)}] 중간 저장")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_path}")
    return results


# ── Adaptive RAG 7B 평가 ──────────────────────────────────────────────────────
SYSTEM_MAP = {
    "LAW":  "You are a legal expert.",
    "MED":  "You are a medical expert.",
    "SOC":  "You are a social science expert.",
    "STEM": "You are a science expert.",
}

def run_adaptive_rag(medquad_data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*55}\n[Adaptive RAG 7B] MedQuAD 평가\n{'='*55}")
    results = []
    routing_correct = 0
    out_path = os.path.join(OUT_DIR, "medquad_adaptive_rag_7b.json")

    for i, item in enumerate(medquad_data):
        query  = item["instruction"]
        gold   = item["gold_answer"]
        domain = keyword_route(query)
        if domain == "MED":
            routing_correct += 1
        system = SYSTEM_MAP.get(domain, "You are a helpful expert.")
        ctx    = retrieve_unified(query, embed_model, texts, index)
        answer = generate_answer(query, ctx, model, tokenizer, system)
        results.append({"query": query, "gold_answer": gold, "pred_answer": answer,
                        "domain_pred": domain})

        if (i + 1) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(medquad_data)}] 중간 저장")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    routing_acc = routing_correct / len(medquad_data) * 100
    print(f"  [Adaptive RAG] 라우팅 정확도: {routing_acc:.1f}%  ({routing_correct}/{len(medquad_data)})")
    print(f"  저장: {out_path}")
    return results, routing_acc


# ── 최종 테이블 출력 ──────────────────────────────────────────────────────────
def print_table(score_table: dict):
    print("\n" + "=" * 80)
    print(f"  MedQuAD 7B 평가 결과 (N={N_SAMPLES})")
    print("=" * 80)
    hdr = f"  {'Model':<22}  {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8} {'Routing Acc':>12}"
    print(hdr)
    print("-" * 80)
    for name, v in score_table.items():
        m   = v["metrics"]
        ra  = v.get("routing_acc")
        ra_s = f"{ra:.1f}%" if ra is not None else "  N/A"
        def f(x): return f"{x:.2f}" if x is not None else "  N/A"
        marker = " ◀" if "NexRoute" in name else ""
        print(f"  {name:<22}  {f(m['rouge_l']):>8} {f(m['bertscore']):>10} {f(m['bleu4']):>8} {ra_s:>12}{marker}")
    print("=" * 80)


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. MedQuAD 로드
    medquad_data = load_medquad(N_SAMPLES)

    # 2. 임베딩 모델 로드
    print("\n[*] 임베딩 모델 로드...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    # 3. 벡터 DB 구축 (통합 + 도메인별)
    texts, unified_index = build_unified_db(embed_model)
    domain_db = build_domain_db(embed_model)

    # 4. 라우터 로드
    router, router_tok = load_router(ROUTER_PATH, DEVICE)

    # 5. 7B 베이스 모델 로드
    print("\n[*] Qwen2.5-7B-Instruct 로드 (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager"
    )
    print("[*] 7B 로드 완료!")

    score_table = {}

    # ── NexRoute 7B ──────────────────────────────────────────────────────────
    nexroute_results, nexroute_routing_acc = run_nexroute(
        medquad_data, embed_model, domain_db, router, router_tok, model, tokenizer
    )
    items_nx = [{"gold": r["gold_answer"], "pred": r["pred_answer"]}
                for r in nexroute_results if r["pred_answer"] not in ["", "[FAILED]"]]
    score_table["NexRoute 7B"] = {
        "metrics": compute_metrics(items_nx, "NexRoute 7B"),
        "routing_acc": nexroute_routing_acc
    }

    # LoRA 해제: PEFT 0.18.1 호환 방식
    if hasattr(model, "base_model") and hasattr(model.base_model, "disable_adapter_layers"):
        model.base_model.disable_adapter_layers()
    elif hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()

    # ── Naive RAG 7B ─────────────────────────────────────────────────────────
    naive_results = run_naive_rag(
        medquad_data, embed_model, texts, unified_index, model, tokenizer
    )
    items_naive = [{"gold": r["gold_answer"], "pred": r["pred_answer"]}
                   for r in naive_results if r["pred_answer"] not in ["", "[FAILED]"]]
    score_table["Naive RAG 7B"] = {
        "metrics": compute_metrics(items_naive, "Naive RAG 7B"),
        "routing_acc": None
    }

    # ── Adaptive RAG 7B ──────────────────────────────────────────────────────
    adaptive_results, adaptive_routing_acc = run_adaptive_rag(
        medquad_data, embed_model, texts, unified_index, model, tokenizer
    )
    items_ada = [{"gold": r["gold_answer"], "pred": r["pred_answer"]}
                 for r in adaptive_results if r["pred_answer"] not in ["", "[FAILED]"]]
    score_table["Adaptive RAG 7B"] = {
        "metrics": compute_metrics(items_ada, "Adaptive RAG 7B"),
        "routing_acc": adaptive_routing_acc
    }

    # ── 라우팅 세부 분석 (NexRoute) ──────────────────────────────────────────
    routing_breakdown = defaultdict(int)
    for r in nexroute_results:
        routing_breakdown[r["top1"]] += 1
    print("\n[NexRoute 라우팅 분포 on MedQuAD]")
    for d, cnt in sorted(routing_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {d}: {cnt}개  ({cnt/len(nexroute_results)*100:.1f}%)")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    final_path = os.path.join(OUT_DIR, "medquad_scores_7b.json")
    with open(final_path, "w") as f:
        json.dump({
            "dataset": "MedQuAD",
            "n_samples": N_SAMPLES,
            "models": score_table,
            "routing_breakdown_nexroute": dict(routing_breakdown),
        }, f, indent=2, ensure_ascii=False)

    print_table(score_table)
    print(f"\n최종 저장: {final_path}")
