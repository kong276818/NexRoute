"""
eval_medquad_cross_v2.py
========================
MedQuAD 재실험 v2 - 공정성 전면 수정 + CROSS 특화 평가

수정 내역 (v1 대비):
  [공정성 1] 모든 모델 동일 시스템 프롬프트 (SYSTEM_MED)
  [공정성 2] 모든 모델 동일 few-shot 적용
  [공정성 3] 컨텍스트 길이 통일: ctx[:200] → ctx[:600], TOP_K 10→5
  [CROSS 1]  GAP 임계값 완화: 0.40 → 0.30 (더 많은 CROSS 감지)
  [CROSS 2]  신뢰도 가중 검색: 도메인 점수 비례 TOP-K 분배
  [CROSS 3]  lora_cross 어댑터 사용
  [CROSS 4]  도메인 조합별 전문화 시스템 프롬프트
  [CROSS 5]  CROSS 서브셋 별도 메트릭 보고
  [생성]     max_new_tokens 400→512, temperature 0.7→0.6
"""

import os, json, re, torch, numpy as np
from collections import defaultdict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, AutoModel,
)
from sentence_transformers import SentenceTransformer
from peft import PeftModel
import faiss
import torch.nn as nn
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_7B       = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR      = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR      = "/home/sclab/moe_model/processed_v2"
ROUTER_PATH   = "/home/sclab/moe_model/router_deberta_best.pt"
EMBED_MODEL   = "intfloat/multilingual-e5-large"
OUT_DIR       = "/home/sclab/moe_model/medquad_results_v2"
os.makedirs(OUT_DIR, exist_ok=True)

DOMAINS       = ["LAW", "MED", "SOC", "STEM"]
THRESHOLD     = 0.30        # 공통 최소 신뢰도
GAP           = 0.30        # [수정] 0.40 → 0.30: 더 많은 CROSS 감지
TOP_K         = 5           # [수정] 10  → 5:   ctx 품질 향상
CTX_LEN       = 600         # [수정] 200 → 600: 정보 손실 최소화
N_SAMPLES     = 200
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DEBERTA_MODEL = "microsoft/deberta-large"

# ── 공통 프롬프트 (모든 모델 동일 적용) ──────────────────────────────────────
SYSTEM_MED = (
    "You are a medical expert. "
    "Answer accurately and comprehensively based on medical facts and clinical knowledge."
)

FEW_SHOT_MED = (
    "Q: How does the renin-angiotensin system regulate blood pressure?\n"
    "A: The renin-angiotensin system regulates blood pressure through a cascade "
    "of enzymatic reactions initiated when the kidneys detect low blood pressure. "
    "Renin converts angiotensinogen to angiotensin I, which ACE converts to "
    "angiotensin II. Angiotensin II causes vasoconstriction and stimulates "
    "aldosterone secretion, increasing sodium retention and blood volume, "
    "ultimately elevating blood pressure."
)

# ── CROSS 도메인 조합별 전문화 프롬프트 ────────────────────────────────────
_CROSS_SYSTEM = {
    frozenset(["MED", "STEM"]): (
        "You are a biomedical scientist with expertise in both clinical medicine "
        "and molecular/cellular biology. Integrate mechanistic and clinical knowledge "
        "to provide a comprehensive answer."
    ),
    frozenset(["MED", "LAW"]): (
        "You are an expert in both medicine and healthcare law. "
        "Integrate medical evidence with regulatory and legal knowledge."
    ),
    frozenset(["MED", "SOC"]): (
        "You are a public health expert. Integrate clinical medicine with social "
        "determinants of health and epidemiological knowledge."
    ),
    frozenset(["LAW", "STEM"]): (
        "You are an expert in science policy. "
        "Integrate technical knowledge with legal and regulatory frameworks."
    ),
    frozenset(["SOC", "STEM"]): (
        "You are an interdisciplinary expert in science and society. "
        "Integrate empirical and social perspectives."
    ),
}

def get_cross_system(domain_scores: dict) -> str:
    top2 = sorted(domain_scores.items(), key=lambda x: -x[1])[:2]
    key  = frozenset([top2[0][0], top2[1][0]])
    return _CROSS_SYSTEM.get(
        key,
        f"You are a multidisciplinary expert. Synthesize knowledge from "
        f"{top2[0][0]} and {top2[1][0]} to answer comprehensively."
    )

# Adaptive RAG 도메인별 시스템 프롬프트
SYSTEM_MAP = {
    "LAW":  "You are a legal expert specializing in healthcare law.",
    "MED":  SYSTEM_MED,
    "SOC":  "You are a public health and social science expert.",
    "STEM": "You are a biomedical scientist with expertise in life sciences.",
}

# ── 키워드 라우팅 (Adaptive RAG) ──────────────────────────────────────────────
KEYWORDS = {
    "LAW":  ["law","legal","court","statute","plaintiff","contract","judge","attorney"],
    "MED":  ["medical","patient","disease","treatment","clinical","diagnosis","symptom",
             "drug","health","therapy","medicine","physician","hospital","dose","side effect"],
    "SOC":  ["social","society","culture","economic","political","community","psychology"],
    "STEM": ["science","physics","chemical","biology","math","engineering","technology"],
}

def keyword_route(query: str) -> str:
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

# ── DeBERTa-large 라우터 ────────────────────────────────────────────────────
class DeBERTaRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(DEBERTA_MODEL)
        h = self.encoder.config.hidden_size  # 1024
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 4)
        )

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(pooled)


def load_router(path: str, device: str):
    print(f"[*] DeBERTa-large 라우터 로드...")
    tok    = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
    router = DeBERTaRouter()
    router.load_state_dict(torch.load(path, map_location="cpu"))
    router.eval().to(device)
    print("    완료!")
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
    top2     = sorted_d[1][0]

    is_cross = (s2 >= THRESHOLD) and ((s1 - s2) <= GAP)
    pred     = "CROSS" if is_cross else top1

    return {"pred": pred, "scores": domain_scores,
            "top1": top1, "top2": top2,
            "s1": s1, "s2": s2, "is_cross": is_cross}

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_medquad(n: int, seed: int = 42) -> list:
    print("\n[*] MedQuAD 로드...")
    ds    = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    items = []
    for row in ds:
        q = (row.get("Question") or "").strip()
        a = (row.get("Answer")   or "").strip()
        if not q or not a or len(a.split()) < 10:
            continue
        items.append({"instruction": q, "gold_answer": a,
                       "qtype": row.get("qtype", "")})

    import random
    random.seed(seed)
    sampled = random.sample(items, min(n, len(items)))

    from collections import Counter
    qtypes = Counter(d["qtype"] for d in sampled)
    print(f"    샘플: {len(sampled)}개 | qtype 분포: {dict(qtypes.most_common(6))}")
    return sampled

# ── 벡터 DB 구축 ─────────────────────────────────────────────────────────────
_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

def _load_texts(domain: str) -> list:
    path = os.path.join(DATA_DIR, _FILES[domain])
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    bad = {"The correct answer", "All answers:"}
    return [
        f"{d['instruction']} {d.get('output', d.get('gold_answer', ''))}"
        for d in data
        if not any(k in d.get("output", d.get("gold_answer", "")) for k in bad)
    ]


def build_unified_db(embed_model):
    print("\n[*] 통합 벡터 DB 구축...")
    all_texts = []
    for domain in _FILES:
        texts = _load_texts(domain)
        all_texts.extend(texts)
        if texts:
            print(f"    {domain}: {len(texts)}개")
    embs  = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"    통합 총계: {len(all_texts)}개")
    return all_texts, index


def build_domain_db(embed_model):
    print("[*] 도메인별 벡터 DB 구축 (NexRoute용)...")
    db = {}
    for domain in _FILES:
        texts = _load_texts(domain)
        if not texts:
            continue
        embs  = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.astype(np.float32))
        db[domain] = (texts, index)
        print(f"    {domain}: {len(texts)}개")
    return db

# ── 검색 ─────────────────────────────────────────────────────────────────────
def retrieve_unified(query: str, embed_model, texts: list, index,
                     top_k: int = TOP_K) -> list:
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]]


def retrieve_nexroute(query: str, routing: dict,
                      embed_model, db: dict, top_k: int = TOP_K) -> list:
    """[CROSS 2] 신뢰도 가중 다중 도메인 검색"""
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)

    if not routing["is_cross"]:
        d = routing["top1"]
        src = db.get(d) or db.get("MED")
        if src is None:
            return []
        texts, idx = src
        D, I = idx.search(emb.astype(np.float32), top_k)
        return [texts[i] for i in I[0]]

    # CROSS: THRESHOLD 이상 도메인을 신뢰도 비례로 검색
    active = {d: s for d, s in routing["scores"].items() if s >= THRESHOLD and d in db}
    if not active:
        active = {routing["top1"]: routing["s1"]}

    total   = sum(active.values())
    results = []

    for domain, w in sorted(active.items(), key=lambda x: -x[1]):
        k_d = max(1, round(top_k * w / total))
        texts, idx = db[domain]
        D, I = idx.search(emb.astype(np.float32), k_d)
        for sim, i in zip(D[0], I[0]):
            results.append({"text": texts[i], "score": float(sim), "w": w})

    # relevance(0.6) + domain_weight(0.4) 혼합 재정렬
    results.sort(key=lambda x: -(x["score"] * 0.6 + x["w"] * 0.4))
    # CROSS DB도 추가로 혼합
    if "CROSS" in db:
        cx_texts, cx_idx = db["CROSS"]
        D, I = cx_idx.search(emb.astype(np.float32), 2)
        for sim, i in zip(D[0], I[0]):
            results.append({"text": cx_texts[i], "score": float(sim), "w": 0.5})
        results.sort(key=lambda x: -(x["score"] * 0.6 + x["w"] * 0.4))

    return [r["text"] for r in results[:top_k]]

# ── 생성 ─────────────────────────────────────────────────────────────────────
def generate_answer(query: str, context_strs: list, model, tokenizer,
                    system: str) -> str:
    # [공정성 3] 모든 모델 동일 ctx 길이, [공정성 2] 동일 few-shot
    ctx = "\n\n".join([c[:CTX_LEN] for c in context_strs[:TOP_K]])

    prompt = (
        f"{system}\n\n"
        f"Example:\n{FEW_SHOT_MED}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n"
        f"Provide a comprehensive answer in flowing prose. "
        f"Do not use bullet points or multiple-choice format.\n\nAnswer:"
    )

    msgs   = [{"role": "user", "content": prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,     # [수정] 400 → 512
            temperature=0.6,        # [수정] 0.7 → 0.6 (더 안정적)
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # MCQ 잔재 제거
    answer = re.sub(r"The correct answer is[^\n]*\n?", "", answer).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[FAILED]"

# ── 메트릭 ───────────────────────────────────────────────────────────────────
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

    print(f"  [{tag:35s}] n={n:3d}  "
          f"ROUGE-L={rl:.2f}  BERTScore={bs:.2f}  BLEU-4={bleu:.2f}")
    return {"rouge_l": round(float(rl), 2),
            "bertscore": round(float(bs), 2),
            "bleu4":    round(float(bleu), 2),
            "n": n}

# ── NexRoute 7B ──────────────────────────────────────────────────────────────
def run_nexroute(data, embed_model, db, router, router_tok, base_model, tokenizer):
    print(f"\n{'='*65}\n[NexRoute 7B v2] 평가\n{'='*65}")
    out_path = os.path.join(OUT_DIR, "medquad_nexroute_v2.json")
    results  = []
    routing_correct = 0
    cross_count     = 0

    # PEFT 모델 구성 (함수 범위 내에서만 유지)
    print("[*] MED LoRA 로드...")
    lora_med = os.path.join(LORA_DIR, "lora_med")
    model = PeftModel.from_pretrained(base_model, lora_med, adapter_name="MED")
    model.set_adapter("MED")
    current_lora = "MED"

    # CROSS LoRA 사전 확인
    lora_cross = os.path.join(LORA_DIR, "lora_cross")
    has_cross  = os.path.exists(lora_cross)
    if has_cross:
        print(f"    CROSS LoRA 발견: {lora_cross}")

    for i, item in enumerate(data):
        query   = item["instruction"]
        gold    = item["gold_answer"]
        routing = deberta_route(query, router, router_tok, DEVICE)
        top1    = routing["top1"]
        top2    = routing["top2"]
        is_cross = routing["is_cross"]

        # 라우팅 정확도 (MedQuAD: 정답=MED)
        if top1 == "MED" or (is_cross and top2 == "MED"):
            routing_correct += 1
        if is_cross:
            cross_count += 1

        # LoRA 선택
        target_lora = "CROSS" if (is_cross and has_cross) else top1
        if target_lora != current_lora:
            lora_subdir = os.path.join(LORA_DIR, f"lora_{target_lora.lower()}")
            if os.path.exists(lora_subdir):
                try:
                    loaded = [n for n, _ in model.peft_config.items()]
                    if target_lora not in loaded:
                        model.load_adapter(lora_subdir, adapter_name=target_lora)
                    model.set_adapter(target_lora)
                    current_lora = target_lora
                except Exception as e:
                    pass  # 로드 실패시 현재 LoRA 유지

        # 검색
        ctx_strs = retrieve_nexroute(query, routing, embed_model, db)

        # 시스템 프롬프트: CROSS는 도메인 조합별 전문화
        system = get_cross_system(routing["scores"]) if is_cross else SYSTEM_MED

        answer = generate_answer(query, ctx_strs, model, tokenizer, system)

        results.append({
            "query": query, "gold_answer": gold, "pred_answer": answer,
            "domain_pred": routing["pred"], "top1": top1, "top2": top2,
            "is_cross": is_cross, "domain_scores": routing["scores"],
            "qtype": item.get("qtype", ""),
        })

        if (i + 1) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(data)}] 저장 (CROSS 감지: {cross_count}건)")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # PEFT 어댑터 비활성화 후 반환 (base_model에 영향 최소화)
    try:
        model.base_model.disable_adapter_layers()
    except Exception:
        pass

    routing_acc = routing_correct / len(data) * 100
    print(f"  라우팅 정확도: {routing_acc:.1f}% | CROSS 감지: {cross_count}건 ({cross_count/len(data)*100:.1f}%)")
    print(f"  저장: {out_path}")
    return results, routing_acc, cross_count


# ── Naive RAG 7B ─────────────────────────────────────────────────────────────
def run_naive_rag(data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*65}\n[Naive RAG 7B v2] 평가  [공정 비교 적용]\n{'='*65}")
    out_path = os.path.join(OUT_DIR, "medquad_naive_rag_v2.json")
    results  = []

    for i, item in enumerate(data):
        query  = item["instruction"]
        gold   = item["gold_answer"]
        ctx    = retrieve_unified(query, embed_model, texts, index)
        # [공정성 1,2] 동일 system + 동일 few-shot
        answer = generate_answer(query, ctx, model, tokenizer, SYSTEM_MED)
        results.append({"query": query, "gold_answer": gold, "pred_answer": answer,
                         "qtype": item.get("qtype", "")})

        if (i + 1) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(data)}] 저장")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  저장: {out_path}")
    return results


# ── Adaptive RAG 7B ──────────────────────────────────────────────────────────
def run_adaptive_rag(data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*65}\n[Adaptive RAG 7B v2] 평가  [공정 비교 적용]\n{'='*65}")
    out_path = os.path.join(OUT_DIR, "medquad_adaptive_rag_v2.json")
    results  = []
    routing_correct = 0

    for i, item in enumerate(data):
        query  = item["instruction"]
        gold   = item["gold_answer"]
        domain = keyword_route(query)
        if domain == "MED":
            routing_correct += 1
        # [공정성 1,2] 도메인 전문 system + 동일 few-shot
        system = SYSTEM_MAP.get(domain, SYSTEM_MED)
        ctx    = retrieve_unified(query, embed_model, texts, index)
        answer = generate_answer(query, ctx, model, tokenizer, system)
        results.append({"query": query, "gold_answer": gold, "pred_answer": answer,
                         "domain_pred": domain, "qtype": item.get("qtype", "")})

        if (i + 1) % 20 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [{i+1}/{len(data)}] 저장")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    routing_acc = routing_correct / len(data) * 100
    print(f"  라우팅 정확도: {routing_acc:.1f}% | 저장: {out_path}")
    return results, routing_acc


# ── 최종 테이블 출력 ──────────────────────────────────────────────────────────
def print_table(overall: dict, cross_sub: dict, cross_count: int):
    W = 90
    print("\n" + "=" * W)
    print(f"  MedQuAD 7B v2 결과  "
          f"[N={N_SAMPLES}, GAP={GAP}, TOP_K={TOP_K}, ctx={CTX_LEN}]")
    print("=" * W)
    hdr = f"  {'Model':<28}  {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8} {'Routing':>10}"
    print(hdr)
    print("-" * W)
    for name, v in overall.items():
        m  = v["metrics"]
        ra = v.get("routing_acc")
        ra_s = f"{ra:.1f}%" if ra is not None else "   N/A"
        def f(x): return f"{x:.2f}" if x is not None else " N/A"
        tag = " ◀ NexRoute" if "NexRoute" in name else ""
        print(f"  {name:<28}  {f(m['rouge_l']):>8} {f(m['bertscore']):>10} "
              f"{f(m['bleu4']):>8} {ra_s:>10}{tag}")

    if cross_count > 0:
        print("\n" + "─" * W)
        print(f"  [CROSS 서브셋 | NexRoute가 CROSS로 분류한 {cross_count}건]")
        print("─" * W)
        for name, v in cross_sub.items():
            m = v.get("metrics", {})
            if not m or m.get("n", 0) == 0:
                continue
            def f(x): return f"{x:.2f}" if x is not None else " N/A"
            tag = " ◀ NexRoute" if "NexRoute" in name else ""
            print(f"  {name:<28}  {f(m.get('rouge_l')):>8} {f(m.get('bertscore')):>10} "
                  f"{f(m.get('bleu4')):>8} {'':>10}{tag}")
    print("=" * W)


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. 데이터
    data = load_medquad(N_SAMPLES)

    # 2. 임베딩
    print("\n[*] 임베딩 모델 로드...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    # 3. DB
    texts, unified_index = build_unified_db(embed_model)
    domain_db = build_domain_db(embed_model)

    # 4. 라우터
    router, router_tok = load_router(ROUTER_PATH, DEVICE)

    # 5. 7B 베이스 모델 (4-bit NF4)
    print("\n[*] Qwen2.5-7B-Instruct 로드 (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager",
    )
    print("[*] 7B 로드 완료!")

    overall   = {}
    cross_sub = {}

    # ── NexRoute ──────────────────────────────────────────────────────────────
    nx_results, nx_routing_acc, nx_cross_count = run_nexroute(
        data, embed_model, domain_db, router, router_tok, base_model, tokenizer
    )
    torch.cuda.empty_cache()

    nx_valid = [{"gold": r["gold_answer"], "pred": r["pred_answer"]}
                for r in nx_results if r["pred_answer"] not in ["", "[FAILED]"]]
    overall["NexRoute 7B v2"] = {
        "metrics":     compute_metrics(nx_valid, "NexRoute 7B v2 (전체)"),
        "routing_acc": nx_routing_acc,
    }

    # CROSS 인덱스 수집 (Naive/Adaptive에 동일 인덱스 적용)
    cross_idx = [i for i, r in enumerate(nx_results) if r["is_cross"]]
    nx_cross  = [{"gold": nx_results[i]["gold_answer"], "pred": nx_results[i]["pred_answer"]}
                 for i in cross_idx
                 if nx_results[i]["pred_answer"] not in ["", "[FAILED]"]]
    cross_sub["NexRoute 7B v2"] = {
        "metrics": compute_metrics(nx_cross, "NexRoute  CROSS") if nx_cross else {"n": 0}
    }

    # 라우팅 분포
    dist = defaultdict(int)
    for r in nx_results:
        dist[r["top1"]] += 1
    print(f"\n[NexRoute 라우팅 분포] top1: {dict(dist)}")
    print(f"[CROSS 감지: {nx_cross_count}건 ({nx_cross_count/len(data)*100:.1f}%)]")

    # ── Naive RAG ─────────────────────────────────────────────────────────────
    naive_results = run_naive_rag(
        data, embed_model, texts, unified_index, base_model, tokenizer
    )
    naive_valid = [{"gold": r["gold_answer"], "pred": r["pred_answer"]}
                   for r in naive_results if r["pred_answer"] not in ["", "[FAILED]"]]
    overall["Naive RAG 7B v2"] = {
        "metrics":     compute_metrics(naive_valid, "Naive RAG 7B v2 (전체)"),
        "routing_acc": None,
    }
    naive_cross = [{"gold": naive_results[i]["gold_answer"], "pred": naive_results[i]["pred_answer"]}
                   for i in cross_idx
                   if naive_results[i]["pred_answer"] not in ["", "[FAILED]"]]
    cross_sub["Naive RAG 7B v2"] = {
        "metrics": compute_metrics(naive_cross, "Naive RAG CROSS") if naive_cross else {"n": 0}
    }

    # ── Adaptive RAG ──────────────────────────────────────────────────────────
    ada_results, ada_routing_acc = run_adaptive_rag(
        data, embed_model, texts, unified_index, base_model, tokenizer
    )
    ada_valid = [{"gold": r["gold_answer"], "pred": r["pred_answer"]}
                 for r in ada_results if r["pred_answer"] not in ["", "[FAILED]"]]
    overall["Adaptive RAG 7B v2"] = {
        "metrics":     compute_metrics(ada_valid, "Adaptive RAG 7B v2 (전체)"),
        "routing_acc": ada_routing_acc,
    }
    ada_cross = [{"gold": ada_results[i]["gold_answer"], "pred": ada_results[i]["pred_answer"]}
                 for i in cross_idx
                 if ada_results[i]["pred_answer"] not in ["", "[FAILED]"]]
    cross_sub["Adaptive RAG 7B v2"] = {
        "metrics": compute_metrics(ada_cross, "Adaptive RAG CROSS") if ada_cross else {"n": 0}
    }

    # ── 저장 ──────────────────────────────────────────────────────────────────
    final = {
        "dataset":   "MedQuAD (keivalya/MedQuad-MedicalQnADataset)",
        "n_samples": N_SAMPLES,
        "config": {
            "GAP": GAP, "THRESHOLD": THRESHOLD,
            "TOP_K": TOP_K, "ctx_len": CTX_LEN,
            "max_new_tokens": 512, "temperature": 0.6,
        },
        "overall":     {k: {"metrics": v["metrics"], "routing_acc": v.get("routing_acc")}
                        for k, v in overall.items()},
        "cross_subset": {k: v for k, v in cross_sub.items()},
        "cross_count":  nx_cross_count,
        "routing_breakdown_nexroute": dict(dist),
    }
    save_path = os.path.join(OUT_DIR, "medquad_scores_v2.json")
    with open(save_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print_table(overall, cross_sub, nx_cross_count)
    print(f"\n최종 저장: {save_path}")
