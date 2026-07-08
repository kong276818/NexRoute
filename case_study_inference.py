"""
Case Study: All 7B baselines on a single cross-domain query
Query: hepatotoxicity / off-label drug use / legal liability
"""
import json, os, sys, torch, faiss, re, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ── 설정 ──────────────────────────────────────────────────────
QUERY = (
    "A patient suffered from severe hepatotoxicity after being prescribed "
    "Drug X for weight loss. The doctor obtained a general treatment consent "
    "form but did not specifically warn about liver damage, as weight loss is "
    "an off-label use for Drug X. Why might the physician be held legally "
    "liable despite the signed consent form?"
)

MODEL_PATH  = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL = "intfloat/multilingual-e5-large"
TOP_K       = 5
GEN_KWARGS  = dict(max_new_tokens=256, temperature=0.1,
                   top_p=0.9, do_sample=False)

DOMAIN_FILES = {
    "LAW":  "law_rewritten.json",
    "MED":  "med_rewritten.json",
    "SOC":  "soc_rewritten.json",
    "STEM": "stem_rewritten.json",
    "CROSS":"cross_rewritten.json",
}

# ── 모델 로드 ─────────────────────────────────────────────────
print("[*] LLM 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
print("[*] 임베딩 모델 로드 중...")
embed = SentenceTransformer(EMBED_MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu")

# ── KB 인덱스 구축 ────────────────────────────────────────────
print("[*] KB 인덱스 구축 중...")
kb = {}
all_texts = []
for dom, fname in DOMAIN_FILES.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    texts = [d.get("output", d.get("gold_answer",""))[:500]
             for d in data if d.get("output") or d.get("gold_answer")]
    texts = [t for t in texts if len(t) > 30]
    embs  = embed.encode(texts, batch_size=256, normalize_embeddings=True,
                         show_progress_bar=False)
    idx   = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    kb[dom]     = (texts, idx)
    all_texts.extend(texts)
    print(f"  {dom}: {len(texts)} passages")

# unified KB (Naive / Adaptive / HyDE / CRAG용)
uni_embs = embed.encode(all_texts, batch_size=256, normalize_embeddings=True,
                        show_progress_bar=False)
uni_idx  = faiss.IndexFlatIP(uni_embs.shape[1])
uni_idx.add(uni_embs.astype(np.float32))
kb["UNIFIED"] = (all_texts, uni_idx)
print(f"  UNIFIED: {len(all_texts)} passages\n")


# ── 검색 유틸 ─────────────────────────────────────────────────
def retrieve(query_text, domain="UNIFIED", k=TOP_K):
    emb = embed.encode([query_text], normalize_embeddings=True)
    texts, idx = kb[domain]
    D, I = idx.search(emb.astype(np.float32), k)
    return [texts[i] for i in I[0]]


def generate(prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **GEN_KWARGS,
            pad_token_id=tokenizer.eos_token_id,
        )
    ans = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    # 잡음 제거
    for kw in ["Choices:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in ans:
            ans = ans[:ans.index(kw)].strip()
    return ans


def fmt_ctx(passages):
    return "\n\n".join(
        f"[Passage {i+1}] {p[:300]}" for i, p in enumerate(passages))


# ══════════════════════════════════════════════════════════════
# 1. Naive RAG 7B — unified KB dense retrieval
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[1/5] Naive RAG 7B")
passages = retrieve(QUERY, "UNIFIED")
prompt = (
    "You are a helpful expert. Use the context below to answer the question.\n\n"
    f"Context:\n{fmt_ctx(passages)}\n\n"
    f"Question: {QUERY}\n\nAnswer:"
)
naive_ans = generate(prompt)
print(f"  → {naive_ans[:120]}...")

# ══════════════════════════════════════════════════════════════
# 2. Adaptive RAG 7B — TF-IDF keyword routing + unified retrieval
# ══════════════════════════════════════════════════════════════
print("[2/5] Adaptive RAG 7B")
# 키워드 기반 간단 라우팅 (법/의료 혼합 → both)
law_kw = ["liab", "consent", "legal", "negligenc", "malpractice", "off-label"]
med_kw = ["hepato", "liver", "drug", "patient", "clinical", "treatment"]
q_lower = QUERY.lower()
domains_ada = []
if any(k in q_lower for k in law_kw): domains_ada.append("LAW")
if any(k in q_lower for k in med_kw): domains_ada.append("MED")
if not domains_ada: domains_ada = ["UNIFIED"]

if len(domains_ada) > 1:
    passages_ada = []
    per = max(1, TOP_K // len(domains_ada))
    for d in domains_ada:
        passages_ada.extend(retrieve(QUERY, d, per))
else:
    passages_ada = retrieve(QUERY, domains_ada[0])

prompt = (
    f"You are an expert in {' and '.join(domains_ada)}. "
    "Use the context to answer precisely.\n\n"
    f"Context:\n{fmt_ctx(passages_ada)}\n\n"
    f"Question: {QUERY}\n\nAnswer:"
)
adaptive_ans = generate(prompt)
print(f"  → {adaptive_ans[:120]}...")

# ══════════════════════════════════════════════════════════════
# 3. HyDE 7B — hypothetical document embedding retrieval
# ══════════════════════════════════════════════════════════════
print("[3/5] HyDE 7B")
# Step 1: 가상 답변 생성
hyde_prompt = (
    "Write a short expert answer (3-4 sentences) to the following question:\n\n"
    f"Question: {QUERY}\n\nHypothetical Answer:"
)
hypothetical_doc = generate(hyde_prompt)
# Step 2: 가상 답변으로 검색
passages_hyde = retrieve(hypothetical_doc, "UNIFIED")
prompt = (
    "You are a helpful expert. Use the retrieved context to answer.\n\n"
    f"Context:\n{fmt_ctx(passages_hyde)}\n\n"
    f"Question: {QUERY}\n\nAnswer:"
)
hyde_ans = generate(prompt)
print(f"  → {hyde_ans[:120]}...")

# ══════════════════════════════════════════════════════════════
# 4. CRAG 7B — corrective retrieval with re-retrieval
# ══════════════════════════════════════════════════════════════
print("[4/5] CRAG 7B")
# Step 1: 초기 검색
passages_init = retrieve(QUERY, "UNIFIED", k=3)
# Step 2: 관련성 평가 → 낮으면 쿼리 재작성 후 재검색
eval_prompt = (
    "Rate the relevance of these passages to the question on a scale 1-5.\n"
    f"Question: {QUERY}\n"
    f"Passages: {passages_init[0][:200]}\n"
    "Relevance (1-5):"
)
rel_score = generate(eval_prompt).strip()
try:
    score = int(re.search(r'\d', rel_score).group())
except: score = 3

if score <= 2:
    # 재작성 후 재검색
    rewrite_prompt = (
        f"Rewrite this question for better document retrieval:\n{QUERY}\nRewritten:"
    )
    rewritten_q = generate(rewrite_prompt)
    passages_crag = retrieve(rewritten_q, "UNIFIED")
else:
    passages_crag = retrieve(QUERY, "UNIFIED")

prompt = (
    "You are a helpful expert. Use the context to provide a thorough answer.\n\n"
    f"Context:\n{fmt_ctx(passages_crag)}\n\n"
    f"Question: {QUERY}\n\nAnswer:"
)
crag_ans = generate(prompt)
print(f"  → {crag_ans[:120]}...")

# ══════════════════════════════════════════════════════════════
# 5. NexRoute 7B — partitioned KB (LAW+MED), CROSS expert, Why scaffold
# ══════════════════════════════════════════════════════════════
print("[5/5] NexRoute 7B")
# CROSS 도메인 (LAW + MED) partitioned KB 검색
passages_law  = retrieve(QUERY, "LAW",  k=3)
passages_med  = retrieve(QUERY, "MED",  k=3)
passages_nr   = passages_law[:2] + passages_med[:2] + retrieve(QUERY, "CROSS", k=2)

ctx_nr = "\n\n".join([
    f"[LAW] {p[:300]}"  for p in passages_law[:2]
] + [
    f"[MED] {p[:300]}"  for p in passages_med[:2]
] + [
    f"[CROSS] {p[:300]}" for p in retrieve(QUERY, "CROSS", k=1)
])

# Why scaffold + CROSS multidisciplinary system prompt
system_nr = (
    "You are a multidisciplinary expert specializing in Medical Ethics and Law. "
    "Integrate knowledge from both domains to answer accurately."
)
why_scaffold = (
    "To answer why X causes Y, identify: "
    "(1) the mechanism/principle involved, "
    "(2) the legal or causal chain, "
    "(3) the broader implication."
)
prompt = (
    f"{system_nr}\n\n"
    f"Analytical Framework: {why_scaffold}\n\n"
    f"Context:\n{ctx_nr}\n\n"
    f"Question: {QUERY}\n\n"
    "Provide a comprehensive analytical answer integrating legal and medical knowledge. "
    "Write in flowing paragraph form only.\n\nAnswer:"
)
nexroute_ans = generate(prompt)
print(f"  → {nexroute_ans[:120]}...")

# ══════════════════════════════════════════════════════════════
# 결과 저장
# ══════════════════════════════════════════════════════════════
results = {
    "query":         QUERY,
    "Naive RAG 7B":     naive_ans,
    "Adaptive RAG 7B":  adaptive_ans,
    "HyDE 7B":          hyde_ans,
    "CRAG 7B":          crag_ans,
    "NexRoute 7B":      nexroute_ans,
}

out_path = "/home/sclab/moe_model/case_study_outputs.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ 저장: {out_path}")
print("\n" + "=" * 70)
print("QUERY:", QUERY[:80], "...")
print("=" * 70)
for name in ["Naive RAG 7B", "Adaptive RAG 7B", "HyDE 7B",
             "CRAG 7B", "NexRoute 7B"]:
    print(f"\n{'─'*70}")
    print(f"[{name}]")
    print(results[name])
print("=" * 70)
