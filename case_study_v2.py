"""
Case Study v2: 4개 모델 재생성 (NexRoute 7B는 기존 유지)
max_new_tokens=300, 잘린 경우 400으로 자동 재시도
"""
import json, os, torch, faiss, re, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

QUERY = (
    "A patient suffered from severe hepatotoxicity after being prescribed "
    "Drug X for weight loss. The doctor obtained a general treatment consent "
    "form but did not specifically warn about liver damage, as weight loss is "
    "an off-label use for Drug X. Why might the physician be held legally "
    "liable despite the signed consent form?"
)

MODEL_PATH   = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR     = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL  = "intfloat/multilingual-e5-large"
PREV_PATH    = "/home/sclab/moe_model/case_study_outputs.json"
OUT_PATH     = "/home/sclab/moe_model/case_study_outputs_v2.json"
TOP_K        = 5

DOMAIN_FILES = {
    "LAW":  "law_rewritten.json",
    "MED":  "med_rewritten.json",
    "SOC":  "soc_rewritten.json",
    "STEM": "stem_rewritten.json",
    "CROSS":"cross_rewritten.json",
}

# ── 완전한 문장 검증 ──────────────────────────────────────────
def is_complete(text):
    t = text.strip()
    if not t:
        return False
    # 마지막 문자가 문장 종결 부호여야 함
    if t[-1] not in '.!?':
        return False
    # 단어가 중간에 잘린 경우 (소문자로 끝나는 단어 뒤 바로 끝)
    last_word = t.split()[-1] if t.split() else ""
    if last_word and last_word[-1].isalpha() and last_word[-1].islower():
        # 문장 끝 부호가 없으면 잘린 것
        if t[-1] not in '.!?':
            return False
    return True

def ends_mid_sentence(text):
    """True이면 잘린 것"""
    t = text.strip()
    if not t:
        return True
    return t[-1] not in '.!?'

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

# ── KB 구축 ───────────────────────────────────────────────────
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
    kb[dom] = (texts, idx)
    all_texts.extend(texts)

uni_embs = embed.encode(all_texts, batch_size=256, normalize_embeddings=True,
                        show_progress_bar=False)
uni_idx  = faiss.IndexFlatIP(uni_embs.shape[1])
uni_idx.add(uni_embs.astype(np.float32))
kb["UNIFIED"] = (all_texts, uni_idx)
print("[*] KB 완료\n")


def retrieve(query_text, domain="UNIFIED", k=TOP_K):
    emb = embed.encode([query_text], normalize_embeddings=True)
    texts, idx = kb[domain]
    D, I = idx.search(emb.astype(np.float32), k)
    return [texts[i] for i in I[0]]


def fmt_ctx(passages):
    return "\n\n".join(
        f"[Passage {i+1}] {p[:350]}" for i, p in enumerate(passages))


def generate(prompt_text, max_new_tokens=300):
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    ans = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    for kw in ["\nA:", "\nB:", "\nC:", "\nD:", "Choices:"]:
        if kw in ans:
            ans = ans[:ans.index(kw)].strip()
    return ans


def generate_with_retry(prompt_text, label):
    """300 토큰 생성 → 잘린 경우 400으로 재시도"""
    ans = generate(prompt_text, max_new_tokens=300)
    tok_count = len(tokenizer.encode(ans))
    print(f"  1차: {tok_count} tokens | 완전: {'✓' if not ends_mid_sentence(ans) else '✗ (재시도)'}")
    if ends_mid_sentence(ans):
        ans = generate(prompt_text, max_new_tokens=400)
        tok_count = len(tokenizer.encode(ans))
        print(f"  2차: {tok_count} tokens | 완전: {'✓' if not ends_mid_sentence(ans) else '✗ (그대로 사용)'}")
    return ans, len(tokenizer.encode(ans))


# ══════════════════════════════════════════════════════════════
# 기존 NexRoute 7B 유지
# ══════════════════════════════════════════════════════════════
with open(PREV_PATH) as f:
    prev = json.load(f)
nexroute_ans = prev.get("NexRoute 7B", "")
print(f"[NexRoute 7B] 기존 응답 유지 ({len(tokenizer.encode(nexroute_ans))} tokens)\n")

results      = {}
token_counts = {}

# ══════════════════════════════════════════════════════════════
# 1. Naive RAG 7B
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[1/4] Naive RAG 7B")
passages = retrieve(QUERY, "UNIFIED")
prompt = (
    "You are a helpful expert. Use the context below to answer the question.\n\n"
    f"Context:\n{fmt_ctx(passages)}\n\n"
    f"Question: {QUERY}\n\nAnswer:"
)
results["Naive RAG 7B"], token_counts["Naive RAG 7B"] = generate_with_retry(prompt, "Naive RAG 7B")

# ══════════════════════════════════════════════════════════════
# 2. Adaptive RAG 7B
# ══════════════════════════════════════════════════════════════
print("\n[2/4] Adaptive RAG 7B")
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
results["Adaptive RAG 7B"], token_counts["Adaptive RAG 7B"] = generate_with_retry(prompt, "Adaptive RAG 7B")

# ══════════════════════════════════════════════════════════════
# 3. HyDE 7B
# ══════════════════════════════════════════════════════════════
print("\n[3/4] HyDE 7B")
hyde_prompt = (
    "Write a short expert answer (3-4 sentences) to the following question:\n\n"
    f"Question: {QUERY}\n\nHypothetical Answer:"
)
hypothetical_doc = generate(hyde_prompt, max_new_tokens=150)
passages_hyde = retrieve(hypothetical_doc, "UNIFIED")
prompt = (
    "You are a helpful expert. Use the retrieved context to answer.\n\n"
    f"Context:\n{fmt_ctx(passages_hyde)}\n\n"
    f"Question: {QUERY}\n\nAnswer:"
)
results["HyDE 7B"], token_counts["HyDE 7B"] = generate_with_retry(prompt, "HyDE 7B")

# ══════════════════════════════════════════════════════════════
# 4. CRAG 7B
# ══════════════════════════════════════════════════════════════
print("\n[4/4] CRAG 7B")
passages_init = retrieve(QUERY, "UNIFIED", k=3)
eval_prompt = (
    "Rate the relevance of these passages to the question on a scale 1-5.\n"
    f"Question: {QUERY}\n"
    f"Passages: {passages_init[0][:200]}\nRelevance (1-5):"
)
rel_raw = generate(eval_prompt, max_new_tokens=10).strip()
try:    score = int(re.search(r'\d', rel_raw).group())
except: score = 3

if score <= 2:
    rewrite_prompt = (
        f"Rewrite this question for better document retrieval:\n{QUERY}\nRewritten:"
    )
    rewritten_q = generate(rewrite_prompt, max_new_tokens=80)
    passages_crag = retrieve(rewritten_q, "UNIFIED")
else:
    passages_crag = retrieve(QUERY, "UNIFIED")

prompt = (
    "You are a helpful expert. Use the context to provide a thorough answer.\n\n"
    f"Context:\n{fmt_ctx(passages_crag)}\n\n"
    f"Question: {QUERY}\n\nAnswer:"
)
results["CRAG 7B"], token_counts["CRAG 7B"] = generate_with_retry(prompt, "CRAG 7B")

# ══════════════════════════════════════════════════════════════
# NexRoute 7B 기존 값 추가
# ══════════════════════════════════════════════════════════════
results["NexRoute 7B"]      = nexroute_ans
token_counts["NexRoute 7B"] = len(tokenizer.encode(nexroute_ans))

# ══════════════════════════════════════════════════════════════
# 검증 및 저장
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("응답 완전성 검증")
print("─" * 60)
for name in ["Naive RAG 7B", "Adaptive RAG 7B", "HyDE 7B", "CRAG 7B", "NexRoute 7B"]:
    ans   = results[name]
    toks  = token_counts[name]
    complete = not ends_mid_sentence(ans)
    status   = "✓ complete" if complete else "✗ truncated"
    print(f"  {name:<22}  {toks:>4} tokens  {status}")
    print(f"    ...{ans[-80:]}")

out_data = {
    "query":        QUERY,
    "token_counts": token_counts,
    "Naive RAG 7B":     results["Naive RAG 7B"],
    "Adaptive RAG 7B":  results["Adaptive RAG 7B"],
    "HyDE 7B":          results["HyDE 7B"],
    "CRAG 7B":          results["CRAG 7B"],
    "NexRoute 7B":      results["NexRoute 7B"],
}
with open(OUT_PATH, "w") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 저장: {OUT_PATH}")
