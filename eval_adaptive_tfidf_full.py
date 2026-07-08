import json, pickle, torch, glob, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from bert_score import score as bscore
import faiss

DATA_DIR    = "/home/sclab/moe_model/processed_v2"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
ROUTER_PATH = "/home/sclab/moe_model/adaptive_tfidf_router.pkl"
OUT_PATH    = "/home/sclab/moe_model/baseline_results/adaptive_rag_tfidf_7b_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"

_cands = glob.glob(
    "/home/sclab/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/*"
)
BASE_7B = _cands[0] if _cands else "Qwen/Qwen2.5-7B-Instruct"

SYSTEMS = {
    'LAW':   "You are a legal expert with deep knowledge of law and jurisprudence.",
    'MED':   "You are a medical expert with deep knowledge of medicine and clinical science.",
    'SOC':   "You are a social science expert with knowledge of sociology, psychology, and culture.",
    'STEM':  "You are a science and technology expert with knowledge of physics, biology, chemistry, and math.",
    'CROSS': "You are a multidisciplinary expert with deep knowledge across law, medicine, social science, and STEM.",
}

# ── TF-IDF+LR 라우터 로드 ────────────────────────────────────
print("[*] TF-IDF+LR 라우터 로드...")
with open(ROUTER_PATH, 'rb') as f:
    router_data = pickle.load(f)
tfidf = router_data['tfidf']
clf   = router_data['clf']

def tfidf_route(query):
    vec = tfidf.transform([query])
    return clf.predict(vec)[0]

# ── Unified Vector DB ─────────────────────────────────────────
print("[*] Vector DB 구축...")
embed_model = SentenceTransformer(EMBED_MODEL)
domain_files = {
    "LAW": "law_rewritten.json", "MED": "med_rewritten.json",
    "SOC": "soc_rewritten.json", "STEM": "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}
all_texts = []
for domain, fname in domain_files.items():
    with open(f"{DATA_DIR}/{fname}") as f:
        data = json.load(f)
    for d in data:
        out = d.get("output", d.get("gold_answer", ""))
        all_texts.append(f"{d['instruction']} {out}")

embs = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs.astype(np.float32))
print(f"  DB: {len(all_texts)}개")

# ── 7B 모델 로드 ─────────────────────────────────────────────
print("\n[*] Qwen2.5-7B 로드 (4-bit NF4)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_7B, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager"
)
print("[*] 로드 완료!")

def retrieve(query, top_k=10):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    _, I = index.search(emb.astype(np.float32), top_k)
    return [all_texts[i] for i in I[0]]

def generate(query, context, system):
    ctx = "\n".join([c[:200] for c in context])
    prompt = (f"{system}\n\nContext:\n{ctx}\n\n"
              f"Question: {query}\n\n"
              f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:", "The correct answer"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[FAILED]"

# ── 평가 실행 ─────────────────────────────────────────────────
with open(GT_PATH) as f: gt_data = json.load(f)
results = []

for i, item in enumerate(gt_data):
    query = item.get("instruction", "")
    gold  = item.get("gold_answer", "")
    domain_pred = tfidf_route(query)
    system = SYSTEMS.get(domain_pred, SYSTEMS['SOC'])
    print(f"[{i+1}/{len(gt_data)}] [{domain_pred}] {query[:50]}...")
    try:
        ctx    = retrieve(query)
        answer = generate(query, ctx, system)
        results.append({
            "query": query, "gold_answer": gold,
            "pred_answer": answer, "routed_domain": domain_pred
        })
    except Exception as e:
        print(f"  [Error] {e}")
    if (i + 1) % 50 == 0:
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  [저장] {i+1}개")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n[+] 생성 완료: {len(results)}개")

# ── BERTScore 계산 ────────────────────────────────────────────
print("\n[*] BERTScore 계산 (전체 + 도메인별)...")
gt_map = {d["instruction"].strip(): d for d in gt_data}
LABELS = ['LAW','MED','SOC','STEM','CROSS']

# 전체
preds = [r["pred_answer"] for r in results]
refs  = [r["gold_answer"]  for r in results]
_, _, F = bscore(preds, refs, lang="en", verbose=False, rescale_with_baseline=True)
print(f"\nAdaptive RAG TF-IDF+LR 전체 BERTScore: {F.mean().item()*100:.2f}%")

# CROSS / Single 분리
for split, label in [("CROSS","CROSS"), ("Single","MED+SOC+STEM")]:
    sp, sr = [], []
    for r in results:
        q = r["query"].strip()
        gt_item = gt_map.get(q)
        if not gt_item: continue
        domain = gt_item.get("expected_domain","")
        if split == "CROSS" and domain != "CROSS": continue
        if split == "Single" and domain in ["CROSS","LAW"]: continue
        sp.append(r["pred_answer"])
        sr.append(r["gold_answer"])
    if sp:
        _, _, F2 = bscore(sp, sr, lang="en", verbose=False, rescale_with_baseline=True)
        print(f"{split} ({label}, N={len(sp)}): {F2.mean().item()*100:.2f}%")

print("\n완료!")