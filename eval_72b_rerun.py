import os, json, time, torch, glob
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

DATA_DIR = "/home/sclab/moe_model/processed_v2"
OUT_DIR  = "/home/sclab/moe_model/baseline_results"
GT_PATH  = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"

cands = glob.glob("/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/*")
BASE_72B = cands[0]
print(f"[*] 72B 모델 경로: {BASE_72B}")

KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':  ['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':  ['social','society','culture','economic','political','community','psychology'],
    'STEM': ['science','physics','chemical','biology','math','engineering','technology'],
}

def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

# ── STEP 1: 임베딩 먼저 GPU로 ────────────────────────────────────────────
print("[*] STEP1: 임베딩 모델 로딩 (GPU:2)...")
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda:2")

print("[*] STEP2: Vector DB 구축 (GPU:2)...")
t0 = time.time()
domain_files = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}
all_texts = []
for domain, fname in domain_files.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    for d in data:
        out = d.get("output", d.get("gold_answer", ""))
        all_texts.append(f"{d['instruction']} {out}")

embs = embed_model.encode(
    all_texts, show_progress_bar=True,
    batch_size=512, convert_to_numpy=True
)
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs.astype(np.float32))
print(f"  DB 완료: {len(all_texts)}개 ({time.time()-t0:.1f}s)")

# ── STEP 3: 임베딩 GPU 해제 ──────────────────────────────────────────────
print("[*] STEP3: 임베딩 모델 CPU로 이동...")
embed_model_cpu = SentenceTransformer(EMBED_MODEL, device="cpu")
del embed_model
torch.cuda.empty_cache()
print("  해제 완료")

def retrieve(query, top_k=10):
    emb = embed_model_cpu.encode([query], convert_to_numpy=True)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [all_texts[i] for i in I[0]]

# ── STEP 4: 72B 모델 로딩 ────────────────────────────────────────────────
print("[*] STEP4: 72B 모델 로딩...")
t0 = time.time()
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
max_memory = {0: "23GiB", 1: "23GiB", 2: "23GiB", "cpu": "30GiB"}
model = AutoModelForCausalLM.from_pretrained(
    BASE_72B,
    quantization_config=bnb,
    device_map="auto",
    max_memory=max_memory,
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_72B)
print(f"  로딩 완료: {time.time()-t0:.1f}s")

# ── STEP 5: 생성 (기존 방식 그대로 - chat_template 사용) ─────────────────
def generate(query, context, system="You are a helpful expert."):
    ctx = "\n".join([c[:200] for c in context])
    prompt = (
        f"{system}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    first_device = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=2048
    ).to(first_device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:", "The correct answer"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer

with open(GT_PATH) as f:
    gt_data = json.load(f)

RUNS = [
    ("naive_rag_72b_500_v2",    "none"),
    ("adaptive_rag_72b_500_v2", "keyword"),
    ("hyde_72b_500_v2",         "hyde"),
    ("crag_72b_500_v2",         "crag"),
]

for out_name, routing in RUNS:
    print(f"\n[*] {out_name} 실행 중...")
    results = []
    t_start = time.time()

    for i, item in enumerate(gt_data):
        query = item["instruction"]
        gold  = item["gold_answer"]

        if routing == "hyde":
            hypo_ctx = retrieve(query)
            hypo = generate(query, hypo_ctx,
                            system="Generate a hypothetical answer to the question.")
            context = retrieve(hypo)
        else:
            context = retrieve(query)

        if routing == "keyword":
            domain = keyword_route(query)
            system = f"You are a helpful {domain} expert."
        else:
            system = "You are a helpful expert."

        pred = generate(query, context, system=system)

        results.append({
            "query":       query,
            "gold_answer": gold,
            "pred_answer": pred,
            "domain":      item.get("expected_domain", ""),
        })

        if (i+1) % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i+1) * (500 - i - 1)
            print(f"  {i+1}/500 | 경과: {elapsed/60:.1f}분 | ETA: {eta/60:.1f}분")

    out_path = os.path.join(OUT_DIR, f"{out_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}")

print("\n✅ 72B 재실험 완료!")
