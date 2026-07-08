import os, json, time, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DATA_DIR    = "/home/sclab/moe_model/processed_v2"
OUT_DIR     = "/home/sclab/moe_model/baseline_results"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"
BASE_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"

import glob
cands = glob.glob(f"{BASE_7B}/*")
BASE_7B = cands[0]

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

def build_db(embed_model):
    print("[*] Unified Vector DB 구축...")
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
    embs = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    return all_texts, index

def retrieve(query, embed_model, texts, index, top_k=10):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    D, I = index.search(emb.astype(np.float32), top_k)
    return "\n".join([texts[i] for i in I[0]])

def generate(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,   # ← 300 → 120
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

print("[*] 임베딩 모델 로딩...")
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda:0")

print("[*] 7B 모델 로딩...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B)

texts, index = build_db(embed_model)

with open(GT_PATH) as f:
    gt_data = json.load(f)

RUNS = [
    ("naive_rag_7b_500",    "none"),
    ("adaptive_rag_7b_500", "keyword"),
    ("hyde_7b_500",         "hyde"),
    ("crag_7b_500",         "crag"),
]

for out_name, routing in RUNS:
    print(f"\n[*] {out_name} 실행 중...")
    results = []
    t_start = time.time()

    for i, item in enumerate(gt_data):
        query = item["instruction"]
        gold  = item["gold_answer"]

        if routing == "hyde":
            hyde_prompt = f"Write a hypothetical answer to: {query}"
            hypo = generate(hyde_prompt, model, tokenizer)
            ctx  = retrieve(hypo, embed_model, texts, index)
        else:
            ctx = retrieve(query, embed_model, texts, index)

        if routing == "keyword":
            domain = keyword_route(query)
            prompt = f"Domain: {domain}\nContext: {ctx}\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Context: {ctx}\nQuestion: {query}\nAnswer:"

        pred = generate(prompt, model, tokenizer)

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

    out_path = os.path.join(OUT_DIR, f"{out_name}_v2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}")

print("\n✅ 7B 재실험 완료!")
