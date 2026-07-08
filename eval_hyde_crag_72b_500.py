import os, json, re, glob, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR    = "/home/sclab/moe_model/processed_v2"
OUT_DIR     = "/home/sclab/moe_model/baseline_results"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"

_cands = glob.glob(
    "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/*"
)
BASE_72B = _cands[0] if _cands else "Qwen/Qwen2.5-72B-Instruct"

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
    print("[*] Unified Vector DB 구축 (non-partitioned)...")
    domain_files = {
        "LAW":   "law_rewritten.json",
        "MED":   "med_rewritten.json",
        "SOC":   "soc_rewritten.json",
        "STEM":  "stem_rewritten.json",
        "CROSS": "cross_rewritten.json",
    }
    all_texts = []
    for domain, fname in domain_files.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
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

def generate_text(prompt, model, tokenizer, max_tokens=300):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:", "The correct answer"]:
        if kw in result:
            result = result[:result.index(kw)].strip()
    return result if result.strip() else "[FAILED]"

def gen_prompt(query, ctx, system="You are a helpful expert."):
    ctx_text = "\n".join([c[:200] for c in ctx])
    return (
        f"{system}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
    )

SYSTEMS = {
    'LAW':  "You are a legal expert.",
    'MED':  "You are a medical expert.",
    'SOC':  "You are a social science expert.",
    'STEM': "You are a science expert.",
}

# ── Naive / Adaptive (routing differs only in system prompt) ──────
def run_naive_adaptive(name, gt_data, embed_model, texts, index, model, tokenizer, routing):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    out_path = f"{OUT_DIR}/{name}.json"
    results = []
    for i, item in enumerate(gt_data):
        query = item.get("instruction", "")
        gold  = item.get("gold_answer", "")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")
        try:
            ctx, _ = retrieve(query, embed_model, texts, index)
            if routing == "keyword":
                domain = keyword_route(query)
                system = SYSTEMS.get(domain, "You are a helpful expert.")
            else:
                system = "You are a helpful expert."
            answer = generate_text(gen_prompt(query, ctx, system), model, tokenizer)
            results.append({"query": query, "gold_answer": gold, "pred_answer": answer})
        except Exception as e:
            print(f"  [Error] {e}")
        if (i + 1) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [저장] {i+1}개")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] {name} 완료: {len(results)}개")

# ── HyDE ────────────────────────────────────────────────────────
def run_hyde(name, gt_data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    out_path = f"{OUT_DIR}/{name}.json"
    results = []
    for i, item in enumerate(gt_data):
        query = item.get("instruction", "")
        gold  = item.get("gold_answer", "")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")
        try:
            hypo_prompt = (
                f"You are a helpful expert.\n\n"
                f"Question: {query}\n\n"
                f"Write a brief hypothetical answer (2-3 sentences) to help retrieve "
                f"relevant documents:\n\nAnswer:"
            )
            hypo_answer = generate_text(hypo_prompt, model, tokenizer, max_tokens=100)

            hypo_emb = embed_model.encode([hypo_answer])
            hypo_emb = hypo_emb / np.linalg.norm(hypo_emb)
            ctx, _ = retrieve(hypo_emb, embed_model, texts, index, is_emb=True)

            answer = generate_text(gen_prompt(query, ctx), model, tokenizer)
            results.append({"query": query, "gold_answer": gold, "pred_answer": answer})
        except Exception as e:
            print(f"  [Error] {e}")
        if (i + 1) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [저장] {i+1}개")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] {name} 완료: {len(results)}개")

# ── CRAG ────────────────────────────────────────────────────────
def run_crag(name, gt_data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    out_path = f"{OUT_DIR}/{name}.json"
    results = []
    stats = {"correct": 0, "ambiguous": 0, "incorrect": 0}
    for i, item in enumerate(gt_data):
        query = item.get("instruction", "")
        gold  = item.get("gold_answer", "")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")
        try:
            ctx, scores = retrieve(query, embed_model, texts, index)
            top_score = float(scores[0])

            if top_score >= 0.85:
                quality, final_ctx = "correct", ctx
            elif top_score >= 0.60:
                quality = "ambiguous"
                expanded = f"{query} explanation details mechanism"
                extra_ctx, _ = retrieve(expanded, embed_model, texts, index, top_k=5)
                final_ctx = ctx[:5] + extra_ctx[:5]
            else:
                quality, final_ctx = "incorrect", []
            stats[quality] += 1

            if final_ctx:
                answer = generate_text(gen_prompt(query, final_ctx), model, tokenizer)
            else:
                prompt = (
                    f"You are a helpful expert.\n\nQuestion: {query}\n\n"
                    f"Based on your knowledge, provide a comprehensive analytical "
                    f"answer in flowing paragraph form.\n\nAnswer:"
                )
                answer = generate_text(prompt, model, tokenizer)

            results.append({
                "query": query, "gold_answer": gold,
                "pred_answer": answer, "retrieval_quality": quality
            })
        except Exception as e:
            print(f"  [Error] {e}")
        if (i + 1) % 50 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [저장] {i+1}개 | {stats}")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] {name} 완료: {len(results)}개 | 품질분포: {stats}")


if __name__ == "__main__":
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    print(f"GT: {len(gt_data)}개")
    print(f"72B 모델 경로: {BASE_72B}")

    embed_model = SentenceTransformer(EMBED_MODEL)
    texts, index = build_db(embed_model)

    print("\n[*] Qwen2.5-72B-Instruct 로드 (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_72B, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_72B, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
        local_files_only=True, attn_implementation="eager"
    )
    print("[*] 로드 완료! HyDE + CRAG 72B만 실행 (Naive/Adaptive는 이미 완료됨)...\n")

    run_hyde("hyde_72b_500", gt_data, embed_model, texts, index, model, tokenizer)
    run_crag("crag_72b_500", gt_data, embed_model, texts, index, model, tokenizer)

    print("\n[+] HyDE + CRAG 72B 완료!")