import os
import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

DATA_DIR    = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL = "intfloat/multilingual-e5-large"
OUT_DIR     = "/home/sclab/moe_model/baseline_results"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_sample_50.json"
QWEN_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':  ['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':  ['social','society','culture','economic','political','community','psychology','policy'],
    'STEM': ['science','physics','chemical','biology','math','engineering','technology','energy'],
}

def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d,kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

def load_embed():
    print("[*] 임베딩 모델 로드...")
    return SentenceTransformer(EMBED_MODEL)

def build_db(embed_model):
    print("[*] 벡터 DB 구축...")
    files = {
        "LAW":   "law.json",   "MED":  "med.json",
        "SOC":   "soc.json",   "STEM": "stem.json",
        "CROSS": "cross_rewritten.json",
    }
    domain_db  = {}
    all_texts  = []

    for domain, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        texts = []
        for d in data:
            out = d.get("output", d.get("gold_answer",""))
            if any(k in out for k in ["The correct answer","All answers:"]):
                continue
            texts.append(f"{d['instruction']} {out}")

        embs  = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.astype(np.float32))
        domain_db[domain] = (texts, index)
        all_texts.extend(texts)
        print(f"  {domain}: {len(texts)}개")

    # 전체 통합 DB
    all_embs = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    all_embs = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)
    all_idx  = faiss.IndexFlatIP(all_embs.shape[1])
    all_idx.add(all_embs.astype(np.float32))
    domain_db["ALL"] = (all_texts, all_idx)
    print(f"  전체: {len(all_texts)}개")
    return domain_db

def retrieve(query, embed_model, db, domain="ALL", top_k=10):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    texts, index = db[domain]
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]]

def load_7b():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(QWEN_7B, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    return model, tokenizer

def generate(query, context, model, tokenizer):
    ctx_text = "\n".join([c[:200] for c in context])
    prompt   = (
        f"You are a helpful expert.\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form.\n\n"
        f"Answer:"
    )
    messages = [{"role": "user", "content": prompt}]
    text     = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=300, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

def run_eval(name, gt_data, embed_model, db, model, tokenizer,
             routing="none", save_path=None):
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")

    results = []
    for i, item in enumerate(gt_data):
        query = item.get("instruction","")
        gold  = item.get("gold_answer","")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")

        try:
            # 라우팅 결정
            if routing == "none":
                domain = "ALL"
            elif routing == "stem_fixed":
                domain = "STEM"
            elif routing == "keyword":
                domain = keyword_route(query)

            context = retrieve(query, embed_model, db, domain)
            answer  = generate(query, context, model, tokenizer)

            results.append({
                "query":       query,
                "gold_answer": gold,
                "pred_answer": answer,
                "domain":      domain
            })
        except Exception as e:
            print(f"  [Error] {e}")

        if (i+1) % 10 == 0 and save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[+] {name} 완료: {len(results)}개")
    return results

if __name__ == "__main__":
    import gc

    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"GT 샘플: {len(gt_data)}개")

    embed_model = load_embed()
    db          = build_db(embed_model)

    print("\n[*] Qwen2.5-7B 로드...")
    model, tokenizer = load_7b()
    print("[*] 로드 완료!")

    # LoRA RAG 7B (STEM 고정, LoRA 없음)
    run_eval(
        "LoRA RAG 7B (STEM 고정, no LoRA)",
        gt_data, embed_model, db, model, tokenizer,
        routing="stem_fixed",
        save_path=f"{OUT_DIR}/lora_rag_7b.json"
    )

    # Adaptive RAG 7B (키워드 라우터, LoRA 없음)
    run_eval(
        "Adaptive RAG 7B (키워드 라우터, no LoRA)",
        gt_data, embed_model, db, model, tokenizer,
        routing="keyword",
        save_path=f"{OUT_DIR}/adaptive_rag_7b.json"
    )

    print("\n[+] 전체 완료!")