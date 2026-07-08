import os, json, re, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from sentence_transformers import SentenceTransformer
import faiss, torch.nn as nn
from collections import defaultdict

BASE_7B   = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR  = "/home/sclab/moe_model/processed_v2"
OUT_DIR   = "/home/sclab/moe_model/baseline_results"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"

KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':  ['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':  ['social','society','culture','economic','political','community','psychology'],
    'STEM': ['science','physics','chemical','biology','math','engineering','technology'],
}

def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d,kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

def build_db(embed_model):
    print("[*] Vector DB 구축...")
    files = {
        "LAW":   "law_rewritten.json",
        "MED":   "med_rewritten.json",
        "SOC":   "soc_rewritten.json",
        "STEM":  "stem_rewritten.json",
        "CROSS": "cross_rewritten.json",
    }
    all_texts = []
    for domain, fname in files.items():
        with open(os.path.join(DATA_DIR, fname)) as f: data = json.load(f)
        for d in data:
            out = d.get("output", d.get("gold_answer",""))
            all_texts.append(f"{d['instruction']} {out}")

    embs  = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"  전체 DB: {len(all_texts)}개")
    return all_texts, index

def retrieve(query, embed_model, texts, index, top_k=10):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]]

def generate(query, context, model, tokenizer, system="You are a helpful expert."):
    ctx = "\n".join([c[:200] for c in context])
    prompt = (
        f"{system}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
    )
    msgs   = [{"role":"user","content":prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for kw in ["Choices:","Options:","\nA:","\nB:","\nC:","\nD:"]:
        if kw in answer: answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[FAILED]"

def run_eval(name, gt_data, embed_model, texts, index, model, tokenizer, routing="none"):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    results = []
    SYSTEMS = {
        'LAW':  "You are a legal expert.",
        'MED':  "You are a medical expert.",
        'SOC':  "You are a social science expert.",
        'STEM': "You are a science expert.",
    }
    for i, item in enumerate(gt_data):
        query = item.get("instruction","")
        gold  = item.get("gold_answer","")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")
        try:
            ctx = retrieve(query, embed_model, texts, index)
            if routing == "keyword":
                domain = keyword_route(query)
                system = SYSTEMS.get(domain, "You are a helpful expert.")
            else:
                system = "You are a helpful expert."
            answer = generate(query, ctx, model, tokenizer, system)
            results.append({"query":query,"gold_answer":gold,"pred_answer":answer})
        except Exception as e:
            print(f"  [Error] {e}")
        if (i+1) % 50 == 0:
            with open(f"{OUT_DIR}/{name}.json","w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    with open(f"{OUT_DIR}/{name}.json","w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] {name} 완료: {len(results)}개")
    return results

if __name__ == "__main__":
    with open(GT_PATH) as f: gt_data = json.load(f)
    print(f"GT: {len(gt_data)}개")

    embed_model = SentenceTransformer(EMBED_MODEL)
    texts, index = build_db(embed_model)

    print("\n[*] 7B 로드...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_7B, quantization_config=bnb,
            device_map="auto", trust_remote_code=True, local_files_only=True,
            attn_implementation="eager")
    print("[*] 로드 완료!")

    run_eval("naive_rag_7b_500",    gt_data, embed_model, texts, index, model, tokenizer, routing="none")
    run_eval("adaptive_rag_7b_500", gt_data, embed_model, texts, index, model, tokenizer, routing="keyword")

    print("\n[+] 전체 완료!")
