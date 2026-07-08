import os, json, torch, numpy as np, faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

BASE_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
OUT_DIR     = "/home/sclab/moe_model/baseline_results"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

def build_db(embed_model):
    print("[*] Vector DB 구축...")
    all_texts = []
    for domain, fname in DOMAIN_FILES.items():
        with open(os.path.join(DATA_DIR, fname)) as f: data = json.load(f)
        for d in data:
            out = d.get("output", d.get("gold_answer",""))
            all_texts.append(f"{d['instruction']} {out}")
    embs  = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"  DB: {len(all_texts)}개")
    return all_texts, index, embs

def retrieve(query_or_emb, embed_model, texts, index, top_k=10, is_emb=False):
    if is_emb:
        emb = query_or_emb
    else:
        emb = embed_model.encode([query_or_emb])
        emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]], D[0]

def generate_text(prompt, model, tokenizer, max_tokens=300):
    msgs   = [{"role":"user","content":prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for kw in ["Choices:","Options:","\nA:","\nB:","\nC:","\nD:"]:
        if kw in result: result = result[:result.index(kw)].strip()
    return result

# ── HyDE ──────────────────────────────────────────────
def run_hyde(gt_data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*50}\nHyDE\n{'='*50}")
    results = []
    for i, item in enumerate(gt_data):
        query = item.get("instruction","")
        gold  = item.get("gold_answer","")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")
        try:
            # Step 1: 가상 답변 생성
            hypo_prompt = (
                f"You are a helpful expert.\n\n"
                f"Question: {query}\n\n"
                f"Write a brief hypothetical answer (2-3 sentences) to help retrieve relevant documents:\n\nAnswer:"
            )
            hypo_answer = generate_text(hypo_prompt, model, tokenizer, max_tokens=100)

            # Step 2: 가상 답변으로 임베딩 검색
            hypo_emb = embed_model.encode([hypo_answer])
            hypo_emb = hypo_emb / np.linalg.norm(hypo_emb)
            ctx, scores = retrieve(hypo_emb, embed_model, texts, index, is_emb=True)

            # Step 3: 실제 답변 생성
            ctx_text = "\n".join([c[:200] for c in ctx])
            answer_prompt = (
                f"You are a helpful expert.\n\n"
                f"Context:\n{ctx_text}\n\n"
                f"Question: {query}\n\n"
                f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
            )
            answer = generate_text(answer_prompt, model, tokenizer, max_tokens=300)
            results.append({"query":query,"gold_answer":gold,"pred_answer":answer})
        except Exception as e:
            print(f"  [Error] {e}")

        if (i+1) % 50 == 0:
            with open(f"{OUT_DIR}/hyde_7b_500.json","w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [저장] {i+1}개")

    with open(f"{OUT_DIR}/hyde_7b_500.json","w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] HyDE 완료: {len(results)}개")
    return results

# ── CRAG ──────────────────────────────────────────────
def run_crag(gt_data, embed_model, texts, index, model, tokenizer):
    print(f"\n{'='*50}\nCRAG\n{'='*50}")
    results = []
    stats = {"correct":0, "ambiguous":0, "incorrect":0}

    for i, item in enumerate(gt_data):
        query = item.get("instruction","")
        gold  = item.get("gold_answer","")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")
        try:
            # Step 1: 검색
            ctx, scores = retrieve(query, embed_model, texts, index)
            top_score   = float(scores[0])

            # Step 2: 검색 품질 판단
            # cosine similarity 기준: >0.85=correct, 0.6~0.85=ambiguous, <0.6=incorrect
            if top_score >= 0.85:
                quality = "correct"
                final_ctx = ctx
            elif top_score >= 0.60:
                quality = "ambiguous"
                # 재검색: 쿼리 확장
                expanded = f"{query} explanation details mechanism"
                extra_ctx, _ = retrieve(expanded, embed_model, texts, index, top_k=5)
                final_ctx = ctx[:5] + extra_ctx[:5]
            else:
                quality = "incorrect"
                # 검색 결과 불신 → 모델 자체 지식 사용
                final_ctx = []

            stats[quality] += 1

            # Step 3: 답변 생성
            if final_ctx:
                ctx_text = "\n".join([c[:200] for c in final_ctx])
                prompt = (
                    f"You are a helpful expert.\n\n"
                    f"Context:\n{ctx_text}\n\n"
                    f"Question: {query}\n\n"
                    f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
                )
            else:
                prompt = (
                    f"You are a helpful expert.\n\n"
                    f"Question: {query}\n\n"
                    f"Based on your knowledge, provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
                )

            answer = generate_text(prompt, model, tokenizer, max_tokens=300)
            results.append({
                "query":query,"gold_answer":gold,
                "pred_answer":answer,"retrieval_quality":quality
            })
        except Exception as e:
            print(f"  [Error] {e}")

        if (i+1) % 50 == 0:
            with open(f"{OUT_DIR}/crag_7b_500.json","w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [저장] {i+1}개 | correct:{stats['correct']} ambiguous:{stats['ambiguous']} incorrect:{stats['incorrect']}")

    with open(f"{OUT_DIR}/crag_7b_500.json","w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[+] CRAG 완료: {len(results)}개")
    print(f"    품질 분포: {stats}")
    return results

# ── 메인 ──────────────────────────────────────────────
if __name__ == "__main__":
    with open(GT_PATH) as f: gt_data = json.load(f)
    print(f"GT: {len(gt_data)}개")

    embed_model = SentenceTransformer(EMBED_MODEL)
    texts, index, _ = build_db(embed_model)

    print("\n[*] 7B 로드...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_7B, quantization_config=bnb,
            device_map="auto", trust_remote_code=True, local_files_only=True,
            attn_implementation="eager")
    print("[*] 로드 완료!")

    run_hyde(gt_data, embed_model, texts, index, model, tokenizer)
    run_crag(gt_data, embed_model, texts, index, model, tokenizer)

    print("\n[+] HyDE + CRAG 전체 완료!")
