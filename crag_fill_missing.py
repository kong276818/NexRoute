"""
CRAG 7B 누락 50개 재생성 (CROSS 40 + STEM 10)
기존 crag_7b_500.json + 새 결과 → crag_7b_500_v2.json
"""
import os, json, torch, numpy as np, faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

BASE_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"
EXISTING    = "/home/sclab/moe_model/baseline_results/crag_7b_500.json"
OUT_PATH    = "/home/sclab/moe_model/baseline_results/crag_7b_500_v2.json"
CKPT_PATH   = "/home/sclab/moe_model/baseline_results/crag_fill_ckpt.json"

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
            out = d.get("output", d.get("gold_answer", ""))
            all_texts.append(f"{d['instruction']} {out}")
    embs  = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"  DB: {len(all_texts)}개")
    return all_texts, index

def retrieve(query, embed_model, texts, index, top_k=10):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]], D[0]

def generate_text(prompt, model, tokenizer, max_tokens=300):
    msgs   = [{"role": "user", "content": prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in result:
            result = result[:result.index(kw)].strip()
    return result

def run_crag_single(query, embed_model, texts, index, model, tokenizer):
    ctx, scores = retrieve(query, embed_model, texts, index)
    top_score   = float(scores[0])
    if top_score >= 0.85:
        quality, final_ctx = "correct", ctx
    elif top_score >= 0.60:
        quality = "ambiguous"
        expanded = f"{query} explanation details mechanism"
        extra_ctx, _ = retrieve(expanded, embed_model, texts, index, top_k=5)
        final_ctx = ctx[:5] + extra_ctx[:5]
    else:
        quality, final_ctx = "incorrect", []

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
    return answer, quality

if __name__ == "__main__":
    # 기존 결과 로드
    with open(EXISTING) as f:
        existing = json.load(f)
    existing_queries = {d.get("query", "").strip() for d in existing}
    print(f"기존 CRAG 결과: {len(existing)}개")

    # GT 로드 → 누락 항목 추출
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    missing_items = [d for d in gt_data if d["instruction"].strip() not in existing_queries]
    print(f"누락 항목: {len(missing_items)}개")
    from collections import Counter
    print(f"도메인 분포: {dict(Counter(d.get('expected_domain','?') for d in missing_items))}")

    # 체크포인트 로드
    filled = {}
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            filled = json.load(f)
        print(f"체크포인트 로드: {len(filled)}개 완료")

    remaining = [d for d in missing_items if d["instruction"].strip() not in filled]
    if not remaining:
        print("모두 완료됨!")
    else:
        # 모델 로드
        print(f"\n[*] {len(remaining)}개 재생성 시작...")
        embed_model = SentenceTransformer(EMBED_MODEL)
        texts, index = build_db(embed_model)

        print("[*] Qwen2.5-7B 로드...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_7B, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, local_files_only=True, attn_implementation="eager")
        print("[*] 로드 완료!\n")

        for i, item in enumerate(remaining):
            q    = item["instruction"].strip()
            gold = item.get("gold_answer", "")
            domain = item.get("expected_domain", "?")
            print(f"[{i+1}/{len(remaining)}] [{domain}] {q[:60]}...")
            try:
                answer, quality = run_crag_single(q, embed_model, texts, index, model, tokenizer)
                filled[q] = {"query": q, "gold_answer": gold, "pred_answer": answer,
                             "retrieval_quality": quality, "expected_domain": domain}
                print(f"  quality={quality}, len={len(answer)}")
            except Exception as e:
                print(f"  [Error] {e}")
                filled[q] = {"query": q, "gold_answer": gold, "pred_answer": "",
                             "retrieval_quality": "error", "expected_domain": domain}

            if (i + 1) % 10 == 0:
                with open(CKPT_PATH, "w") as f:
                    json.dump(filled, f, ensure_ascii=False, indent=2)
                print(f"  [체크포인트 저장] {i+1}개")

        with open(CKPT_PATH, "w") as f:
            json.dump(filled, f, ensure_ascii=False, indent=2)

    # 기존 + 새 결과 병합 → v2
    new_items = list(filled.values())
    merged = existing + new_items
    # GT 순서로 정렬
    gt_order = {d["instruction"].strip(): i for i, d in enumerate(gt_data)}
    merged.sort(key=lambda d: gt_order.get(d.get("query", "").strip(), 9999))

    with open(OUT_PATH, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 저장 완료: {OUT_PATH}")
    print(f"   총 {len(merged)}개 (기존 {len(existing)} + 신규 {len(new_items)})")

    # 통계
    from collections import Counter
    domains = Counter(d.get("expected_domain", "?") for d in new_items)
    qualities = Counter(d.get("retrieval_quality", "?") for d in new_items)
    print(f"   신규 도메인: {dict(domains)}")
    print(f"   신규 품질:   {dict(qualities)}")
