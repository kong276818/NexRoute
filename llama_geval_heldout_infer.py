"""
Phase 1: Held-out Cross Inference
- cross_rewritten.json에서 GT500에 없는 100개 샘플링
- 5개 모델로 추론: NexRoute 7B / Naive RAG / Adaptive RAG / HyDE / CRAG
- 출력: ~/moe_model/heldout_cross_predictions.json
- 체크포인트: ~/moe_model/heldout_infer_checkpoint.json (매 10개)
"""
import os, sys, json, random, re, torch, numpy as np, faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from peft import PeftModel

sys.path.insert(0, "/home/sclab/moe_model")

SEED        = 42
N_SAMPLES   = 100
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
CROSS_RW    = "/home/sclab/moe_model/processed_v2/cross_rewritten.json"
OUT_PATH    = "/home/sclab/moe_model/heldout_cross_predictions.json"
CKPT_PATH   = "/home/sclab/moe_model/heldout_infer_checkpoint.json"
BASE_7B_DIR = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
EMBED_MODEL = "intfloat/multilingual-e5-large"
ROUTER_PATH = "/home/sclab/moe_model/router_multilabel.pt"
LORA_DIR    = "/home/sclab/moe_model/lora_weights_7b_v3"
DOMAINS     = ["LAW", "MED", "SOC", "STEM"]
THRESHOLD   = 0.30
GAP         = 0.40

import glob
BASE_7B = glob.glob(f"{BASE_7B_DIR}/*")[0]

# ── Held-out 샘플 선택 ──────────────────────────────────────────────────
random.seed(SEED)
with open(GT_PATH) as f:
    gt500 = json.load(f)
gt500_instr = {d["instruction"].strip() for d in gt500}

with open(CROSS_RW) as f:
    cross_rw = json.load(f)
pool = [item for item in cross_rw if item["instruction"].strip() not in gt500_instr]
held_out = random.sample(pool, min(N_SAMPLES, len(pool)))
print(f"Held-out pool: {len(pool)}, 샘플: {len(held_out)}")

# ── 체크포인트 로드 ──────────────────────────────────────────────────────
results = {}
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH) as f:
        results = json.load(f)
    print(f"기존 체크포인트 로드: {list(results.keys())}")

def save_ckpt():
    with open(CKPT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# ── 키워드 라우터 (Adaptive RAG용) ──────────────────────────────────────
KEYWORDS = {
    "LAW":  ["law","legal","court","statute","plaintiff","contract","judge","attorney"],
    "MED":  ["medical","patient","disease","treatment","clinical","diagnosis","symptom","drug"],
    "SOC":  ["social","society","culture","economic","political","community","psychology"],
    "STEM": ["science","physics","chemical","biology","math","engineering","technology"],
}

def keyword_route(query):
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

# ── 벡터 DB 구축 ─────────────────────────────────────────────────────────
def build_unified_db(embed_model):
    print("[*] 벡터 DB 구축...")
    domain_files = {
        "LAW":   "law_rewritten.json",
        "MED":   "med_rewritten.json",
        "SOC":   "soc_rewritten.json",
        "STEM":  "stem_rewritten.json",
        "CROSS": "cross_rewritten.json",
    }
    all_texts = []
    for fname in domain_files.values():
        with open(os.path.join(DATA_DIR, fname)) as f:
            data = json.load(f)
        for d in data:
            out = d.get("output", d.get("gold_answer", ""))
            if "The correct answer" not in out:
                all_texts.append(f"{d['instruction']} {out}")
    embs = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"[*] DB 완료: {len(all_texts)}개")
    return all_texts, index

def retrieve(query_or_emb, embed_model, texts, index, top_k=10, precomputed=False):
    if precomputed:
        emb = query_or_emb
    else:
        emb = embed_model.encode([query_or_emb])
        emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]], D[0]

def generate_answer(query, context, model, tokenizer, max_new_tokens=200):
    ctx_text = "\n".join([c[:200] for c in context])
    prompt = (
        f"You are a helpful expert.\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form. "
        f"Do NOT use multiple choice format. Do NOT begin with 'The correct answer is'.\n\n"
        f"Answer:"
    )
    msgs   = [{"role": "user", "content": prompt}]
    text   = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in pred:
            pred = pred[:pred.index(kw)].strip()
    pred = re.sub(r"The correct answer is[^\n]*\n?", "", pred).strip()
    return pred

# ── 4-bit 7B 모델 로드 ──────────────────────────────────────────────────
needs_7b = any(m not in results for m in ["Naive RAG 7B", "Adaptive RAG 7B", "HyDE 7B", "CRAG 7B"])

embed_model = None
texts = index = None
model_7b = tokenizer_7b = None

if needs_7b:
    print("[*] 임베딩 모델 로드...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    texts, index = build_unified_db(embed_model)

    print("[*] Qwen2.5-7B 4-bit 로드...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer_7b = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True)
    tokenizer_7b.pad_token = tokenizer_7b.eos_token
    model_7b = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    print("[*] 7B 로드 완료!")

# ── Naive RAG ───────────────────────────────────────────────────────────
if "Naive RAG 7B" not in results:
    print(f"\n{'='*50}\nNaive RAG 7B\n{'='*50}")
    naive_res = []
    for i, item in enumerate(held_out):
        q    = item["instruction"]
        gold = item["output"]
        try:
            ctx, _ = retrieve(q, embed_model, texts, index)
            pred   = generate_answer(q, ctx, model_7b, tokenizer_7b)
        except Exception as e:
            print(f"  [Error] {e}"); pred = ""
        naive_res.append({"query": q, "gold_answer": gold, "pred_answer": pred,
                          "cross_pair": item.get("crossed_domains", "")})
        if (i + 1) % 10 == 0:
            results["Naive RAG 7B"] = naive_res
            save_ckpt()
            print(f"  [{i+1}/100]")
    results["Naive RAG 7B"] = naive_res
    save_ckpt()

# ── Adaptive RAG ────────────────────────────────────────────────────────
if "Adaptive RAG 7B" not in results:
    print(f"\n{'='*50}\nAdaptive RAG 7B\n{'='*50}")
    adaptive_res = []
    for i, item in enumerate(held_out):
        q      = item["instruction"]
        gold   = item["output"]
        domain = keyword_route(q)
        try:
            ctx, _ = retrieve(q, embed_model, texts, index)
            pred   = generate_answer(q, ctx, model_7b, tokenizer_7b)
        except Exception as e:
            print(f"  [Error] {e}"); pred = ""
        adaptive_res.append({"query": q, "gold_answer": gold, "pred_answer": pred,
                             "routed_domain": domain, "cross_pair": item.get("crossed_domains", "")})
        if (i + 1) % 10 == 0:
            results["Adaptive RAG 7B"] = adaptive_res
            save_ckpt()
            print(f"  [{i+1}/100]")
    results["Adaptive RAG 7B"] = adaptive_res
    save_ckpt()

# ── HyDE ────────────────────────────────────────────────────────────────
if "HyDE 7B" not in results:
    print(f"\n{'='*50}\nHyDE 7B\n{'='*50}")
    hyde_res = []
    for i, item in enumerate(held_out):
        q    = item["instruction"]
        gold = item["output"]
        try:
            hyp_prompt = f"Write a brief hypothetical answer (2-3 sentences) to: {q}\n\nAnswer:"
            hyp_msgs   = [{"role": "user", "content": hyp_prompt}]
            hyp_text   = tokenizer_7b.apply_chat_template(hyp_msgs, tokenize=False, add_generation_prompt=True)
            hyp_inputs = tokenizer_7b(hyp_text, return_tensors="pt",
                                      truncation=True, max_length=512).to(model_7b.device)
            with torch.no_grad():
                hyp_out = model_7b.generate(**hyp_inputs, max_new_tokens=80, temperature=0.7,
                                             do_sample=True, pad_token_id=tokenizer_7b.eos_token_id)
            hypo     = tokenizer_7b.decode(hyp_out[0][hyp_inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True).strip()
            hyp_emb  = embed_model.encode([hypo])
            hyp_emb  = hyp_emb / np.linalg.norm(hyp_emb)
            ctx, _   = retrieve(hyp_emb, embed_model, texts, index, precomputed=True)
            pred     = generate_answer(q, ctx, model_7b, tokenizer_7b)
        except Exception as e:
            print(f"  [Error] {e}"); pred = ""
        hyde_res.append({"query": q, "gold_answer": gold, "pred_answer": pred,
                         "cross_pair": item.get("crossed_domains", "")})
        if (i + 1) % 10 == 0:
            results["HyDE 7B"] = hyde_res
            save_ckpt()
            print(f"  [{i+1}/100]")
    results["HyDE 7B"] = hyde_res
    save_ckpt()

# ── CRAG ─────────────────────────────────────────────────────────────────
if "CRAG 7B" not in results:
    print(f"\n{'='*50}\nCRAG 7B\n{'='*50}")
    crag_res = []
    for i, item in enumerate(held_out):
        q    = item["instruction"]
        gold = item["output"]
        try:
            ctx, scores = retrieve(q, embed_model, texts, index)
            top_score   = float(scores[0])
            if top_score >= 0.85:
                final_ctx = ctx
            elif top_score >= 0.60:
                extra_ctx, _ = retrieve(f"{q} explanation details mechanism",
                                         embed_model, texts, index, top_k=5)
                final_ctx = ctx[:5] + extra_ctx[:5]
            else:
                final_ctx = []
            pred = generate_answer(q, final_ctx, model_7b, tokenizer_7b)
        except Exception as e:
            print(f"  [Error] {e}"); pred = ""
        crag_res.append({"query": q, "gold_answer": gold, "pred_answer": pred,
                         "retrieval_score": float(scores[0]) if len(scores) > 0 else 0,
                         "cross_pair": item.get("crossed_domains", "")})
        if (i + 1) % 10 == 0:
            results["CRAG 7B"] = crag_res
            save_ckpt()
            print(f"  [{i+1}/100]")
    results["CRAG 7B"] = crag_res
    save_ckpt()

# ── 7B 모델 해제 후 NexRoute 로드 ────────────────────────────────────────
if "NexRoute 7B" not in results:
    if model_7b is not None:
        del model_7b
        torch.cuda.empty_cache()
        print("[*] 7B 메모리 해제")

    print(f"\n{'='*50}\nNexRoute 7B\n{'='*50}")

    from pipeline import DualMoEPipeline, DOMAIN_SYSTEM, FEW_SHOT

    # BERT 라우터 정의 (pipeline.py와 동일)
    class DomainRouter(nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import AutoModel as _AM
            self.encoder = _AM.from_pretrained("bert-base-uncased")
            h = self.encoder.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 4))
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :])

    from transformers import AutoTokenizer as _ATok
    router_tok = _ATok.from_pretrained("bert-base-uncased")
    router_model = DomainRouter()
    router_model.load_state_dict(torch.load(ROUTER_PATH, map_location="cpu"))
    router_model.eval().cuda()

    def classify_domain(query):
        enc = router_tok(query, max_length=256, truncation=True,
                         padding="max_length", return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = router_model(enc["input_ids"], enc["attention_mask"])
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()
        domain_scores = dict(zip(DOMAINS, scores))
        sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
        top1, s1 = sorted_d[0]; top2, s2 = sorted_d[1]
        is_cross = (s2 >= THRESHOLD) and ((s1 - s2) <= GAP)
        return {"is_cross": is_cross, "domains": [top1, top2] if is_cross else [top1]}

    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL)

    # 도메인별 DB (NexRoute는 도메인 분리 DB 사용)
    nx_db = {}
    for dom, fname in {"LAW": "law_rewritten.json", "MED": "med_rewritten.json",
                       "SOC": "soc_rewritten.json", "STEM": "stem_rewritten.json",
                       "CROSS": "cross_rewritten.json"}.items():
        with open(os.path.join(DATA_DIR, fname)) as f:
            data = json.load(f)
        txts = [f"{d['instruction']} {d.get('output', d.get('gold_answer',''))}"
                for d in data if "The correct answer" not in d.get("output", d.get("gold_answer",""))]
        embs = embed_model.encode(txts, show_progress_bar=False, batch_size=64)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        idx  = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs.astype(np.float32))
        nx_db[dom] = (txts, idx)

    def nx_retrieve(query, domains, is_cross, top_k=10):
        emb_q = embed_model.encode([query])
        emb_q = emb_q / np.linalg.norm(emb_q)
        res = []
        if is_cross:
            per = max(1, top_k // len(domains))
            for d in domains:
                if d not in nx_db: continue
                txts, idx = nx_db[d]
                D, I = idx.search(emb_q.astype(np.float32), per)
                for sc, ii in zip(D[0], I[0]):
                    res.append({"domain": d, "text": txts[ii], "score": float(sc)})
        else:
            txts, idx = nx_db.get(domains[0], nx_db["CROSS"])
            D, I = idx.search(emb_q.astype(np.float32), top_k)
            for sc, ii in zip(D[0], I[0]):
                res.append({"domain": domains[0], "text": txts[ii], "score": float(sc)})
        res.sort(key=lambda x: -x["score"])
        return res[:top_k]

    # Base model + LoRA 로드
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    nx_tok = AutoTokenizer.from_pretrained(
        BASE_7B, trust_remote_code=True, local_files_only=True)
    nx_tok.pad_token = nx_tok.eos_token
    nx_base = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")

    LORA_MAP = {"LAW": "lora_law", "MED": "lora_med",
                "SOC": "lora_soc", "STEM": "lora_stem", "CROSS": "lora_cross"}
    nx_base = PeftModel.from_pretrained(
        nx_base, os.path.join(LORA_DIR, "lora_cross"), adapter_name="CROSS")
    for dom, lname in LORA_MAP.items():
        if dom == "CROSS": continue
        nx_base.load_adapter(os.path.join(LORA_DIR, lname), adapter_name=dom)
    current_lora = "CROSS"
    nx_base.set_adapter("CROSS")
    print("[*] NexRoute 로드 완료!")

    def nx_generate(query, context, is_cross, domains):
        ctx_text = "\n".join([f"[{r['domain']}] {r['text'][:200]}" for r in context])
        primary  = domains[0] if not is_cross else "CROSS"
        if is_cross:
            system   = (f"You are a multidisciplinary expert specializing in "
                        f"{' and '.join(domains)}. Integrate knowledge from both domains to answer accurately.")
            few_shot = FEW_SHOT["CROSS"]
        else:
            system   = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")
            few_shot = FEW_SHOT.get(domains[0], FEW_SHOT["STEM"])
        domain_instr = (
            "Provide a comprehensive explanation of the legal holding, principle, or rule. "
            "Describe the court's reasoning, the applicable legal standard, any relevant exceptions, "
            "and the practical implications. Write as a detailed legal analysis in full paragraph form."
            if primary == "LAW" else
            "provide a comprehensive analytical answer with specific mechanisms and implications"
        )
        prompt = (
            f"{system}\n\n"
            f"Here is an example of a high-quality answer:\n{few_shot}\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Question: {query}\n\n"
            f"Following the example style above, {domain_instr}. "
            f"Do NOT use multiple choice format. Do NOT list choices A, B, C, D. "
            f"Do NOT begin with 'The correct answer is'. Write in flowing paragraph form only.\n\nAnswer:"
        )
        msgs   = [{"role": "user", "content": prompt}]
        text   = nx_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = nx_tok(text, return_tensors="pt", truncation=True, max_length=2048).to(nx_base.device)
        max_tok = 800 if primary == "LAW" else 512
        with torch.no_grad():
            out = nx_base.generate(**inputs, max_new_tokens=max_tok, temperature=0.7,
                                   do_sample=True, pad_token_id=nx_tok.eos_token_id)
        answer = nx_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        answer = re.sub(r"The correct answer is[^\n]*\n?", "", answer).strip()
        answer = re.sub(r"The correct holding is option\s*\d+\s*:\s*", "", answer).strip()
        return answer

    nexroute_res = []
    for i, item in enumerate(held_out):
        q    = item["instruction"]
        gold = item["output"]
        try:
            routing  = classify_domain(q)
            is_cross = routing["is_cross"]
            domains  = routing["domains"]
            target   = "CROSS" if is_cross else domains[0]
            if current_lora != target:
                nx_base.set_adapter(target)
                current_lora = target
            ctx  = nx_retrieve(q, domains, is_cross)
            pred = nx_generate(q, ctx, is_cross, domains)
        except Exception as e:
            print(f"  [Error] {e}"); pred = ""
        nexroute_res.append({"query": q, "gold_answer": gold, "pred_answer": pred,
                             "is_cross": is_cross, "domains": domains,
                             "cross_pair": item.get("crossed_domains", "")})
        if (i + 1) % 10 == 0:
            results["NexRoute 7B"] = nexroute_res
            save_ckpt()
            print(f"  [{i+1}/100]")
    results["NexRoute 7B"] = nexroute_res
    save_ckpt()

# ── 최종 저장 ────────────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*60)
print("Phase 1 완료!")
for name, res in results.items():
    print(f"  {name}: {len(res)}개")
print(f"저장: {OUT_PATH}")
