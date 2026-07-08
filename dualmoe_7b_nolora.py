import os
import re
import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import torch.nn as nn
from sentence_transformers import SentenceTransformer

DATA_DIR    = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL = "intfloat/multilingual-e5-large"
ROUTER_PATH = "/home/sclab/moe_model/router_multilabel.pt"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_sample_50.json"
OUT_PATH    = "/home/sclab/moe_model/baseline_results/dualmoe_7b_nolora.json"

QWEN_7B  = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DOMAINS  = ["LAW", "MED", "SOC", "STEM"]
THRESHOLD = 0.30
GAP       = 0.40
TOP_K     = 10

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":  "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":  "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM": "You are a science and engineering expert. Answer accurately based on scientific principles.",
}

# ── BERT 라우터 ──────────────────────────────────
class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, 4)
        )
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

def load_router():
    print("[*] BERT 라우터 로드...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    router    = DomainRouter()
    router.load_state_dict(torch.load(ROUTER_PATH, map_location="cpu"))
    router.eval().to("cuda")
    return router, tokenizer

def classify(router, tokenizer, query):
    enc = tokenizer(query, max_length=256, truncation=True,
                    padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = router(enc["input_ids"], enc["attention_mask"])
        scores = torch.sigmoid(logits).squeeze().cpu().tolist()
    domain_scores = dict(zip(DOMAINS, scores))
    sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
    top1, s1 = sorted_d[0]
    top2, s2 = sorted_d[1]
    is_cross  = (s2 >= THRESHOLD) and ((s1 - s2) <= GAP)
    return ("CROSS", [top1,top2]) if is_cross else (top1, [top1])

# ── 벡터 DB ──────────────────────────────────────
def build_db(embed_model):
    print("[*] 벡터 DB 구축...")
    files = {
        "LAW":"law.json", "MED":"med.json",
        "SOC":"soc.json", "STEM":"stem.json",
        "CROSS":"cross_rewritten.json",
    }
    db = {}
    for domain, fname in files.items():
        with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        texts = [f"{d['instruction']} {d.get('output', d.get('gold_answer',''))}"
                 for d in data
                 if "The correct answer" not in d.get("output", d.get("gold_answer",""))]
        embs  = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.astype(np.float32))
        db[domain] = (texts, index)
        print(f"  {domain}: {len(texts)}개")
    return db

def retrieve(query, embed_model, db, domains, is_cross, top_k=TOP_K):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    results = []
    if is_cross:
        per = max(1, top_k // len(domains))
        for d in domains:
            if d not in db: continue
            texts, index = db[d]
            D, I = index.search(emb.astype(np.float32), per)
            for score, idx in zip(D[0], I[0]):
                results.append({"domain":d, "text":texts[idx], "score":float(score)})
    else:
        texts, index = db[domains[0]]
        D, I = index.search(emb.astype(np.float32), top_k)
        for score, idx in zip(D[0], I[0]):
            results.append({"domain":domains[0], "text":texts[idx], "score":float(score)})
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]

# ── 생성 ─────────────────────────────────────────
def generate(query, context, model, tokenizer, is_cross, domains):
    ctx_text = "\n".join([f"[{r['domain']}] {r['text'][:200]}" for r in context])
    if is_cross:
        system = (f"You are a multidisciplinary expert in {' and '.join(domains)}. "
                  f"Integrate knowledge from both domains.")
    else:
        system = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")

    prompt = (
        f"{system}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form. "
        f"Do NOT use multiple choice format.\n\n"
        f"Answer:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=300, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    for kw in ["Choices:", "All answers:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[GENERATION_FAILED]"

# ── 메인 ─────────────────────────────────────────
if __name__ == "__main__":
    with open(GT_PATH, encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"GT 샘플: {len(gt_data)}개")

    # 라우터
    router, r_tokenizer = load_router()

    # 임베딩 + DB
    embed_model = SentenceTransformer(EMBED_MODEL)
    db          = build_db(embed_model)

    # 7B 베이스 모델 (LoRA 없음)
    print("[*] Qwen2.5-7B 로드 (LoRA 없음)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    gen_tokenizer = AutoTokenizer.from_pretrained(QWEN_7B, trust_remote_code=True)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    print("[*] 로드 완료!")

    results = []
    for i, item in enumerate(gt_data):
        query = item.get("instruction", "")
        gold  = item.get("gold_answer", "")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")

        try:
            domain, domains = classify(router, r_tokenizer, query)
            is_cross        = (domain == "CROSS")
            context         = retrieve(query, embed_model, db, domains, is_cross)
            answer          = generate(query, context, model, gen_tokenizer,
                                       is_cross, domains)
            results.append({
                "query":       query,
                "gold_answer": gold,
                "is_cross":    is_cross,
                "domains":     domains,
                "pred_answer": answer,
            })
        except Exception as e:
            print(f"  [Error] {e}")

        if (i+1) % 10 == 0:
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[+] 완료! {len(results)}개 → {OUT_PATH}")