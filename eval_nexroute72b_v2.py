"""
NexRoute 72B v2 — max_new_tokens=120, 베이스라인과 동일한 생성 조건
기존 eval_nexroute_72b.py와 라우터/검색 로직은 동일하고
generate()만 baseline(eval_72b_rerun.py) 기준으로 맞춤
"""
import os, json, time, glob, re
import torch
import numpy as np
import torch.nn as nn
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

DATA_DIR    = "/home/sclab/moe_model/processed_v2"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_PATH    = "/home/sclab/moe_model/baseline_results/nexroute_72b_500_v2.json"
ROUTER_PATH = "/home/sclab/moe_model/router_multilabel_v6_best.pt"
EMBED_MODEL = "intfloat/multilingual-e5-large"

cands  = glob.glob("/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/*")
BASE_72B = cands[0]
print(f"[*] 72B 경로: {BASE_72B}")

DOMAINS   = ["LAW", "MED", "SOC", "STEM"]
THRESHOLD = 0.30
GAP       = 0.40
TOP_K     = 10

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert. Answer accurately based on legal principles.",
    "MED":  "You are a medical expert. Answer accurately based on medical facts.",
    "SOC":  "You are a social science expert. Answer based on social context and research.",
    "STEM": "You are a science and engineering expert. Answer based on scientific principles.",
    "CROSS":"You are a multidisciplinary expert. Integrate knowledge from multiple domains.",
}


# ── 라우터 ────────────────────────────────────────────────────────────
class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 4)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


def classify_domain(query, router, router_tok, device):
    enc = router_tok(
        query, max_length=256, truncation=True,
        padding="max_length", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = router(enc["input_ids"], enc["attention_mask"])
        scores = torch.sigmoid(logits).squeeze().cpu().tolist()
    domain_scores = dict(zip(DOMAINS, scores))
    sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
    top1, s1 = sorted_d[0]
    top2, s2 = sorted_d[1]
    is_cross = (s2 >= THRESHOLD) and ((s1 - s2) <= GAP)
    return {
        "is_cross": is_cross,
        "domains":  [top1, top2] if is_cross else [top1],
        "scores":   domain_scores,
    }


# ── 벡터 DB ───────────────────────────────────────────────────────────
def build_db(embed_model):
    domain_files = {
        "LAW":  "law_rewritten.json",  "MED":  "med_rewritten.json",
        "SOC":  "soc_rewritten.json",  "STEM": "stem_rewritten.json",
        "CROSS":"cross_rewritten.json",
    }
    db = {}
    for domain, fname in domain_files.items():
        with open(os.path.join(DATA_DIR, fname)) as f:
            data = json.load(f)
        filtered = [d for d in data if not any(
            k in d.get("output", d.get("gold_answer", ""))
            for k in ["The correct answer", "All answers:"]
        )]
        texts = [
            f"{d['instruction']} {d.get('output', d.get('gold_answer', ''))}"
            for d in filtered
        ]
        embs = embed_model.encode(texts, show_progress_bar=False, batch_size=256)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs.astype(np.float32))
        db[domain] = (texts, index)
        print(f"  {domain}: {len(texts)}개")
    return db


def retrieve(query, embed_model, db, domains, is_cross, top_k=TOP_K):
    emb_q = embed_model.encode([query])
    emb_q = emb_q / np.linalg.norm(emb_q)
    results = []
    if is_cross:
        per = max(1, top_k // len(domains))
        for d in domains:
            if d not in db: continue
            texts, index = db[d]
            D, I = index.search(emb_q.astype(np.float32), per)
            for sc, idx in zip(D[0], I[0]):
                results.append({"domain": d, "text": texts[idx], "score": float(sc)})
    else:
        texts, index = db[domains[0]]
        D, I = index.search(emb_q.astype(np.float32), top_k)
        for sc, idx in zip(D[0], I[0]):
            results.append({"domain": domains[0], "text": texts[idx], "score": float(sc)})
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ── 생성 (baseline 동일: max_new_tokens=120, 간결 프롬프트) ──────────
def generate(query, context, system, model, tokenizer):
    ctx_text = "\n".join([r["text"][:200] for r in context])
    prompt = (
        f"{system}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    first_device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=2048).to(first_device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:", "The correct answer"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer


# ── 모델 로드 ─────────────────────────────────────────────────────────
print("[*] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")

print("[*] 라우터 로드...")
router_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
router     = DomainRouter()
router.load_state_dict(torch.load(ROUTER_PATH, map_location="cpu"))
router.eval()
router_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
router.to(router_device)

print("[*] 벡터 DB 구축...")
db = build_db(embed_model)

print("[*] 72B LLM 로드...")
t0  = time.time()
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
max_memory = {0: "23GiB", 1: "23GiB", 2: "23GiB", "cpu": "30GiB"}
model = AutoModelForCausalLM.from_pretrained(
    BASE_72B, quantization_config=bnb, device_map="auto",
    max_memory=max_memory, trust_remote_code=True, local_files_only=True,
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_72B, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"  로드 완료: {time.time()-t0:.1f}s")

# ── 추론 ──────────────────────────────────────────────────────────────
with open(GT_PATH) as f:
    gt_data = json.load(f)

results = []
for i, item in enumerate(gt_data):
    query = item["instruction"]
    gold  = item["gold_answer"]

    routing   = classify_domain(query, router, router_tok, router_device)
    is_cross  = routing["is_cross"]
    domains   = routing["domains"]
    context   = retrieve(query, embed_model, db, domains, is_cross)

    key  = "CROSS" if is_cross else domains[0]
    sys_ = DOMAIN_SYSTEM.get(key, "You are a helpful expert.")
    pred = generate(query, context, sys_, model, tokenizer)

    results.append({
        "id":           i,
        "query":        query,
        "gold_answer":  gold,
        "domain_gt":    item.get("expected_domain", ""),
        "domain_pred":  "CROSS" if is_cross else domains[0],
        "is_cross":     is_cross,
        "domain_scores":routing["scores"],
        "pred_answer":  pred,
    })

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/500] domain={domains}  cross={is_cross}")
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ 저장 완료: {OUT_PATH}  (총 {len(results)}개)")
