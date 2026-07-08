import os, json, time, glob, re
import torch
import numpy as np
import torch.nn as nn
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# ── 경로 ─────────────────────────────────────────────────────────────
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_PATH    = "/home/sclab/moe_model/baseline_results/nexroute_72b_500.json"
EMBED_MODEL = "intfloat/multilingual-e5-large"
ROUTER_PATH = "/home/sclab/moe_model/router_multilabel_v6_best.pt"

cands = glob.glob("/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/*")
BASE_72B = cands[0]
print(f"[*] 72B 모델 경로: {BASE_72B}")

# ── 라우터 설정 ───────────────────────────────────────────────────────
DOMAINS   = ["LAW", "MED", "SOC", "STEM"]
THRESHOLD = 0.30
GAP       = 0.40
TOP_K     = 10

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":  "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":  "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM": "You are a science and engineering expert. Answer accurately based on scientific principles.",
}

FEW_SHOT = {
    "LAW": (
        "Q: What is the doctrine of promissory estoppel?\n"
        "A: Promissory estoppel is a legal doctrine that prevents a party from "
        "withdrawing a promise when another party has reasonably relied on it to "
        "their detriment. The doctrine requires a clear and definite promise, "
        "reasonable reliance by the promisee, and actual detriment suffered as a "
        "result of that reliance. Courts apply this principle to enforce promises "
        "even without formal consideration, thereby ensuring fairness and preventing "
        "injustice in contractual relationships."
    ),
    "MED": (
        "Q: How does the renin-angiotensin system regulate blood pressure?\n"
        "A: The renin-angiotensin system regulates blood pressure through a cascade "
        "of enzymatic reactions initiated when the kidneys detect low blood pressure "
        "or reduced sodium levels. Renin converts angiotensinogen to angiotensin I, "
        "which is subsequently converted to angiotensin II by ACE in the lungs. "
        "Angiotensin II causes vasoconstriction and stimulates aldosterone secretion, "
        "increasing sodium retention and blood volume, ultimately elevating blood pressure."
    ),
    "SOC": (
        "Q: How does social stratification affect educational outcomes?\n"
        "A: Social stratification significantly influences educational outcomes by "
        "creating unequal access to resources, opportunities, and cultural capital. "
        "Children from higher socioeconomic backgrounds benefit from better-funded "
        "schools, private tutoring, and enriched home environments that foster "
        "academic achievement. Conversely, those from lower strata face systemic "
        "barriers including underfunded schools and economic pressures that hinder "
        "educational attainment, perpetuating intergenerational inequality."
    ),
    "STEM": (
        "Q: How does CRISPR-Cas9 technology work?\n"
        "A: CRISPR-Cas9 is a revolutionary gene-editing technology that uses a "
        "guide RNA to direct the Cas9 endonuclease to specific DNA sequences. "
        "The guide RNA binds complementarily to the target DNA, enabling Cas9 to "
        "create a precise double-strand break at the desired location. The cell "
        "then repairs this break through either non-homologous end joining, which "
        "can disrupt gene function, or homology-directed repair, which can introduce "
        "specific genetic modifications with high precision."
    ),
    "CROSS": (
        "Q: How might advances in neuroscience impact criminal justice policy?\n"
        "A: Advances in neuroscience are profoundly reshaping criminal justice policy "
        "by providing empirical insights into the biological underpinnings of behavior, "
        "decision-making, and culpability. Neuroimaging studies reveal that certain "
        "criminal behaviors may be associated with structural or functional brain "
        "abnormalities, raising questions about moral responsibility and the "
        "appropriate balance between punishment and rehabilitation. Policymakers must "
        "therefore integrate neuroscientific evidence with established legal principles "
        "to develop more nuanced, evidence-based approaches to sentencing and "
        "treatment that reflect both individual accountability and neurological realities."
    ),
}


# ── BERT 도메인 라우터 ─────────────────────────────────────────────────
class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
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


# ── 벡터 DB 구축 ──────────────────────────────────────────────────────
def build_db(embed_model):
    domain_files = {
        "LAW":   "law_rewritten.json",
        "MED":   "med_rewritten.json",
        "SOC":   "soc_rewritten.json",
        "STEM":  "stem_rewritten.json",
        "CROSS": "cross_rewritten.json",
    }
    db = {}
    for domain, fname in domain_files.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        filtered = [
            d for d in data
            if not any(k in d.get("output", d.get("gold_answer", ""))
                       for k in ["The correct answer", "All answers:"])
        ]
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
            if d not in db:
                continue
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


def generate(query, context, is_cross, domains, model, tokenizer):
    ctx_text = "\n".join([f"[{r['domain']}] {r['text'][:200]}" for r in context])

    if is_cross:
        system   = (f"You are a multidisciplinary expert specializing in "
                    f"{' and '.join(domains)}. "
                    f"Integrate knowledge from both domains to answer accurately.")
        few_shot = FEW_SHOT["CROSS"]
    else:
        system   = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")
        few_shot = FEW_SHOT.get(domains[0], FEW_SHOT["STEM"])

    prompt = (
        f"{system}\n\n"
        f"Here is an example of a high-quality answer:\n"
        f"{few_shot}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Following the example style above, provide a comprehensive "
        f"analytical answer with specific mechanisms and implications. "
        f"Do NOT use multiple choice format. "
        f"Do NOT list choices A, B, C, D. "
        f"Write in flowing paragraph form only.\n\n"
        f"Answer:"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=3072).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # MCQ 형식 제거
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
    for kw in ["Choices:", "All answers:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    answer = re.sub(r'\n\d+:.*', '', answer).strip()

    return answer if answer else "[GENERATION_FAILED]"


# ── 메인 ─────────────────────────────────────────────────────────────
print("[*] STEP 1: 임베딩 모델 로딩 (cuda:2)...")
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda:2")

print("[*] STEP 2: Vector DB 구축...")
db = build_db(embed_model)

print("[*] STEP 3: BERT 라우터 로딩 (cuda:0)...")
router_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
router = DomainRouter()
router.load_state_dict(torch.load(ROUTER_PATH, map_location="cpu"))
router.eval().to("cuda:0")

print("[*] STEP 4: 72B 모델 로딩 (4bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_72B)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_72B, quantization_config=bnb, device_map="auto"
)
print("[*] 72B 모델 로딩 완료!")

with open(GT_PATH) as f:
    gt_data = json.load(f)

print(f"\n[*] NexRoute 72B 평가 시작: {len(gt_data)}개")
results = []
t_start = time.time()

for i, item in enumerate(gt_data):
    query = item["instruction"]
    gold  = item["gold_answer"]

    try:
        routing = classify_domain(query, router, router_tok, "cuda:0")
        is_cross = routing["is_cross"]
        domains  = routing["domains"]

        context = retrieve(query, embed_model, db, domains, is_cross)
        answer  = generate(query, context, is_cross, domains, model, tokenizer)
    except Exception as e:
        print(f"  [Error #{i}] {e}")
        answer   = "[ERROR]"
        routing  = {"is_cross": False, "domains": ["STEM"], "scores": {}}
        is_cross = False
        domains  = ["STEM"]

    results.append({
        "id":            i,
        "query":         query,
        "gold_answer":   gold,
        "domain_gt":     item.get("expected_domain", ""),
        "domain_pred":   domains,
        "is_cross":      is_cross,
        "domain_scores": routing["scores"],
        "pred_answer":   answer,
    })

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(gt_data) - i - 1)
        print(f"  [{i+1}/{len(gt_data)}] 경과: {elapsed/60:.1f}분 | ETA: {eta/60:.1f}분")
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

elapsed = time.time() - t_start
print(f"\n✅ 완료! {len(results)}개 | 총 {elapsed/60:.1f}분")
print(f"저장: {OUT_PATH}")
