"""
FActScore (NLI-based hallucination rate) for reinference_results/
Split: overall / cross / single / clean / clean_cross / clean_single / exposed / exposed_cross / exposed_single
Output: reinference_results/factscore_summary.json
"""
import json, os, torch, faiss, numpy as np
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

RESULT_FILES = {
    "naive_rag":    ("/home/sclab/moe_model/reinference_results/preds_500_naive_rag.json",    False),
    "tfidf_router": ("/home/sclab/moe_model/reinference_results/preds_500_tfidf_router.json", False),
    "hyde":         ("/home/sclab/moe_model/reinference_results/preds_500_hyde.json",         False),
    "crag":         ("/home/sclab/moe_model/reinference_results/preds_500_crag.json",         False),
    "nexroute":     ("/home/sclab/moe_model/reinference_results/preds_500_nexroute.json",     True),
}

DATA_DIR = "/home/sclab/moe_model/processed_v2"
TOP_K    = 5
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",  "MED":  "med_rewritten.json",
    "SOC":   "soc_rewritten.json",  "STEM": "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

print("[*] NLI 모델 로드...")
nli = hf_pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small",
                  device=0 if DEVICE == "cuda" else -1, truncation=True, max_length=512)

print("[*] 임베딩 모델 로드...")
enc = SentenceTransformer("intfloat/multilingual-e5-large", device=DEVICE)

print("[*] KB 인덱스 구축...")
domain_texts = {}
domain_index = {}
for dom, fname in DOMAIN_FILES.items():
    texts = [d.get("output", d.get("gold_answer", ""))
             for d in json.load(open(os.path.join(DATA_DIR, fname)))]
    embs = enc.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    domain_texts[dom] = texts
    domain_index[dom] = idx
    print(f"  {dom}: {len(texts)}개")

all_texts = sum(domain_texts.values(), [])
all_embs = enc.encode(all_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
uni_idx = faiss.IndexFlatIP(all_embs.shape[1])
uni_idx.add(all_embs.astype(np.float32))

def retrieve(sent, domains):
    emb = enc.encode([sent], normalize_embeddings=True).astype(np.float32)
    if not domains or domains == ["UNIFIED"]:
        _, I = uni_idx.search(emb, TOP_K)
        return [all_texts[i] for i in I[0] if i >= 0]
    per = max(1, TOP_K // len(domains))
    results = []
    for dom in domains:
        if dom in domain_index:
            _, I = domain_index[dom].search(emb, per)
            results += [domain_texts[dom][i] for i in I[0] if i >= 0]
    return results[:TOP_K]

def check(sent, passages):
    if not passages:
        return "neutral"
    premise = " ".join(passages)[:1000]
    try:
        r = nli(f"{premise} [SEP] {sent}")[0]
        return "contradiction" if r["label"].lower() == "contradiction" else "neutral"
    except:
        return "neutral"

def pct(c, t):
    return round(100 * c / t, 2) if t > 0 else None

results_out = {}

for sys_name, (path, use_part) in RESULT_FILES.items():
    print(f"\n[▶] {sys_name}")
    data = json.load(open(path))

    counters = {k: [0, 0] for k in [
        "overall", "cross", "single",
        "clean", "clean_cross", "clean_single",
        "exposed", "exposed_cross", "exposed_single"
    ]}

    for i, item in enumerate(data):
        pred = item.get("pred_answer", "").strip()
        if not pred:
            continue
        is_cross = item.get("is_cross", False)
        idx = item.get("idx", 9999)
        is_clean = (idx < 300)

        if use_part:
            doms = item.get("pred_domains", item.get("pred_domain", ["UNIFIED"]))
            if isinstance(doms, str):
                doms = [doms]
            if not doms:
                doms = ["UNIFIED"]
        else:
            doms = ["UNIFIED"]

        sents = [s.strip() for s in sent_tokenize(pred) if len(s.strip()) > 20]
        if not sents:
            continue

        n_contra = sum(1 for s in sents if check(s, retrieve(s, doms)) == "contradiction")
        ns = len(sents)

        counters["overall"][0] += ns;  counters["overall"][1] += n_contra
        if is_cross:
            counters["cross"][0] += ns;  counters["cross"][1] += n_contra
        else:
            counters["single"][0] += ns; counters["single"][1] += n_contra
        if is_clean:
            counters["clean"][0] += ns;  counters["clean"][1] += n_contra
            if is_cross:
                counters["clean_cross"][0] += ns;  counters["clean_cross"][1] += n_contra
            else:
                counters["clean_single"][0] += ns; counters["clean_single"][1] += n_contra
        else:
            counters["exposed"][0] += ns; counters["exposed"][1] += n_contra
            if is_cross:
                counters["exposed_cross"][0] += ns;  counters["exposed_cross"][1] += n_contra
            else:
                counters["exposed_single"][0] += ns; counters["exposed_single"][1] += n_contra

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/500] overall={pct(counters['overall'][1], counters['overall'][0])}%")

    results_out[sys_name] = {
        k: {"hal": pct(v[1], v[0]), "n_sents": v[0]}
        for k, v in counters.items()
    }
    r = results_out[sys_name]
    print(f"  overall={r['overall']['hal']}%  cross={r['cross']['hal']}%  single={r['single']['hal']}%")
    print(f"  clean={r['clean']['hal']}%  clean_cross={r['clean_cross']['hal']}%  clean_single={r['clean_single']['hal']}%")

out_path = "/home/sclab/moe_model/reinference_results/factscore_summary.json"
with open(out_path, "w") as f:
    json.dump(results_out, f, indent=2, ensure_ascii=False)
print(f"\n저장 완료: {out_path}")
