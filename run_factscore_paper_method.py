"""
FActScore (논문 동일 방법) — eval/eval_factscore_hallucination.py 설정 재현
NLI: cross-encoder/nli-deberta-v3-large
KB:  domain-partitioned FAISS, TOP_K=3
     NexRoute → pred_domains (CROSS = 두 도메인 합집합)
     baselines → UNIFIED KB
판정: ENTAILMENT 있으면 supported, CONTRADICTION만 있으면 contradiction
출력:
  reinference_results/factscore_paper_per_sample_{sys}.csv  (per-sample)
  reinference_results/factscore_paper_summary.json          (집계)
"""
import json, os, csv, torch, faiss, numpy as np
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
OUT_DIR  = "/home/sclab/moe_model/reinference_results"
NLI_MODEL   = "cross-encoder/nli-deberta-v3-large"   # 논문과 동일
EMBED_MODEL = "intfloat/multilingual-e5-large"
TOP_K       = 3                                        # 논문과 동일
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",  "MED":  "med_rewritten.json",
    "SOC":   "soc_rewritten.json",  "STEM": "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

print(f"[*] NLI 모델 로드: {NLI_MODEL}")
nli = hf_pipeline("text-classification", model=NLI_MODEL,
                  device=0 if DEVICE == "cuda" else -1, truncation=True, max_length=512)

print(f"[*] 임베딩 모델 로드: {EMBED_MODEL}")
enc = SentenceTransformer(EMBED_MODEL, device=DEVICE)

print("[*] KB 인덱스 구축...")
kb = {}       # domain -> (texts, index)
all_texts = []
for dom, fname in DOMAIN_FILES.items():
    texts = [d.get("output", d.get("gold_answer", ""))[:400]
             for d in json.load(open(os.path.join(DATA_DIR, fname)))
             if d.get("output") or d.get("gold_answer")]
    texts = [t for t in texts if len(t) > 20]
    embs = enc.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    kb[dom] = (texts, idx)
    all_texts.extend(texts)
    print(f"  {dom}: {len(texts)}개")

uni_embs = enc.encode(all_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
uni_idx  = faiss.IndexFlatIP(uni_embs.shape[1])
uni_idx.add(uni_embs.astype(np.float32))
kb["UNIFIED"] = (all_texts, uni_idx)
print(f"  UNIFIED: {len(all_texts)}개\n")


def retrieve(sent, domains):
    emb = enc.encode([sent], normalize_embeddings=True).astype(np.float32)
    per = max(1, TOP_K // len(domains))
    results = []
    for dom in domains:
        if dom not in kb:
            continue
        texts, index = kb[dom]
        D, I = index.search(emb, per)
        for score, i in zip(D[0], I[0]):
            if i >= 0:
                results.append((float(score), texts[i]))
    results.sort(key=lambda x: -x[0])
    return [t for _, t in results[:TOP_K]]


def check_sentence(sent, passages):
    """논문과 동일: ENTAILMENT 있으면 supported, CONTRADICTION만 있으면 contradiction."""
    has_entail = has_contra = False
    for passage in passages:
        try:
            out = nli({"text": passage[:600], "text_pair": sent[:300]}, truncation=True)
            label = (out["label"] if isinstance(out, dict) else out[0]["label"]).upper()
            if label == "ENTAILMENT":
                has_entail = True
            elif label == "CONTRADICTION":
                has_contra = True
        except Exception:
            continue
    if has_entail:
        return "supported"
    elif has_contra:
        return "contradiction"
    return "neutral"


def pct(c, t):
    return round(100 * c / t, 2) if t > 0 else None


summary = {}

for sys_name, (path, use_part) in RESULT_FILES.items():
    print(f"\n[▶] {sys_name}")
    data = json.load(open(path))

    per_sample_rows = []

    # 집계 카운터: [total_sents, contra_sents]
    C = {k: [0, 0] for k in [
        "overall", "cross", "single",
        "clean", "clean_cross", "clean_single",
        "exposed", "exposed_cross", "exposed_single",
    ]}

    for i, item in enumerate(data):
        pred     = item.get("pred_answer", "").strip()
        idx      = item.get("idx", i)
        is_cross = bool(item.get("is_cross", False))
        is_clean = (idx < 300)
        exp_dom  = item.get("expected_domain", "")

        if use_part:
            doms = item.get("pred_domains", item.get("pred_domain", ["UNIFIED"]))
            if isinstance(doms, str):
                doms = [doms]
            if not doms:
                doms = ["UNIFIED"]
        else:
            doms = ["UNIFIED"]

        sents = [s.strip() for s in sent_tokenize(pred) if len(s.strip()) > 20] if pred else []

        n_sents = len(sents)
        n_contra = 0
        for sent in sents:
            passages = retrieve(sent, doms)
            if passages and check_sentence(sent, passages) == "contradiction":
                n_contra += 1

        hal_rate = round(100 * n_contra / n_sents, 2) if n_sents > 0 else None

        per_sample_rows.append({
            "idx": idx,
            "expected_domain": exp_dom,
            "is_cross": is_cross,
            "is_clean": is_clean,
            "pred_domains": "|".join(doms) if isinstance(doms, list) else doms,
            "n_sents": n_sents,
            "n_contra": n_contra,
            "hal_rate": hal_rate,
        })

        def add(key, ns, nc):
            C[key][0] += ns
            C[key][1] += nc

        add("overall", n_sents, n_contra)
        if is_cross:
            add("cross", n_sents, n_contra)
        else:
            add("single", n_sents, n_contra)
        if is_clean:
            add("clean", n_sents, n_contra)
            add("clean_cross" if is_cross else "clean_single", n_sents, n_contra)
        else:
            add("exposed", n_sents, n_contra)
            add("exposed_cross" if is_cross else "exposed_single", n_sents, n_contra)

        if (i + 1) % 100 == 0:
            ov = pct(C["overall"][1], C["overall"][0])
            print(f"  [{i+1}/500] overall={ov}%")

    # per-sample CSV 저장
    csv_path = os.path.join(OUT_DIR, f"factscore_paper_per_sample_{sys_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"  CSV 저장: {csv_path}")

    summary[sys_name] = {
        k: {"hal": pct(v[1], v[0]), "n_sents": v[0], "n_contra": v[1]}
        for k, v in C.items()
    }
    r = summary[sys_name]
    print(f"  overall={r['overall']['hal']}%  cross={r['cross']['hal']}%  single={r['single']['hal']}%")
    print(f"  clean={r['clean']['hal']}%  clean_cross={r['clean_cross']['hal']}%  clean_single={r['clean_single']['hal']}%")

out_path = os.path.join(OUT_DIR, "factscore_paper_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n집계 저장: {out_path}")
