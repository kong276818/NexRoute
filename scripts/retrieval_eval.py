"""
retrieval_eval.py
=================
어노테이션 없이 계산 가능한 retrieval-level 지표
(a) Cross-domain contamination rate (unified retrieval)
(b) Domain-mismatch ratio 분포
(c) CROSS 질의의 도메인별 evidence coverage
(d) 오염 passage 수 vs NaiveRAG hallucination 상관
(e) Recall@10, nDCG@10 proxy (gold-answer embedding 기반)

생성 없음 — 검색 + 임베딩만 수행.
"""
import json, os, csv, math
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict

GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
PREDS_CSV  = "/home/sclab/moe_model/preds_500.csv"
NAIVE_PREDS= "/home/sclab/moe_model/reinference_results/preds_500_naive_rag.json"
EMBED_NAME = "intfloat/multilingual-e5-large"
OUT_DIR    = "/home/sclab/moe_model/retrieval_eval_results"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_K   = 10
PSEUDO_K = 50   # pseudo-relevant set 크기

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}
DOMAIN_ORDER = ["LAW", "MED", "SOC", "STEM", "CROSS"]
CLEAN_IDX = set(range(0, 300))
EXPOSED_IDX = set(range(300, 500))

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("[1] GT 및 라우팅 로드...")
with open(GT_PATH) as f:
    gt_list = json.load(f)
print(f"  GT: {len(gt_list)}개")

routing = {}
with open(PREDS_CSV) as f:
    for row in csv.DictReader(f):
        routing[int(row['sample_id'])] = {
            'pred_domain': row['pred_domain_nexroute'],
            'is_cross':    row['is_cross'].strip().lower() == 'true',
            'top1':        row['top1'],
            'top2':        row['top2'],
        }

# ── 임베딩 모델 ───────────────────────────────────────────────────────────────
print("[2] 임베딩 모델 로드...")
embed = SentenceTransformer(EMBED_NAME, device="cuda")

# ── 코퍼스 인덱스 구축 ─────────────────────────────────────────────────────────
print("[3] 코퍼스 인덱스 구축...")
domain_texts  = {}   # domain -> [text, ...]
domain_index  = {}   # domain -> faiss index
corpus_domain = []   # unified corpus에서 각 passage의 domain 레이블
corpus_texts  = []   # unified corpus 전체 텍스트

for domain in DOMAIN_ORDER:
    fname = DOMAIN_FILES[domain]
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    texts = []
    for d in data:
        out = d.get("output", d.get("gold_answer", ""))
        if any(k in out for k in ["The correct answer", "All answers:"]):
            continue
        t = f"{d['instruction']} {out}"
        texts.append(t)
        corpus_texts.append(t)
        corpus_domain.append(domain)
    embs = embed.encode(texts, batch_size=256, show_progress_bar=False,
                        normalize_embeddings=True)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    domain_texts[domain] = texts
    domain_index[domain] = idx
    print(f"  {domain}: {len(texts)}개")

print("[4] Unified 인덱스 구축...")
uni_embs = embed.encode(corpus_texts, batch_size=256, show_progress_bar=True,
                        normalize_embeddings=True)
uni_index = faiss.IndexFlatIP(uni_embs.shape[1])
uni_index.add(uni_embs.astype(np.float32))
corpus_domain = np.array(corpus_domain)
print(f"  UNIFIED: {len(corpus_texts)}개")

# ── 질의 임베딩 ───────────────────────────────────────────────────────────────
print("[5] 500개 질의 임베딩...")
queries    = [item["instruction"] for item in gt_list]
gold_ans   = [item["gold_answer"] for item in gt_list]
gold_doms  = [item["expected_domain"] for item in gt_list]
query_embs = embed.encode(queries, batch_size=128, show_progress_bar=True,
                          normalize_embeddings=True)

print("[6] Gold answer 임베딩 (recall proxy용)...")
gold_embs = embed.encode(gold_ans, batch_size=128, show_progress_bar=True,
                         normalize_embeddings=True)

# ── Pseudo-relevant set 구축 ─────────────────────────────────────────────────
print("[7] Pseudo-relevant set 구축 (gold answer → unified 상위 50)...")
_, pseudo_I = uni_index.search(gold_embs.astype(np.float32), PSEUDO_K)
# pseudo_I[i] = i번 질의의 pseudo-relevant passage 인덱스 top-50

# ── 검색 ─────────────────────────────────────────────────────────────────────
print("[8] Unified 검색 (top-10)...")
uni_D, uni_I = uni_index.search(query_embs.astype(np.float32), TOP_K)

print("[9] Partitioned 검색 (gold domain 기준 top-10)...")
part_I_list = []  # 질의별 retrieved passage 인덱스 (unified 기준 offset)
part_D_list = []
# domain → unified corpus offset 계산
domain_offset = {}
offset = 0
for domain in DOMAIN_ORDER:
    domain_offset[domain] = offset
    offset += len(domain_texts[domain])

for i in range(len(gt_list)):
    gold_dom  = gold_doms[i]
    r = routing.get(i, {})
    is_cross  = r.get('is_cross', False)
    top1      = r.get('top1', gold_dom)
    top2      = r.get('top2', gold_dom)
    domains   = [top1, top2] if is_cross else [top1]

    emb = query_embs[i:i+1]
    retrieved_global = []

    if is_cross:
        per = max(1, TOP_K // len(domains))
        for dom in domains:
            if dom not in domain_index:
                continue
            D, I = domain_index[dom].search(emb.astype(np.float32), per)
            off  = domain_offset[dom]
            for score, local_idx in zip(D[0], I[0]):
                retrieved_global.append((float(score), off + local_idx))
        retrieved_global.sort(key=lambda x: -x[0])
        retrieved_global = retrieved_global[:TOP_K]
    else:
        dom = top1 if top1 in domain_index else gold_dom
        if dom not in domain_index:
            dom = "STEM"
        D, I = domain_index[dom].search(emb.astype(np.float32), TOP_K)
        off = domain_offset[dom]
        retrieved_global = [(float(D[0][j]), off + I[0][j]) for j in range(len(I[0]))]

    scores = [x[0] for x in retrieved_global]
    idxs   = [x[1] for x in retrieved_global]
    part_D_list.append(scores)
    part_I_list.append(idxs)
    if (i+1) % 100 == 0:
        print(f"  partitioned: {i+1}/500")

# ── 지표 계산 ─────────────────────────────────────────────────────────────────

def dcg(relevances):
    return sum(r / math.log2(k + 2) for k, r in enumerate(relevances))

records = []
contamination_by_domain = defaultdict(list)   # gold_domain → [mismatch count per query]
cross_coverage = {"unified": [], "partitioned": []}  # CROSS 질의 도메인 coverage

for i, item in enumerate(gt_list):
    gold_dom = item["expected_domain"]
    is_cross = (gold_dom == "CROSS")
    r = routing.get(i, {})
    pred_top1 = r.get('top1', gold_dom)
    pred_top2 = r.get('top2', gold_dom)
    pred_cross = r.get('is_cross', False)

    # ── unified 검색 결과 도메인 분류 ──────────────────────────────────────────
    uni_retrieved_domains = [corpus_domain[j] for j in uni_I[i]]

    # 오염 = gold_domain이 아닌 도메인에서 온 passage
    # CROSS 질의의 경우 어느 단일 도메인도 "정답"이 아니므로
    # 오염 정의: CROSS 질의는 pred_top1/top2 이외 도메인을 오염으로 간주
    if is_cross:
        target_domains = {pred_top1, pred_top2} if pred_cross else {pred_top1}
    else:
        target_domains = {gold_dom}

    uni_mismatch = sum(1 for d in uni_retrieved_domains if d not in target_domains)
    uni_mismatch_rate = uni_mismatch / TOP_K

    # ── partitioned 검색 결과 도메인 분류 ──────────────────────────────────────
    part_domains = [corpus_domain[j] for j in part_I_list[i]]
    part_mismatch = sum(1 for d in part_domains if d not in target_domains)
    part_mismatch_rate = part_mismatch / TOP_K

    contamination_by_domain[gold_dom].append(uni_mismatch)

    # ── CROSS evidence coverage ─────────────────────────────────────────────
    if is_cross:
        uni_cov = {d: uni_retrieved_domains.count(d) for d in DOMAIN_ORDER}
        part_cov = {d: part_domains.count(d) for d in DOMAIN_ORDER}
        cross_coverage["unified"].append(uni_cov)
        cross_coverage["partitioned"].append(part_cov)

    # ── Recall@10, nDCG@10 proxy ────────────────────────────────────────────
    pseudo_set = set(pseudo_I[i])

    uni_rel   = [1 if j in pseudo_set else 0 for j in uni_I[i]]
    part_rel  = [1 if j in pseudo_set else 0 for j in part_I_list[i]]

    denom = min(TOP_K, len(pseudo_set))
    uni_recall   = sum(uni_rel) / denom if denom > 0 else 0
    part_recall  = sum(part_rel) / denom if denom > 0 else 0

    ideal_dcg_val = dcg([1] * min(TOP_K, len(pseudo_set)))
    uni_ndcg  = dcg(uni_rel)  / ideal_dcg_val if ideal_dcg_val > 0 else 0
    part_ndcg = dcg(part_rel) / ideal_dcg_val if ideal_dcg_val > 0 else 0

    split = "clean" if i in CLEAN_IDX else "exposed"

    records.append({
        "idx": i,
        "gold_domain": gold_dom,
        "is_cross": is_cross,
        "split": split,
        "uni_mismatch":      uni_mismatch,
        "uni_mismatch_rate": round(uni_mismatch_rate, 4),
        "part_mismatch":     part_mismatch,
        "part_mismatch_rate": round(part_mismatch_rate, 4),
        "uni_recall10":  round(uni_recall, 4),
        "part_recall10": round(part_recall, 4),
        "uni_ndcg10":    round(uni_ndcg, 4),
        "part_ndcg10":   round(part_ndcg, 4),
    })

# ── FActScore 연동 (d): Naive RAG hallucination per query 추정 ───────────────
print("[10] Naive RAG per-query hallucination 추정...")
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

nli_pipe = pipeline("text-classification",
                    model="cross-encoder/nli-deberta-v3-large",
                    device=0, truncation=True, max_length=512)

kb_texts = {}
kb_index = {}
for domain in DOMAIN_ORDER:
    with open(os.path.join(DATA_DIR, DOMAIN_FILES[domain])) as f:
        data = json.load(f)
    texts = [d.get("output", d.get("gold_answer",""))[:400]
             for d in data if d.get("output") or d.get("gold_answer")]
    texts = [t for t in texts if len(t) > 20]
    embs  = embed.encode(texts, batch_size=256, show_progress_bar=False,
                         normalize_embeddings=True)
    idx   = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    kb_texts[domain] = texts
    kb_index[domain] = idx
# Unified KB
all_kb_texts = []
for d in DOMAIN_ORDER:
    all_kb_texts.extend(kb_texts[d])
uni_kb_embs = embed.encode(all_kb_texts, batch_size=256, show_progress_bar=False,
                           normalize_embeddings=True)
uni_kb_idx = faiss.IndexFlatIP(uni_kb_embs.shape[1])
uni_kb_idx.add(uni_kb_embs.astype(np.float32))

with open(NAIVE_PREDS) as f:
    naive_preds = json.load(f)
naive_map = {item["idx"]: item for item in naive_preds}

hall_records = []
for rec in records:
    i = rec["idx"]
    pred_item = naive_map.get(i)
    if not pred_item:
        hall_records.append(None)
        continue
    pred_text = pred_item.get("pred_answer", "").strip()
    if not pred_text or pred_text in ("[FAILED]", "[GENERATION_FAILED]"):
        hall_records.append(None)
        continue
    sents = [s.strip() for s in sent_tokenize(pred_text) if len(s.strip()) > 20]
    if not sents:
        hall_records.append(None)
        continue

    contra = 0
    total  = len(sents)
    for sent in sents:
        emb_s = embed.encode([sent], normalize_embeddings=True).astype(np.float32)
        D_kb, I_kb = uni_kb_idx.search(emb_s, 3)
        for kb_idx_val in I_kb[0]:
            try:
                out = nli_pipe({"text": all_kb_texts[kb_idx_val][:600],
                               "text_pair": sent[:300]}, truncation=True)
                label = (out["label"] if isinstance(out, dict) else out[0]["label"]).upper()
                if label == "CONTRADICTION":
                    contra += 1
                    break
            except:
                pass

    hall_records.append(round(contra / total, 4) if total > 0 else None)
    if (i + 1) % 50 == 0:
        print(f"  hallucination: {i+1}/500")

# ── 결과 집계 및 저장 ─────────────────────────────────────────────────────────
print("\n[11] 집계 및 저장...")

def avg(lst):
    lst = [x for x in lst if x is not None]
    return round(float(np.mean(lst)), 4) if lst else None

def group(records, filter_fn):
    return [r for r in records if filter_fn(r)]

def report(label, recs):
    uni_mis   = avg([r["uni_mismatch_rate"]  for r in recs])
    part_mis  = avg([r["part_mismatch_rate"] for r in recs])
    uni_r10   = avg([r["uni_recall10"]   for r in recs])
    part_r10  = avg([r["part_recall10"]  for r in recs])
    uni_ndcg  = avg([r["uni_ndcg10"]     for r in recs])
    part_ndcg = avg([r["part_ndcg10"]    for r in recs])
    n = len(recs)
    return {
        "group": label, "n": n,
        "unified_mismatch_rate":     uni_mis,
        "partitioned_mismatch_rate": part_mis,
        "unified_recall10":          uni_r10,
        "partitioned_recall10":      part_r10,
        "unified_ndcg10":            uni_ndcg,
        "partitioned_ndcg10":        part_ndcg,
    }

groups = [
    ("All-500",       lambda r: True),
    ("Clean-300",     lambda r: r["split"] == "clean"),
    ("Exposed-200",   lambda r: r["split"] == "exposed"),
    ("CROSS",         lambda r: r["is_cross"]),
    ("Single",        lambda r: not r["is_cross"]),
    ("Clean-CROSS",   lambda r: r["split"] == "clean" and r["is_cross"]),
    ("Clean-Single",  lambda r: r["split"] == "clean" and not r["is_cross"]),
]
for domain in DOMAIN_ORDER:
    groups.append((f"Single-{domain}",
                   lambda r, d=domain: r["gold_domain"] == d and not r["is_cross"]))

summary = []
for label, fn in groups:
    recs = group(records, fn)
    summary.append(report(label, recs))

# Histogram: 질의당 unified mismatch passage 수 분포
hist = defaultdict(int)
for r in records:
    hist[r["uni_mismatch"]] += 1
hist = dict(sorted(hist.items()))

# CROSS coverage 집계
def avg_cov(cov_list):
    if not cov_list:
        return {}
    agg = defaultdict(list)
    for c in cov_list:
        for d, v in c.items():
            agg[d].append(v)
    return {d: round(float(np.mean(vs)), 2) for d, vs in agg.items()}

cross_uni_cov  = avg_cov(cross_coverage["unified"])
cross_part_cov = avg_cov(cross_coverage["partitioned"])

# Contamination vs Hallucination 상관
corr_data = [(records[i]["uni_mismatch"], hall_records[i])
             for i in range(len(records)) if hall_records[i] is not None]
if corr_data:
    x = np.array([c[0] for c in corr_data])
    y = np.array([c[1] for c in corr_data])
    corr = float(np.corrcoef(x, y)[0, 1])
else:
    corr = None

# Per-domain contamination
domain_cont = {}
for dom, counts in contamination_by_domain.items():
    domain_cont[dom] = {
        "mean_mismatch_passages": round(float(np.mean(counts)), 2),
        "zero_mismatch_pct":      round(100 * sum(1 for c in counts if c == 0) / len(counts), 1),
        "n": len(counts),
    }

# 저장
result = {
    "summary_table":           summary,
    "mismatch_histogram":      hist,
    "cross_unified_coverage":  cross_uni_cov,
    "cross_partitioned_coverage": cross_part_cov,
    "contamination_hallucination_corr": corr,
    "n_corr_pairs":            len(corr_data),
    "domain_contamination":    domain_cont,
    "embed_model":             EMBED_NAME,
    "pseudo_k":                PSEUDO_K,
    "top_k":                   TOP_K,
    "note": ("Recall and nDCG are proxy metrics based on gold-answer embedding similarity. "
             "Pseudo-relevant set = top-50 passages from unified corpus by gold-answer embedding. "
             "These are relative comparisons only; absolute values do not represent true relevance."),
}
out_path = os.path.join(OUT_DIR, "retrieval_eval_results.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

# 샘플별 CSV
csv_path = os.path.join(OUT_DIR, "retrieval_eval_per_query.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) + ["naive_hallucination"])
    writer.writeheader()
    for i, r in enumerate(records):
        row = dict(r)
        row["naive_hallucination"] = hall_records[i]
        writer.writerow(row)

print(f"\n[✓] 저장: {out_path}")
print(f"[✓] 저장: {csv_path}")
print("\n" + "="*80)
print("RETRIEVAL EVALUATION SUMMARY")
print("="*80)
print(f"{'Group':<16} {'N':>5}  {'UniMis':>7} {'PrtMis':>7}  {'UniR@10':>8} {'PrtR@10':>8}  {'UniDCG':>7} {'PrtDCG':>7}")
print("-"*80)
for row in summary[:7]:
    print(f"  {row['group']:<14} {row['n']:>5}  "
          f"{str(row['unified_mismatch_rate']):>7} {str(row['partitioned_mismatch_rate']):>7}  "
          f"{str(row['unified_recall10']):>8} {str(row['partitioned_recall10']):>8}  "
          f"{str(row['unified_ndcg10']):>7} {str(row['partitioned_ndcg10']):>7}")
print("="*80)
print(f"\nContamination-Hallucination Pearson r = {corr} (n={len(corr_data)})")
print(f"\nCROSS evidence coverage (avg passages per query, top-10):")
print(f"  Unified:    {cross_uni_cov}")
print(f"  Partitioned:{cross_part_cov}")
print(f"\nMismatch histogram (unified, passages per query):")
for k, v in hist.items():
    print(f"  {k} mismatch passages: {v} queries ({100*v/500:.1f}%)")
