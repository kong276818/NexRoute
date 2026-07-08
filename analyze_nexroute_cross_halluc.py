"""
NexRoute CROSS Hallucination 원인 분석
STEP 1: 응답 길이 vs hallucination 상관
STEP 2: KB 파티션 전략 비교 (합산 / 도메인별 / budget 분배)
STEP 3: isCross false positive 분리
STEP 4: per-100-token 정규화
STEP 5: 최종 리포트
"""
import json, os, torch, numpy as np, faiss
from scipy.stats import pearsonr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# ── 경로 ──────────────────────────────────────────────────────────────
NEXROUTE_PATH = "/home/sclab/moe_model/eval_results_7b_v3_500.json"
GT_PATH       = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
DATA_DIR      = "/home/sclab/moe_model/processed_v2"
NLI_MODEL     = "cross-encoder/nli-deberta-v3-large"
EMBED_MODEL   = "intfloat/multilingual-e5-large"
TOP_K = 3

# ── 데이터 로드 ────────────────────────────────────────────────────────
with open(NEXROUTE_PATH) as f: nexroute = json.load(f)
with open(GT_PATH) as f: gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

# CROSS 샘플만 추출
cross_samples = [x for x in nexroute if x.get("is_cross") == True]
print(f"[*] NexRoute CROSS 샘플: {len(cross_samples)}개\n")

# ── 모델 로드 ──────────────────────────────────────────────────────────
device_nli = 0 if torch.cuda.is_available() else -1
print("[*] NLI 모델 로드...")
nli_pipe = pipeline("text-classification", model=NLI_MODEL, device=device_nli,
                    truncation=True, max_length=512)
print("[*] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu")

# ── KB 구축 ────────────────────────────────────────────────────────────
DOMAIN_FILES = {
    "LAW": "law_rewritten.json", "MED": "med_rewritten.json",
    "SOC": "soc_rewritten.json", "STEM": "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}
print("[*] KB 인덱스 구축 중...")
kb = {}
for dom, fname in DOMAIN_FILES.items():
    with open(os.path.join(DATA_DIR, fname)) as f: data = json.load(f)
    texts = [d.get("output", d.get("gold_answer",""))[:400] for d in data
             if d.get("output") or d.get("gold_answer")]
    texts = [t for t in texts if len(t) > 20]
    embs  = embed_model.encode(texts, batch_size=256,
                               show_progress_bar=False, normalize_embeddings=True)
    idx   = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    kb[dom] = (texts, idx)
    print(f"  {dom}: {len(texts)}개")
print()


def retrieve(sent, domains, top_k=TOP_K):
    emb = embed_model.encode([sent], normalize_embeddings=True)
    results = []
    per = max(1, top_k // len(domains))
    for d in domains:
        if d not in kb: continue
        texts, idx = kb[d]
        D, I = idx.search(emb.astype(np.float32), per)
        for sc, i in zip(D[0], I[0]):
            results.append((float(sc), texts[i]))
    results.sort(key=lambda x: -x[0])
    return [t for _, t in results[:top_k]]


def check_sent(sent, passages):
    has_ent = has_con = False
    for p in passages:
        try:
            out   = nli_pipe({"text": p[:600], "text_pair": sent[:300]}, truncation=True)
            label = (out["label"] if isinstance(out, dict) else out[0]["label"]).upper()
            if label == "ENTAILMENT":   has_ent = True
            elif label == "CONTRADICTION": has_con = True
        except: continue
    if has_ent:  return "supported"
    if has_con:  return "contradiction"
    return "neutral"


def factscore_item(pred, domains):
    """
    returns: (total_sents, contra_sents, per_sent_labels)
    """
    sents  = [s.strip() for s in sent_tokenize(pred) if len(s.strip()) > 20]
    contra = 0
    for s in sents:
        passages = retrieve(s, domains, TOP_K)
        if not passages: continue
        if check_sent(s, passages) == "contradiction":
            contra += 1
    return len(sents), contra


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 + 3: 샘플별 분석 (길이, halluc, false-positive 여부)
# ═══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1+3: 샘플별 per-sample FActScore (domain_pred KB)")
print("=" * 60)

records = []
for i, item in enumerate(cross_samples):
    pred    = item.get("pred_answer", "").strip()
    q       = item.get("query", "").strip()
    domains = item.get("domain_pred", [])
    if isinstance(domains, str): domains = [domains]

    # GT 도메인 (false-positive 판별용)
    gt_item    = gt_map.get(q) or {}
    true_domain = gt_item.get("expected_domain", "UNKNOWN")
    is_fp = (true_domain != "CROSS")   # 실제론 single이지만 CROSS로 잘못 라우팅

    if not pred or not domains: continue

    n_sents, n_contra = factscore_item(pred, domains)
    token_len = len(pred.split())

    records.append({
        "id": item.get("id", i),
        "n_sents":    n_sents,
        "n_contra":   n_contra,
        "token_len":  token_len,
        "halluc_rate": n_contra / n_sents * 100 if n_sents else 0,
        "is_fp":      is_fp,
        "true_domain": true_domain,
        "domains":    domains,
    })
    if (i + 1) % 20 == 0:
        done = [r for r in records if not r["is_fp"]]
        cur  = sum(r["n_contra"] for r in done) / max(sum(r["n_sents"] for r in done),1) * 100
        print(f"  [{i+1}/{len(cross_samples)}]  TP cross halluc={cur:.1f}%")

# ── STEP 1: 상관 분석 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: 응답 길이 vs Hallucination 상관")
print("="*60)
lens   = [r["token_len"]  for r in records]
rates  = [r["halluc_rate"] for r in records]
r_val, p_val = pearsonr(lens, rates)
print(f"  Pearson r={r_val:.3f}  p={p_val:.4f}")
# 4분위별
lens_arr = np.array(lens); rates_arr = np.array(rates)
qs = np.percentile(lens_arr, [25, 50, 75])
for lo, hi, label in [(0, qs[0], "Q1(short)"), (qs[0], qs[1], "Q2"),
                       (qs[1], qs[2], "Q3"), (qs[2], 1e9, "Q4(long)")]:
    mask = (lens_arr >= lo) & (lens_arr < hi)
    sub  = rates_arr[mask]
    print(f"  {label:<12}  n={mask.sum():>3}  halluc={sub.mean():.1f}%  "
          f"tok_len={lens_arr[mask].mean():.0f}")

# ── STEP 3: false-positive 분석 ───────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: isCross False-Positive 분리")
print("="*60)
fp_records = [r for r in records if r["is_fp"]]
tp_records = [r for r in records if not r["is_fp"]]

def agg_rate(recs):
    t = sum(r["n_sents"] for r in recs); c = sum(r["n_contra"] for r in recs)
    return c/t*100 if t else 0, t

fp_rate, fp_n = agg_rate(fp_records)
tp_rate, tp_n = agg_rate(tp_records)
all_rate, _   = agg_rate(records)
print(f"  전체  CROSS:               {all_rate:.1f}%  (N={len(records)})")
print(f"  False-Positive (잘못라우팅): {fp_rate:.1f}%  (N={len(fp_records)})")
print(f"  True  CROSS (TP only):     {tp_rate:.1f}%  (N={len(tp_records)})")
for r in fp_records[:5]:
    print(f"    FP example: true={r['true_domain']}  pred_domains={r['domains']}  "
          f"halluc={r['halluc_rate']:.1f}%")

# ── STEP 2: KB 파티션 전략 비교 ───────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: KB 파티션 전략 비교 (TP CROSS 샘플, N=20 샘플로 추정)")
print("="*60)
# TP CROSS 샘플 20개로 빠르게 비교
sample20 = tp_records[:20]
ids_20   = {r["id"] for r in sample20}
items_20 = [item for item in cross_samples
            if item.get("id") in ids_20][:20]

def run_strategy(items, strategy_fn, label):
    tot = con = 0
    for item in items:
        pred    = item.get("pred_answer", "").strip()
        domains = item.get("domain_pred", [])
        if isinstance(domains, str): domains = [domains]
        doms    = strategy_fn(item, domains)
        n, c    = factscore_item(pred, doms)
        tot += n; con += c
    rate = con/tot*100 if tot else 0
    print(f"  {label:<35}  halluc={rate:.1f}%  (sents={tot})")
    return rate

# Case A: 현재 방식 (domain_pred 합산)
rate_A = run_strategy(items_20,
    lambda it, d: d,
    "Case A: domain_pred 합산 (current)")

# Case B: domain_pred[0]만
rate_B = run_strategy(items_20,
    lambda it, d: [d[0]] if d else d,
    "Case B: 주 도메인만")

# Case C: domain_pred[1]만 (있을 때)
rate_C = run_strategy(items_20,
    lambda it, d: [d[1]] if len(d)>1 else d,
    "Case C: 부 도메인만")

# Case D: CROSS KB 추가
rate_D = run_strategy(items_20,
    lambda it, d: list(set(d + ["CROSS"])),
    "Case D: domain_pred + CROSS KB")

# Case E: confidence score 비율로 budget 분배
def budget_strategy(item, domains):
    scores = item.get("domain_scores", {})
    if len(domains) < 2: return domains
    s1 = scores.get(domains[0], 0.5); s2 = scores.get(domains[1], 0.5)
    # 두 도메인 모두 포함하되 점수 비율 정보만 로그
    return domains

rate_E = run_strategy(items_20,
    budget_strategy,
    "Case E: confidence 비율 budget (≈A)")

# ── STEP 4: per-100-token 정규화 ──────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Per-100-token Hallucination Rate")
print("="*60)
tp_lens   = np.array([r["token_len"]  for r in tp_records])
tp_contra = np.array([r["n_contra"]   for r in tp_records])
# per-100-token: n_contra / token_len * 100
norm_rates = tp_contra / np.maximum(tp_lens, 1) * 100
print(f"  TP CROSS  per-100-tok halluc: {norm_rates.mean():.2f}%")
# Single 기준값 (nexroute single 샘플)
single_items = [x for x in nexroute if not x.get("is_cross")][:len(tp_records)]
# 빠른 추정: token_len 기반으로 normalization만 비교
single_lens = np.array([len(x.get("pred_answer","").split()) for x in single_items])
print(f"  Single 평균 응답 길이: {single_lens.mean():.0f} tok  "
      f"vs CROSS: {tp_lens.mean():.0f} tok")

# ── STEP 5: 최종 리포트 ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: NexRoute CROSS Hallucination 원인 분석 리포트")
print("="*60)
rate_fixed_kb = rate_D   # CROSS KB 추가
rate_no_fp    = tp_rate
rate_combined = None     # STEP2+3 동시 적용 (TP 샘플 + CROSS KB) 추정

# 주요 원인 판별
causes = []
if abs(r_val) > 0.2 and p_val < 0.05:
    causes.append(f"응답 길이 상관 (r={r_val:.2f})")
if len(fp_records) > 0 and abs(fp_rate - tp_rate) > 2:
    causes.append(f"false-positive 라우팅 ({len(fp_records)}개, FP halluc={fp_rate:.1f}%)")
if rate_D < all_rate - 1:
    causes.append(f"KB 파티션 부족 (CROSS KB 추가 시 {rate_D:.1f}%)")
primary = " / ".join(causes) if causes else "복합적 요인 (명확한 단일 원인 없음)"

print(f"  원본 CROSS hallucination:        {all_rate:.1f}%")
print(f"  CROSS KB 추가 후 (Step2 Case D): {rate_fixed_kb:.1f}%")
print(f"  False-Positive 제외 후 (Step3):  {rate_no_fp:.1f}%")
print(f"  주요 원인: {primary}")
print(f"\n  응답 길이 CROSS: {tp_lens.mean():.0f}tok  Single: {single_lens.mean():.0f}tok")
print(f"  Per-100-tok halluc(CROSS TP): {norm_rates.mean():.2f}%")

# JSON 저장
result = {
    "all_cross_rate":   round(all_rate, 1),
    "tp_cross_rate":    round(tp_rate, 1),
    "fp_cross_rate":    round(fp_rate, 1),
    "n_fp":             len(fp_records),
    "n_tp":             len(tp_records),
    "kb_case_A":        round(rate_A, 1),
    "kb_case_D_cross":  round(rate_D, 1),
    "pearson_r":        round(float(r_val), 3),
    "pearson_p":        round(float(p_val), 4),
    "primary_cause":    primary,
}
with open("/home/sclab/moe_model/nexroute_cross_halluc_analysis.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\n✅ 저장: nexroute_cross_halluc_analysis.json")
