"""
FActScore-style Hallucination Rate
- pred_answer를 문장 단위로 분리
- 각 문장을 domain KB에서 top-k 패시지 검색 후 NLI 검증
- NexRoute: domain_pred 기반 partitioned KB
- 베이스라인: unified KB (전 도메인 통합)
- 결과: contradiction 비율 = hallucination rate
"""
import json, os, torch, numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# ── 경로 ──────────────────────────────────────────────────────────────
DATA_DIR  = "/home/sclab/moe_model/processed_v2"
OUT_PATH  = "/home/sclab/moe_model/factscore_hallucination_results.json"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
EMBED_MODEL = "intfloat/multilingual-e5-large"
TOP_K = 3  # 문장당 검색할 KB 패시지 수

_GEN = "/home/sclab/moe_model/results/full_ablation/generations"

RESULT_FILES = {
    "Naive RAG 7B":     "/home/sclab/moe_model/baseline_results/naive_rag_7b_500.json",
    "Adaptive RAG 7B":  "/home/sclab/moe_model/baseline_results/adaptive_rag_7b_500.json",
    "HyDE 7B":          "/home/sclab/moe_model/baseline_results/hyde_7b_500.json",
    "CRAG 7B":          "/home/sclab/moe_model/baseline_results/crag_7b_500.json",
    "NexRoute 7B":      "/home/sclab/moe_model/eval_results_7b_v3_500.json",
    "Naive RAG 72B":    "/home/sclab/moe_model/baseline_results/naive_rag_72b_500_v2.json",
    "Adaptive RAG 72B": "/home/sclab/moe_model/baseline_results/adaptive_rag_72b_500_v2.json",
    "HyDE 72B":         "/home/sclab/moe_model/baseline_results/hyde_72b_500_v2.json",
    "CRAG 72B":         "/home/sclab/moe_model/baseline_results/crag_72b_500_v2.json",
    "NexRoute 72B":     "/home/sclab/moe_model/baseline_results/nexroute_72b_500_v2.json",
    # ── Ablation variants ──────────────────────────────────────────────
    "V01_full_nexroute":            f"{_GEN}/variant_01_full_nexroute.json",
    "V02_wo_function_axis":         f"{_GEN}/variant_02_wo_function_axis.json",
    "V03_wo_partitioned_retrieval": f"{_GEN}/variant_03_wo_partitioned_retrieval.json",
    "V04_wo_lora":                  f"{_GEN}/variant_04_wo_lora.json",
    "V05_wo_domain_axis":           f"{_GEN}/variant_05_wo_domain_axis.json",
    "V07_cross_weight_avg":         f"{_GEN}/variant_07_cross_weight_avg.json",
    "V08_cross_task_arithmetic":    f"{_GEN}/variant_08_cross_task_arithmetic.json",
    "V09_cross_dominant_single":    f"{_GEN}/variant_09_cross_dominant_single.json",
}

GT_PATH = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"

# ── GT 로드 ───────────────────────────────────────────────────────────
with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

# ── 모델 로드 ─────────────────────────────────────────────────────────
device_nli = 0 if torch.cuda.is_available() else -1
print("[*] NLI 모델 로드...")
nli_pipe = pipeline("text-classification", model=NLI_MODEL, device=device_nli,
                    truncation=True, max_length=512)
print("[*] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
print("[*] 완료\n")

# ── KB 인덱스 구축 ────────────────────────────────────────────────────
DOMAIN_FILES = {
    "LAW":  "law_rewritten.json",
    "MED":  "med_rewritten.json",
    "SOC":  "soc_rewritten.json",
    "STEM": "stem_rewritten.json",
    "CROSS":"cross_rewritten.json",
}

print("[*] KB 인덱스 구축 중...")
kb_indices = {}  # domain -> (texts, faiss_index)
all_texts = []   # unified KB용

for domain, fname in DOMAIN_FILES.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    # output 필드 사용 (rewritten 텍스트)
    texts = [d.get("output", d.get("gold_answer", ""))[:400] for d in data if d.get("output") or d.get("gold_answer")]
    texts = [t for t in texts if len(t) > 20]

    embs = embed_model.encode(texts, batch_size=256, show_progress_bar=False,
                              normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    kb_indices[domain] = (texts, index)
    all_texts.extend(texts)
    print(f"  {domain}: {len(texts)}개")

# unified KB (베이스라인용)
print("[*] Unified KB 인덱스 구축 중...")
unified_embs = embed_model.encode(all_texts, batch_size=256, show_progress_bar=False,
                                  normalize_embeddings=True)
unified_index = faiss.IndexFlatIP(unified_embs.shape[1])
unified_index.add(unified_embs.astype(np.float32))
kb_indices["UNIFIED"] = (all_texts, unified_index)
print(f"  UNIFIED: {len(all_texts)}개\n")


def retrieve_passages(sentence: str, domains: list, top_k: int = TOP_K) -> list:
    """문장과 관련된 KB 패시지 검색"""
    emb = embed_model.encode([sentence], normalize_embeddings=True)
    results = []
    per = max(1, top_k // len(domains))
    for dom in domains:
        if dom not in kb_indices:
            continue
        texts, index = kb_indices[dom]
        D, I = index.search(emb.astype(np.float32), per)
        for score, idx in zip(D[0], I[0]):
            results.append((float(score), texts[idx]))
    # 점수 기준 정렬 후 top_k 반환
    results.sort(key=lambda x: -x[0])
    return [t for _, t in results[:top_k]]


def check_sentence(sentence: str, passages: list) -> str:
    """
    문장을 passages로 NLI 검증.
    - 하나라도 ENTAILMENT → 'supported'
    - 하나라도 CONTRADICTION (entailment 없이) → 'contradiction'
    - 모두 NEUTRAL → 'neutral'
    """
    has_entail = False
    has_contra = False
    for passage in passages:
        try:
            out = nli_pipe({"text": passage[:600], "text_pair": sentence[:300]},
                           truncation=True)
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


def eval_model(model_name: str, pred_path: str, use_partitioned_kb: bool) -> dict:
    with open(pred_path) as f:
        data = json.load(f)

    total_sents = 0
    contra_sents = 0
    cross_total = cross_contra = 0
    single_total = single_contra = 0

    for i, item in enumerate(data):
        q    = item.get("query", item.get("instruction", "")).strip()
        pred = item.get("pred_answer", "").strip()
        if not pred:
            continue

        # 도메인 결정
        if use_partitioned_kb:
            # NexRoute / V01: domain_pred 또는 pred_domain 필드 사용
            domains = item.get("domain_pred", item.get("pred_domain", ["UNIFIED"]))
            if isinstance(domains, str):
                domains = [domains]
            if not domains:
                domains = ["UNIFIED"]
            is_cross = item.get("is_cross", False)
        else:
            domains = ["UNIFIED"]
            # is_cross 판단 (GT 룩업)
            gt_item = gt_map.get(q) or {}
            is_cross = (gt_item.get("expected_domain", "") == "CROSS") or item.get("is_cross", False)

        # 문장 분리
        sents = [s.strip() for s in sent_tokenize(pred) if len(s.strip()) > 20]
        if not sents:
            continue

        item_contra = 0
        for sent in sents:
            passages = retrieve_passages(sent, domains, top_k=TOP_K)
            if not passages:
                continue
            result = check_sentence(sent, passages)
            if result == "contradiction":
                item_contra += 1

        total_sents  += len(sents)
        contra_sents += item_contra
        if is_cross:
            cross_total  += len(sents)
            cross_contra += item_contra
        else:
            single_total  += len(sents)
            single_contra += item_contra

        if (i + 1) % 50 == 0:
            cur = contra_sents / total_sents * 100 if total_sents else 0
            print(f"    [{i+1}/{len(data)}]  running halluc={cur:.1f}%  "
                  f"(total_sents={total_sents})")

    def rate(c, t): return round(c / t * 100, 1) if t > 0 else None

    result = {
        "all":    {"rate": rate(contra_sents, total_sents),   "sents": total_sents},
        "cross":  {"rate": rate(cross_contra, cross_total),   "sents": cross_total},
        "single": {"rate": rate(single_contra, single_total), "sents": single_total},
    }
    fmt = lambda v: f"{v:.1f}" if v is not None else "N/A"
    print(f"  {model_name:<22}  ALL={fmt(result['all']['rate']):>5}%  "
          f"CROSS={fmt(result['cross']['rate']):>5}%  "
          f"Single={fmt(result['single']['rate']):>5}%")
    return result


# ── 평가 실행 ─────────────────────────────────────────────────────────
NEXROUTE_MODELS = {"NexRoute 7B", "NexRoute 72B", "V01_full_nexroute"}

print(f"{'Model':<22}  {'ALL':>7}  {'CROSS':>7}  {'Single':>8}")
print("─" * 55)

all_results = {}
for name, path in RESULT_FILES.items():
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: 파일 없음")
        continue
    use_partitioned = name in NEXROUTE_MODELS
    print(f"\n[▶] {name}  (KB={'partitioned' if use_partitioned else 'unified'})")
    all_results[name] = eval_model(name, path, use_partitioned)

with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n\n✅ 저장: {OUT_PATH}")
print("\n" + "=" * 62)
print(f"{'Model':<22}  {'ALL(%)':>8}  {'CROSS(%)':>10}  {'Single(%)':>11}")
print("─" * 62)
for name, v in all_results.items():
    a = v["all"]["rate"];  c = v["cross"]["rate"];  s = v["single"]["rate"]
    fmt = lambda x: f"{x:.1f}" if x is not None else "N/A"
    marker = " ◀" if "NexRoute" in name else ""
    print(f"  {name:<22}  {fmt(a):>8}  {fmt(c):>10}  {fmt(s):>11}{marker}")
