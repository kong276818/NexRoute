"""
run_phase5_cached.py
====================
V03/V04/V05 hallucination 계산 (KB 임베딩 캐시 포함).
dualmoe 환경에서 실행: CUDA 12.8, PyTorch 2.11.0+cu128

캐시: results/ablation/kb_cache/
  - {dom}_texts.json     : KB 텍스트
  - {dom}_embs.npy       : 임베딩 (float32)
  - unified_texts.json   / unified_embs.npy
  캐시 키: EMBED_NAME + 텍스트 count (모델/데이터 변경 시 자동 무효화)
"""

import os, json, csv, gc, sys
import numpy as np
import torch
from pathlib import Path

BASE_DIR   = "/home/sclab/moe_model"
DATA_DIR   = os.path.join(BASE_DIR, "processed_v2")
GT_PATH    = os.path.join(DATA_DIR, "gt_dataset_500.json")
PREDS_CSV  = os.path.join(BASE_DIR, "preds_500.csv")
OUT_DIR    = os.path.join(BASE_DIR, "results/ablation")
CACHE_DIR  = os.path.join(OUT_DIR, "kb_cache")
EMBED_NAME = "intfloat/multilingual-e5-large"
NLI_MODEL  = "cross-encoder/nli-deberta-v3-large"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_NLI  = 3
MIN_SENT   = 20

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

os.makedirs(CACHE_DIR, exist_ok=True)
print(f"[init] DEVICE={DEVICE}  EMBED={EMBED_NAME}")
print(f"[init] CACHE_DIR={CACHE_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
def load_routing():
    import pandas as pd
    df = pd.read_csv(PREDS_CSV)
    routing = {}
    for _, row in df.iterrows():
        idx = int(row['sample_id'])
        routing[idx] = {
            'pred_domain': str(row.get('pred_domain','')),
            'is_cross':    bool(row.get('is_cross_pred', row.get('is_cross', False))),
            'top1':        str(row.get('top1_domain', row.get('pred_domain','SOC'))),
            'top2':        str(row.get('top2_domain', 'STEM')),
        }
    return routing

def load_gt():
    return json.load(open(GT_PATH))


# ─────────────────────────────────────────────────────────────────────────────
def build_or_load_kb():
    """KB 임베딩 캐시 로드 또는 빌드. 캐시 미스 시 encode 후 저장."""
    import faiss
    from sentence_transformers import SentenceTransformer

    # 캐시 유효성 확인용 메타
    meta_path = os.path.join(CACHE_DIR, "meta.json")
    texts_total = {}
    for dom, fname in DOMAIN_FILES.items():
        raw = json.load(open(os.path.join(DATA_DIR, fname)))
        texts_total[dom] = [d.get("output", d.get("gold_answer","")) for d in raw]
        texts_total[dom] = [t for t in texts_total[dom] if len(t) > 20]

    counts = {dom: len(t) for dom, t in texts_total.items()}
    current_meta = {"embed_model": EMBED_NAME, "counts": counts}

    cache_valid = False
    if os.path.exists(meta_path):
        saved_meta = json.load(open(meta_path))
        cache_valid = (saved_meta == current_meta)

    if cache_valid:
        print("  [KB cache] HIT — 캐시에서 로드...")
        enc = None  # 인코딩 불필요
    else:
        print("  [KB cache] MISS — 임베딩 계산 (GPU)...")
        enc = SentenceTransformer(EMBED_NAME, device=DEVICE)

    kb_domain = {}
    all_texts = []

    for dom in DOMAIN_FILES:
        texts = texts_total[dom]
        emb_path  = os.path.join(CACHE_DIR, f"{dom}_embs.npy")
        text_path = os.path.join(CACHE_DIR, f"{dom}_texts.json")

        if cache_valid and os.path.exists(emb_path):
            embs = np.load(emb_path)
            print(f"    {dom}: loaded {len(texts)} texts from cache")
        else:
            print(f"    {dom}: encoding {len(texts)} texts...")
            embs = enc.encode(texts, batch_size=256, normalize_embeddings=True,
                              show_progress_bar=True).astype(np.float32)
            np.save(emb_path, embs)
            json.dump(texts, open(text_path, 'w'), ensure_ascii=False)

        idx_f = faiss.IndexFlatIP(embs.shape[1])
        idx_f.add(embs)
        kb_domain[dom] = {'texts': texts, 'idx': idx_f}
        all_texts.extend(texts)

    # unified
    uni_emb_path  = os.path.join(CACHE_DIR, "unified_embs.npy")
    uni_text_path = os.path.join(CACHE_DIR, "unified_texts.json")

    if cache_valid and os.path.exists(uni_emb_path):
        all_embs = np.load(uni_emb_path)
        print(f"    UNIFIED: loaded {len(all_texts)} texts from cache")
    else:
        print(f"    UNIFIED: encoding {len(all_texts)} texts...")
        all_embs = enc.encode(all_texts, batch_size=256, normalize_embeddings=True,
                              show_progress_bar=True).astype(np.float32)
        np.save(uni_emb_path, all_embs)
        json.dump(all_texts, open(uni_text_path, 'w'), ensure_ascii=False)

    uni_idx = faiss.IndexFlatIP(all_embs.shape[1])
    uni_idx.add(all_embs)
    kb_unified = {'texts': all_texts, 'idx': uni_idx}

    # 메타 저장 (성공 시)
    json.dump(current_meta, open(meta_path, 'w'))

    if enc is not None:
        del enc; gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    return kb_domain, kb_unified


# ─────────────────────────────────────────────────────────────────────────────
def hal_pool(rows):
    ns = sum(int(r.get('n_sents', 0) or 0) for r in rows)
    nc = sum(int(r.get('n_contra', 0) or 0) for r in rows)
    rate = round(100 * nc / ns, 2) if ns else None
    return rate, ns, nc


def compute_hallucination(gen_path, variant_name, routing, kb_domain, kb_unified,
                          use_subdomain_for_cross):
    from transformers import pipeline as hf_pipeline
    from sentence_transformers import SentenceTransformer
    import nltk
    nltk.download('punkt_tab', quiet=True)

    out_csv = os.path.join(OUT_DIR, f"{variant_name}_hallucination_per_sample.csv")
    out_sum = os.path.join(OUT_DIR, f"{variant_name}_hallucination_summary.json")

    if os.path.exists(out_csv):
        rows = list(csv.DictReader(open(out_csv)))
        if len(rows) == 500:
            print(f"  [{variant_name}] 이미 완료 — skip")
            return
        done_ids = {int(r['idx']) for r in rows}
        print(f"  [{variant_name}] 재개: {len(done_ids)}개 완료")
    else:
        rows = []
        done_ids = set()

    print(f"  [{variant_name}] hallucination 계산 중...")
    data    = json.load(open(gen_path))
    gt_list = load_gt()

    nli = hf_pipeline("text-classification", model=NLI_MODEL,
                      device=0 if DEVICE == "cuda" else -1,
                      truncation=True, max_length=512)
    enc = SentenceTransformer(EMBED_NAME, device=DEVICE)

    VALID_DOMAINS = set(kb_domain.keys())  # LAW, MED, SOC, STEM, CROSS

    def get_domains(idx, pred_domain, is_cross):
        if not is_cross:
            # pred_domain이 유효한 KB 도메인이 아니면 (예: 'UNIFIED') → unified KB 사용
            if pred_domain and pred_domain in VALID_DOMAINS:
                return [pred_domain]
            return None  # → unified KB
        if use_subdomain_for_cross:
            r = routing.get(idx, {})
            return [r.get('top1', 'SOC'), r.get('top2', 'STEM')]
        return None  # unified

    def retrieve(sent, domains):
        emb = enc.encode([sent], normalize_embeddings=True).astype(np.float32)
        if domains is None:
            _, I = kb_unified['idx'].search(emb, TOP_K_NLI)
            return [kb_unified['texts'][i] for i in I[0] if i >= 0]
        per = max(1, TOP_K_NLI // len(domains))
        res = []
        for d in domains:
            if d in kb_domain:
                _, I = kb_domain[d]['idx'].search(emb, per)
                res += [kb_domain[d]['texts'][i] for i in I[0] if i >= 0]
        return res[:TOP_K_NLI]

    def check(sent, passages):
        if not passages: return "neutral"
        has_ent = has_con = False
        for p in passages:
            try:
                r = nli({"text": p[:800], "text_pair": sent[:400]})
                lbl = (r["label"] if isinstance(r, dict) else r[0]["label"]).upper()
                if lbl == "ENTAILMENT":    has_ent = True
                elif lbl == "CONTRADICTION": has_con = True
            except: pass
        if has_ent: return "supported"
        if has_con: return "contradiction"
        return "neutral"

    csv_rows = list(rows)  # already-done rows
    for item in data:
        idx = int(item.get('sample_id', item.get('id', item.get('idx', -1))))
        if idx in done_ids:
            continue

        pred        = item.get('pred_answer', '').strip()
        domain_gt   = item.get('domain', item.get('expected_domain',
                      gt_list[idx].get('expected_domain', '?')))
        is_cross_gt = (domain_gt == 'CROSS')
        is_clean    = idx < 300
        pred_dom    = item.get('pred_domain', routing.get(idx, {}).get('pred_domain', 'CROSS'))
        is_cross_pred = bool(item.get('is_cross', routing.get(idx, {}).get('is_cross', False)))

        domains = get_domains(idx, pred_dom, is_cross_pred)

        from nltk.tokenize import sent_tokenize
        sents = [s.strip() for s in sent_tokenize(pred) if len(s.strip()) >= MIN_SENT] if pred else []
        n_sents = len(sents); n_contra = 0
        for s in sents:
            ps = retrieve(s, domains)
            if check(s, ps) == "contradiction":
                n_contra += 1

        hal_rate = round(100 * n_contra / n_sents, 2) if n_sents else None
        csv_rows.append({
            'idx': idx, 'expected_domain': domain_gt,
            'is_cross': is_cross_gt, 'is_clean': is_clean,
            'pred_domains': "UNIFIED" if domains is None else "|".join(domains or [pred_dom]),
            'n_sents': n_sents, 'n_contra': n_contra,
            'hal_rate': hal_rate if hal_rate is not None else '',
        })
        done_ids.add(idx)

        if len(csv_rows) % 50 == 0:
            print(f"    [{len(csv_rows)}/500]")
            with open(out_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                w.writeheader(); w.writerows(csv_rows)

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)

    # 집계
    def sub(cond_fn):
        rs = [r for r in csv_rows if cond_fn(r)]
        rate, n, c = hal_pool(rs)
        return {"hal": rate, "n_sents": n, "n_contra": c}

    summary = {
        "overall":      sub(lambda r: True),
        "cross":        sub(lambda r: str(r['is_cross']) in ('True','true')),
        "single":       sub(lambda r: str(r['is_cross']) not in ('True','true')),
        "clean":        sub(lambda r: str(r['is_clean']) in ('True','true')),
        "clean_cross":  sub(lambda r: str(r['is_clean']) in ('True','true') and str(r['is_cross']) in ('True','true')),
        "clean_single": sub(lambda r: str(r['is_clean']) in ('True','true') and str(r['is_cross']) not in ('True','true')),
        "excl_law":     sub(lambda r: r['expected_domain'] != 'LAW'),
        "by_domain":    {d: sub(lambda r, d=d: r['expected_domain'] == d)
                         for d in ['LAW','MED','SOC','STEM','CROSS']},
    }
    json.dump(summary, open(out_sum, 'w'), indent=2)

    print(f"  [{variant_name}] 완료: overall={summary['overall']['hal']}%  "
          f"CROSS={summary['cross']['hal']}%  single={summary['single']['hal']}%  "
          f"excl_LAW={summary['excl_law']['hal']}%")

    del nli, enc; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Phase5 (cached) — DEVICE={DEVICE}")
    print(f"{'='*60}\n")

    print("[Step 1] KB 인덱스 구축 (캐시 우선)...")
    kb_domain, kb_unified = build_or_load_kb()

    routing = load_routing()
    print(f"\n[Step 2] Hallucination 계산...")

    for variant, use_sub in [("v03", True), ("v04", True), ("v05", False)]:
        gen_path = os.path.join(OUT_DIR, f"{variant}_gen_500.json")
        if not os.path.exists(gen_path):
            print(f"  [{variant}] gen 파일 없음 — skip")
            continue
        compute_hallucination(gen_path, variant, routing, kb_domain, kb_unified,
                              use_subdomain_for_cross=use_sub)

    del kb_domain, kb_unified; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
    print("\n[완료] Phase5 hallucination 계산 완료")
