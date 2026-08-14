"""
Exp2 hallucination: sentence-level NLI for 4 conditions × 300 samples.
Same method as eval_factscore_hallucination.py:
  - top-3 passages from appropriate KB per sentence
  - cross-encoder/nli-deberta-v3-large
  - contradiction rate = hallucination rate
KB assignment:
  predicted        → pred_domain (CROSS KB if is_cross, else single pred_domain)
  oracle_domain    → gold_domain (CROSS KB if gold=CROSS, else single gold_domain)
  always_cross     → CROSS KB
  always_single_best → top1 domain KB
"""
import json, numpy as np, os, time
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk; nltk.download('punkt_tab', quiet=True)

KB_CACHE = "/home/sclab/moe_model/results/ablation/kb_cache"
EXP2_PATH = "/home/sclab/moe_model/results/exp2_domain_router.json"
OUT_PATH  = "/home/sclab/moe_model/results/exp2_hallucination.json"
TOP_K = 3

# ── 모델 로드 ──────────────────────────────────────────────────────────
print("[1] 모델 로드...")
nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-large",
               device=-1, truncation=True, max_length=512, batch_size=16)
embed = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
print("    완료")

# ── KB 인덱스 ──────────────────────────────────────────────────────────
print("[2] KB 인덱스 로드...")
domains = ['LAW','MED','SOC','STEM','CROSS']
kb_texts = {}; kb_index = {}
for dom in domains:
    kb_texts[dom] = json.load(open(f"{KB_CACHE}/{dom}_texts.json"))
    embs = np.load(f"{KB_CACHE}/{dom}_embs.npy").astype(np.float32)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    kb_index[dom] = idx
    print(f"    {dom}: {len(kb_texts[dom])} passages")

def retrieve(sentence, dom_list):
    emb = embed.encode([sentence], normalize_embeddings=True).astype(np.float32)
    results = []
    per = max(1, TOP_K // len(dom_list))
    for dom in dom_list:
        if dom not in kb_index: continue
        D, I = kb_index[dom].search(emb, per)
        for score, i in zip(D[0], I[0]):
            results.append((float(score), kb_texts[dom][i]))
    results.sort(key=lambda x: -x[0])
    return [t for _, t in results[:TOP_K]]

def nli_label(passage, sentence):
    try:
        out = nli({"text": passage[:600], "text_pair": sentence[:300]})
        lbl = out["label"] if isinstance(out, dict) else out[0]["label"]
        return lbl.upper()
    except Exception:
        return "NEUTRAL"

def check_sentence(sentence, passages):
    has_entail = has_contra = False
    for passage in passages:
        lbl = nli_label(passage, sentence)
        if lbl == "ENTAILMENT": has_entail = True
        elif lbl == "CONTRADICTION": has_contra = True
    if has_entail: return "supported"
    if has_contra: return "contradiction"
    return "neutral"

# ── 데이터 ─────────────────────────────────────────────────────────────
data = json.load(open(EXP2_PATH))['results']
print(f"\n[3] 평가 시작: {len(data)} samples × 4 conditions")

# condition → KB 결정 함수
def get_kb(r, cond):
    if cond == 'predicted':
        dom = r['pred_domain']
        return ['CROSS'] if dom == 'CROSS' else [dom]
    elif cond == 'oracle_domain':
        dom = r['gold_domain']
        return ['CROSS'] if dom == 'CROSS' else [dom]
    elif cond == 'always_cross':
        return ['CROSS']
    elif cond == 'always_single_best':
        return [r['top1']]

conds = ['predicted', 'oracle_domain', 'always_cross', 'always_single_best']

# per-sample results
per_sample = []
cross_idx  = [i for i,r in enumerate(data) if r['gold_domain']=='CROSS']
single_idx = [i for i,r in enumerate(data) if r['gold_domain']!='CROSS']

t0 = time.perf_counter()
for n, r in enumerate(data):
    row = {'sample_id': r['sample_id'], 'gold_domain': r['gold_domain'], 'is_cross': r['gold_domain']=='CROSS'}
    for cond in conds:
        text = r[cond]
        kb   = get_kb(r, cond)
        sents = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]
        n_sents = len(sents)
        n_contra = 0
        for sent in sents:
            passages = retrieve(sent, kb)
            if not passages: continue
            if check_sentence(sent, passages) == "contradiction":
                n_contra += 1
        row[cond] = {'n_sents': n_sents, 'n_contra': n_contra,
                     'hal_rate': round(n_contra/n_sents*100, 2) if n_sents else None}
    per_sample.append(row)

    if (n+1) % 25 == 0 or (n+1) == len(data):
        elapsed = time.perf_counter() - t0
        rate = elapsed / (n+1)
        remaining = rate * (len(data) - n - 1)
        # print running hal rates
        rates = {c: np.mean([x[c]['hal_rate'] for x in per_sample if x[c]['hal_rate'] is not None]) for c in conds}
        print(f"  [{n+1:3}/{len(data)}] elapsed={elapsed/60:.1f}min remaining={remaining/60:.1f}min")
        print(f"    running hal%: " + " | ".join(f"{c}={v:.1f}" for c,v in rates.items()))
        # save checkpoint
        json.dump({'partial': (n+1)<len(data), 'n': len(per_sample), 'per_sample': per_sample},
                  open(OUT_PATH,'w'), indent=2)

# ── 집계 ───────────────────────────────────────────────────────────────
def summarize(indices):
    out = {}
    for cond in conds:
        ns = sum(per_sample[i][cond]['n_sents'] for i in indices if per_sample[i][cond]['n_sents'])
        nc = sum(per_sample[i][cond]['n_contra'] for i in indices if per_sample[i][cond]['n_sents'])
        out[cond] = {'n_sents': ns, 'n_contra': nc, 'hal_rate': round(nc/ns*100,2) if ns else None}
    return out

summary = {
    'overall': summarize(list(range(len(data)))),
    'cross':   summarize(cross_idx),
    'single':  summarize(single_idx),
}

result = {'partial': False, 'n': len(data), 'summary': summary, 'per_sample': per_sample}
json.dump(result, open(OUT_PATH,'w'), indent=2)
print(f"\n[done] Saved: {OUT_PATH}")

print("\n=== HALLUCINATION SUMMARY ===")
for split, d in summary.items():
    print(f"\n  {split.upper()}:")
    for cond, v in d.items():
        print(f"    {cond:22}: {v['hal_rate']:5.2f}% ({v['n_contra']}/{v['n_sents']} sents)")
