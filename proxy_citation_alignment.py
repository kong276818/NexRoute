"""
proxy_citation_alignment.py — Phase 3
"Retrieved evidence grounding ratio" — a proxy for factual citation alignment.

For each generated answer:
1. Split into sentences (atomic claim units)
2. Encode each sentence with multilingual-E5-large
3. Compute cosine similarity against each retrieved passage (top-K=10)
4. A sentence is "grounded" if max cosine sim ≥ THRESHOLD (default 0.75)
5. Report grounded_sentences / total_sentences per variant × domain

NOTE: This is NOT FActScore. It checks whether generated claims are
lexically/semantically consistent with the retrieved evidence passages,
not whether they are factually correct. It is a proxy for retrieval
utilization, reported separately from FActScore.
Interpretation: low score → model is generating from parametric memory
rather than the retrieved context.

Reads: results/full_ablation/generations/*.json
       (must include 'retrieved_context' field — Variant 1 may lack this)
"""

import json
import os
from collections import defaultdict

import nltk
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

EMBED_MODEL = "intfloat/multilingual-e5-large"
GEN_DIR     = "/home/sclab/moe_model/results/full_ablation/generations"
OUT_PATH    = "/home/sclab/moe_model/results/full_ablation/citation_alignment.json"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
THRESHOLD   = 0.75
ALL_DOMAINS = ["LAW", "MED", "SOC", "STEM", "CROSS"]

print("Loading multilingual-E5-large ...")
device     = "cuda" if torch.cuda.is_available() else "cpu"
embed      = SentenceTransformer(EMBED_MODEL, device=device)
print("Ready")

gt_data      = json.load(open(GT_PATH))
instr2domain = {g["instruction"].strip(): g["expected_domain"] for g in gt_data}

gen_files = sorted(f for f in os.listdir(GEN_DIR) if f.endswith(".json"))
results   = {}

for fname in gen_files:
    vname = fname.replace(".json", "")
    data  = json.load(open(os.path.join(GEN_DIR, fname)))

    # Check if retrieved_context is present
    has_context = any(d.get("retrieved_context") for d in data)
    if not has_context:
        print(f"\n[{vname}]  No 'retrieved_context' field — skipping "
              f"(Variant 1 reuse; context not stored in original run)")
        results[vname] = {
            "note": "retrieved_context not available in this generation file; "
                    "citation alignment cannot be computed",
            "OVERALL": None,
        }
        continue

    print(f"\n[{vname}]  N={len(data)}")

    domain_scores = defaultdict(list)

    for item in tqdm(data, desc=f"  {vname}", ncols=80):
        pred     = item.get("pred_answer", "").strip()
        ctx_list = item.get("retrieved_context", [])
        if not pred or "[GENERATION_FAILED]" in pred or not ctx_list:
            continue

        dom = item.get("expected_domain") or instr2domain.get(
            item.get("query", "").strip(), "UNKNOWN")

        # Encode predicted answer sentences
        pred_sents = nltk.sent_tokenize(pred)
        if not pred_sents:
            continue

        # Encode retrieved passages (truncated)
        passages = [c[:512] for c in ctx_list if c.strip()]
        if not passages:
            continue

        pred_embs = embed.encode(
            pred_sents, batch_size=32, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        pass_embs = embed.encode(
            passages, batch_size=32, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # cosine sim matrix [n_sents × n_passages]
        sim = pred_embs @ pass_embs.T  # shape: [n_sents, n_passages]
        max_sim = sim.max(axis=1)      # best passage match per sentence

        grounded = float((max_sim >= THRESHOLD).sum()) / len(pred_sents)
        domain_scores[dom].append(grounded)

    # Aggregate
    variant_result = {}
    all_scores = []
    for dom in ALL_DOMAINS:
        if dom not in domain_scores:
            continue
        sc = domain_scores[dom]
        variant_result[dom] = {
            "n":                   len(sc),
            "grounded_ratio":      round(float(np.mean(sc)), 4),
            "threshold":           THRESHOLD,
        }
        all_scores.extend(sc)
        print(f"  {dom}: n={len(sc)}, grounded={np.mean(sc):.4f}")

    variant_result["OVERALL"] = {
        "n":               len(all_scores),
        "grounded_ratio":  round(float(np.mean(all_scores)), 4) if all_scores else None,
        "threshold":       THRESHOLD,
        "note": (
            "Grounding checked against top-10 retrieved passages "
            "(not a general KB). Low score indicates parametric generation "
            "rather than retrieval utilization."
        ),
    }
    print(f"  OVERALL grounded_ratio={variant_result['OVERALL']['grounded_ratio']}")
    results[vname] = variant_result

json.dump(results, open(OUT_PATH, "w"), indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT_PATH}")

# ── Print Table B ─────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  TABLE B  —  Citation / Evidence Grounding Ratio (threshold=0.75)")
print("  NOTE: Proxy only — grounding against retrieved passages, not factual KB")
print("=" * 65)
print(f"{'Variant':<42}  {'OVERALL':>8}", " ".join(f"{d:>7}" for d in ALL_DOMAINS))
print("-" * 90)
for vname, vm in results.items():
    if vm.get("note") and "OVERALL" in vm and vm["OVERALL"] is None:
        print(f"  {vname:<40}  {'N/A':>8}")
        continue
    ov = vm.get("OVERALL", {}).get("grounded_ratio")
    ov_str = f"{ov:.4f}" if ov is not None else "N/A"
    dom_strs = []
    for d in ALL_DOMAINS:
        v = vm.get(d, {}).get("grounded_ratio")
        dom_strs.append(f"{v:.4f}" if v is not None else " N/A ")
    print(f"  {vname:<40}  {ov_str:>8}", " ".join(f"{s:>7}" for s in dom_strs))
