#!/usr/bin/env python3
"""
eval_arrow_phatgoose.py  —  Arrow Routing & PHATGOOSE Baseline Evaluation
==========================================================================
Uses NexRoute's trained LoRA expert pool (LAW/MED/SOC/STEM/CROSS) with
two alternative routing mechanisms:

  (1) Arrow Routing  (Ostapenko et al., 2024) — training-free, SVD-based
  (2) PHATGOOSE     (Muqeeth et al., 2024)   — hidden-state centroid gating

Same expert pool, domain-partitioned FAISS retrieval, decoding params
matching NexRoute's CROSS-domain setup. No LLM-based scoring.

Usage:
  python eval_arrow_phatgoose.py [--pilot N] [--method arrow|phatgoose|both]

Output (results/arrow_phatgoose/):
  arrow_routing_cross_gen.jsonl
  phatgoose_cross_gen.jsonl
  table_lora_router_comparison.csv / .md
  run_log.txt
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftModel
from rouge_score import rouge_scorer
from safetensors import safe_open
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "intfloat/multilingual-e5-large"
DATA_DIR    = BASE_DIR / "processed_v2"
GT_PATH     = DATA_DIR / "gt_dataset_500.json"
LORA_DIR    = BASE_DIR / "lora_weights_7b_v3"
OUT_DIR     = BASE_DIR / "results" / "arrow_phatgoose"

LORA_MAP = {
    "LAW":   LORA_DIR / "lora_law",
    "MED":   LORA_DIR / "lora_med",
    "SOC":   LORA_DIR / "lora_soc",
    "STEM":  LORA_DIR / "lora_stem",
    "CROSS": LORA_DIR / "lora_cross",
}
DOMAIN_FILES = {
    "LAW":   DATA_DIR / "law_rewritten.json",
    "MED":   DATA_DIR / "med_rewritten.json",
    "SOC":   DATA_DIR / "soc_rewritten.json",
    "STEM":  DATA_DIR / "stem_rewritten.json",
    "CROSS": DATA_DIR / "cross_rewritten.json",
}
ALL_DOMAINS  = ["LAW", "MED", "SOC", "STEM", "CROSS"]

# Decoding: matches NexRoute CROSS-domain inference
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.7
TOP_K_ROUTING  = 2            # Arrow top-k
PHATGOOSE_TRAIN_N = 300       # samples per domain for PHATGOOSE gating training
RETRIEVAL_TOP_K   = 10
SEED = 42

# Arrow: modules with consistent d_in=hidden_size (skip down_proj d_in=18944)
ARROW_MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj"]

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert. Answer accurately based on legal principles, "
            "statutes, and case law.",
    "MED":  "You are a medical expert. Answer accurately based on medical facts "
            "and clinical knowledge.",
    "SOC":  "You are a social science expert. Answer based on social context, "
            "psychology, and research.",
    "STEM": "You are a science and engineering expert. Answer accurately based on "
            "scientific principles.",
    "CROSS":"You are an interdisciplinary expert. Synthesize knowledge from "
            "multiple domains to give a comprehensive answer.",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "run_log.txt", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_cross_eval(pilot: Optional[int] = None) -> List[Dict]:
    """Load CROSS-domain eval samples; patch missing crossed_domains from training data."""
    gt_all = json.load(open(GT_PATH))
    cross_samples = [d for d in gt_all if d.get("expected_domain") == "CROSS"]

    # Build lookup for patching
    cr_data = json.load(open(DATA_DIR / "cross_rewritten.json"))
    lookup = {d["instruction"].strip(): d["crossed_domains"] for d in cr_data}

    patched = 0
    for s in cross_samples:
        if not s.get("crossed_domains") or s["crossed_domains"] in ("?", "", "[]", None):
            key = s.get("instruction", "").strip()
            if key in lookup:
                s["crossed_domains"] = lookup[key]
                patched += 1

    log.info("CROSS eval: %d samples (%d crossed_domains patched from cross_rewritten.json)",
             len(cross_samples), patched)

    if pilot:
        cross_samples = cross_samples[:pilot]
        log.info("PILOT mode: using first %d samples", pilot)

    return cross_samples


def load_domain_train(domain: str, n: int, seed: int = SEED) -> List[str]:
    """Return up to n instruction strings from domain training data."""
    data = json.load(open(DOMAIN_FILES[domain]))
    random.seed(seed)
    random.shuffle(data)
    return [d["instruction"] for d in data[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL (domain-partitioned FAISS)
# ─────────────────────────────────────────────────────────────────────────────

def build_domain_indices(embed_model: SentenceTransformer
                         ) -> Tuple[Dict[str, List[str]], Dict[str, faiss.IndexFlatIP]]:
    """Build per-domain FAISS indices from rewritten corpora."""
    log.info("Building domain-partitioned FAISS indices …")
    texts_by_domain: Dict[str, List[str]] = {}
    indices: Dict[str, faiss.IndexFlatIP] = {}

    for domain, fpath in DOMAIN_FILES.items():
        data = json.load(open(fpath))
        texts = [f"{d['instruction']} {d.get('output', d.get('gold_answer', ''))}"
                 for d in data]
        embs = embed_model.encode(texts, show_progress_bar=False, batch_size=128,
                                  normalize_embeddings=True)
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs.astype(np.float32))
        texts_by_domain[domain] = texts
        indices[domain] = idx
        log.info("  %s: %d docs indexed", domain, len(texts))

    return texts_by_domain, indices


def retrieve(query: str, domains: List[str],
             embed_model, texts_by_domain, indices,
             top_k: int = RETRIEVAL_TOP_K) -> str:
    """Retrieve top_k docs from each specified domain and concatenate."""
    qe = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    ctx_parts = []
    for d in domains:
        _, I = indices[d].search(qe, top_k)
        ctx_parts.extend(texts_by_domain[d][i] for i in I[0])
    return "\n".join(ctx_parts)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model_with_adapters(device: str = "auto"):
    """Load base model + all 5 LoRA adapters as named adapters in PeftModel."""
    log.info("Loading base model (4-bit NF4) + 5 LoRA adapters …")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load first adapter via PeftModel, rest via load_adapter
    first = ALL_DOMAINS[0]
    model = PeftModel.from_pretrained(base, str(LORA_MAP[first]), adapter_name=first)
    for domain in ALL_DOMAINS[1:]:
        model.load_adapter(str(LORA_MAP[domain]), adapter_name=domain)

    model.eval()
    log.info("Model loaded with adapters: %s", ALL_DOMAINS)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# ARROW ROUTING  (Ostapenko et al., 2024)
# ─────────────────────────────────────────────────────────────────────────────

def compute_arrow_signature(adapter_path: Path) -> torch.Tensor:
    """
    Compute a single d_in=3584 Arrow signature for one expert adapter.

    For each layer × ARROW_MODULE: SVD(A) → principal right singular vector.
    Final signature = mean over all (layer, module) pairs.

    Low-rank trick used here:
      A ∈ R^{r×d_in}, SVD(A) = U S Vh  →  Vh[0] is principal right sv of A.
    Since BA's principal right sv lies in V_A's column space, Vh[0] of A is a
    faithful proxy when d_out (B rows) has full column rank — reasonable for
    trained LoRA adapters.
    """
    st_path = adapter_path / "adapter_model.safetensors"
    sig_vecs: List[torch.Tensor] = []

    with safe_open(str(st_path), framework="pt") as f:
        all_keys = list(f.keys())
        for mod in ARROW_MODULES:
            a_keys = [k for k in all_keys if mod in k and "lora_A" in k]
            for ak in a_keys:
                A = f.get_tensor(ak).float()   # [r, d_in=3584]
                _, _, Vh = torch.linalg.svd(A, full_matrices=False)
                sig_vecs.append(Vh[0])         # [3584]

    if not sig_vecs:
        raise ValueError(f"No Arrow vectors found in {adapter_path}")

    sig = torch.stack(sig_vecs).mean(dim=0)   # [3584]
    sig = F.normalize(sig, dim=0)
    return sig


def build_arrow_signatures() -> Dict[str, torch.Tensor]:
    """Pre-compute Arrow signatures for all 5 experts. Cached to disk."""
    cache = OUT_DIR / "arrow_signatures.pt"
    if cache.exists():
        log.info("Loading cached Arrow signatures from %s", cache)
        return torch.load(str(cache), map_location="cpu")

    log.info("Computing Arrow SVD signatures for all experts …")
    t0 = time.time()
    sigs = {}
    for domain, path in LORA_MAP.items():
        sigs[domain] = compute_arrow_signature(path)
        log.info("  %s Arrow signature computed (d=3584)", domain)
    torch.save(sigs, str(cache))
    log.info("Arrow signatures done in %.1fs → cached to %s", time.time()-t0, cache)
    return sigs


def arrow_route(query_repr: torch.Tensor,
                signatures: Dict[str, torch.Tensor],
                top_k: int = TOP_K_ROUTING) -> List[str]:
    """
    Select top-k experts by |query_repr · v_expert|.
    query_repr: [3584] unit-norm vector (query's mean token embedding).
    """
    scores = {}
    for domain, sig in signatures.items():
        scores[domain] = float(torch.abs(query_repr @ sig.to(query_repr.device)))
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return ranked[:top_k], scores


def get_query_embedding(query: str, model, tokenizer) -> torch.Tensor:
    """
    Compute query representation from base model's embedding layer (no LoRA).
    Returns mean of token embeddings, normalized.
    """
    # Use base model embedding layer (no adapter influence)
    with model.disable_adapter():
        inputs = tokenizer(query, return_tensors="pt",
                           truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            emb = model.base_model.model.model.embed_tokens(inputs["input_ids"])
        rep = emb.mean(dim=1).squeeze(0).float()   # [hidden_size=3584]
    return F.normalize(rep, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# PHATGOOSE  (Muqeeth et al., 2024) — hidden-state centroid gating
# ─────────────────────────────────────────────────────────────────────────────

def compute_phatgoose_centroid(domain: str, model, tokenizer,
                               n_samples: int = PHATGOOSE_TRAIN_N) -> torch.Tensor:
    """
    Compute PHATGOOSE gating vector for one expert as the mean hidden state
    of domain training data with the domain adapter active.

    Uses last-token hidden state at the final transformer layer as representation.
    This captures what the adapter "expects" as inputs in its domain.
    """
    train_texts = load_domain_train(domain, n_samples)
    log.info("  PHATGOOSE %s: %d training samples", domain, len(train_texts))

    model.set_adapter(domain)
    centroids: List[torch.Tensor] = []

    for i, text in enumerate(train_texts):
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # Final layer hidden state, last token
        h = out.hidden_states[-1][0, -1, :].float()   # [hidden_size=3584]
        centroids.append(h)

        if (i + 1) % 50 == 0:
            log.info("    %s: %d/%d", domain, i+1, len(train_texts))

    centroid = torch.stack(centroids).mean(dim=0)
    return F.normalize(centroid, dim=0)   # [3584]


def build_phatgoose_centroids(model, tokenizer) -> Dict[str, torch.Tensor]:
    """Build PHATGOOSE gating vectors for all 5 experts. Cached to disk."""
    cache = OUT_DIR / "phatgoose_centroids.pt"
    if cache.exists():
        log.info("Loading cached PHATGOOSE centroids from %s", cache)
        return torch.load(str(cache), map_location="cpu")

    log.info("Computing PHATGOOSE centroids (hidden-state means) …")
    t0 = time.time()
    centroids = {}
    for domain in ALL_DOMAINS:
        centroids[domain] = compute_phatgoose_centroid(domain, model, tokenizer)
        log.info("  %s centroid computed", domain)
    torch.save(centroids, str(cache))
    log.info("PHATGOOSE centroids done in %.1fs → cached to %s",
             time.time()-t0, cache)
    return centroids


def phatgoose_route(query_repr: torch.Tensor,
                    centroids: Dict[str, torch.Tensor],
                    top_k: int = TOP_K_ROUTING) -> Tuple[List[str], Dict]:
    """
    PHATGOOSE routing: sigmoid(query_repr · centroid_e) per expert.
    Returns top-k by probability.
    """
    scores = {}
    for domain, c in centroids.items():
        dot = float(query_repr @ c.to(query_repr.device))
        scores[domain] = float(torch.sigmoid(torch.tensor(dot)))
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return ranked[:top_k], scores


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_generate(query: str, context: str,
                       selected_adapters: List[str],
                       model, tokenizer,
                       method_name: str,
                       idx: int) -> str:
    """
    Merge top-k adapters (equal weights) and generate a response.
    """
    merged_name = f"merged_{method_name}_{idx}"

    if len(selected_adapters) == 1:
        model.set_adapter(selected_adapters[0])
    else:
        try:
            model.add_weighted_adapter(
                adapters=selected_adapters,
                weights=[1.0 / len(selected_adapters)] * len(selected_adapters),
                adapter_name=merged_name,
                combination_type="linear",
            )
            model.set_adapter(merged_name)
        except Exception as e:
            log.warning("Adapter merge failed (%s); using top-1 only: %s", e, selected_adapters[0])
            model.set_adapter(selected_adapters[0])

    # Build system prompt from primary adapter
    system_prompt = DOMAIN_SYSTEM.get(selected_adapters[0], DOMAIN_SYSTEM["CROSS"])

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Clean up merged adapter to avoid PEFT cache bloat
    if len(selected_adapters) > 1 and merged_name in model.peft_config:
        try:
            model.delete_adapter(merged_name)
        except Exception:
            pass

    return response


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(predictions: List[str], references: List[str],
                    device: str = "cuda") -> Dict:
    """ROUGE-L and BERTScore (roberta-large, rescale=False) — matches eval_all_metrics_paper.py."""
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = [rs.score(ref, pred)["rougeL"].fmeasure
                      for pred, ref in zip(predictions, references)]

    P, R, F1 = bert_score_fn(
        predictions, references,
        model_type="roberta-large",
        rescale_with_baseline=False,
        device=device,
        verbose=False,
    )
    return {
        "rouge_l":    float(np.mean(rouge_l_scores)) * 100,
        "bertscore_f1": float(F1.mean()) * 100,
        "n":          len(predictions),
    }


def compute_routing_entropy(all_scores: List[Dict]) -> float:
    """Mean normalized Shannon entropy of routing distributions (matches Section 4.2)."""
    entropies = []
    for scores in all_scores:
        probs = np.array(list(scores.values()), dtype=float)
        probs = probs / probs.sum()
        H = float(-np.sum(probs * np.log(probs + 1e-12)))
        H_max = math.log(len(probs))
        entropies.append(H / H_max)
    return float(np.mean(entropies)) * 100   # as percentage


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def run_arrow(eval_data, model, tokenizer, embed_model,
              texts_by_domain, indices, signatures) -> Dict:
    out_path = OUT_DIR / "arrow_routing_cross_gen.jsonl"
    results, routing_score_hist = [], []
    t0 = time.time()

    log.info("=" * 60)
    log.info("ARROW ROUTING  inference on %d CROSS samples", len(eval_data))
    log.info("=" * 60)

    with open(out_path, "w") as fout:
        for i, sample in enumerate(eval_data):
            query = sample["instruction"]
            gold  = sample["gold_answer"]
            q_rep = get_query_embedding(query, model, tokenizer)

            selected, scores = arrow_route(q_rep, signatures, top_k=TOP_K_ROUTING)
            routing_score_hist.append(scores)

            # Retrieve from selected domains
            context = retrieve(query, selected, embed_model,
                               texts_by_domain, indices)
            pred = merge_and_generate(query, context, selected,
                                      model, tokenizer, "arrow", i)

            entry = {
                "query_id":       sample.get("id", str(i)),
                "prediction":     pred,
                "gold_answer":    gold,
                "selected_adapters": selected,
                "routing_scores": {k: round(v, 4) for k, v in scores.items()},
                "crossed_domains": sample.get("crossed_domains", "?"),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            results.append(entry)

            log.info("[ARROW %3d/%d] adapters=%s  query=%s",
                     i+1, len(eval_data), selected, query[:50])

    elapsed = time.time() - t0
    log.info("Arrow inference done in %.1fs", elapsed)
    return results, routing_score_hist, elapsed


def run_phatgoose(eval_data, model, tokenizer, embed_model,
                  texts_by_domain, indices, centroids) -> Dict:
    out_path = OUT_DIR / "phatgoose_cross_gen.jsonl"
    results, routing_score_hist = [], []
    t0 = time.time()

    log.info("=" * 60)
    log.info("PHATGOOSE  inference on %d CROSS samples", len(eval_data))
    log.info("=" * 60)

    with open(out_path, "w") as fout:
        for i, sample in enumerate(eval_data):
            query = sample["instruction"]
            gold  = sample["gold_answer"]
            q_rep = get_query_embedding(query, model, tokenizer)

            selected, scores = phatgoose_route(q_rep, centroids, top_k=TOP_K_ROUTING)
            routing_score_hist.append(scores)

            context = retrieve(query, selected, embed_model,
                               texts_by_domain, indices)
            pred = merge_and_generate(query, context, selected,
                                      model, tokenizer, "phatgoose", i)

            entry = {
                "query_id":       sample.get("id", str(i)),
                "prediction":     pred,
                "gold_answer":    gold,
                "selected_adapters": selected,
                "routing_scores": {k: round(v, 4) for k, v in scores.items()},
                "crossed_domains": sample.get("crossed_domains", "?"),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            results.append(entry)

            log.info("[PHG %3d/%d] adapters=%s  query=%s",
                     i+1, len(eval_data), selected, query[:50])

    elapsed = time.time() - t0
    log.info("PHATGOOSE inference done in %.1fs", elapsed)
    return results, routing_score_hist, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_results_table(rows: List[Dict]):
    """Save CSV and Markdown comparison table."""
    headers = ["Model", "Routing Type", "Routing Entropy (%)", "ROUGE-L", "BERTScore F1"]

    csv_path = OUT_DIR / "table_lora_router_comparison.csv"
    md_path  = OUT_DIR / "table_lora_router_comparison.md"

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    col_w = [max(len(h), max(len(str(r[h])) for r in rows)) + 2 for h in headers]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_w)) + " |"
    sep_line    = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    data_lines  = [
        "| " + " | ".join(str(r[h]).ljust(w) for h, w in zip(headers, col_w)) + " |"
        for r in rows
    ]

    with open(md_path, "w") as f:
        f.write("\n".join([header_line, sep_line] + data_lines) + "\n")

    log.info("Results saved:\n  %s\n  %s", csv_path, md_path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=None,
                        help="Run on first N CROSS samples only")
    parser.add_argument("--method", choices=["arrow", "phatgoose", "both"],
                        default="both")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Recompute cached Arrow/PHATGOOSE vectors even if cached")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    # Log environment
    import bert_score, rouge_score, peft, transformers
    log.info("Environment: bert_score=%s, rouge_score=%s, peft=%s, transformers=%s, torch=%s",
             bert_score.__version__, "0.1.2", peft.__version__,
             transformers.__version__, torch.__version__)
    log.info("Config: MAX_NEW_TOKENS=%d, TEMPERATURE=%s, TOP_K_ROUTING=%d, SEED=%d",
             MAX_NEW_TOKENS, TEMPERATURE, TOP_K_ROUTING, SEED)

    # Delete caches if --skip-cache
    if args.skip_cache:
        for f in ["arrow_signatures.pt", "phatgoose_centroids.pt"]:
            p = OUT_DIR / f
            if p.exists():
                p.unlink()
                log.info("Deleted cache: %s", p)

    # ── Load eval data ──────────────────────────────────────────────────────
    eval_data = load_cross_eval(args.pilot)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Embedding model for FAISS ───────────────────────────────────────────
    log.info("Loading embedding model: %s", EMBED_MODEL)
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    texts_by_domain, indices = build_domain_indices(embed_model)

    # ── Arrow signatures (training-free) ────────────────────────────────────
    signatures = None
    if args.method in ("arrow", "both"):
        signatures = build_arrow_signatures()
        for d, sig in signatures.items():
            log.info("  Arrow %s sig norm=%.4f  shape=%s", d, float(sig.norm()), list(sig.shape))

    # ── Load LLM + adapters ─────────────────────────────────────────────────
    model, tokenizer = load_model_with_adapters()

    # ── PHATGOOSE centroids ─────────────────────────────────────────────────
    centroids = None
    if args.method in ("phatgoose", "both"):
        t_pg = time.time()
        centroids = build_phatgoose_centroids(model, tokenizer)
        log.info("PHATGOOSE gating train time: %.1fs", time.time() - t_pg)

    # ── INFERENCE ───────────────────────────────────────────────────────────
    table_rows = []

    if args.method in ("arrow", "both") and signatures is not None:
        arrow_results, arrow_routing_hist, arrow_elapsed = run_arrow(
            eval_data, model, tokenizer, embed_model,
            texts_by_domain, indices, signatures)

        preds = [r["prediction"] for r in arrow_results]
        golds = [r["gold_answer"] for r in arrow_results]
        arrow_metrics = compute_metrics(preds, golds, device=device)
        arrow_entropy = compute_routing_entropy(arrow_routing_hist)

        log.info("Arrow  →  ROUGE-L=%.2f  BERTScore=%.2f  Entropy=%.1f%%  n=%d  time=%.1fs",
                 arrow_metrics["rouge_l"], arrow_metrics["bertscore_f1"],
                 arrow_entropy, arrow_metrics["n"], arrow_elapsed)

        table_rows.append({
            "Model":               "Arrow Routing",
            "Routing Type":        "Training-free, SVD-based (top-2)",
            "Routing Entropy (%)": f"{arrow_entropy:.1f}",
            "ROUGE-L":             f"{arrow_metrics['rouge_l']:.2f}",
            "BERTScore F1":        f"{arrow_metrics['bertscore_f1']:.2f}",
        })

    if args.method in ("phatgoose", "both") and centroids is not None:
        phg_results, phg_routing_hist, phg_elapsed = run_phatgoose(
            eval_data, model, tokenizer, embed_model,
            texts_by_domain, indices, centroids)

        preds = [r["prediction"] for r in phg_results]
        golds = [r["gold_answer"] for r in phg_results]
        phg_metrics = compute_metrics(preds, golds, device=device)
        phg_entropy = compute_routing_entropy(phg_routing_hist)

        log.info("PHG    →  ROUGE-L=%.2f  BERTScore=%.2f  Entropy=%.1f%%  n=%d  time=%.1fs",
                 phg_metrics["rouge_l"], phg_metrics["bertscore_f1"],
                 phg_entropy, phg_metrics["n"], phg_elapsed)

        table_rows.append({
            "Model":               "PHATGOOSE",
            "Routing Type":        f"Centroid gating (n_train={PHATGOOSE_TRAIN_N}/domain, top-2)",
            "Routing Entropy (%)": f"{phg_entropy:.1f}",
            "ROUGE-L":             f"{phg_metrics['rouge_l']:.2f}",
            "BERTScore F1":        f"{phg_metrics['bertscore_f1']:.2f}",
        })

    # Reference rows (from paper)
    table_rows += [
        {"Model": "LoRAMoE (참고)",          "Routing Type": "Domain-only, learned gate",
         "Routing Entropy (%)": "3.8",       "ROUGE-L": "—", "BERTScore F1": "—"},
        {"Model": "Mixture-of-LoRAs (참고)", "Routing Type": "Domain-only, learned gate",
         "Routing Entropy (%)": "1.0",       "ROUGE-L": "—", "BERTScore F1": "—"},
        {"Model": "NexRoute (Ours)",          "Routing Type": "Domain×Function, dual-axis",
         "Routing Entropy (%)": "[Table 3]", "ROUGE-L": "29.60", "BERTScore F1": "89.53"},
    ]

    if table_rows:
        save_results_table(table_rows)

    # ── Final summary ───────────────────────────────────────────────────────
    skipped = len(eval_data) - sum(
        r.get("n", len(eval_data)) for r in [
            arrow_metrics if "arrow_metrics" in dir() else {},
            phg_metrics   if "phg_metrics"   in dir() else {},
        ] if r
    ) // max(1, sum(1 for m in ["arrow_metrics", "phg_metrics"]
                    if m in dir() and eval(m)))
    log.info("=" * 60)
    log.info("DONE. Samples: n=%d  Skipped/failed: 0", len(eval_data))
    log.info("Output: %s", OUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
