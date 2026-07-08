#!/usr/bin/env python3
"""
evaluate_baselines.py  —  NexRoute LoRA-axis Comparison
========================================================
Tasks
-----
  1  Gold Answer PPL        base / oracle-NexRoute / LoRAMoE-routed / MoL-routed
  2  Router Shannon Entropy NexRoute vs LoRAMoE-gate vs MoL (DeBERTa classifiers)
  3  Latent Space Similarity cosine sim of last hidden states per routing config

Paths (all local, no HF download needed except base model)
-----
  LLM backbone  : Qwen/Qwen2.5-7B-Instruct  (4-bit NF4)
  LoRA adapters : lora_weights_7b_v3/lora_{law,med,soc,stem,cross}
  Routers       : router_deberta_best.pt   (NexRoute  — sigmoid 4-class + τ/δ)
                  ablation_loramoe_gate.pt  (LoRAMoE  — softmax 5-class MLP)
                  ablation_mol_router.pt    (MoL      — softmax 5-class Linear)
  Eval data     : processed_v2/gt_dataset_500.json

Usage
-----
  export HF_TOKEN=<token>
  conda run -n dualmoe python evaluate_baselines.py [--pilot N] [--tasks 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
DEBERTA_NAME = "microsoft/deberta-large"

LORA_DIR = BASE_DIR / "lora_weights_7b_v3"
LORA_MAP: Dict[str, Path] = {
    "LAW":   LORA_DIR / "lora_law",
    "MED":   LORA_DIR / "lora_med",
    "SOC":   LORA_DIR / "lora_soc",
    "STEM":  LORA_DIR / "lora_stem",
    "CROSS": LORA_DIR / "lora_cross",
}

ROUTER_PATHS = {
    "nexroute": BASE_DIR / "router_deberta_best.pt",
    "loramoe":  BASE_DIR / "ablation_loramoe_gate.pt",
    "mol":      BASE_DIR / "ablation_mol_router.pt",
}

DATA_PATH   = BASE_DIR / "processed_v2" / "gt_dataset_500.json"
RESULTS_DIR = BASE_DIR / "eval_results"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT    = True
MAX_SEQ_LEN = 1024

# NexRoute τ/δ thresholds (must match training)
TAU, DELTA = 0.30, 0.40

DOMAINS     = ["LAW", "MED", "SOC", "STEM"]
ALL_DOMAINS = ["LAW", "MED", "SOC", "STEM", "CROSS"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: Path, pilot: Optional[int] = None) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if pilot:
        # stratified pilot: up to pilot//5 samples per domain
        per = max(1, pilot // len(ALL_DOMAINS))
        buckets: Dict[str, List] = defaultdict(list)
        for d in data:
            buckets[d.get("expected_domain", "CROSS")].append(d)
        data = []
        for dom in ALL_DOMAINS:
            data.extend(buckets[dom][:per])
    log.info("Loaded %d samples from %s", len(data), path)
    return data


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER MODELS  (DeBERTa classifiers)
# ──────────────────────────────────────────────────────────────────────────────

class NexRouteRouter(nn.Module):
    """MLP(1024→512→4) + sigmoid, CROSS via τ/δ thresholding"""
    def __init__(self):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(DEBERTA_NAME)
        h               = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 4)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        return self.classifier(self.dropout(
            self.encoder(**kw).last_hidden_state[:, 0, :]
        ))


class LoRAMoERouter(nn.Module):
    """MLP(1024→512→5) + softmax, Focal CE — direct 5-class gate"""
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(DEBERTA_NAME)
        h            = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.gate    = nn.Sequential(
            nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 5)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        return self.gate(self.dropout(
            self.encoder(**kw).last_hidden_state[:, 0, :]
        ))


class MoLRouter(nn.Module):
    """Linear(1024→5) + softmax, standard CE — lightweight task-label routing"""
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(DEBERTA_NAME)
        h            = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head    = nn.Linear(h, 5)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        return self.head(self.dropout(
            self.encoder(**kw).last_hidden_state[:, 0, :]
        ))


def load_router(name: str) -> nn.Module:
    cls_map = {"nexroute": NexRouteRouter, "loramoe": LoRAMoERouter, "mol": MoLRouter}
    path    = ROUTER_PATHS[name]
    log.info("Loading %s router from %s", name, path)
    model = cls_map[name]()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


def route_nexroute(logits: torch.Tensor) -> str:
    """NexRoute τ/δ CROSS detection on sigmoid 4-class output."""
    probs = torch.sigmoid(logits[0])          # (4,)
    top2  = probs.topk(2).values
    # CROSS: top score < τ  OR  gap between top-2 < δ
    if top2[0].item() < TAU or (top2[0] - top2[1]).item() < DELTA:
        return "CROSS"
    return DOMAINS[probs.argmax().item()]


def route_softmax(logits: torch.Tensor) -> str:
    """Standard softmax argmax for LoRAMoE and MoL (5-class)."""
    return ALL_DOMAINS[logits[0].argmax().item()]


def batch_route(
    router: nn.Module,
    tokenizer: AutoTokenizer,
    queries: List[str],
    router_name: str,
) -> List[str]:
    """Run router on all queries; return predicted domain per query."""
    decisions = []
    for q in queries:
        enc = tokenizer(
            q, return_tensors="pt", truncation=True,
            max_length=256, padding=False,
        ).to(DEVICE)
        with torch.no_grad():
            logits = router(**enc)
        if router_name == "nexroute":
            decisions.append(route_nexroute(logits))
        else:
            decisions.append(route_softmax(logits))
    return decisions


# ──────────────────────────────────────────────────────────────────────────────
# LLM LOADING — multi-adapter via PEFT
# ──────────────────────────────────────────────────────────────────────────────

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_llm_with_adapters() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen2.5-7B-Instruct with all 5 domain LoRA adapters pre-loaded.
    Switch between them with model.set_adapter(domain).
    Disable all adapters with model.disable_adapters() for base evaluation.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("export HF_TOKEN=<token>")

    log.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, token=hf_token, padding_side="left", trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kw: Dict = dict(device_map="auto", trust_remote_code=True, token=hf_token)
    if USE_4BIT:
        kw["quantization_config"] = _bnb_config()
    else:
        kw["torch_dtype"] = torch.bfloat16

    log.info("Loading base LLM (4bit=%s)", USE_4BIT)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kw)

    # Load the first domain adapter via PeftModel, then add the rest
    domains_to_load = [d for d in ALL_DOMAINS if LORA_MAP[d].exists()]
    if not domains_to_load:
        raise FileNotFoundError(f"No LoRA adapters found under {LORA_DIR}")

    first_domain = domains_to_load[0]
    log.info("Attaching first adapter: %s", first_domain)
    model = PeftModel.from_pretrained(
        model, str(LORA_MAP[first_domain]),
        adapter_name=first_domain, is_trainable=False,
    )

    for domain in domains_to_load[1:]:
        log.info("Adding adapter: %s", domain)
        model.load_adapter(str(LORA_MAP[domain]), adapter_name=domain)

    model.eval()
    log.info("All adapters loaded: %s", domains_to_load)
    return model, tokenizer


def unload_llm(model: AutoModelForCausalLM) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("LLM unloaded; VRAM cleared.")


# ──────────────────────────────────────────────────────────────────────────────
# TASK 1 — GOLD ANSWER PPL
# ──────────────────────────────────────────────────────────────────────────────

def compute_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: str,
    gold_answer: str,
) -> float:
    """PPL of gold_answer conditioned on query. Only answer tokens are scored."""
    q_ids = tokenizer.encode(query, add_special_tokens=True)
    a_ids = tokenizer.encode(gold_answer, add_special_tokens=False)

    if not a_ids:
        return float("inf")

    input_ids = torch.tensor([q_ids + a_ids], device=DEVICE)
    labels    = torch.full_like(input_ids, -100)
    labels[0, len(q_ids):] = input_ids[0, len(q_ids):]

    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, :MAX_SEQ_LEN]
        labels    = labels[:, :MAX_SEQ_LEN]

    if (labels != -100).sum() == 0:
        return float("inf")

    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss
    return math.exp(loss.item())


EVAL_CONFIGS = {
    "base":           "no adapter — pure base model",
    "nexroute_oracle":"correct domain adapter per query (oracle routing)",
    "loramoe_routed": "adapter selected by LoRAMoE-gate router",
    "mol_routed":     "adapter selected by MoL router",
}


def run_task1(
    data: List[Dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    routing_cache: Dict[str, List[str]],
) -> Dict:
    """
    Task 1: PPL for each routing config.

    routing_cache keys: "loramoe", "mol"  → list of predicted domains (len == len(data))
    """
    log.info("=" * 60)
    log.info("TASK 1  Gold Answer Perplexity")
    log.info("=" * 60)

    results: Dict[str, Dict] = {}

    for config_name, desc in EVAL_CONFIGS.items():
        log.info("[PPL] Config: %s — %s", config_name, desc)
        scores: Dict[str, float] = {}

        for i, sample in enumerate(data):
            sid    = sample.get("id", str(i))
            query  = sample["instruction"]
            answer = sample["gold_answer"]
            domain = sample.get("expected_domain", "LAW")

            # Select adapter
            if config_name == "base":
                with model.disable_adapter():
                    ppl = compute_ppl(model, tokenizer, query, answer)
            else:
                if config_name == "nexroute_oracle":
                    adapter = domain
                elif config_name == "loramoe_routed":
                    adapter = routing_cache["loramoe"][i]
                elif config_name == "mol_routed":
                    adapter = routing_cache["mol"][i]
                else:
                    adapter = domain

                model.set_adapter(adapter)
                ppl = compute_ppl(model, tokenizer, query, answer)

            scores[sid] = {
                "ppl":            ppl,
                "domain":         domain,
                "adapter_used":   config_name if config_name == "base" else adapter,
                "correct_adapter": (adapter == domain) if config_name != "base" else None,
            }
            log.info(
                "[PPL] %-20s  sid=%-4s  domain=%-5s  adapter=%-6s  PPL=%.3f",
                config_name, sid, domain,
                scores[sid]["adapter_used"], ppl,
            )

        results[config_name] = _summarize_ppl(scores, config_name)

    out = RESULTS_DIR / "task1_ppl.json"
    _save_json(results, out)
    log.info("Task 1 saved → %s", out)
    return results


def _summarize_ppl(scores: Dict, config_name: str) -> Dict:
    ppl_vals   = [v["ppl"] for v in scores.values() if math.isfinite(v["ppl"])]
    per_domain: Dict[str, List[float]] = defaultdict(list)
    for v in scores.values():
        if math.isfinite(v["ppl"]):
            per_domain[v["domain"]].append(v["ppl"])

    summary = {
        "config":       config_name,
        "mean_ppl":     float(np.mean(ppl_vals)) if ppl_vals else float("nan"),
        "median_ppl":   float(np.median(ppl_vals)) if ppl_vals else float("nan"),
        "n":            len(ppl_vals),
        "per_domain":   {d: round(float(np.mean(v)), 4) for d, v in per_domain.items()},
        "per_sample":   scores,
    }
    log.info(
        "[PPL] %-20s  mean=%.3f  median=%.3f  n=%d",
        config_name, summary["mean_ppl"], summary["median_ppl"], summary["n"],
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# TASK 2 — ROUTER SHANNON ENTROPY
# ──────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(probs: torch.Tensor) -> float:
    """H = -Σ p·log(p), expects a 1-D probability tensor."""
    p = probs.clamp_min(1e-9)
    return -(p * p.log()).sum().item()


def compute_router_entropy_batch(
    router: nn.Module,
    tokenizer: AutoTokenizer,
    queries: List[str],
    router_name: str,
) -> List[Dict]:
    """
    For each query return:
      logits      (list of floats — raw logits)
      probs       (list of floats — after softmax / sigmoid→softmax)
      entropy     (float — Shannon H in nats)
      prediction  (str  — predicted domain)
    """
    results = []
    deberta_tok = AutoTokenizer.from_pretrained(DEBERTA_NAME)

    for q in queries:
        enc = deberta_tok(
            q, return_tensors="pt", truncation=True,
            max_length=256, padding=False,
        ).to(DEVICE)

        with torch.no_grad():
            logits = router(**enc)[0]   # (n_classes,)

        if router_name == "nexroute":
            # 4-class sigmoid → treat as unnormalised scores, apply softmax for entropy
            sig_probs = torch.sigmoid(logits)   # (4,) domain probs
            # Append implicit CROSS "score": determined by τ/δ rule, not a 5th output
            # For entropy we use the 4-class softmax (captures uncertainty among DOMAINS)
            probs = F.softmax(logits, dim=-1)   # (4,)
            pred  = route_nexroute(logits.unsqueeze(0))
        else:
            # 5-class softmax (LoRAMoE / MoL)
            probs = F.softmax(logits, dim=-1)   # (5,)
            pred  = route_softmax(logits.unsqueeze(0))

        H = _shannon_entropy(probs)

        results.append({
            "logits":     logits.cpu().tolist(),
            "probs":      probs.cpu().tolist(),
            "entropy":    H,
            "prediction": pred,
        })

    return results


def run_task2(data: List[Dict]) -> Dict:
    """
    Task 2: router entropy per model (NexRoute / LoRAMoE / MoL).

    Expected pattern:
      CROSS queries    → high entropy in LoRAMoE & MoL  (router is uncertain)
      Single-domain    → low entropy in all (clear winner class)
      NexRoute         → lower overall entropy even on CROSS (τ/δ absorbs uncertainty)
    """
    log.info("=" * 60)
    log.info("TASK 2  Router Shannon Entropy")
    log.info("=" * 60)

    queries     = [s["instruction"] for s in data]
    true_domains= [s.get("expected_domain", "CROSS") for s in data]
    all_results: Dict = {}

    for name in ["nexroute", "loramoe", "mol"]:
        path = ROUTER_PATHS[name]
        if not path.exists():
            log.warning("Router not found: %s — skipping.", path)
            continue

        router  = load_router(name)
        deberta_tok = AutoTokenizer.from_pretrained(DEBERTA_NAME)
        per_q   = compute_router_entropy_batch(router, deberta_tok, queries, name)

        # Group entropy by true domain
        by_domain: Dict[str, List[float]] = defaultdict(list)
        correct, total = 0, 0
        for dom, r in zip(true_domains, per_q):
            by_domain[dom].append(r["entropy"])
            total += 1
            if r["prediction"] == dom:
                correct += 1

        n_classes = 4 if name == "nexroute" else 5
        max_H     = math.log(n_classes)

        all_results[name] = {
            "routing_accuracy": round(correct / total * 100, 2) if total else 0,
            "n_classes":        n_classes,
            "max_entropy_nats": round(max_H, 4),
            "mean_entropy_overall": round(float(np.mean([r["entropy"] for r in per_q])), 4),
            "mean_entropy_per_domain": {
                d: round(float(np.mean(vals)), 4) for d, vals in by_domain.items()
            },
            "per_sample": [
                {
                    "id":            data[i].get("id", str(i)),
                    "true_domain":   true_domains[i],
                    "prediction":    per_q[i]["prediction"],
                    "entropy":       round(per_q[i]["entropy"], 4),
                    "probs":         [round(p, 4) for p in per_q[i]["probs"]],
                }
                for i in range(len(data))
            ],
        }

        log.info("[Entropy] %-10s  accuracy=%.1f%%  mean_H=%.4f  max_H=%.4f",
                 name,
                 all_results[name]["routing_accuracy"],
                 all_results[name]["mean_entropy_overall"],
                 max_H)
        for dom, h in all_results[name]["mean_entropy_per_domain"].items():
            log.info("           domain=%-5s  mean_H=%.4f  (%.1f%% of max)",
                     dom, h, h / max_H * 100)

        del router
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = RESULTS_DIR / "task2_router_entropy.json"
    _save_json(all_results, out)
    log.info("Task 2 saved → %s", out)
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# TASK 3 — LATENT SPACE COSINE SIMILARITY
# ──────────────────────────────────────────────────────────────────────────────

def extract_hidden(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
) -> np.ndarray:
    """Last hidden state of the last non-padding token. Shape: (hidden_dim,)."""
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_SEQ_LEN, padding=False,
    ).to(DEVICE)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, return_dict=True)

    last_layer = out.hidden_states[-1]        # (1, seq, hidden)
    vec        = last_layer[0, -1, :].float() # last token
    return vec.cpu().numpy()


def pairwise_cosine(E: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-9)
    N     = E / norms
    return (N @ N.T).astype(np.float32)


def run_task3(
    data: List[Dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    routing_cache: Dict[str, List[str]],
) -> Dict:
    """
    Task 3: pairwise cosine similarity of last hidden states.

    Expected pattern:
      NexRoute oracle  → high same-domain sim, low cross-domain sim (disentangled)
      MoL/LoRAMoE routed → more entangled (wrong adapter for cross-domain queries)
      Base             → most entangled (no domain specialisation)
    """
    log.info("=" * 60)
    log.info("TASK 3  Latent Space Cosine Similarity")
    log.info("=" * 60)

    configs = {
        "base":           None,
        "nexroute_oracle": "oracle",
        "loramoe_routed":  "loramoe",
        "mol_routed":      "mol",
    }
    all_results: Dict = {}

    for config_name, routing_key in configs.items():
        log.info("[Sim] Config: %s", config_name)
        embeddings, labels = [], []

        for i, sample in enumerate(data):
            domain = sample.get("expected_domain", "CROSS")
            func   = sample.get("expected_func", "?")
            text   = sample["instruction"]

            if config_name == "base":
                with model.disable_adapter():
                    emb = extract_hidden(model, tokenizer, text)
            else:
                if routing_key == "oracle":
                    adapter = domain
                else:
                    adapter = routing_cache[routing_key][i]
                model.set_adapter(adapter)
                emb = extract_hidden(model, tokenizer, text)

            embeddings.append(emb)
            labels.append(f"{domain}/{func}")
            log.info("[Sim] %-20s  %s/%s  shape=%s",
                     config_name, domain, func, emb.shape)

        E       = np.stack(embeddings)
        sim     = pairwise_cosine(E)
        summary = _sim_summary(sim, labels, config_name)

        all_results[config_name] = {
            "labels":      labels,
            "sim_matrix":  sim.tolist(),
            "summary":     summary,
        }

    out = RESULTS_DIR / "task3_similarity.json"
    _save_json(all_results, out)
    log.info("Task 3 saved → %s", out)

    _visualize_similarity(all_results)
    return all_results


def _sim_summary(sim: np.ndarray, labels: List[str], name: str) -> Dict:
    n = len(labels)
    same_vals, cross_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dom_i = labels[i].split("/")[0]
            dom_j = labels[j].split("/")[0]
            (same_vals if dom_i == dom_j else cross_vals).append(float(sim[i, j]))

    avg_same  = float(np.mean(same_vals))  if same_vals  else float("nan")
    avg_cross = float(np.mean(cross_vals)) if cross_vals else float("nan")
    gap       = avg_same - avg_cross

    log.info(
        "[SimSummary] %-20s  same=%.4f  cross=%.4f  gap=%.4f",
        name, avg_same, avg_cross, gap,
    )
    return {"avg_same_domain": avg_same, "avg_cross_domain": avg_cross, "gap": gap}


def _visualize_similarity(all_results: Dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping heatmap.")
        return

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_results.items()):
        labels = res["labels"]
        sim    = np.array(res["sim_matrix"])
        gap    = res["summary"].get("gap", float("nan"))

        im = ax.imshow(sim, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{name}\ngap={gap:.4f}", fontsize=9)
        plt.colorbar(im, ax=ax)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{sim[i,j]:.2f}",
                        ha="center", va="center", fontsize=6)

    plt.suptitle("Pairwise Cosine Similarity — Last Hidden State", fontsize=11)
    plt.tight_layout()
    out = RESULTS_DIR / "task3_similarity_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Heatmap saved → %s", out)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _save_json(obj: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _print_summary(task1: Dict, task2: Dict, task3: Dict) -> None:
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if task1:
        print("\n[Task 1] Mean PPL by routing config (lower = better):")
        for cfg, v in task1.items():
            print(f"  {cfg:<22} mean={v['mean_ppl']:.3f}  median={v['median_ppl']:.3f}")

    if task2:
        print("\n[Task 2] Router entropy (nats) — mean per domain (higher = uncertain):")
        for name, v in task2.items():
            print(f"  {name} (acc={v['routing_accuracy']:.1f}%, max_H={v['max_entropy_nats']:.3f})")
            for dom, h in v["mean_entropy_per_domain"].items():
                pct = h / v["max_entropy_nats"] * 100
                print(f"    {dom:<6} H={h:.4f}  ({pct:.1f}% of max)")

    if task3:
        print("\n[Task 3] Representation disentanglement (gap = same_H - cross_H, higher = better):")
        for cfg, v in task3.items():
            s = v["summary"]
            print(f"  {cfg:<22} same={s['avg_same_domain']:.4f}  "
                  f"cross={s['avg_cross_domain']:.4f}  gap={s['gap']:.4f}")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=int, default=None,
                   help="Stratified pilot: N total samples (e.g. 25 = 5/domain)")
    p.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3],
                   help="Which tasks to run (e.g. --tasks 1 2 3)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    tasks  = set(args.tasks)
    log.info("Device=%s  4bit=%s  tasks=%s  pilot=%s",
             DEVICE, USE_4BIT, sorted(tasks), args.pilot)

    data = load_data(DATA_PATH, pilot=args.pilot)

    # ── Pre-compute routing decisions for all queries (Tasks 1 & 3) ──────────
    routing_cache: Dict[str, List[str]] = {}
    if tasks & {1, 3}:
        log.info("Pre-computing routing decisions for LoRAMoE and MoL routers...")
        deberta_tok = AutoTokenizer.from_pretrained(DEBERTA_NAME)
        for name in ["loramoe", "mol"]:
            if not ROUTER_PATHS[name].exists():
                log.warning("Router not found: %s — skipping routing cache.", name)
                routing_cache[name] = [s.get("expected_domain", "CROSS") for s in data]
                continue
            router = load_router(name)
            routing_cache[name] = batch_route(
                router, deberta_tok, [s["instruction"] for s in data], name
            )
            del router
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        log.info("Routing pre-computation done.")

    # ── Task 2: DeBERTa routers only — no LLM needed ─────────────────────────
    task2_results: Dict = {}
    if 2 in tasks:
        task2_results = run_task2(data)

    # ── Tasks 1 & 3: load LLM with all adapters ───────────────────────────────
    task1_results: Dict = {}
    task3_results: Dict = {}

    if tasks & {1, 3}:
        model, tokenizer = load_llm_with_adapters()

        if 1 in tasks:
            task1_results = run_task1(data, model, tokenizer, routing_cache)

        if 3 in tasks:
            task3_results = run_task3(data, model, tokenizer, routing_cache)

        unload_llm(model)

    _print_summary(task1_results, task2_results, task3_results)
    log.info("All done. Results in %s/", RESULTS_DIR)


if __name__ == "__main__":
    main()
