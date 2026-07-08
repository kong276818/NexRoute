"""
run_ablation_matrix.py — NexRoute ablation generation pipeline
Phase 1: 4-switch ablation matrix

SWITCH_RETRIEVAL: partitioned | unified
SWITCH_LORA:      domain_specific | none
SWITCH_SCAFFOLD:  on | off
SWITCH_CROSS:     dedicated | weight_avg | task_arithmetic | dominant_single

Usage:
  python run_ablation_matrix.py --variant variant_04_wo_lora [--pilot N]

Variant 1 (Full NexRoute) is NOT re-run; the existing
eval_results_7b_v3_500.json is reused.

Notes on CROSS merge strategies:
  weight_avg:       PEFT add_weighted_adapter, combination_type="linear",
                    weights=[0.5, 0.5] → weighted sum of LoRA deltas (ΔW = 0.5*B_MED@A_MED + 0.5*B_LAW@A_LAW)
  task_arithmetic:  Same API, weights=[1.0, 1.0] → full-strength task vector sum
                    (Ilharco et al. 2023 λ=1.0 for each domain), conceptually
                    "add the full improvement of each domain adapter"
  For LoRA, task vectors (θ_ft − θ_base) ARE the LoRA deltas (B@A), so
  both strategies operate on delta space. The distinction: weight_avg
  normalises (λ=0.5 each), task_arithmetic does not (λ=1.0 each).
"""

import argparse
import copy
import json
import os
import re
import time

import faiss
import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
LORA_DIR     = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR     = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL  = "intfloat/multilingual-e5-large"
ROUTER_PT    = "/home/sclab/moe_model/router_deberta_best.pt"
ROUTER_BERT  = "microsoft/deberta-large"
GT_PATH      = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_DIR      = "/home/sclab/moe_model/results/full_ablation/generations"

DOMAINS      = ["LAW", "MED", "SOC", "STEM"]
ALL_DOMAINS  = ["LAW", "MED", "SOC", "STEM", "CROSS"]
TAU, DELTA   = 0.30, 0.40
TOP_K        = 10

LORA_MAP = {
    "LAW":   "lora_law",
    "MED":   "lora_med",
    "SOC":   "lora_soc",
    "STEM":  "lora_stem",
    "CROSS": "lora_cross",
}

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":  "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":  "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM": "You are a science and engineering expert. Answer accurately based on scientific principles.",
}

FEW_SHOT = {
    "LAW": (
        "Q: What is the doctrine of promissory estoppel?\n"
        "A: Promissory estoppel is a legal doctrine that prevents a party from withdrawing a promise "
        "when another party has reasonably relied on it to their detriment. The doctrine requires a "
        "clear and definite promise, reasonable reliance by the promisee, and actual detriment suffered "
        "as a result of that reliance. Courts apply this principle to enforce promises even without "
        "formal consideration, thereby ensuring fairness and preventing injustice in contractual relationships."
    ),
    "MED": (
        "Q: How does the renin-angiotensin system regulate blood pressure?\n"
        "A: The renin-angiotensin system regulates blood pressure through a cascade of enzymatic "
        "reactions initiated when the kidneys detect low blood pressure or reduced sodium levels. "
        "Renin converts angiotensinogen to angiotensin I, which is subsequently converted to "
        "angiotensin II by ACE in the lungs. Angiotensin II causes vasoconstriction and stimulates "
        "aldosterone secretion, increasing sodium retention and blood volume, ultimately elevating blood pressure."
    ),
    "SOC": (
        "Q: How does social stratification affect educational outcomes?\n"
        "A: Social stratification significantly influences educational outcomes by creating unequal "
        "access to resources, opportunities, and cultural capital. Children from higher socioeconomic "
        "backgrounds benefit from better-funded schools, private tutoring, and enriched home environments "
        "that foster academic achievement. Conversely, those from lower strata face systemic barriers "
        "including underfunded schools and economic pressures that hinder educational attainment, "
        "perpetuating intergenerational inequality."
    ),
    "STEM": (
        "Q: How does CRISPR-Cas9 technology work?\n"
        "A: CRISPR-Cas9 is a revolutionary gene-editing technology that uses a guide RNA to direct "
        "the Cas9 endonuclease to specific DNA sequences. The guide RNA binds complementarily to the "
        "target DNA, enabling Cas9 to create a precise double-strand break at the desired location. "
        "The cell then repairs this break through either non-homologous end joining, which can disrupt "
        "gene function, or homology-directed repair, which can introduce specific genetic modifications "
        "with high precision."
    ),
    "CROSS": (
        "Q: How might advances in neuroscience impact criminal justice policy?\n"
        "A: Advances in neuroscience are profoundly reshaping criminal justice policy by providing "
        "empirical insights into the biological underpinnings of behavior, decision-making, and "
        "culpability. Neuroimaging studies reveal that certain criminal behaviors may be associated "
        "with structural or functional brain abnormalities, raising questions about moral responsibility "
        "and the appropriate balance between punishment and rehabilitation."
    ),
}

# ── Variant definitions ────────────────────────────────────────────────────
VARIANTS = {
    "variant_02_wo_function_axis": {
        "retrieval":       "partitioned",
        "lora":            "domain_specific",
        "scaffold":        False,
        "cross_strategy":  "dedicated",
        "cross_only":      False,
        "desc":            "w/o Function Axis (no few-shot, no domain system prompt)",
    },
    "variant_03_wo_partitioned_retrieval": {
        "retrieval":       "unified",
        "lora":            "domain_specific",
        "scaffold":        True,
        "cross_strategy":  "dedicated",
        "cross_only":      False,
        "desc":            "w/o Domain-Partitioned Retrieval (unified FAISS index)",
    },
    "variant_04_wo_lora": {
        "retrieval":       "partitioned",
        "lora":            "none",
        "scaffold":        True,
        "cross_strategy":  "dedicated",
        "cross_only":      False,
        "desc":            "w/o LoRA Experts (base Qwen2.5-7B, no adapter)",
    },
    "variant_05_wo_domain_axis": {
        "retrieval":       "unified",
        "lora":            "none",
        "scaffold":        True,
        "cross_strategy":  "dedicated",
        "cross_only":      False,
        "desc":            "w/o Domain Axis = Prompt Scaffold Only "
                           "(unified retrieval + base model + function scaffold only; "
                           "Variant 5 and 6 are identical configs — merged into one)",
    },
    "variant_07_cross_weight_avg": {
        "retrieval":       "partitioned",
        "lora":            "domain_specific",
        "scaffold":        True,
        "cross_strategy":  "weight_avg",
        "cross_only":      True,
        "desc":            "CROSS: Weight-Averaged LoRA Merge (MED+LAW, λ=0.5 each)",
    },
    "variant_08_cross_task_arithmetic": {
        "retrieval":       "partitioned",
        "lora":            "domain_specific",
        "scaffold":        True,
        "cross_strategy":  "task_arithmetic",
        "cross_only":      True,
        "desc":            "CROSS: Task Arithmetic Merge (MED+LAW, λ=1.0 each, Ilharco 2023)",
    },
    "variant_09_cross_dominant_single": {
        "retrieval":       "partitioned",
        "lora":            "domain_specific",
        "scaffold":        True,
        "cross_strategy":  "dominant_single",
        "cross_only":      True,
        "desc":            "w/o CROSS Adapter: dominant single-domain adapter",
    },
}


# ── DeBERTa Router ────────────────────────────────────────────────────────
class NexRouteRouter(nn.Module):
    def __init__(self, model_name=ROUTER_BERT):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(512, 4),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.encoder(**kw)
        return self.classifier(self.dropout(out.last_hidden_state[:, 0, :]))


# ── Pipeline ──────────────────────────────────────────────────────────────
class AblationPipeline:
    def __init__(self, cfg: dict):
        self.cfg              = cfg
        self.router           = None
        self.router_tok       = None
        self.base_model       = None
        self.base_tok         = None
        self.embed_model      = None
        self.db               = {}          # partitioned: domain → (texts, faiss_index)
        self.unified_db       = None        # unified: (texts, faiss_index)
        self._loaded_adapters = set()
        self._current_adapter = None

    # ── Model loading ──────────────────────────────────────────────────────
    def load_router(self):
        print("[Router] Loading DeBERTa router ...")
        self.router_tok = AutoTokenizer.from_pretrained(ROUTER_BERT)
        self.router     = NexRouteRouter(ROUTER_BERT).cuda()
        state = torch.load(ROUTER_PT, map_location="cuda")
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        self.router.load_state_dict(state, strict=False)
        self.router.eval()
        print("[Router] Ready")

    def load_base(self):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("[LLM] Loading Qwen2.5-7B NF4 ...")
        self.base_tok = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, local_files_only=True)
        self.base_tok.pad_token = self.base_tok.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )
        print("[LLM] Ready")

    def load_embed(self):
        print("[Embed] Loading multilingual-E5-large ...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print("[Embed] Ready")

    # ── FAISS index building ───────────────────────────────────────────────
    def _load_domain_texts(self):
        domain_files = {
            "LAW":   "law_rewritten.json",
            "MED":   "med_rewritten.json",
            "SOC":   "soc_rewritten.json",
            "STEM":  "stem_rewritten.json",
            "CROSS": "cross_rewritten.json",
        }
        all_texts_by_domain = {}
        for domain, fname in domain_files.items():
            path = os.path.join(DATA_DIR, fname)
            items = json.load(open(path))
            texts = []
            for d in items:
                out = d.get("output", d.get("gold_answer", ""))
                if any(k in out for k in ["The correct answer", "All answers:"]):
                    continue
                texts.append(
                    f"{d['instruction']} {out}"
                )
            all_texts_by_domain[domain] = texts
        return all_texts_by_domain

    def build_partitioned_db(self):
        print("[FAISS] Building partitioned indices ...")
        domain_texts = self._load_domain_texts()
        for domain, texts in domain_texts.items():
            embs = self.embed_model.encode(
                texts, show_progress_bar=False, batch_size=64
            ).astype(np.float32)
            embs /= np.linalg.norm(embs, axis=1, keepdims=True)
            idx = faiss.IndexFlatIP(embs.shape[1])
            idx.add(embs)
            self.db[domain] = (texts, idx)
            print(f"  {domain}: {len(texts)}")
        print("[FAISS] Partitioned DB ready")

    def build_unified_db(self):
        print("[FAISS] Building unified index ...")
        domain_texts = self._load_domain_texts()
        all_texts = []
        for domain_texts_list in domain_texts.values():
            all_texts.extend(domain_texts_list)
        embs = self.embed_model.encode(
            all_texts, show_progress_bar=True, batch_size=64
        ).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        self.unified_db = (all_texts, idx)
        print(f"[FAISS] Unified DB ready ({len(all_texts)} docs)")

    # ── Routing ───────────────────────────────────────────────────────────
    def route(self, query: str) -> dict:
        enc = self.router_tok(
            query, max_length=256, truncation=True,
            padding="max_length", return_tensors="pt",
        ).to("cuda")
        kw = {"input_ids": enc["input_ids"],
              "attention_mask": enc["attention_mask"]}
        if "token_type_ids" in enc:
            kw["token_type_ids"] = enc["token_type_ids"]
        with torch.no_grad():
            scores = torch.sigmoid(self.router(**kw)).squeeze().cpu().tolist()
        domain_scores = dict(zip(DOMAINS, scores))
        sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
        s1, s2 = sorted_d[0][1], sorted_d[1][1]
        is_cross = (s2 >= TAU) and (s1 - s2 <= DELTA)
        return {
            "is_cross":     is_cross,
            "domains":      [sorted_d[0][0], sorted_d[1][0]] if is_cross
                             else [sorted_d[0][0]],
            "scores":       domain_scores,
            "top_domain":   sorted_d[0][0],
        }

    # ── Retrieval ─────────────────────────────────────────────────────────
    def retrieve(self, query: str, routing: dict) -> list:
        emb = self.embed_model.encode([query]).astype(np.float32)
        emb /= np.linalg.norm(emb)

        if self.cfg["retrieval"] == "unified":
            texts, idx = self.unified_db
            D, I = idx.search(emb, TOP_K)
            return [{"domain": "UNIFIED", "text": texts[i], "score": float(d)}
                    for d, i in zip(D[0], I[0])]
        else:
            # partitioned
            results = []
            domains = routing["domains"]
            per_domain = max(1, TOP_K // len(domains))
            for domain in domains:
                if domain not in self.db:
                    domain = domains[0]
                texts, idx = self.db[domain]
                D, I = idx.search(emb, per_domain)
                for d, i in zip(D[0], I[0]):
                    results.append({"domain": domain, "text": texts[i],
                                    "score": float(d)})
            results.sort(key=lambda x: -x["score"])
            return results[:TOP_K]

    # ── LoRA adapter management ───────────────────────────────────────────
    def _ensure_adapter_loaded(self, adapter_name: str):
        if adapter_name in self._loaded_adapters:
            return
        lora_path = os.path.join(LORA_DIR, LORA_MAP[adapter_name])
        if not self._loaded_adapters:
            self.base_model = PeftModel.from_pretrained(
                self.base_model, lora_path, adapter_name=adapter_name)
        else:
            self.base_model.load_adapter(lora_path, adapter_name=adapter_name)
        self._loaded_adapters.add(adapter_name)

    def set_lora(self, routing: dict, cross_strategy: str):
        """Set the active LoRA adapter based on routing result and CROSS strategy."""
        is_cross    = routing["is_cross"]
        top_domain  = routing["top_domain"]

        if not is_cross:
            # Single domain: load & activate the domain adapter
            self._ensure_adapter_loaded(top_domain)
            if self._current_adapter != top_domain:
                self.base_model.set_adapter(top_domain)
                self._current_adapter = top_domain
            return

        # CROSS query: apply the selected strategy
        if cross_strategy == "dedicated":
            self._ensure_adapter_loaded("CROSS")
            if self._current_adapter != "CROSS":
                self.base_model.set_adapter("CROSS")
                self._current_adapter = "CROSS"

        elif cross_strategy == "dominant_single":
            # Use highest-scoring domain adapter, ignore CROSS adapter
            self._ensure_adapter_loaded(top_domain)
            if self._current_adapter != top_domain:
                self.base_model.set_adapter(top_domain)
                self._current_adapter = top_domain

        elif cross_strategy in ("weight_avg", "task_arithmetic"):
            # Merge two domain adapters
            merge_domains = routing["domains"][:2]
            w = 0.5 if cross_strategy == "weight_avg" else 1.0
            weights = [w, w]
            merge_name = f"CROSS_MERGE_{cross_strategy.upper()}"

            if merge_name not in self._loaded_adapters:
                for d in merge_domains:
                    self._ensure_adapter_loaded(d)
                # PEFT LoraModel.add_weighted_adapter is on base_model.base_model
                lora_model = self.base_model.base_model
                lora_model.add_weighted_adapter(
                    adapters=merge_domains,
                    weights=weights,
                    adapter_name=merge_name,
                    combination_type="linear",
                )
                self._loaded_adapters.add(merge_name)

            if self._current_adapter != merge_name:
                self.base_model.set_adapter(merge_name)
                self._current_adapter = merge_name

    # ── Prompt construction ────────────────────────────────────────────────
    def build_prompt(self, query: str, context: list,
                     routing: dict, scaffold_on: bool) -> str:
        ctx_text = "\n".join(
            [f"[{r['domain']}] {r['text'][:200]}" for r in context]
        )
        is_cross   = routing["is_cross"]
        domains    = routing["domains"]
        primary    = "CROSS" if is_cross else domains[0]

        if not scaffold_on:
            return (
                f"Context:\n{ctx_text}\n\n"
                f"Question: {query}\n\nAnswer:"
            )

        if is_cross:
            system = (
                f"You are a multidisciplinary expert specializing in "
                f"{' and '.join(domains)}. "
                f"Integrate knowledge from both domains to answer accurately."
            )
            few_shot = FEW_SHOT["CROSS"]
        else:
            system   = DOMAIN_SYSTEM.get(primary, "You are a helpful expert.")
            few_shot = FEW_SHOT.get(primary, FEW_SHOT["STEM"])

        if primary == "LAW":
            instr = (
                "Provide a comprehensive explanation of the legal holding, "
                "principle, or rule. Describe the court's reasoning, the "
                "applicable legal standard, any relevant exceptions, and the "
                "practical implications. Write as a detailed legal analysis "
                "in full paragraph form."
            )
        else:
            instr = (
                "provide a comprehensive analytical answer with specific "
                "mechanisms and implications"
            )

        return (
            f"{system}\n\n"
            f"Here is an example of a high-quality answer:\n{few_shot}\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Question: {query}\n\n"
            f"Following the example style above, {instr}. "
            f"Do NOT use multiple choice format. "
            f"Do NOT list choices A, B, C, D. "
            f"Do NOT begin with 'The correct answer is'. "
            f"Write in flowing paragraph form only.\n\nAnswer:"
        )

    # ── Generation ────────────────────────────────────────────────────────
    def generate(self, prompt: str, primary_domain: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.base_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.base_tok(
            text, return_tensors="pt", truncation=True, max_length=3072,
        ).to(self.base_model.device)

        max_new = 800 if primary_domain == "LAW" else 512
        with torch.no_grad():
            out = self.base_model.generate(
                **inputs, max_new_tokens=max_new,
                temperature=0.7, do_sample=True,
                pad_token_id=self.base_tok.eos_token_id,
            )
        answer = self.base_tok.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Post-process
        answer = re.sub(r"The correct answer is[^\n]*\n?", "", answer)
        answer = re.sub(r"Therefore,? the answer is[^\n]*\n?", "", answer)
        answer = re.sub(r"The correct holding is option\s*\d+\s*:\s*", "", answer)
        for kw in ["Choices:", "All answers:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
            if kw in answer:
                answer = answer[: answer.index(kw)]
        answer = re.sub(r"\n\d+:.*", "", answer).strip()
        return answer if answer else "[GENERATION_FAILED]"

    # ── Main run ──────────────────────────────────────────────────────────
    def run_variant(self, variant_name: str, pilot: int = 0):
        cfg = self.cfg
        out_path = os.path.join(OUT_DIR, f"{variant_name}.json")

        gt = json.load(open(GT_PATH))
        if cfg.get("cross_only", False):
            samples = [g for g in gt if g["expected_domain"] == "CROSS"]
        else:
            samples = gt
        if pilot:
            samples = samples[:pilot]

        print(f"\n{'='*65}")
        print(f"  {variant_name}")
        print(f"  {cfg['desc']}")
        print(f"  N={len(samples)}, retrieval={cfg['retrieval']}, "
              f"lora={cfg['lora']}, scaffold={cfg['scaffold']}, "
              f"cross={cfg['cross_strategy']}")
        print(f"{'='*65}")

        results = []
        t0 = time.time()

        for idx, item in enumerate(samples):
            query  = item["instruction"].strip()
            gold   = item["gold_answer"]
            domain = item["expected_domain"]

            try:
                routing = self.route(query)

                context = self.retrieve(query, routing)

                if cfg["lora"] == "domain_specific":
                    self.set_lora(routing, cfg["cross_strategy"])

                prompt = self.build_prompt(
                    query, context, routing, cfg["scaffold"])

                primary = "CROSS" if routing["is_cross"] else routing["domains"][0]
                answer  = self.generate(prompt, primary)

            except Exception as e:
                print(f"  [Error] {idx}: {e}")
                routing  = {"is_cross": False, "domains": ["UNKNOWN"],
                             "scores": {}, "top_domain": "UNKNOWN"}
                context  = []
                answer   = "[GENERATION_FAILED]"

            results.append({
                "id":             item.get("id", idx),
                "query":          query,
                "gold_answer":    gold,
                "expected_domain": domain,
                "pred_domain":    "CROSS" if routing["is_cross"]
                                  else routing["domains"][0],
                "is_cross":       routing["is_cross"],
                "domain_scores":  routing.get("scores", {}),
                "pred_answer":    answer,
                "retrieved_context": [r["text"] for r in context],
            })

            elapsed = time.time() - t0
            eta_min = elapsed / (idx + 1) * (len(samples) - idx - 1) / 60
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx+1:>4}/{len(samples)}]  "
                      f"pred={results[-1]['pred_domain']:<6}  "
                      f"ETA {eta_min:.0f}min")

            # Intermediate save every 50
            if (idx + 1) % 50 == 0:
                json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)

        json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=2)
        elapsed_min = (time.time() - t0) / 60
        print(f"\n  Done. N={len(results)}, time={elapsed_min:.0f}min")
        print(f"  Saved → {out_path}")
        return results


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()),
                    help="Variant name to run")
    ap.add_argument("--pilot", type=int, default=0,
                    help="Run only first N samples (0 = full run)")
    args = ap.parse_args()

    cfg  = VARIANTS[args.variant]
    pipe = AblationPipeline(cfg)

    pipe.load_router()
    pipe.load_embed()

    if cfg["retrieval"] == "partitioned":
        pipe.build_partitioned_db()
    else:
        pipe.build_unified_db()

    pipe.load_base()

    pipe.run_variant(args.variant, pilot=args.pilot)


if __name__ == "__main__":
    main()
