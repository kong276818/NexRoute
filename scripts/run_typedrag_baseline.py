"""
Typed-RAG 베이스라인 구현 및 평가 스크립트

Typed-RAG (NAACL 2024):
  - Query function type 분류 (What/How/Why/Compare)
  - Unified corpus RAG (도메인 파티션 없음)
  - Function-specific prompt scaffold 적용
  - LoRA adapter 없음 (base 모델만 사용)

NexRoute vs Typed-RAG 핵심 차이:
  - Typed-RAG: 단일 function axis + 통합 corpus
  - NexRoute:  domain axis + function axis + 도메인 파티션 corpus + LoRA

실행:
    python scripts/run_typedrag_baseline.py \
        --gt_path processed_v2/gt_dataset_500.json \
        --output_dir results/typedrag \
        --n_samples 300
"""
import os, sys, json, re, argparse, gc, random
import numpy as np
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

BASE_MODEL = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
EMBED_NAME = "intfloat/multilingual-e5-large"

# NexRoute의 DeBERTa function router 재사용
FUNC_ROUTER_DIR = "/home/sclab/moe_model/routers/func_router"

# ── Function scaffold (Typed-RAG 스타일) ──────────────────
SCAFFOLDS = {
    "WHAT":    (
        "Based on the following context, identify and define the key concepts. "
        "Answer factually and concisely:"
    ),
    "HOW":     (
        "Using the following context, describe the mechanism or procedure "
        "step-by-step in logical order:"
    ),
    "WHY":     (
        "Based on the following context, explain the underlying causes and reasons. "
        "Construct a clear causal chain:"
    ),
    "COMPARE": (
        "Using the following context, compare and contrast the entities mentioned. "
        "Highlight key similarities and differences:"
    ),
}
SYSTEM_PROMPT = (
    "You are a knowledgeable expert. Answer questions accurately and comprehensively "
    "based on the provided context. Write in fluent paragraph form."
)


# ── Function Classifier (DeBERTa 재사용) ──────────────────

class FuncClassifier:
    """NexRoute 학습된 DeBERTa function router를 재사용."""

    def __init__(self):
        import pickle
        # pipeline.pkl에서 function head만 로드
        try:
            with open(os.path.join(FUNC_ROUTER_DIR, "pipeline.pkl"), "rb") as f:
                self.pipe = pickle.load(f)
            self.use_pipe = True
            print("[*] Function router (pipeline.pkl) loaded")
        except Exception:
            self.use_pipe = False
            print("[!] Function router not found, using rule-based fallback")

    def classify(self, query: str) -> str:
        if self.use_pipe:
            try:
                label = self.pipe.predict([query])[0]
                return str(label).upper()
            except Exception:
                pass
        return self._rule_based(query)

    @staticmethod
    def _rule_based(query: str) -> str:
        q = query.lower()
        if any(q.startswith(w) for w in ["compare", "contrast", "differ", "similar", "versus", "vs"]):
            return "COMPARE"
        if any(q.startswith(w) for w in ["why", "reason", "cause", "explain why", "what causes"]):
            return "WHY"
        if any(q.startswith(w) for w in ["how", "describe how", "what is the process", "what are the steps"]):
            return "HOW"
        return "WHAT"


# ── Typed-RAG Pipeline ────────────────────────────────────

class TypedRAGPipeline:
    def __init__(self):
        self.func_clf  = None
        self.tokenizer = None
        self.model     = None
        self.embed     = None
        self.unified_index = None
        self.unified_texts = []

    def load_all(self):
        self.func_clf = FuncClassifier()
        self._load_model()
        self._load_embed()
        self._build_unified_db()

    def _load_model(self):
        print("[*] Typed-RAG: base model 로드...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )
        self.model.config.use_cache = False
        print("[*] 로드 완료!")

    def _load_embed(self):
        print("[*] 임베딩 모델 로드...")
        self.embed = SentenceTransformer(EMBED_NAME)
        print("[*] 임베딩 완료!")

    def _build_unified_db(self):
        """모든 도메인 corpus를 하나의 통합 index로 구성 (Typed-RAG 방식)."""
        print("[*] Unified corpus 구축...")
        domain_files = {
            "LAW":   "law_rewritten.json",
            "MED":   "med_rewritten.json",
            "SOC":   "soc_rewritten.json",
            "STEM":  "stem_rewritten.json",
        }
        all_texts = []
        for domain, fname in domain_files.items():
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            for d in data:
                out = d.get("output", d.get("gold_answer", ""))
                if any(k in out for k in ["The correct answer", "All answers:", "All holdings:"]):
                    continue
                text = f"{d['instruction']} {out}"
                all_texts.append(text)

        print(f"  총 {len(all_texts)}개 문서 임베딩 중...")
        embs = self.embed.encode(all_texts, show_progress_bar=True, batch_size=64)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        self.unified_index = faiss.IndexFlatIP(embs.shape[1])
        self.unified_index.add(embs.astype(np.float32))
        self.unified_texts = all_texts
        print(f"[*] Unified DB 완료: {len(all_texts)}개")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        emb_q = self.embed.encode([query])
        emb_q = emb_q / np.linalg.norm(emb_q)
        D, I = self.unified_index.search(emb_q.astype(np.float32), top_k)
        return [{"text": self.unified_texts[idx], "score": float(sc)}
                for sc, idx in zip(D[0], I[0])]

    def generate(self, query: str, context: list, func_label: str) -> str:
        scaffold = SCAFFOLDS.get(func_label, SCAFFOLDS["WHAT"])
        ctx_text = "\n".join(f"[{i+1}] {r['text'][:200]}" for i, r in enumerate(context))
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"{scaffold}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # MCQ artifact 제거
        for kw in ["The correct answer is", "All answers:", "\nA:", "\nB:", "\nC:", "\nD:"]:
            if kw in answer:
                answer = answer[:answer.index(kw)].strip()
        return answer if answer else "[GENERATION_FAILED]"

    def run(self, query: str) -> dict:
        func_label = self.func_clf.classify(query)
        context    = self.retrieve(query)
        answer     = self.generate(query, context, func_label)
        return {"func_label": func_label, "context": context, "answer": answer}


# ── 평가 ─────────────────────────────────────────────────

def compute_metrics(preds, golds, desc=""):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = [scorer.score(g, p)["rougeL"].fmeasure for p, g in zip(preds, golds)]
    P, R, F = bert_score_fn(
        preds, golds, lang="en", model_type="roberta-large", verbose=False
    )
    metrics = {
        "rouge_l": float(np.mean(rl)),
        "bert_score_f1": float(F.mean().item()),
        "n": len(preds),
    }
    print(f"  {desc} ROUGE-L: {metrics['rouge_l']:.4f} | BERTScore: {metrics['bert_score_f1']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", default="processed_v2/gt_dataset_500.json")
    parser.add_argument("--output_dir", default="results/typedrag")
    parser.add_argument("--n_samples", type=int, default=300,
                        help="clean-300 subset (idx 0-299)")
    parser.add_argument("--skip_inference", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.gt_path) as f:
        gt_all = json.load(f)
    # clean-300: 인덱스 0~299 (LoRA adapter training에 노출 안 됨)
    gt = gt_all[: args.n_samples]
    print(f"평가 샘플: {len(gt)}개 (clean-{args.n_samples})")

    out_path  = os.path.join(args.output_dir, "typedrag_results.json")
    score_path = os.path.join(args.output_dir, "typedrag_scores.json")

    if not args.skip_inference:
        pipe = TypedRAGPipeline()
        pipe.load_all()

        results = []
        for i, s in enumerate(gt):
            print(f"  [{i+1}/{len(gt)}] id={s['id']}", end="\r")
            try:
                out = pipe.run(s["instruction"])
                answer = out["answer"]
                func_label = out["func_label"]
            except Exception as e:
                answer = f"[ERROR] {e}"
                func_label = "WHAT"
            results.append({
                "id": s["id"],
                "expected_domain": s["expected_domain"],
                "expected_func":   s.get("expected_func", ""),
                "instruction":     s["instruction"],
                "gold_answer":     s["gold_answer"],
                "prediction":      answer,
                "func_label":      func_label,
            })

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  → 저장: {out_path}")

        del pipe; torch.cuda.empty_cache(); gc.collect()
    else:
        with open(out_path) as f:
            results = json.load(f)

    # ── 스코어링 ──────────────────────────────────
    preds = [r["prediction"] for r in results]
    golds = [r["gold_answer"] for r in results]

    overall  = compute_metrics(preds, golds, "Overall")
    cross_r  = [r for r in results if r["expected_domain"] == "CROSS"]
    single_r = [r for r in results if r["expected_domain"] != "CROSS"]
    cross_m  = compute_metrics(
        [r["prediction"] for r in cross_r],
        [r["gold_answer"] for r in cross_r], "CROSS"
    ) if cross_r else {}
    single_m = compute_metrics(
        [r["prediction"] for r in single_r],
        [r["gold_answer"] for r in single_r], "Single"
    ) if single_r else {}

    scores = {
        "model": "Typed-RAG",
        "n_samples": len(results),
        "overall":  overall,
        "cross":    cross_m,
        "single":   single_m,
    }
    with open(score_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n[저장] {score_path}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
