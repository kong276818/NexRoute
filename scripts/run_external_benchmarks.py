"""
외부 벤치마크 평가 스크립트: MedQA (USMLE) + MMLU-Law

평가 방식:
- MCQ → NexRoute가 open-ended 답변 생성
- 정확도: 생성 텍스트에서 정답 option letter/text 일치 여부 추출
- 생성 품질: ROUGE-L, BERTScore (vs 정답 option 텍스트)

실행:
    python scripts/run_external_benchmarks.py \
        --benchmark medqa \
        --n_samples 200 \
        --output_dir results/external_benchmarks

    python scripts/run_external_benchmarks.py \
        --benchmark mmlu_law \
        --n_samples 200 \
        --output_dir results/external_benchmarks
"""
import os, sys, json, re, argparse, random
import torch
import pyarrow as pa
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

MEDQA_TEST_ARROW = (
    "/home/sclab/moe_model/datasets/GBaker___med_qa-usmle-4-options"
    "/default/0.0.0/0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"
    "/med_qa-usmle-4-options-test.arrow"
)
MMLU_LAW_JSON = "/home/sclab/moe_model/eval_data/law_mmlu.json"

# ── 데이터 로드 ────────────────────────────────────

def load_medqa(n_samples=200):
    reader = pa.ipc.open_stream(MEDQA_TEST_ARROW)
    table = reader.read_all()
    rows = []
    for i in range(len(table)):
        row = {col: table[col][i].as_py() for col in table.column_names}
        q = row["question"]
        opts = row["options"]
        correct_letter = row["answer_idx"]
        correct_text = row["answer"]

        # MCQ → open-ended 변환
        options_str = "\n".join(f"  {k}: {v}" for k, v in opts.items())
        instruction = (
            f"{q}\n\nOptions:\n{options_str}\n\n"
            "Please select the best answer and explain your reasoning."
        )
        rows.append({
            "id": i,
            "domain": "MED",
            "instruction": instruction,
            "gold_answer": correct_text,
            "correct_letter": correct_letter,
            "options": opts,
        })
    rows = rows[:n_samples]
    print(f"[MedQA] {len(rows)} samples loaded")
    return rows


def load_mmlu_law(n_samples=200):
    with open(MMLU_LAW_JSON) as f:
        data = json.load(f)
    rows = []
    for i, d in enumerate(data):
        inst = d.get("instruction", "")
        out = d.get("output", "")
        # MCQ output → extract correct answer text
        m = re.search(r"The correct answer is [A-D]:\s*(.+?)(?:\n|$|Choices:)", out, re.IGNORECASE)
        if m:
            correct_text = m.group(1).strip()
        else:
            correct_text = out[:200]  # fallback
        letter_m = re.search(r"The correct answer is ([A-D]):", out, re.IGNORECASE)
        correct_letter = letter_m.group(1) if letter_m else "A"
        rows.append({
            "id": i,
            "domain": "LAW",
            "instruction": inst,
            "gold_answer": correct_text,
            "correct_letter": correct_letter,
        })
    rows = rows[:n_samples]
    print(f"[MMLU-Law] {len(rows)} samples loaded")
    return rows


# ── 정답 추출 ───────────────────────────────────────

def extract_letter(text):
    """생성 텍스트에서 선택지 letter 추출."""
    # 명시적 패턴
    for pat in [
        r"\b([A-D])\b is (?:correct|the (?:best|right))",
        r"answer is\s*[:\-]?\s*\*?\*?([A-D])\b",
        r"\bOption\s+([A-D])\b",
        r"^([A-D])[:.)]",
        r"\b([A-D])\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def compute_metrics(preds, golds):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(g, p)["rougeL"].fmeasure for p, g in zip(preds, golds)]
    P, R, F = bert_score_fn(preds, golds, lang="en", model_type="roberta-large", verbose=False)
    return {
        "rouge_l": float(np.mean(rouge_scores)),
        "bert_score": float(F.mean().item()),
    }


# ── 파이프라인 실행 ─────────────────────────────────

def run_nexroute(samples, out_path):
    from src.pipeline import DualMoEPipeline

    pipe = DualMoEPipeline()
    pipe.load_router()
    pipe.load_base_model()
    pipe.load_embed_model()
    pipe.build_db()

    results = []
    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] id={s['id']}", end="\r")
        try:
            routing = pipe.classify_domain(s["instruction"])
            is_cross = routing["is_cross"]
            domains  = routing["domains"]
            pipe.load_lora("CROSS" if is_cross else domains[0])
            context  = pipe.retrieve(s["instruction"], domains, is_cross)
            answer   = pipe.generate(s["instruction"], context, is_cross, domains)
        except Exception as e:
            answer = f"[ERROR] {e}"
        results.append({**s, "prediction": answer, "routing": routing})

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  → saved: {out_path}")
    return results


def run_naive_rag(samples, out_path, pipe):
    """Naive RAG baseline: unified corpus retrieval, no routing."""
    results = []
    # unified DB: all domains concatenated
    all_texts = []
    for domain_texts, _ in pipe.db.values():
        all_texts.extend(domain_texts)
    import faiss
    from sentence_transformers import SentenceTransformer
    embs = pipe.embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    unified_index = faiss.IndexFlatIP(embs.shape[1])
    unified_index.add(embs.astype(np.float32))

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}]", end="\r")
        try:
            emb_q = pipe.embed_model.encode([s["instruction"]])
            emb_q = emb_q / np.linalg.norm(emb_q)
            D, I = unified_index.search(emb_q.astype(np.float32), 10)
            context = [{"domain": "UNI", "text": all_texts[idx], "score": float(sc)}
                       for sc, idx in zip(D[0], I[0])]
            answer = pipe.generate(s["instruction"], context, False, [s["domain"]])
        except Exception as e:
            answer = f"[ERROR] {e}"
        results.append({**s, "prediction": answer})

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  → saved: {out_path}")
    return results


# ── 스코어링 ────────────────────────────────────────

def score_results(results, out_path):
    preds = [r["prediction"] for r in results]
    golds = [r["gold_answer"] for r in results]

    metrics = compute_metrics(preds, golds)

    # 정확도 (letter 기반)
    correct = 0
    for r in results:
        pred_letter = extract_letter(r["prediction"])
        gold_letter = r.get("correct_letter", "")
        # text 일치 fallback
        if pred_letter and gold_letter:
            if pred_letter.upper() == gold_letter.upper():
                correct += 1
        elif r["gold_answer"].lower() in r["prediction"].lower():
            correct += 1
    metrics["accuracy"] = correct / len(results) if results else 0.0

    with open(out_path, "w") as f:
        json.dump({"n": len(results), **metrics}, f, indent=2)
    print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"  BERTScore: {metrics['bert_score']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    return metrics


# ── 메인 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["medqa", "mmlu_law"], required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results/external_benchmarks")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference, only score existing results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.benchmark == "medqa":
        samples = load_medqa(args.n_samples)
        tag = "medqa"
    else:
        samples = load_mmlu_law(args.n_samples)
        tag = "mmlu_law"

    nexroute_path = os.path.join(args.output_dir, f"{tag}_nexroute_results.json")
    naive_path    = os.path.join(args.output_dir, f"{tag}_naive_rag_results.json")

    if not args.skip_inference:
        print("\n[1] NexRoute 추론...")
        nexroute_results = run_nexroute(samples, nexroute_path)

        print("\n[2] Naive RAG 추론...")
        # NexRoute pipeline 재사용 (모델 이미 로드됨)
        from src.pipeline import DualMoEPipeline
        pipe = DualMoEPipeline()
        pipe.load_base_model()
        pipe.load_embed_model()
        pipe.build_db()
        naive_results = run_naive_rag(samples, naive_path, pipe)
    else:
        with open(nexroute_path) as f: nexroute_results = json.load(f)
        with open(naive_path) as f: naive_results = json.load(f)

    print("\n[스코어링] NexRoute:")
    nr_metrics = score_results(
        nexroute_results,
        os.path.join(args.output_dir, f"{tag}_nexroute_scores.json")
    )
    print("\n[스코어링] Naive RAG:")
    naive_metrics = score_results(
        naive_results,
        os.path.join(args.output_dir, f"{tag}_naive_rag_scores.json")
    )

    # 비교 요약
    summary = {
        "benchmark": args.benchmark,
        "n_samples": len(samples),
        "nexroute": nr_metrics,
        "naive_rag": naive_metrics,
        "delta": {
            "rouge_l": nr_metrics["rouge_l"] - naive_metrics["rouge_l"],
            "bert_score": nr_metrics["bert_score"] - naive_metrics["bert_score"],
            "accuracy": nr_metrics["accuracy"] - naive_metrics["accuracy"],
        }
    }
    summary_path = os.path.join(args.output_dir, f"{tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[요약 저장] {summary_path}")
    print(json.dumps(summary["delta"], indent=2))


if __name__ == "__main__":
    main()
