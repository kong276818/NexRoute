import os
import re
import json
import torch
import faiss
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

BASE_MODEL  = "Qwen/Qwen2.5-72B-Instruct"
LORA_DIR    = "/home/sclab/moe_model/lora_weights_v2"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL = "intfloat/multilingual-e5-large"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_rewritten.json"
OUT_DIR     = "/home/sclab/moe_model/baseline_results"
os.makedirs(OUT_DIR, exist_ok=True)

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert.",
    "MED":  "You are a medical expert.",
    "SOC":  "You are a social science expert.",
    "STEM": "You are a science and engineering expert.",
}

# ── 공통 유틸 ─────────────────────────────────────
def load_base_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager"
    )
    return model, tokenizer

def load_embed():
    return SentenceTransformer(EMBED_MODEL)

def build_single_db(embed_model):
    """Naive RAG / Adaptive RAG 용 단일 통합 DB"""
    all_texts = []
    files = ["law.json", "med.json", "soc.json", "stem.json",
             "cross_rewritten.json"]
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        for d in data:
            inst = d.get("instruction", "")
            out  = d.get("output", d.get("gold_answer", ""))
            if any(k in out for k in ["The correct answer", "All answers:"]):
                continue
            all_texts.append(f"{inst} {out}")

    embs  = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    print(f"단일 DB: {len(all_texts)}개")
    return all_texts, index

def retrieve_single(query, embed_model, texts, index, top_k=10):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    D, I = index.search(emb.astype(np.float32), top_k)
    return [texts[i] for i in I[0]]

def generate_answer(query, context, model, tokenizer, system="You are a helpful expert."):
    ctx_text = "\n".join([f"{c[:200]}" for c in context])
    prompt = (
        f"{system}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"- Write a detailed explanatory answer in 3-5 sentences.\n"
        f"- Do NOT use multiple choice format.\n"
        f"- Do NOT list choices A, B, C, D.\n"
        f"- Write a complete paragraph answer only.\n\n"
        f"Answer:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # MCQ 후처리
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    for kw in ["Choices:", "All answers:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[GENERATION_FAILED]"

def calc_metrics(results):
    golds = [x["gold_answer"] for x in results]
    preds = [x["pred_answer"] for x in results]

    # ROUGE-L
    sc     = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge  = np.mean([sc.score(g, p)["rougeL"].fmeasure for g, p in zip(golds, preds)])

    # BERTScore
    _, _, F1 = bert_score(preds, golds, lang="en",
                          model_type="bert-base-uncased", verbose=False)
    bscore = F1.mean().item()

    # BLEU-4
    smooth = SmoothingFunction().method1
    bleus  = []
    for g, p in zip(golds, preds):
        ref = [nltk.word_tokenize(g.lower())]
        hyp = nltk.word_tokenize(p.lower())
        bleus.append(sentence_bleu(ref, hyp, smoothing_function=smooth))
    bleu = np.mean(bleus)

    return {
        "rouge_l":   round(rouge * 100, 2),
        "bertscore": round(bscore * 100, 2),
        "bleu_4":    round(bleu * 100, 2),
    }

# ════════════════════════════════════════════════
# Baseline 1: Naive RAG (라우터 없음 + 단일 DB + 베이스 모델)
# ════════════════════════════════════════════════
def run_naive_rag(gt_data, embed_model, texts, index, model, tokenizer):
    print("\n" + "="*50)
    print("Baseline 1: Naive RAG")
    print("="*50)
    results = []
    for i, item in enumerate(gt_data):
        query = item.get("instruction", "")
        gold  = item.get("gold_answer", "")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")
        try:
            context = retrieve_single(query, embed_model, texts, index)
            answer  = generate_answer(query, context, model, tokenizer)
            results.append({
                "id": i, "query": query,
                "gold_answer": gold, "pred_answer": answer
            })
        except Exception as e:
            print(f"  [Error] {e}")

        if (i + 1) % 50 == 0:
            with open(f"{OUT_DIR}/naive_rag.json", "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(f"{OUT_DIR}/naive_rag.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

# ════════════════════════════════════════════════
# Baseline 2: LoRA RAG (라우터 없음 + 단일 DB + STEM LoRA 고정)
# ════════════════════════════════════════════════
def run_lora_rag(gt_data, embed_model, texts, index, model, tokenizer):
    print("\n" + "="*50)
    print("Baseline 2: LoRA RAG (STEM LoRA 고정)")
    print("="*50)

    # STEM LoRA 고정 로드
    lora_path = os.path.join(LORA_DIR, "lora_stem")
    model = PeftModel.from_pretrained(model, lora_path, adapter_name="STEM")
    print("[*] STEM LoRA loaded")

    results = []
    for i, item in enumerate(gt_data):
        query = item.get("instruction", "")
        gold  = item.get("gold_answer", "")
        print(f"[{i+1}/{len(gt_data)}] {query[:50]}...")
        try:
            context = retrieve_single(query, embed_model, texts, index)
            answer  = generate_answer(query, context, model, tokenizer)
            results.append({
                "id": i, "query": query,
                "gold_answer": gold, "pred_answer": answer
            })
        except Exception as e:
            print(f"  [Error] {e}")

        if (i + 1) % 50 == 0:
            with open(f"{OUT_DIR}/lora_rag.json", "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(f"{OUT_DIR}/lora_rag.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results, model

# ════════════════════════════════════════════════
# Baseline 3: Adaptive RAG (단순 키워드 라우터 + 단일 DB + LoRA 스위칭)
# ════════════════════════════════════════════════
KEYWORDS = {
    "LAW":  ["law", "legal", "court", "statute", "plaintiff", "defendant",
              "contract", "judge", "attorney", "jurisdiction"],
    "MED":  ["medical", "patient", "disease", "treatment", "clinical",
              "diagnosis", "symptom", "drug", "therapy", "health"],
    "SOC":  ["social", "society", "culture", "economic", "political",
              "community", "behavior", "psychology", "policy", "inequality"],
    "STEM": ["science", "physics", "chemical", "biology", "math",
              "engineering", "technology", "energy", "molecule", "data"],
}

def simple_router(query: str) -> str:
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q)
              for d, kws in KEYWORDS.items()}
    return max(scores, key=scores.get)

def run_adaptive_rag(gt_data, embed_model, texts, index, model, tokenizer):
    print("\n" + "="*50)
    print("Baseline 3: Adaptive RAG (키워드 라우터)")
    print("="*50)

    current_lora = None
    lora_map = {
        "LAW": "lora_law", "MED": "lora_med",
        "SOC": "lora_soc", "STEM": "lora_stem"
    }

    results = []
    for i, item in enumerate(gt_data):
        query  = item.get("instruction", "")
        gold   = item.get("gold_answer", "")
        domain = simple_router(query)
        print(f"[{i+1}/{len(gt_data)}] [{domain}] {query[:50]}...")

        try:
            # LoRA 스위칭
            lora_path = os.path.join(LORA_DIR, lora_map[domain])
            if current_lora is None:
                model = PeftModel.from_pretrained(
                    model, lora_path, adapter_name=domain
                )
            elif current_lora != domain:
                model.load_adapter(lora_path, adapter_name=domain)
                model.set_adapter(domain)
            current_lora = domain

            context = retrieve_single(query, embed_model, texts, index)
            system  = DOMAIN_SYSTEM.get(domain, "You are a helpful expert.")
            answer  = generate_answer(query, context, model, tokenizer, system)

            results.append({
                "id": i, "query": query,
                "gold_answer": gold,
                "pred_answer": answer,
                "domain_pred": domain
            })
        except Exception as e:
            print(f"  [Error] {e}")

        if (i + 1) % 50 == 0:
            with open(f"{OUT_DIR}/adaptive_rag.json", "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(f"{OUT_DIR}/adaptive_rag.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

# ════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════
if __name__ == "__main__":
    nltk.download('punkt_tab', quiet=True)

    # GT 로드
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    print(f"GT 샘플: {len(gt_data)}개")

    # 공통 인프라 로드
    print("\n[*] 인프라 로드 중...")
    embed_model    = load_embed()
    texts, index   = build_single_db(embed_model)
    model, tokenizer = load_base_model()

    # 베이스라인 순차 실행
    naive_results          = run_naive_rag(gt_data, embed_model, texts, index, model, tokenizer)
    lora_results, model    = run_lora_rag(gt_data, embed_model, texts, index, model, tokenizer)
    adaptive_results       = run_adaptive_rag(gt_data, embed_model, texts, index, model, tokenizer)

    # 지표 계산
    print("\n" + "="*60)
    print("           최종 비교 결과 (100점 만점)")
    print("="*60)
    print(f"{'모델':<20} {'ROUGE-L':>10} {'BERTScore':>10} {'BLEU-4':>10}")
    print("-"*55)

    for name, results in [
        ("Naive RAG",    naive_results),
        ("LoRA RAG",     lora_results),
        ("Adaptive RAG", adaptive_results),
    ]:
        clean = [x for x in results
                 if "[GENERATION_FAILED]" not in x.get("pred_answer", "")]
        m = calc_metrics(clean)
        print(f"{name:<20} {m['rouge_l']:>10} {m['bertscore']:>10} {m['bleu_4']:>10}")

    # DualMoE 결과 (기존)
    with open("/home/sclab/moe_model/eval_results.json") as f:
        dual_data = json.load(f)
    dual_clean = [x for x in dual_data
                  if "[GENERATION_FAILED]" not in x.get("pred_answer", "")]
    dm = calc_metrics(dual_clean)
    print(f"{'DualMoE (Ours)':<20} {dm['rouge_l']:>10} {dm['bertscore']:>10} {dm['bleu_4']:>10}")
    print("="*60)

    # 저장
    summary = {
        "Naive RAG":    calc_metrics([x for x in naive_results if "[GENERATION_FAILED]" not in x.get("pred_answer","")]),
        "LoRA RAG":     calc_metrics([x for x in lora_results if "[GENERATION_FAILED]" not in x.get("pred_answer","")]),
        "Adaptive RAG": calc_metrics([x for x in adaptive_results if "[GENERATION_FAILED]" not in x.get("pred_answer","")]),
        "DualMoE":      dm
    }
    with open(f"{OUT_DIR}/comparison.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_DIR}/comparison.json")