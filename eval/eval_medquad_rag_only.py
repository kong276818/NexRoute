"""
eval_medquad_rag_only.py
========================
NexRoute가 이미 완료됐으므로, Naive RAG + Adaptive RAG만 실행하고
전체 스코어 테이블을 출력한다.
"""

import os, json, re, torch, numpy as np, random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_7B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL = "intfloat/multilingual-e5-large"
OUT_DIR     = "/home/sclab/moe_model/medquad_results"
NX_PATH     = os.path.join(OUT_DIR, "medquad_nexroute_7b.json")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES   = 200
TOP_K       = 10
SEED        = 42

KEYWORDS = {
    "LAW":  ["law","legal","court","statute","plaintiff","contract","judge","attorney"],
    "MED":  ["medical","patient","disease","treatment","clinical","diagnosis","symptom","drug",
             "health","therapy","medicine","physician","hospital","dose","side effect"],
    "SOC":  ["social","society","culture","economic","political","community","psychology"],
    "STEM": ["science","physics","chemical","biology","math","engineering","technology"],
}
SYSTEM_MAP = {
    "LAW": "You are a legal expert.", "MED": "You are a medical expert.",
    "SOC": "You are a social science expert.", "STEM": "You are a science expert.",
}

def keyword_route(query):
    q = query.lower()
    s = {d: sum(1 for k in kws if k in q) for d, kws in KEYWORDS.items()}
    return max(s, key=s.get)

def load_medquad():
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    items = [{"instruction": r["Question"], "gold_answer": r["Answer"], "qtype": r["qtype"]}
             for r in ds if r["Question"] and r["Answer"] and len(r["Answer"].split()) >= 10]
    random.seed(SEED)
    return random.sample(items, min(N_SAMPLES, len(items)))

def build_db(embed_model):
    files = {"LAW":"law_rewritten.json","MED":"med_rewritten.json",
             "SOC":"soc_rewritten.json","STEM":"stem_rewritten.json","CROSS":"cross_rewritten.json"}
    all_texts = []
    for _, fname in files.items():
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p): continue
        with open(p) as f: data = json.load(f)
        for d in data:
            out = d.get("output", d.get("gold_answer", ""))
            if any(k in out for k in ["The correct answer", "All answers:"]): continue
            all_texts.append(f"{d['instruction']} {out}")
    embs  = embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs.astype(np.float32))
    print(f"    통합 DB: {len(all_texts)}개")
    return all_texts, idx

def retrieve(query, embed_model, texts, index):
    emb = embed_model.encode([query]); emb /= np.linalg.norm(emb)
    _, I = index.search(emb.astype(np.float32), TOP_K)
    return [texts[i] for i in I[0]]

def generate(query, context, model, tokenizer, system):
    ctx = "\n".join([c[:200] for c in context])
    prompt = (f"{system}\n\nContext:\n{ctx}\n\nQuestion: {query}\n\n"
              f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:")
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    answer = re.sub(r"The correct answer is[^\n]*\n?", "", answer).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer: answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[FAILED]"

def compute_metrics(items, tag):
    golds = [x["gold"] for x in items]; preds = [x["pred"] for x in items]
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl  = np.mean([rscorer.score(g, p)["rougeL"].fmeasure for g, p in zip(golds, preds)]) * 100
    _, _, F1 = bert_score_fn(preds, golds, lang="en", model_type="roberta-large",
                             rescale_with_baseline=False, verbose=False, device=DEVICE)
    bs   = F1.mean().item() * 100
    sm   = SmoothingFunction().method1
    bleu = np.mean([sentence_bleu([nltk.word_tokenize(g.lower())], nltk.word_tokenize(p.lower()),
                    smoothing_function=sm) for g, p in zip(golds, preds)]) * 100
    print(f"  [{tag}] n={len(items)}  ROUGE-L={rl:.2f}  BERTScore={bs:.2f}  BLEU-4={bleu:.2f}")
    return {"rouge_l": round(float(rl),2), "bertscore": round(float(bs),2),
            "bleu4": round(float(bleu),2), "n": len(items)}

# ── 메인 ─────────────────────────────────────────────────────────────────────
medquad = load_medquad()
print(f"\nMedQuAD 샘플: {len(medquad)}개")

embed_model = SentenceTransformer(EMBED_MODEL)
texts, index = build_db(embed_model)

print("\n[*] Qwen2.5-7B-Instruct 로드 (4-bit NF4)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_7B, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
print("[*] 로드 완료!")

score_table = {}

# ── Naive RAG 7B ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}\n[Naive RAG 7B] MedQuAD 평가\n{'='*55}")
naive_results = []
for i, item in enumerate(medquad):
    ctx    = retrieve(item["instruction"], embed_model, texts, index)
    answer = generate(item["instruction"], ctx, model, tokenizer, "You are a helpful expert.")
    naive_results.append({"query": item["instruction"], "gold_answer": item["gold_answer"],
                          "pred_answer": answer})
    if (i+1) % 20 == 0:
        with open(os.path.join(OUT_DIR, "medquad_naive_rag_7b.json"), "w") as f:
            json.dump(naive_results, f, ensure_ascii=False, indent=2)
        print(f"  [{i+1}/{len(medquad)}] 중간 저장")

with open(os.path.join(OUT_DIR, "medquad_naive_rag_7b.json"), "w") as f:
    json.dump(naive_results, f, ensure_ascii=False, indent=2)

items_naive = [{"gold": r["gold_answer"], "pred": r["pred_answer"]} for r in naive_results
               if r["pred_answer"] not in ["", "[FAILED]"]]
score_table["Naive RAG 7B"] = {"metrics": compute_metrics(items_naive, "Naive RAG 7B"),
                               "routing_acc": None}

# ── Adaptive RAG 7B ──────────────────────────────────────────────────────────
print(f"\n{'='*55}\n[Adaptive RAG 7B] MedQuAD 평가\n{'='*55}")
adaptive_results = []
routing_correct  = 0
for i, item in enumerate(medquad):
    domain = keyword_route(item["instruction"])
    if domain == "MED": routing_correct += 1
    system = SYSTEM_MAP.get(domain, "You are a helpful expert.")
    ctx    = retrieve(item["instruction"], embed_model, texts, index)
    answer = generate(item["instruction"], ctx, model, tokenizer, system)
    adaptive_results.append({"query": item["instruction"], "gold_answer": item["gold_answer"],
                             "pred_answer": answer, "domain_pred": domain})
    if (i+1) % 20 == 0:
        with open(os.path.join(OUT_DIR, "medquad_adaptive_rag_7b.json"), "w") as f:
            json.dump(adaptive_results, f, ensure_ascii=False, indent=2)
        print(f"  [{i+1}/{len(medquad)}] 중간 저장")

with open(os.path.join(OUT_DIR, "medquad_adaptive_rag_7b.json"), "w") as f:
    json.dump(adaptive_results, f, ensure_ascii=False, indent=2)

ada_routing_acc = routing_correct / len(medquad) * 100
print(f"  [Adaptive RAG] 라우팅 정확도: {ada_routing_acc:.1f}%")
items_ada = [{"gold": r["gold_answer"], "pred": r["pred_answer"]} for r in adaptive_results
             if r["pred_answer"] not in ["", "[FAILED]"]]
score_table["Adaptive RAG 7B"] = {"metrics": compute_metrics(items_ada, "Adaptive RAG 7B"),
                                  "routing_acc": ada_routing_acc}

# ── NexRoute 기존 결과 불러와 메트릭 재계산 ───────────────────────────────────
print(f"\n{'='*55}\n[NexRoute 7B] 기존 결과 로드 & 메트릭 계산\n{'='*55}")
with open(NX_PATH) as f: nx_data = json.load(f)
items_nx = [{"gold": r["gold_answer"], "pred": r["pred_answer"]} for r in nx_data
            if r["pred_answer"] not in ["", "[FAILED]"]]
from collections import Counter
routing_bd = Counter(r["top1"] for r in nx_data)
nx_routing_acc = sum(1 for r in nx_data if r["top1"] == "MED") / len(nx_data) * 100
print(f"  [NexRoute] 라우팅 분포: {dict(routing_bd)}")
print(f"  [NexRoute] MED 라우팅 정확도: {nx_routing_acc:.1f}%")
score_table["NexRoute 7B"] = {"metrics": compute_metrics(items_nx, "NexRoute 7B"),
                              "routing_acc": nx_routing_acc}

# ── 최종 결과 저장 & 출력 ─────────────────────────────────────────────────────
final_path = os.path.join(OUT_DIR, "medquad_scores_7b.json")
with open(final_path, "w") as f:
    json.dump({"dataset": "MedQuAD (keivalya/MedQuad-MedicalQnADataset)",
               "n_samples": N_SAMPLES, "router": "DeBERTa-large (router_deberta_best.pt)",
               "models": score_table, "routing_breakdown_nexroute": dict(routing_bd)},
              f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"  MedQuAD 7B 평가 결과 (N={N_SAMPLES})")
print("=" * 80)
print(f"  {'Model':<22}  {'ROUGE-L':>8} {'BERTScore':>10} {'BLEU-4':>8} {'Routing Acc':>12}")
print("-" * 80)
order = ["NexRoute 7B", "Naive RAG 7B", "Adaptive RAG 7B"]
for name in order:
    if name not in score_table: continue
    v  = score_table[name]
    m  = v["metrics"]
    ra = v.get("routing_acc")
    ra_s = f"{ra:.1f}%" if ra is not None else "  N/A"
    def f(x): return f"{x:.2f}" if x is not None else "  N/A"
    mark = " ◀" if "NexRoute" in name else ""
    print(f"  {name:<22}  {f(m['rouge_l']):>8} {f(m['bertscore']):>10} {f(m['bleu4']):>8} {ra_s:>12}{mark}")
print("=" * 80)
print(f"\n저장: {final_path}")
