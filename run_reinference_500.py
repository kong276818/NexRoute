"""
run_reinference_500.py
======================
500샘플 전체 재추론 (seed=1)
시스템: Naive RAG / TF-IDF Router / HyDE / Simplified CRAG / NexRoute 7B

출력:
  /home/sclab/moe_model/reinference_results/
    preds_500_naive_rag.json
    preds_500_tfidf_router.json
    preds_500_hyde.json
    preds_500_crag.json
    preds_500_nexroute.json

규칙:
  - 모든 수치는 예측 파일에서 계산; 하드코딩 금지
  - 불일치 발생 시 그대로 보고
  - seed=1 고정
  - 체크포인트: 50샘플마다 중간 저장
"""
import os, json, re, csv, torch, random, gc
import numpy as np
from pathlib import Path
from peft import PeftModel
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
import faiss

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_7B    = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR   = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV  = "/home/sclab/moe_model/preds_500.csv"   # 사전 계산된 라우팅
EMBED_NAME = "intfloat/multilingual-e5-large"
OUT_DIR    = "/home/sclab/moe_model/reinference_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEED       = 1
TOP_K      = 10
CRAG_HIGH  = 0.70
CRAG_LOW   = 0.45

random.seed(SEED)
torch.manual_seed(SEED)

# ── 도메인 파일 ───────────────────────────────────────────────────────────────
DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

# ── 시스템 프롬프트 ────────────────────────────────────────────────────────────
DOMAIN_SYSTEM = {
    "LAW":   "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":   "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":   "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM":  "You are a science and engineering expert. Answer accurately based on scientific principles.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains to answer accurately.",
}
SYSTEM_GENERIC = "You are a helpful expert. Provide a comprehensive analytical answer."

TFIDF_KEYWORDS = {
    'LAW':  ['law','legal','court','statute','plaintiff','contract','judge','attorney'],
    'MED':  ['medical','patient','disease','treatment','clinical','diagnosis','symptom','drug'],
    'SOC':  ['social','society','culture','economic','political','community','psychology'],
    'STEM': ['science','physics','chemical','biology','math','engineering','technology'],
}

# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def keyword_route(query: str) -> str:
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in TFIDF_KEYWORDS.items()}
    return max(scores, key=scores.get)


def clean_answer(answer: str) -> str:
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'The correct holding is option\s*\d+\s*:\s*', '', answer).strip()
    answer = re.sub(r'The correct holding is\s*:\s*', '', answer).strip()
    for kw in ["Choices:", "All answers:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    answer = re.sub(r'\n\d+:.*', '', answer).strip()
    return answer if answer.strip() else "[FAILED]"


# ── GT 로드 ───────────────────────────────────────────────────────────────────
print("[*] GT 로드...")
with open(GT_PATH) as f:
    gt_list = json.load(f)
print(f"  GT: {len(gt_list)}개")

# ── preds_500.csv 라우팅 로드 (NexRoute용) ─────────────────────────────────────
print("[*] preds_500.csv 라우팅 로드...")
routing_map = {}   # idx -> {pred_domain, is_cross}
with open(PREDS_CSV) as f:
    for row in csv.DictReader(f):
        idx = int(row['sample_id'])
        routing_map[idx] = {
            'pred_domain': row['pred_domain_nexroute'],
            'is_cross':    row['is_cross'].strip().lower() == 'true',
            'top1':        row['top1'],
            'top2':        row['top2'],
        }
print(f"  라우팅 캐시: {len(routing_map)}개")

# ── 임베딩 모델 ───────────────────────────────────────────────────────────────
print("[*] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_NAME, device="cuda")

# ── FAISS 인덱스 구축 (도메인별 파티션 + 통합) ──────────────────────────────────
print("[*] FAISS 인덱스 구축...")
domain_texts  = {}   # domain -> [str, ...]
domain_index  = {}   # domain -> faiss.Index

for domain, fname in DOMAIN_FILES.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    texts = []
    for d in data:
        out = d.get("output", d.get("gold_answer", ""))
        if any(k in out for k in ["The correct answer", "All answers:"]):
            continue
        texts.append(f"{d['instruction']} {out}")
    embs  = embed_model.encode(texts, batch_size=128, show_progress_bar=False)
    embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    idx   = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    domain_texts[domain] = texts
    domain_index[domain] = idx
    print(f"  {domain}: {len(texts)}개")

# 통합 인덱스 (Naive RAG / HyDE / CRAG 용)
all_texts = []
for domain in ["LAW", "MED", "SOC", "STEM", "CROSS"]:
    all_texts.extend(domain_texts[domain])
all_embs  = embed_model.encode(all_texts, batch_size=128, show_progress_bar=True)
all_embs  = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)
uni_index = faiss.IndexFlatIP(all_embs.shape[1])
uni_index.add(all_embs.astype(np.float32))
print(f"  UNIFIED: {len(all_texts)}개")


def retrieve_unified(query: str, top_k: int = TOP_K):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    D, I = uni_index.search(emb.astype(np.float32), top_k)
    return [all_texts[i] for i in I[0]], D[0]


def retrieve_partitioned(query: str, domains: list, is_cross: bool, top_k: int = TOP_K):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    if is_cross:
        results = []
        per = max(1, top_k // len(domains))
        for dom in domains:
            if dom not in domain_index:
                continue
            D, I = domain_index[dom].search(emb.astype(np.float32), per)
            for score, i in zip(D[0], I[0]):
                results.append((float(score), dom, domain_texts[dom][i]))
        results.sort(key=lambda x: -x[0])
        return [f"[{d}] {t}" for _, d, t in results[:top_k]]
    else:
        dom = domains[0]
        if dom not in domain_index:
            return retrieve_unified(query)[0]
        D, I = domain_index[dom].search(emb.astype(np.float32), top_k)
        return [domain_texts[dom][i] for i in I[0]]


# ── 언어 모델 로드 ────────────────────────────────────────────────────────────
print("\n[*] Qwen2.5-7B-Instruct 4bit 로드...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager"
)
print("[*] 베이스 모델 로드 완료")


def _generate(prompt: str, max_new_tokens: int = 300) -> str:
    msgs  = [{"role": "user", "content": prompt}]
    text  = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp   = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(base_model.device)
    with torch.no_grad():
        out = base_model.generate(
            **inp, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return clean_answer(answer)


def generate_from_context(query: str, context: list, system: str,
                          max_new_tokens: int = 300) -> str:
    ctx = "\n".join([c[:200] for c in context])
    prompt = (
        f"{system}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive analytical answer in flowing paragraph form.\n\nAnswer:"
    )
    return _generate(prompt, max_new_tokens=max_new_tokens)


def save_checkpoint(results: list, path: str):
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Naive RAG 7B
# ══════════════════════════════════════════════════════════════════════════════
def run_naive_rag():
    name     = "naive_rag"
    out_path = os.path.join(OUT_DIR, f"preds_500_{name}.json")
    print(f"\n{'='*60}\n[1/5] Naive RAG 7B\n{'='*60}")

    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing) == len(gt_list):
            print(f"  이미 완료: {out_path}")
            return existing

    results = []
    for i, item in enumerate(gt_list):
        query  = item["instruction"]
        gold   = item["gold_answer"]
        domain = item["expected_domain"]
        ctx, _ = retrieve_unified(query)
        answer = generate_from_context(query, ctx, SYSTEM_GENERIC)
        results.append({
            "idx": i,
            "instruction": query,
            "gold_answer": gold,
            "expected_domain": domain,
            "is_cross": (domain == "CROSS"),
            "pred_domain": "UNIFIED",
            "pred_answer": answer,
        })
        if (i + 1) % 50 == 0:
            save_checkpoint(results, out_path)
            print(f"  [{i+1}/500] 중간 저장")

    save_checkpoint(results, out_path)
    print(f"  [완료] {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. TF-IDF Domain Router 7B
# ══════════════════════════════════════════════════════════════════════════════
def run_tfidf_router():
    name     = "tfidf_router"
    out_path = os.path.join(OUT_DIR, f"preds_500_{name}.json")
    print(f"\n{'='*60}\n[2/5] TF-IDF Domain Router 7B\n{'='*60}")

    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing) == len(gt_list):
            print(f"  이미 완료: {out_path}")
            return existing

    results = []
    for i, item in enumerate(gt_list):
        query      = item["instruction"]
        gold       = item["gold_answer"]
        domain_gt  = item["expected_domain"]
        pred_dom   = keyword_route(query)
        system     = DOMAIN_SYSTEM.get(pred_dom, SYSTEM_GENERIC)
        # TF-IDF 라우터는 CROSS 없이 단일 도메인으로 라우팅 → 파티션 검색
        ctx        = retrieve_partitioned(query, [pred_dom], is_cross=False)
        answer     = generate_from_context(query, ctx, system)
        results.append({
            "idx": i,
            "instruction": query,
            "gold_answer": gold,
            "expected_domain": domain_gt,
            "is_cross": (domain_gt == "CROSS"),
            "pred_domain": pred_dom,
            "pred_answer": answer,
        })
        if (i + 1) % 50 == 0:
            save_checkpoint(results, out_path)
            print(f"  [{i+1}/500] 중간 저장")

    save_checkpoint(results, out_path)
    print(f"  [완료] {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. HyDE 7B
# ══════════════════════════════════════════════════════════════════════════════
def run_hyde():
    name     = "hyde"
    out_path = os.path.join(OUT_DIR, f"preds_500_{name}.json")
    print(f"\n{'='*60}\n[3/5] HyDE 7B\n{'='*60}")

    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing) == len(gt_list):
            print(f"  이미 완료: {out_path}")
            return existing

    results = []
    for i, item in enumerate(gt_list):
        query     = item["instruction"]
        gold      = item["gold_answer"]
        domain_gt = item["expected_domain"]

        # 1. 가상 답변 생성
        hypo_prompt = (
            f"{SYSTEM_GENERIC}\n\n"
            f"Question: {query}\n\n"
            f"Write a short, factually plausible hypothetical answer (2-3 sentences) "
            f"that could appear in an academic reference text. "
            f"Do NOT use multiple choice format.\n\nAnswer:"
        )
        hypo_doc = _generate(hypo_prompt, max_new_tokens=150)

        # 2. 가상 답변 임베딩으로 통합 검색
        hypo_emb = embed_model.encode([hypo_doc])
        hypo_emb = hypo_emb / np.linalg.norm(hypo_emb)
        D, I     = uni_index.search(hypo_emb.astype(np.float32), TOP_K)
        ctx      = [all_texts[j] for j in I[0]]

        # 3. 최종 생성
        answer = generate_from_context(query, ctx, SYSTEM_GENERIC)

        results.append({
            "idx": i,
            "instruction": query,
            "gold_answer": gold,
            "expected_domain": domain_gt,
            "is_cross": (domain_gt == "CROSS"),
            "pred_domain": "UNIFIED",
            "hypo_doc": hypo_doc,
            "pred_answer": answer,
        })
        if (i + 1) % 50 == 0:
            save_checkpoint(results, out_path)
            print(f"  [{i+1}/500] 중간 저장")

    save_checkpoint(results, out_path)
    print(f"  [완료] {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. Simplified CRAG 7B
# ══════════════════════════════════════════════════════════════════════════════
def crag_refine(docs: list, query: str) -> list:
    q_words = set(query.lower().split())
    strips  = []
    for doc in docs:
        for sent in re.split(r'(?<=[.!?])\s+', doc):
            sent = sent.strip()
            if not sent:
                continue
            if len(q_words & set(sent.lower().split())) >= 2:
                strips.append(sent)
    return strips[:TOP_K] if strips else docs[:3]


def run_crag():
    name     = "crag"
    out_path = os.path.join(OUT_DIR, f"preds_500_{name}.json")
    print(f"\n{'='*60}\n[4/5] Simplified CRAG 7B\n{'='*60}")

    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing) == len(gt_list):
            print(f"  이미 완료: {out_path}")
            return existing

    results    = []
    judgments  = {"correct": 0, "ambiguous": 0, "incorrect": 0}

    for i, item in enumerate(gt_list):
        query     = item["instruction"]
        gold      = item["gold_answer"]
        domain_gt = item["expected_domain"]

        # 1. 초기 검색
        ctx, scores = retrieve_unified(query)
        max_score   = float(scores[0]) if len(scores) > 0 else 0.0

        # 2. 관련성 판정
        if max_score >= CRAG_HIGH:
            judgment   = "correct"
            final_ctx  = ctx
        elif max_score < CRAG_LOW:
            judgment = "incorrect"
            # 파라메트릭 폴백
            fallback_prompt = (
                f"{SYSTEM_GENERIC}\n\n"
                f"The retrieved documents are not relevant. "
                f"Generate factual background knowledge about the following question "
                f"(3-4 sentences, no multiple choice).\n\n"
                f"Question: {query}\n\nKnowledge:"
            )
            fallback   = _generate(fallback_prompt, max_new_tokens=200)
            final_ctx  = [fallback] + ctx[:3]
        else:
            judgment   = "ambiguous"
            final_ctx  = crag_refine(ctx, query)

        judgments[judgment] += 1
        answer = generate_from_context(query, final_ctx, SYSTEM_GENERIC)

        results.append({
            "idx": i,
            "instruction": query,
            "gold_answer": gold,
            "expected_domain": domain_gt,
            "is_cross": (domain_gt == "CROSS"),
            "pred_domain": "UNIFIED",
            "crag_judgment": judgment,
            "crag_max_score": float(max_score),
            "pred_answer": answer,
        })
        if (i + 1) % 50 == 0:
            save_checkpoint(results, out_path)
            print(f"  [{i+1}/500] 중간 저장  판정: {judgments}")

    save_checkpoint(results, out_path)
    print(f"  [완료] {out_path}  최종 판정: {judgments}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. NexRoute 7B
# ══════════════════════════════════════════════════════════════════════════════
# Few-shot (pipeline.py 동일)
FEW_SHOT = {
    "LAW": (
        "Q: What is the doctrine of promissory estoppel?\n"
        "A: Promissory estoppel is a legal doctrine that prevents a party from "
        "withdrawing a promise when another party has reasonably relied on it to "
        "their detriment. The doctrine requires a clear and definite promise, "
        "reasonable reliance by the promisee, and actual detriment suffered as a "
        "result of that reliance. Courts apply this principle to enforce promises "
        "even without formal consideration, thereby ensuring fairness and preventing "
        "injustice in contractual relationships."
    ),
    "MED": (
        "Q: How does the renin-angiotensin system regulate blood pressure?\n"
        "A: The renin-angiotensin system regulates blood pressure through a cascade "
        "of enzymatic reactions initiated when the kidneys detect low blood pressure "
        "or reduced sodium levels. Renin converts angiotensinogen to angiotensin I, "
        "which is subsequently converted to angiotensin II by ACE in the lungs. "
        "Angiotensin II causes vasoconstriction and stimulates aldosterone secretion, "
        "increasing sodium retention and blood volume, ultimately elevating blood pressure."
    ),
    "SOC": (
        "Q: How does social stratification affect educational outcomes?\n"
        "A: Social stratification significantly influences educational outcomes by "
        "creating unequal access to resources, opportunities, and cultural capital. "
        "Children from higher socioeconomic backgrounds benefit from better-funded "
        "schools, private tutoring, and enriched home environments that foster "
        "academic achievement. Conversely, those from lower strata face systemic "
        "barriers including underfunded schools and economic pressures that hinder "
        "educational attainment, perpetuating intergenerational inequality."
    ),
    "STEM": (
        "Q: How does CRISPR-Cas9 technology work?\n"
        "A: CRISPR-Cas9 is a revolutionary gene-editing technology that uses a "
        "guide RNA to direct the Cas9 endonuclease to specific DNA sequences. "
        "The guide RNA binds complementarily to the target DNA, enabling Cas9 to "
        "create a precise double-strand break at the desired location. The cell "
        "then repairs this break through either non-homologous end joining, which "
        "can disrupt gene function, or homology-directed repair, which can introduce "
        "specific genetic modifications with high precision."
    ),
    "CROSS": (
        "Q: How might advances in neuroscience impact criminal justice policy?\n"
        "A: Advances in neuroscience are profoundly reshaping criminal justice policy "
        "by providing empirical insights into the biological underpinnings of behavior, "
        "decision-making, and culpability. Neuroimaging studies reveal that certain "
        "criminal behaviors may be associated with structural or functional brain "
        "abnormalities, raising questions about moral responsibility and the "
        "appropriate balance between punishment and rehabilitation. Policymakers must "
        "therefore integrate neuroscientific evidence with established legal principles "
        "to develop more nuanced, evidence-based approaches to sentencing and "
        "treatment that reflect both individual accountability and neurological realities."
    ),
}


def nexroute_generate(query: str, context: list, is_cross: bool, domains: list) -> str:
    """pipeline.py generate()와 동일한 로직."""
    ctx_text       = "\n".join([c[:200] for c in context])
    primary_domain = "CROSS" if is_cross else domains[0]

    if is_cross:
        system   = (f"You are a multidisciplinary expert specializing in "
                    f"{' and '.join(domains)}. "
                    f"Integrate knowledge from both domains to answer accurately.")
        few_shot = FEW_SHOT["CROSS"]
    else:
        system   = DOMAIN_SYSTEM.get(domains[0], SYSTEM_GENERIC)
        few_shot = FEW_SHOT.get(domains[0], FEW_SHOT["STEM"])

    if primary_domain == "LAW":
        domain_instruction = (
            "Provide a comprehensive explanation of the legal holding, "
            "principle, or rule. Describe the court's reasoning, the "
            "applicable legal standard, any relevant exceptions, and the "
            "practical implications. Write as a detailed legal analysis "
            "in full paragraph form."
        )
        max_toks = 800
    else:
        domain_instruction = (
            "provide a comprehensive analytical answer with specific "
            "mechanisms and implications"
        )
        max_toks = 512

    prompt = (
        f"{system}\n\n"
        f"Here is an example of a high-quality answer:\n"
        f"{few_shot}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Following the example style above, {domain_instruction}. "
        f"Do NOT use multiple choice format. "
        f"Do NOT list choices A, B, C, D. "
        f"Do NOT begin with 'The correct answer is' or 'The correct holding is option'. "
        f"Write in flowing paragraph form only.\n\n"
        f"Answer:"
    )
    return _generate(prompt, max_new_tokens=max_toks)


def run_nexroute():
    name     = "nexroute"
    out_path = os.path.join(OUT_DIR, f"preds_500_{name}.json")
    print(f"\n{'='*60}\n[5/5] NexRoute 7B\n{'='*60}")

    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        if len(existing) == len(gt_list):
            print(f"  이미 완료: {out_path}")
            return existing

    # LoRA 로드
    print("[*] LoRA 어댑터 로드...")
    lora_map = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc",
                "STEM":"lora_stem","CROSS":"lora_cross"}
    current_lora = [None]   # 리스트로 감싸 클로저에서 수정 가능

    def switch_lora(domain: str):
        global base_model
        if current_lora[0] == domain:
            return
        lora_path = os.path.join(LORA_DIR, lora_map[domain])
        if current_lora[0] is None:
            base_model = PeftModel.from_pretrained(base_model, lora_path,
                                                   adapter_name=domain)
        else:
            base_model.load_adapter(lora_path, adapter_name=domain)
            base_model.set_adapter(domain)
        current_lora[0] = domain
        print(f"  LoRA 전환: {domain}")

    results = []
    for i, item in enumerate(gt_list):
        query     = item["instruction"]
        gold      = item["gold_answer"]
        domain_gt = item["expected_domain"]

        # 사전 계산된 라우팅 사용
        r = routing_map.get(i, {})
        pred_domain = r.get('pred_domain', 'STEM')
        is_cross    = r.get('is_cross', False)
        top1        = r.get('top1', pred_domain)
        top2        = r.get('top2', pred_domain)
        domains     = [top1, top2] if is_cross else [top1]

        lora_target = "CROSS" if is_cross else top1
        switch_lora(lora_target)

        ctx = retrieve_partitioned(query, domains, is_cross)
        answer = nexroute_generate(query, ctx, is_cross, domains)

        results.append({
            "idx": i,
            "instruction": query,
            "gold_answer": gold,
            "expected_domain": domain_gt,
            "is_cross": (domain_gt == "CROSS"),
            "pred_domain": pred_domain,
            "pred_is_cross": is_cross,
            "pred_domains": domains,
            "pred_answer": answer,
        })

        if (i + 1) % 50 == 0:
            save_checkpoint(results, out_path)
            print(f"  [{i+1}/500] 중간 저장")

    save_checkpoint(results, out_path)
    print(f"  [완료] {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    # 특정 시스템만 실행 가능: python run_reinference_500.py naive_rag
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    runners = {
        "naive_rag":    run_naive_rag,
        "tfidf_router": run_tfidf_router,
        "hyde":         run_hyde,
        "crag":         run_crag,
        "nexroute":     run_nexroute,
    }

    if target == "all":
        order = ["naive_rag", "tfidf_router", "hyde", "crag", "nexroute"]
    elif target in runners:
        order = [target]
    else:
        print(f"ERROR: 알 수 없는 시스템 '{target}'. 선택: {list(runners.keys())}")
        sys.exit(1)

    for sys_name in order:
        runners[sys_name]()
        torch.cuda.empty_cache()
        gc.collect()

    print("\n[✓] 전체 추론 완료")
    print(f"출력 위치: {OUT_DIR}")
