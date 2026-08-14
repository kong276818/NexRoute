"""
nexroute_resume_400.py
======================
NexRoute 재추론 재개: 인덱스 100-499 (100개 이미 완료)
기존 preds_500_nexroute.json (100개)에 이어서 저장
"""
import os, json, csv, re, gc
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

BASE_7B   = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR  = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR  = "/home/sclab/moe_model/processed_v2"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV = "/home/sclab/moe_model/preds_500.csv"
EMBED_NAME = "intfloat/multilingual-e5-large"
OUT_PATH  = "/home/sclab/moe_model/reinference_results/preds_500_nexroute.json"

TOP_K  = 10
SEED   = 1

import random
random.seed(SEED)
torch.manual_seed(SEED)

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",
    "MED":   "med_rewritten.json",
    "SOC":   "soc_rewritten.json",
    "STEM":  "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}

DOMAIN_SYSTEM = {
    "LAW":   "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":   "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":   "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM":  "You are a science and engineering expert. Answer accurately based on scientific principles.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains to answer accurately.",
}
SYSTEM_GENERIC = "You are a helpful expert. Provide a comprehensive analytical answer."

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

# ── 기존 결과 로드 ────────────────────────────────────────────────────────────
print("[*] 기존 NexRoute 결과 로드...")
with open(OUT_PATH) as f:
    results = json.load(f)
done_indices = {r["idx"] for r in results}
print(f"  이미 완료: {len(done_indices)}개 ({min(done_indices)}-{max(done_indices)})")

# ── GT 로드 ───────────────────────────────────────────────────────────────────
print("[*] GT 로드...")
with open(GT_PATH) as f:
    gt_list = json.load(f)

remaining = [(i, item) for i, item in enumerate(gt_list) if i not in done_indices]
print(f"  남은 샘플: {len(remaining)}개")

# ── 라우팅 캐시 로드 ──────────────────────────────────────────────────────────
print("[*] 라우팅 캐시 로드...")
routing_map = {}
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

# ── FAISS 인덱스 구축 ─────────────────────────────────────────────────────────
print("[*] FAISS 인덱스 구축...")
domain_texts = {}
domain_index = {}

for domain, fname in DOMAIN_FILES.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    texts = []
    for d in data:
        out = d.get("output", d.get("gold_answer", ""))
        if any(k in out for k in ["The correct answer", "All answers:"]):
            continue
        texts.append(f"{d['instruction']} {out}")
    embs = embed_model.encode(texts, batch_size=128, show_progress_bar=False)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    idx  = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    domain_texts[domain] = texts
    domain_index[domain] = idx
    print(f"  {domain}: {len(texts)}개")


def retrieve_partitioned(query, domains, is_cross, top_k=TOP_K):
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    if is_cross:
        results_list = []
        per = max(1, top_k // len(domains))
        for dom in domains:
            if dom not in domain_index:
                continue
            D, I = domain_index[dom].search(emb.astype(np.float32), per)
            for score, i in zip(D[0], I[0]):
                results_list.append((float(score), dom, domain_texts[dom][i]))
        results_list.sort(key=lambda x: -x[0])
        return [f"[{d}] {t}" for _, d, t in results_list[:top_k]]
    else:
        dom = domains[0]
        if dom not in domain_index:
            return []
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


def clean_answer(answer):
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'The correct holding is option\s*\d+\s*:\s*', '', answer).strip()
    answer = re.sub(r'The correct holding is\s*:\s*', '', answer).strip()
    for kw in ["Choices:", "All answers:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    answer = re.sub(r'\n\d+:.*', '', answer).strip()
    return answer if answer.strip() else "[FAILED]"


def _generate(prompt, max_new_tokens=300):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(base_model.device)
    with torch.no_grad():
        out = base_model.generate(
            **inp, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return clean_answer(answer)


def nexroute_generate(query, context, is_cross, domains):
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


# ── LoRA 로드 ─────────────────────────────────────────────────────────────────
print("[*] LoRA 어댑터 로드...")
lora_map = {
    "LAW":   "lora_law",
    "MED":   "lora_med",
    "SOC":   "lora_soc",
    "STEM":  "lora_stem",
    "CROSS": "lora_cross",
}
current_lora = [None]


def switch_lora(domain):
    global base_model
    if current_lora[0] == domain:
        return
    lora_path = os.path.join(LORA_DIR, lora_map[domain])
    if current_lora[0] is None:
        base_model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=domain)
    else:
        base_model.load_adapter(lora_path, adapter_name=domain)
        base_model.set_adapter(domain)
    current_lora[0] = domain
    print(f"  LoRA 전환: {domain}")


# ── 재추론 루프 ───────────────────────────────────────────────────────────────
start_idx = remaining[0][0] if remaining else 'N/A'
print(f"\n[*] NexRoute 재추론 시작 (인덱스 {start_idx} ~)")

for step, (i, item) in enumerate(remaining):
    query     = item["instruction"]
    gold      = item["gold_answer"]
    domain_gt = item["expected_domain"]

    r           = routing_map.get(i, {})
    pred_domain = r.get('pred_domain', 'STEM')
    is_cross    = r.get('is_cross', False)
    top1        = r.get('top1', pred_domain)
    top2        = r.get('top2', pred_domain)
    domains     = [top1, top2] if is_cross else [top1]

    lora_target = "CROSS" if is_cross else top1
    switch_lora(lora_target)

    ctx    = retrieve_partitioned(query, domains, is_cross)
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

    total_done = len(done_indices) + step + 1
    if total_done % 50 == 0:
        sorted_results = sorted(results, key=lambda x: x["idx"])
        with open(OUT_PATH, "w") as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)
        print(f"  [{total_done}/500] 중간 저장")

# ── 최종 저장 ─────────────────────────────────────────────────────────────────
sorted_results = sorted(results, key=lambda x: x["idx"])
with open(OUT_PATH, "w") as f:
    json.dump(sorted_results, f, ensure_ascii=False, indent=2)

print(f"\n[✓] NexRoute 재추론 완료: {len(sorted_results)}개")
print(f"  저장: {OUT_PATH}")

gc.collect()
torch.cuda.empty_cache()
