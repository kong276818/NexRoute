"""
run_reinference_nexroute_v2.py
================================
NexRoute v2 어댑터로 500샘플 재추론.
어댑터 변경:
  LAW   → lora_law_v4  (unchanged from lawv4 baseline)
  MED   → lora_med_v2  (CROSS 30% mix-in 재학습)
  SOC   → lora_soc_v2  (CROSS 30% mix-in 재학습)
  STEM  → lora_stem_v2 (CROSS 30% mix-in 재학습)
  CROSS → lora_cross   (unchanged)

출력: reinference_results/preds_500_nexroute_v2.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# run_reinference_500.py에서 공통 코드 재사용
# lora_map만 v2로 교체

import json, re, csv, torch, random, gc
import numpy as np
from peft import PeftModel
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
import faiss

BASE_7B    = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR   = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV  = "/home/sclab/moe_model/results/preds_500.csv"
EMBED_NAME = "intfloat/multilingual-e5-large"
OUT_DIR    = "/home/sclab/moe_model/reinference_results"
OUT_PATH   = os.path.join(OUT_DIR, "preds_500_nexroute_v2.json")
os.makedirs(OUT_DIR, exist_ok=True)

SEED   = 1
TOP_K  = 10
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

# ── GT 로드 ───────────────────────────────────────────────────────────────────
print("[*] GT 로드...")
with open(GT_PATH) as f:
    gt_list = json.load(f)
print(f"  GT: {len(gt_list)}개")

# ── preds_500.csv 라우팅 로드 ──────────────────────────────────────────────────
print("[*] 라우팅 CSV 로드...")
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
print(f"  라우팅: {len(routing_map)}개")

# ── 임베딩 모델 + FAISS 구축 (원본 스크립트와 동일 방식) ──────────────────────────
print("[*] 임베딩 모델 로드 (cuda)...")
embedder = SentenceTransformer(EMBED_NAME, device="cuda")

print("[*] FAISS 인덱스 구축...")
domain_texts = {}
index_map    = {}
MCQ_SKIP_KEYS = ["The correct answer", "All answers:"]

for domain, fname in DOMAIN_FILES.items():
    with open(os.path.join(DATA_DIR, fname)) as f:
        data = json.load(f)
    texts = []
    for d in data:
        out = d.get("output", d.get("gold_answer", ""))
        if any(k in out for k in MCQ_SKIP_KEYS):
            continue
        texts.append(f"{d['instruction']} {out}")
    embs = embedder.encode(texts, batch_size=128, show_progress_bar=False)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    idx  = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    domain_texts[domain] = texts
    index_map[domain]    = idx
    print(f"  {domain}: {len(texts)}개")

print("[*] FAISS 구축 완료")

# ── 생성 모델 로드 ────────────────────────────────────────────────────────────
print("[*] 모델 로드 (4-bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="cuda:0",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager",
)
base_model.eval()

# v2 어댑터 매핑 (핵심 변경)
lora_map = {
    "LAW":   "lora_law_v4",   # 이미 최적화된 LAW 어댑터
    "MED":   "lora_med_v2",   # CROSS 30% mix-in 재학습
    "SOC":   "lora_soc_v2",   # CROSS 30% mix-in 재학습
    "STEM":  "lora_stem_v2",  # CROSS 30% mix-in 재학습
    "CROSS": "lora_cross",    # 변경 없음
}
current_lora = [None]


def switch_lora(domain: str):
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


def retrieve_partitioned(query: str, domains: list, is_cross: bool):
    emb = embedder.encode([query])
    emb = emb / np.linalg.norm(emb)
    emb = emb.astype(np.float32)
    if is_cross:
        ctx_parts = []
        per = max(1, TOP_K // len(domains))
        for dom in domains:
            if dom not in index_map:
                continue
            D, I = index_map[dom].search(emb, per)
            for score, j in zip(D[0], I[0]):
                if 0 <= j < len(domain_texts[dom]):
                    ctx_parts.append((score, domain_texts[dom][j]))
        ctx_parts.sort(key=lambda x: -x[0])
        return [t for _, t in ctx_parts[:TOP_K]]
    else:
        dom = domains[0]
        if dom in index_map:
            D, I = index_map[dom].search(emb, TOP_K)
            return [domain_texts[dom][j] for j in I[0] if 0 <= j < len(domain_texts[dom])]
        return []


def clean_answer(answer: str) -> str:
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[FAILED]"


def nexroute_generate(query: str, context: list, is_cross: bool, domains: list) -> str:
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
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc  = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        out = base_model.generate(
            **enc, max_new_tokens=max_toks,
            temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return clean_answer(answer)


def save_checkpoint(results, path):
    with open(path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── 체크포인트 확인 ────────────────────────────────────────────────────────────
existing_results = []
start_idx = 0
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        existing_results = json.load(f)
    start_idx = len(existing_results)
    if start_idx >= len(gt_list):
        print(f"[+] 이미 완료: {OUT_PATH}")
        exit(0)
    print(f"[*] 체크포인트 재개: {start_idx}/{len(gt_list)}")

# ── 추론 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"NexRoute v2 추론 ({start_idx}~{len(gt_list)})")
print(f"{'='*60}")

results = list(existing_results)
for i, item in enumerate(gt_list[start_idx:], start=start_idx):
    query     = item["instruction"]
    gold      = item["gold_answer"]
    domain_gt = item["expected_domain"]

    r = routing_map.get(i, {})
    pred_domain = r.get('pred_domain', 'STEM')
    is_cross    = r.get('is_cross', False)
    top1        = r.get('top1', pred_domain)
    top2        = r.get('top2', pred_domain)
    domains     = [top1, top2] if is_cross else [top1]

    lora_target = "CROSS" if is_cross else top1
    switch_lora(lora_target)

    context = retrieve_partitioned(query, domains, is_cross)
    answer  = nexroute_generate(query, context, is_cross, domains)

    results.append({
        "idx":            i,
        "instruction":    query,
        "gold_answer":    gold,
        "expected_domain": domain_gt,
        "is_cross":       (domain_gt == "CROSS"),
        "pred_domain":    pred_domain,
        "pred_is_cross":  is_cross,
        "pred_domains":   domains,
        "pred_answer":    answer,
    })

    if (i + 1) % 50 == 0:
        save_checkpoint(results, OUT_PATH)
        print(f"  [{i+1}/500] 중간 저장")

save_checkpoint(results, OUT_PATH)
print(f"\n[+] 완료: {OUT_PATH} ({len(results)}개)")
