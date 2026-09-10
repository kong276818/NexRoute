"""
reinfer_thresh010.py
====================
CROSS threshold를 0.39 → 0.10 (diff 기준)으로 낮춰서
17개 영향 샘플만 재생성.

기반: preds_500_nexroute_lawv4.json (LAW v4 포함 500샘플)
출력: reinference_results/preds_500_nexroute_thresh010.json

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/reinfer_thresh010.py
"""
import os, json, re, csv, torch, random, gc
import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

BASE_7B    = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR   = "/home/sclab/moe_model/lora_weights_7b_v3"
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV  = "/home/sclab/moe_model/results/preds_500.csv"
BASE_JSON  = "/home/sclab/moe_model/reinference_results/preds_500_nexroute_lawv4.json"
OUT_PATH   = "/home/sclab/moe_model/reinference_results/preds_500_nexroute_thresh010.json"
EMBED_NAME = "intfloat/multilingual-e5-large"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"

NEW_CROSS_THRESH = 0.10   # top1-top2 diff < 이 값이면 CROSS
SEED = 1
TOP_K = 10
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

FEW_SHOT = {
    "LAW": (
        "Q: What is the legal standard for negligence?\n"
        "A: Negligence is established by proving four elements: duty, breach, causation, and damages. "
        "The defendant must owe a legal duty of care to the plaintiff, breach that duty by failing to "
        "act as a reasonably prudent person would under similar circumstances, and that breach must "
        "directly cause the plaintiff's damages."
    ),
    "MED": (
        "Q: How does aspirin reduce fever?\n"
        "A: Aspirin reduces fever by inhibiting the cyclooxygenase (COX) enzymes, which decreases "
        "prostaglandin synthesis. Prostaglandins act on the hypothalamus to raise the body temperature "
        "set point, so their inhibition allows the body temperature to return to normal levels."
    ),
    "SOC": (
        "Q: What is social capital and why does it matter?\n"
        "A: Social capital refers to the networks of relationships among people who live and work in "
        "a particular society, enabling that society to function effectively. It encompasses trust, "
        "reciprocity, and norms of cooperation that facilitate collective action and economic prosperity."
    ),
    "STEM": (
        "Q: How does CRISPR-Cas9 technology work?\n"
        "A: CRISPR-Cas9 is a revolutionary gene-editing technology that uses a guide RNA to direct "
        "the Cas9 endonuclease to specific DNA sequences. The guide RNA binds complementarily to the "
        "target DNA, enabling Cas9 to create a precise double-strand break at the desired location."
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

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("[*] GT 로드...")
with open(GT_PATH) as f:
    gt_list = json.load(f)

print("[*] 기존 preds (lawv4) 로드...")
with open(BASE_JSON) as f:
    base_preds = json.load(f)
base_map = {p["idx"]: p for p in base_preds}

print("[*] 라우팅 CSV 로드...")
csv_rows = list(csv.DictReader(open(PREDS_CSV)))

def get_diff(r):
    scores = sorted([float(r['s_law']), float(r['s_med']),
                     float(r['s_soc']), float(r['s_stem'])], reverse=True)
    return scores[0] - scores[1]

# 새 threshold로 라우팅 재계산 + 영향 샘플 식별
new_routing = {}
changed_idxs = set()
for r in csv_rows:
    idx = int(r['sample_id'])
    diff = get_diff(r)
    old_cross = r['is_cross'].strip().lower() == 'true'
    new_cross = diff < NEW_CROSS_THRESH
    top1 = r['top1']
    top2 = r['top2']
    new_routing[idx] = {
        'is_cross': new_cross,
        'top1': top1,
        'top2': top2,
        'pred_domain': 'CROSS' if new_cross else top1,
    }
    if old_cross != new_cross:
        changed_idxs.add(idx)

print(f"  라우팅 변경 샘플: {sorted(changed_idxs)} ({len(changed_idxs)}개)")

if not changed_idxs:
    print("변경 없음. 종료.")
    exit(0)

# ── 임베딩 / FAISS ─────────────────────────────────────────────────────────────
print("[*] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_NAME, device="cuda")

print("[*] FAISS 인덱스 구축...")
domain_texts = {}
domain_index = {}
for domain, fname in DOMAIN_FILES.items():
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  skip {fname}")
        continue
    with open(fpath) as f:
        items = json.load(f)
    texts = [it.get("output", it.get("text", ""))[:300] for it in items]
    domain_texts[domain] = texts
    embs = embed_model.encode(texts, batch_size=256, show_progress_bar=False,
                               normalize_embeddings=True).astype("float32")
    idx_faiss = faiss.IndexFlatIP(embs.shape[1])
    idx_faiss.add(embs)
    domain_index[domain] = idx_faiss
    print(f"  {domain}: {len(texts)}개")

def retrieve(query, domains, is_cross, top_k=TOP_K):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    results = []
    for dom in domains:
        if dom not in domain_index:
            continue
        D, I = domain_index[dom].search(q_emb, top_k)
        for score, i in zip(D[0], I[0]):
            if i >= 0 and i < len(domain_texts[dom]):
                results.append(domain_texts[dom][i])
    return results[:top_k * len(domains)]

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
print("[*] 모델 로드 (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True,
                                           local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_7B, quantization_config=bnb, device_map="cuda:0",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager")
base_model.eval()

# LoRA 관리
LORA_MAP = {"LAW": "lora_law_v4", "MED": "lora_med", "SOC": "lora_soc",
            "STEM": "lora_stem", "CROSS": "lora_cross"}
current_lora = [None]

def switch_lora(domain):
    global base_model
    if current_lora[0] == domain:
        return
    lora_path = os.path.join(LORA_DIR, LORA_MAP[domain])
    if current_lora[0] is None:
        base_model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=domain)
    else:
        base_model.load_adapter(lora_path, adapter_name=domain)
        base_model.set_adapter(domain)
    current_lora[0] = domain
    print(f"  → LoRA: {domain}")


def _generate(prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=4096).to(base_model.device)
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   temperature=0.1, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()


def clean_answer(answer):
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'The correct holding is option\s*\d+\s*:\s*', '', answer).strip()
    for kw in ["Choices:", "All answers:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer.strip() else "[FAILED]"


def nexroute_generate(query, context, is_cross, domains):
    ctx_text = "\n".join([c[:200] for c in context])
    primary_domain = "CROSS" if is_cross else domains[0]

    if is_cross:
        system = (f"You are a multidisciplinary expert specializing in "
                  f"{' and '.join(domains)}. "
                  f"Integrate knowledge from both domains to answer accurately.")
        few_shot = FEW_SHOT["CROSS"]
    else:
        system = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")
        few_shot = FEW_SHOT.get(domains[0], FEW_SHOT["STEM"])

    if primary_domain == "LAW":
        domain_instruction = (
            "Provide a comprehensive explanation of the legal holding, "
            "principle, or rule. Write as a detailed legal analysis in full paragraph form."
        )
        max_toks = 800
    else:
        domain_instruction = (
            "provide a comprehensive analytical answer with specific mechanisms and implications"
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
    raw = _generate(prompt, max_new_tokens=max_toks)
    return clean_answer(raw)


# ── 재생성 ─────────────────────────────────────────────────────────────────────
print(f"\n[*] {len(changed_idxs)}개 샘플 재생성...")
new_preds = dict(base_map)   # 복사본

for i, idx in enumerate(sorted(changed_idxs)):
    item = gt_list[idx]
    query = item["instruction"]
    gold  = item["gold_answer"]
    domain_gt = item["expected_domain"]

    rt = new_routing[idx]
    is_cross = rt['is_cross']
    top1     = rt['top1']
    top2     = rt['top2']
    domains  = [top1, top2] if is_cross else [top1]
    lora_target = "CROSS" if is_cross else top1

    print(f"\n  [{i+1}/{len(changed_idxs)}] idx={idx} gt={domain_gt} "
          f"is_cross={is_cross} lora={lora_target} domains={domains}")

    switch_lora(lora_target)
    ctx    = retrieve(query, domains, is_cross)
    answer = nexroute_generate(query, ctx, is_cross, domains)
    print(f"  답변: {answer[:100]!r}")

    new_preds[idx] = {
        "idx": idx,
        "instruction": query,
        "gold_answer": gold,
        "expected_domain": domain_gt,
        "is_cross": (domain_gt == "CROSS"),
        "pred_domain": rt['pred_domain'],
        "pred_is_cross": is_cross,
        "pred_domains": domains,
        "pred_answer": answer,
    }

# ── 저장 ─────────────────────────────────────────────────────────────────────
out_list = [new_preds[i] for i in range(len(gt_list))]
with open(OUT_PATH, "w") as f:
    json.dump(out_list, f, ensure_ascii=False, indent=2)
print(f"\n[+] 저장: {OUT_PATH}  ({len(out_list)}개)")

gc.collect()
torch.cuda.empty_cache()
