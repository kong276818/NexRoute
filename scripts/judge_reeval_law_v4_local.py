"""
judge_reeval_law_v4_local.py
============================
Qwen2.5-72B-Instruct (로컬, 4-bit) judge로 LAW 5개 샘플 재평가.
비-LAW 20개 결과(orig score_sheet.csv)와 합산.

실행:
    CUDA_VISIBLE_DEVICES=0,1,2 python scripts/judge_reeval_law_v4_local.py
"""
import json, csv, os, re, random, gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import Counter

RANDOM_SEED  = 20260829
LAW_SAMPLES  = ["S001","S002","S003","S004","S005"]
JUDGE_MODEL  = "Qwen2.5-72B-Instruct-local"
JUDGE_DATE   = "2026-08-29"

BASE_72B     = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
V4_PREDS     = "reinference_results/preds_500_nexroute_lawv4.json"
NAIVE_PREDS  = "reinference_results/preds_500_naive_rag.json"
GT_PATH      = "processed_v2/gt_dataset_500.json"
MANIFEST_CSV = "results/llm_judge/sample_manifest.csv"
ORIG_SCORES  = "results/llm_judge/score_sheet.csv"
OUT_DIR      = "results/llm_judge"
OUT_CSV      = os.path.join(OUT_DIR, "score_sheet_v4_local.csv")
OUT_RAW      = os.path.join(OUT_DIR, "raw/law_v4_local_raw_responses.json")

os.makedirs(os.path.join(OUT_DIR, "raw"), exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
with open(V4_PREDS)    as f: v4_list    = json.load(f)
with open(NAIVE_PREDS) as f: naive_list = json.load(f)
with open(GT_PATH)     as f: gt_list    = json.load(f)

v4_map    = {p["idx"]: p for p in v4_list}
naive_map = {p["idx"]: p for p in naive_list}
gt_map    = {i: g for i, g in enumerate(gt_list)}

manifest = []
with open(MANIFEST_CSV) as f:
    manifest = list(csv.DictReader(f))
law_samples_manifest = {m["sample_id"]: m for m in manifest if m["domain"] == "LAW"}

orig_rows = []
with open(ORIG_SCORES) as f:
    orig_rows = list(csv.DictReader(f))

# ── Judge 프롬프트 ─────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are an independent evaluator of two AI-generated answers.

Your task is to evaluate answer quality, not writing-style similarity.
You do not know which system produced either response. Do not infer or guess system identity.

IMPORTANT EVALUATION PRINCIPLES
1. Judge factual and task-level quality rather than lexical overlap.
2. The reference answer is a guide to expected substance, not a required wording template.
3. An answer may be correct even if its structure or terminology differs from the reference.
4. Do not reward verbosity. Do not penalize concise answers if they contain the required substance.
5. Do not penalize citation-style or professional-domain formatting by itself.
6. However, if formatting causes substantive information to be missing, unclear, or incorrect, penalize accordingly.
7. For cross-domain questions, reward actual integration between domains.
8. Treat Response A and Response B symmetrically.
9. Evaluate each response independently before making a pairwise preference decision.

QUESTION
{question}

REFERENCE ANSWER
{reference}

RESPONSE A
{response_A}

RESPONSE B
{response_B}

Evaluate each response on the following 1-5 dimensions:
- factual_correctness: Is the factual content accurate?
- cross_domain_synthesis: For cross-domain questions only (null for single-domain): Does the response integrate knowledge across domains?
- task_adequacy: Does the response actually perform the requested task (how/why/what/compare)?
- reasoning_coherence: Is the reasoning logical and well-structured?
- completeness: Does the response cover the key points?
- relevance: Is the response focused on what was asked?
- overall_quality: Overall assessment.

SCORING ANCHORS
5=excellent, 4=good/minor limitations, 3=acceptable/noticeable limitations, 2=weak/important deficiencies, 1=poor/major failure

PAIRWISE PREFERENCE
After scoring both responses independently:
- A: prefer Response A
- B: prefer Response B
- TIE: no reliable preference

CONFIDENCE: 1-5 scale.

OUTPUT: Return ONLY valid JSON, no other text:
{{
  "A": {{"factual_correctness": <int>, "cross_domain_synthesis": <int or null>, "task_adequacy": <int>, "reasoning_coherence": <int>, "completeness": <int>, "relevance": <int>, "overall_quality": <int>}},
  "B": {{"factual_correctness": <int>, "cross_domain_synthesis": <int or null>, "task_adequacy": <int>, "reasoning_coherence": <int>, "completeness": <int>, "relevance": <int>, "overall_quality": <int>}},
  "preference": "<A|B|TIE>",
  "confidence": <int>,
  "rationale": "<brief explanation>"
}}"""


# ── Qwen2.5-72B 로드 ────────────────────────────────────────────────────────────
print("[*] Qwen2.5-72B-Instruct 4-bit 로드 (3 GPU)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_72B, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_72B,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="eager",
)
model.eval()
print("[*] 72B 로드 완료!")


def run_judge(question, reference, response_A, response_B):
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference[:800],
        response_A=response_A[:800],
        response_B=response_B[:800],
    )
    msgs = [
        {"role": "system", "content": "You are an expert evaluator. Output only valid JSON."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inp  = {k: v.to(model.device) for k, v in inp.items()}

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(
        out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # JSON 파싱
    m = re.search(r'\{.*\}', response_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), response_text
        except json.JSONDecodeError:
            pass
    print(f"  [JSON 파싱 실패] 응답: {response_text[:200]!r}")
    return None, response_text


# ── A/B 순서 결정 ──────────────────────────────────────────────────────────────
rng = random.Random(RANDOM_SEED)
law_ab = {sid: ("A" if rng.random() < 0.5 else "B") for sid in LAW_SAMPLES}
print("LAW A/B 배정:", law_ab)

# ── 평가 실행 ──────────────────────────────────────────────────────────────────
new_law_rows = []
raw_responses = {}

for sid in LAW_SAMPLES:
    m = law_samples_manifest[sid]
    raw_idx  = int(m["raw_sample_id"])
    domain   = m["domain"]
    func     = m["function"]
    is_cross = int(m["is_cross"])

    question  = gt_map[raw_idx]["instruction"]
    reference = gt_map[raw_idx]["gold_answer"]
    nr_pos    = law_ab[sid]

    nr_answer   = v4_map.get(raw_idx, {}).get("pred_answer", "")
    naive_answer = naive_map.get(raw_idx, {}).get("pred_answer", "")

    if nr_pos == "A":
        resp_A, resp_B = nr_answer, naive_answer
        sys_A, sys_B   = "NexRoute_v4", "NaiveRAG"
    else:
        resp_A, resp_B = naive_answer, nr_answer
        sys_A, sys_B   = "NaiveRAG", "NexRoute_v4"

    print(f"\n[{sid}] {domain} {func}  NexRoute_v4={nr_pos}")
    print(f"  Q: {question[:90]!r}")
    print(f"  NexRoute_v4 ({nr_pos}): {nr_answer[:80]!r}")

    result, raw_text = run_judge(question, reference, resp_A, resp_B)
    if result is None:
        print(f"  → FAILED")
        continue

    raw_responses[sid] = {
        "system_A": sys_A, "system_B": sys_B,
        "response": result, "raw_text": raw_text[:500],
    }

    nr_scores  = result[nr_pos]
    bl_pos     = "B" if nr_pos=="A" else "A"
    bl_scores  = result[bl_pos]
    pref_raw   = result.get("preference","TIE")
    pref_sys   = "TIE" if pref_raw=="TIE" else ("NexRoute_v4" if pref_raw==nr_pos else "NaiveRAG")

    print(f"  NexRoute_v4: factual={nr_scores.get('factual_correctness')}  "
          f"task={nr_scores.get('task_adequacy')}  overall={nr_scores.get('overall_quality')}")
    print(f"  NaiveRAG:    factual={bl_scores.get('factual_correctness')}  "
          f"task={bl_scores.get('task_adequacy')}  overall={bl_scores.get('overall_quality')}")
    print(f"  Preference: {pref_raw} → {pref_sys}  conf={result.get('confidence')}")
    print(f"  Rationale: {result.get('rationale','')[:120]}")

    row = {
        "sample_id": sid, "domain": domain, "function": func, "is_cross": is_cross,
        "judge_provider": "Local", "judge_model": JUDGE_MODEL, "judge_date": JUDGE_DATE,
        "system_A": sys_A, "system_B": sys_B,
        "A_factual":      result["A"].get("factual_correctness"),
        "A_synthesis":    result["A"].get("cross_domain_synthesis"),
        "A_task":         result["A"].get("task_adequacy"),
        "A_reasoning":    result["A"].get("reasoning_coherence"),
        "A_completeness": result["A"].get("completeness"),
        "A_relevance":    result["A"].get("relevance"),
        "A_overall":      result["A"].get("overall_quality"),
        "B_factual":      result["B"].get("factual_correctness"),
        "B_synthesis":    result["B"].get("cross_domain_synthesis"),
        "B_task":         result["B"].get("task_adequacy"),
        "B_reasoning":    result["B"].get("reasoning_coherence"),
        "B_completeness": result["B"].get("completeness"),
        "B_relevance":    result["B"].get("relevance"),
        "B_overall":      result["B"].get("overall_quality"),
        "preference": pref_raw, "confidence": result.get("confidence"),
        "rationale": result.get("rationale",""),
    }
    new_law_rows.append(row)

# ── 저장 ──────────────────────────────────────────────────────────────────────
with open(OUT_RAW, "w") as f:
    json.dump(raw_responses, f, ensure_ascii=False, indent=2)
print(f"\n[+] 원본 저장: {OUT_RAW}")

non_law = [r for r in orig_rows if r["sample_id"] not in LAW_SAMPLES]
all_rows = non_law + new_law_rows
all_fnames = list(orig_rows[0].keys())

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_fnames)
    writer.writeheader()
    for r in all_rows:
        writer.writerow({k: r.get(k, "") for k in all_fnames})
print(f"[+] 합산 CSV 저장: {OUT_CSV}  ({len(all_rows)}행)")

# ── 집계 통계 ──────────────────────────────────────────────────────────────────
print("\n=== V4 (Qwen2.5-72B judge) 통계 (전체 25샘플) ===")

def get_score(row, dim, system_name):
    sys_a = row.get("system_A", "")
    is_a = system_name in sys_a
    val = row.get(f"{'A' if is_a else 'B'}_{dim}", "")
    try: return float(val) if val and val != "" else None
    except: return None

dims = ["factual", "task", "reasoning", "completeness", "relevance", "overall"]
for dim in dims:
    nr = [get_score(r, dim, "NexRoute") for r in all_rows]
    na = [get_score(r, dim, "NaiveRAG") for r in all_rows]
    nr_ok = [v for v in nr if v is not None]
    na_ok = [v for v in na if v is not None]
    if nr_ok and na_ok:
        d = round(np.mean(nr_ok) - np.mean(na_ok), 3)
        print(f"  {dim:<14} NexRoute={np.mean(nr_ok):.2f}  NaiveRAG={np.mean(na_ok):.2f}  Δ={d:+.3f}")

# synthesis (CROSS only)
cross_rows = [r for r in all_rows if str(r.get("is_cross","")) in ("1","True","true")]
nr_syn = [get_score(r, "synthesis", "NexRoute") for r in cross_rows]
na_syn = [get_score(r, "synthesis", "NaiveRAG") for r in cross_rows]
nr_syn_ok = [v for v in nr_syn if v is not None]
na_syn_ok = [v for v in na_syn if v is not None]
if nr_syn_ok and na_syn_ok:
    print(f"  {'synthesis':<14} NexRoute={np.mean(nr_syn_ok):.2f}  NaiveRAG={np.mean(na_syn_ok):.2f}  "
          f"Δ={np.mean(nr_syn_ok)-np.mean(na_syn_ok):+.3f}  (n={len(nr_syn_ok)} CROSS)")

# preference count
pref_counter = Counter()
for r in all_rows:
    pref = r.get("preference","")
    sys_a = r.get("system_A","")
    if pref == "TIE":
        pref_counter["TIE"] += 1
    elif pref == "A":
        pref_counter["NexRoute" if "NexRoute" in sys_a else "NaiveRAG"] += 1
    elif pref == "B":
        pref_counter["NaiveRAG" if "NexRoute" in sys_a else "NexRoute"] += 1
total = len(all_rows)
print(f"\n  NexRoute_v4: {pref_counter['NexRoute']}/{total} ({100*pref_counter['NexRoute']/total:.0f}%)")
print(f"  NaiveRAG:    {pref_counter['NaiveRAG']}/{total} ({100*pref_counter['NaiveRAG']/total:.0f}%)")
print(f"  TIE:         {pref_counter['TIE']}/{total} ({100*pref_counter['TIE']/total:.0f}%)")

# LAW-only
print("\n=== LAW 5개 샘플 (v4 전후 비교) ===")
law_new = [r for r in new_law_rows]
if law_new:
    for dim in ["factual", "task", "reasoning", "overall"]:
        nr = [get_score(r, dim, "NexRoute") for r in law_new]
        na = [get_score(r, dim, "NaiveRAG") for r in law_new]
        nr_ok = [v for v in nr if v is not None]
        na_ok = [v for v in na if v is not None]
        if nr_ok and na_ok:
            print(f"  {dim:<14} NexRoute_v4={np.mean(nr_ok):.2f}  NaiveRAG={np.mean(na_ok):.2f}  "
                  f"Δ={np.mean(nr_ok)-np.mean(na_ok):+.3f}")

gc.collect()
torch.cuda.empty_cache()
