"""
judge_reeval_law_v4.py
======================
LAW 5개 샘플을 NexRoute v4 예측으로 재평가 (Claude Sonnet 4.6 judge).
기존 non-LAW 20개 결과와 합산 → 새로운 score_sheet_v4.csv 생성.

실행:
    python scripts/judge_reeval_law_v4.py
"""
import json, csv, os, re, random, time
import anthropic

RANDOM_SEED  = 20260829   # v4 평가용 새 시드
LAW_SAMPLES  = ["S001","S002","S003","S004","S005"]
JUDGE_MODEL  = "claude-sonnet-4-6"
JUDGE_DATE   = "2026-08-29"
MAX_RETRIES  = 3

# 파일 경로
V4_PREDS     = "reinference_results/preds_500_nexroute_lawv4.json"
NAIVE_PREDS  = "reinference_results/preds_500_naive_rag.json"
GT_PATH      = "processed_v2/gt_dataset_500.json"
MANIFEST_CSV = "results/llm_judge/sample_manifest.csv"
ORIG_SCORES  = "results/llm_judge/score_sheet.csv"
OUT_DIR      = "results/llm_judge"
OUT_CSV      = os.path.join(OUT_DIR, "score_sheet_v4.csv")
OUT_RAW      = os.path.join(OUT_DIR, "raw/law_v4_raw_responses.json")

os.makedirs(os.path.join(OUT_DIR, "raw"), exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
with open(V4_PREDS)    as f: v4_list   = json.load(f)
with open(NAIVE_PREDS) as f: naive_list = json.load(f)
with open(GT_PATH)     as f: gt_list   = json.load(f)

v4_map    = {p["idx"]: p for p in v4_list}
naive_map = {p["idx"]: p for p in naive_list}
gt_map    = {i: g for i, g in enumerate(gt_list)}

manifest = []
with open(MANIFEST_CSV) as f:
    manifest = list(csv.DictReader(f))
law_samples = {m["sample_id"]: m for m in manifest if m["domain"] == "LAW"}

# 기존 scores 로드 (non-LAW 20개)
orig_rows = []
with open(ORIG_SCORES) as f:
    orig_rows = list(csv.DictReader(f))
fieldnames = orig_rows[0].keys() if orig_rows else None

# ── Judge 프롬프트 ─────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are an independent evaluator of two AI-generated answers.

Your task is to evaluate answer quality, not writing-style similarity.

You do not know which system produced either response.
Do not infer or guess system identity.

IMPORTANT EVALUATION PRINCIPLES

1. Judge factual and task-level quality rather than lexical overlap.
2. The reference answer is a guide to the expected substance, not a required wording template.
3. An answer may be correct even if its structure or terminology differs from the reference.
4. Do not reward verbosity.
5. Do not penalize concise answers if they contain the required substance.
6. Do not penalize citation-style or professional-domain formatting by itself.
7. However, if formatting causes substantive information to be missing, unclear, or incorrect,
   penalize the appropriate quality dimension.
8. For cross-domain questions, reward actual integration between domains rather than merely
   listing facts from each domain.
9. Treat Response A and Response B symmetrically.
10. Evaluate each response independently before making a pairwise preference decision.

QUESTION

{question}

REFERENCE ANSWER

{reference}

RESPONSE A

{response_A}

RESPONSE B

{response_B}

Evaluate each response on the following 1-5 dimensions:

- factual_correctness
- cross_domain_synthesis  (integer 1-5 for cross-domain questions; null for single-domain)
- task_adequacy
- reasoning_coherence
- completeness
- relevance
- overall_quality

SCORING ANCHORS

5 = excellent / fully satisfies the criterion
4 = good / minor limitations only
3 = acceptable / noticeable limitations
2 = weak / important deficiencies
1 = poor / major failure

PAIRWISE PREFERENCE

After scoring both responses independently, state your overall preference:
- A (prefer Response A)
- B (prefer Response B)
- TIE (no reliable preference)

Use TIE when both responses are roughly equivalent in overall quality.
Do not force a preference when you cannot reliably distinguish them.

CONFIDENCE: Rate your confidence in the preference on a 1-5 scale.

OUTPUT FORMAT (JSON only, no extra text):

{{
  "A": {{
    "factual_correctness": <int>,
    "cross_domain_synthesis": <int or null>,
    "task_adequacy": <int>,
    "reasoning_coherence": <int>,
    "completeness": <int>,
    "relevance": <int>,
    "overall_quality": <int>
  }},
  "B": {{
    "factual_correctness": <int>,
    "cross_domain_synthesis": <int or null>,
    "task_adequacy": <int>,
    "reasoning_coherence": <int>,
    "completeness": <int>,
    "relevance": <int>,
    "overall_quality": <int>
  }},
  "preference": "<A|B|TIE>",
  "confidence": <int>,
  "rationale": "<brief explanation>"
}}"""


def run_judge(question, reference, response_A, response_B):
    client = anthropic.Anthropic()
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        reference=reference,
        response_A=response_A,
        response_B=response_B,
    )
    for attempt in range(MAX_RETRIES):
        try:
            msg = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            # JSON 파싱
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
    return None


# ── A/B 순서 결정 (v4 재평가용 새 시드) ────────────────────────────────────────
rng = random.Random(RANDOM_SEED)
law_ab = {}
for sid in LAW_SAMPLES:
    law_ab[sid] = "A" if rng.random() < 0.5 else "B"
print("LAW A/B 배정:", law_ab)

# ── 평가 실행 ──────────────────────────────────────────────────────────────────
new_law_rows = []
raw_responses = {}

for sid in LAW_SAMPLES:
    m = law_samples[sid]
    raw_idx   = int(m["raw_sample_id"])
    domain    = m["domain"]
    func      = m["function"]
    is_cross  = int(m["is_cross"])

    question  = gt_map[raw_idx]["instruction"]
    reference = gt_map[raw_idx]["gold_answer"]
    nexroute_pos = law_ab[sid]  # v4 NexRoute가 A 또는 B

    nexroute_answer = v4_map.get(raw_idx, {}).get("pred_answer", "")
    naive_answer    = naive_map.get(raw_idx, {}).get("pred_answer", "")

    if nexroute_pos == "A":
        resp_A, resp_B = nexroute_answer, naive_answer
        system_A, system_B = "NexRoute_v4", "NaiveRAG"
    else:
        resp_A, resp_B = naive_answer, nexroute_answer
        system_A, system_B = "NaiveRAG", "NexRoute_v4"

    print(f"\n[{sid}] {domain} {func}  NexRoute_v4={nexroute_pos}")
    print(f"  Q: {question[:80]!r}")

    result = run_judge(question, reference, resp_A, resp_B)
    if result is None:
        print(f"  → FAILED")
        continue

    raw_responses[sid] = {"request": {"question": question, "reference": reference,
                                       "response_A": resp_A, "response_B": resp_B,
                                       "system_A": system_A, "system_B": system_B},
                           "response": result}

    # NexRoute scores
    nr_pos = nexroute_pos          # "A" or "B"
    bl_pos = "B" if nr_pos=="A" else "A"
    nr_scores = result[nr_pos]
    bl_scores = result[bl_pos]
    pref = result.get("preference", "TIE")

    # preference: A or B → translate to NexRoute vs NaiveRAG
    if pref == "TIE":
        pref_system = "TIE"
    elif pref == nr_pos:
        pref_system = "NexRoute_v4"
    else:
        pref_system = "NaiveRAG"

    print(f"  NexRoute_v4: factual={nr_scores.get('factual_correctness')}  "
          f"task={nr_scores.get('task_adequacy')}  overall={nr_scores.get('overall_quality')}")
    print(f"  NaiveRAG:    factual={bl_scores.get('factual_correctness')}  "
          f"task={bl_scores.get('task_adequacy')}  overall={bl_scores.get('overall_quality')}")
    print(f"  Preference: {pref} → {pref_system}  confidence={result.get('confidence')}")

    row = {
        "sample_id": sid,
        "domain": domain,
        "function": func,
        "is_cross": is_cross,
        "judge_provider": "Anthropic",
        "judge_model": JUDGE_MODEL,
        "judge_date": JUDGE_DATE,
        "system_A": system_A,
        "system_B": system_B,
        "A_factual": result["A"].get("factual_correctness"),
        "A_synthesis": result["A"].get("cross_domain_synthesis"),
        "A_task": result["A"].get("task_adequacy"),
        "A_reasoning": result["A"].get("reasoning_coherence"),
        "A_completeness": result["A"].get("completeness"),
        "A_relevance": result["A"].get("relevance"),
        "A_overall": result["A"].get("overall_quality"),
        "B_factual": result["B"].get("factual_correctness"),
        "B_synthesis": result["B"].get("cross_domain_synthesis"),
        "B_task": result["B"].get("task_adequacy"),
        "B_reasoning": result["B"].get("reasoning_coherence"),
        "B_completeness": result["B"].get("completeness"),
        "B_relevance": result["B"].get("relevance"),
        "B_overall": result["B"].get("overall_quality"),
        "preference": pref,
        "confidence": result.get("confidence"),
        "rationale": result.get("rationale", ""),
    }
    new_law_rows.append(row)

# ── 결과 저장 ──────────────────────────────────────────────────────────────────
with open(OUT_RAW, "w") as f:
    json.dump(raw_responses, f, ensure_ascii=False, indent=2)
print(f"\n[+] 원본 저장: {OUT_RAW}")

# non-LAW 행 유지 + LAW v4 행 교체
non_law_rows = [r for r in orig_rows if r["sample_id"] not in LAW_SAMPLES]
all_rows = non_law_rows + new_law_rows

# CSV 저장
all_fnames = list(orig_rows[0].keys())
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_fnames)
    writer.writeheader()
    for r in all_rows:
        writer.writerow({k: r.get(k, "") for k in all_fnames})
print(f"[+] 합산 CSV 저장: {OUT_CSV}  ({len(all_rows)}행)")

# ── 집계 통계 ──────────────────────────────────────────────────────────────────
print("\n=== V4 Judge 통계 (전체 25샘플) ===")
import numpy as np

def get_nexroute_score(row, dim):
    # NexRoute가 A이면 A_dim, B이면 B_dim
    sys_a = row.get("system_A", "")
    if "NexRoute" in sys_a:
        val = row.get(f"A_{dim}")
    else:
        val = row.get(f"B_{dim}")
    try:
        return float(val) if val and val != "" else None
    except:
        return None

def get_naive_score(row, dim):
    sys_a = row.get("system_A", "")
    if "NaiveRAG" in sys_a or "Naive" in sys_a:
        val = row.get(f"A_{dim}")
    else:
        val = row.get(f"B_{dim}")
    try:
        return float(val) if val and val != "" else None
    except:
        return None

dims = ["factual", "task", "reasoning", "completeness", "relevance", "overall"]
for dim in dims:
    nr = [get_nexroute_score(r, dim) for r in all_rows]
    na = [get_naive_score(r, dim) for r in all_rows]
    nr_ok = [v for v in nr if v is not None]
    na_ok = [v for v in na if v is not None]
    delta = round(np.mean(nr_ok) - np.mean(na_ok), 3) if nr_ok and na_ok else None
    print(f"  {dim:<14} NexRoute={np.mean(nr_ok):.2f}  NaiveRAG={np.mean(na_ok):.2f}  Δ={delta:+.3f}")

# synthesis (CROSS only)
cross_rows = [r for r in all_rows if r.get("is_cross") in ("1", 1, True)]
nr_syn = [get_nexroute_score(r, "synthesis") for r in cross_rows]
na_syn = [get_naive_score(r, "synthesis") for r in cross_rows]
nr_syn_ok = [v for v in nr_syn if v is not None]
na_syn_ok = [v for v in na_syn if v is not None]
if nr_syn_ok and na_syn_ok:
    print(f"  {'synthesis':<14} NexRoute={np.mean(nr_syn_ok):.2f}  NaiveRAG={np.mean(na_syn_ok):.2f}  "
          f"Δ={np.mean(nr_syn_ok)-np.mean(na_syn_ok):+.3f}  (n={len(nr_syn_ok)} CROSS)")

# preference
prefs = [r.get("preference", "") for r in all_rows]
print(f"\n  Preference: NexRoute={prefs.count('A') + sum(1 for r in all_rows if r.get('preference')==r.get('system_A','').replace('_v4',''))}/"
      f"{len(all_rows)}")
from collections import Counter
pref_counter = Counter()
for r in all_rows:
    pref = r.get("preference", "")
    sys_a = r.get("system_A", "")
    if pref == "TIE":
        pref_counter["TIE"] += 1
    elif pref == "A":
        pref_counter["NexRoute" if "NexRoute" in sys_a else "NaiveRAG"] += 1
    elif pref == "B":
        pref_counter["NaiveRAG" if "NexRoute" in sys_a else "NexRoute"] += 1
total = len(all_rows)
print(f"  NexRoute: {pref_counter['NexRoute']}/{total} ({100*pref_counter['NexRoute']/total:.0f}%)")
print(f"  NaiveRAG: {pref_counter['NaiveRAG']}/{total} ({100*pref_counter['NaiveRAG']/total:.0f}%)")
print(f"  TIE:      {pref_counter['TIE']}/{total} ({100*pref_counter['TIE']/total:.0f}%)")
