"""
Multi-Model Judge Evaluation — Prompt Generator v3

Design:
  - Pairwise blind comparison: Response A vs B (system identity hidden)
  - A/B order counterbalanced with fixed seed (RANDOM_SEED = 20260827)
  - 25 samples from Clean-300: 5 per domain (LAW, MED, SOC, STEM, CROSS)
  - 7 evaluation dimensions + pairwise preference + confidence
  - cross_domain_synthesis = null for single-domain samples

Outputs (all in results/llm_judge/):
  prompts/
    prompts_all.md          — all 25 prompts, copy-paste ready
    prompts_S001.md ...     — per-sample files (optional)
  randomization_key.json    — {sample_id: "A"|"B"} (NexRoute position)
  score_sheet.csv           — blank template for manual score entry
  evaluation_manifest.csv   — judge metadata form (fill after eval)
  sample_manifest.csv       — which 25 samples were selected and why

Usage:
  cd /home/sclab/moe_model
  python scripts/gen_llm_judge_prompts.py
"""

import json
import os
import csv
import random

RANDOM_SEED = 20260827
N_PER_DOMAIN = 5
DOMAINS = ["LAW", "MED", "SOC", "STEM", "CROSS"]

NEXROUTE_FILE = "results/ablation/v01_gen_500.json"
BASELINE_FILE = "results/ablation/b3_gen_500.json"
OUT_DIR = "results/llm_judge"

# --------------------------------------------------------------------------
# Judge prompt template (spec section 15, reference-only version)
# --------------------------------------------------------------------------

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

Choose: "A", "B", or "TIE"

Choose A or B only when one answer is meaningfully better overall.
Use TIE when differences are small or involve mixed trade-offs.

CONFIDENCE

Rate your confidence in the evaluation: 1 (very uncertain) to 5 (very high confidence).

Return ONLY valid JSON using exactly this schema:

{{
  "response_A": {{
    "factual_correctness": 1,
    "cross_domain_synthesis": null,
    "task_adequacy": 1,
    "reasoning_coherence": 1,
    "completeness": 1,
    "relevance": 1,
    "overall_quality": 1
  }},
  "response_B": {{
    "factual_correctness": 1,
    "cross_domain_synthesis": null,
    "task_adequacy": 1,
    "reasoning_coherence": 1,
    "completeness": 1,
    "relevance": 1,
    "overall_quality": 1
  }},
  "preference": "TIE",
  "confidence": 1,
  "brief_rationale": "Maximum 60 words."
}}

All non-null scores must be integers from 1 to 5.
Do not output markdown.
Do not output commentary outside the JSON.
"""


def load_data():
    nexroute = {d["sample_id"]: d for d in json.load(open(NEXROUTE_FILE))}
    baseline = {d["sample_id"]: d for d in json.load(open(BASELINE_FILE))}
    return nexroute, baseline


def select_samples(nexroute):
    """5 samples per domain from Clean-300 (sample_id < 300), fixed seed."""
    rng = random.Random(RANDOM_SEED)
    selected = []
    for dom in DOMAINS:
        pool = [
            sid for sid, d in nexroute.items()
            if d["domain"] == dom and int(sid) < 300
        ]
        pool.sort()
        picks = rng.sample(pool, N_PER_DOMAIN)
        selected.extend(picks)
    return selected


def assign_positions(selected):
    """Randomly assign NexRoute to A or B, ~half each, fixed seed."""
    rng = random.Random(RANDOM_SEED)
    n_A = len(selected) // 2  # 12
    nexroute_is_A = set(rng.sample(selected, n_A))
    return {sid: ("A" if sid in nexroute_is_A else "B") for sid in selected}


def make_sample_id(i, domain):
    return f"S{i+1:03d}"


def build_prompt(sample_id, nexroute_d, baseline_d, nexroute_pos, is_cross):
    q = nexroute_d["query"]
    ref = nexroute_d["gold_answer"]
    nex_resp = nexroute_d["pred_answer"]
    base_resp = baseline_d["pred_answer"]

    if nexroute_pos == "A":
        resp_A, resp_B = nex_resp, base_resp
    else:
        resp_A, resp_B = base_resp, nex_resp

    synthesis_note = (
        "cross_domain_synthesis: use an integer 1-5 (this IS a cross-domain question)"
        if is_cross
        else "cross_domain_synthesis: set to null (this is a single-domain question)"
    )

    prompt = PROMPT_TEMPLATE.format(
        question=q,
        reference=ref,
        response_A=resp_A,
        response_B=resp_B,
    )
    # inject synthesis hint just before the schema
    prompt = prompt.replace(
        "- cross_domain_synthesis  (integer 1-5 for cross-domain questions; null for single-domain)",
        f"- cross_domain_synthesis  ({synthesis_note})",
    )
    return prompt


def write_prompts(samples_info, nexroute, baseline, rand_key):
    prompt_dir = os.path.join(OUT_DIR, "prompts")
    os.makedirs(prompt_dir, exist_ok=True)

    all_md_lines = [
        "# NexRoute Multi-Model Judge Evaluation Prompts\n",
        f"Seed: {RANDOM_SEED} | Samples: {len(samples_info)} | Generated: 2026-08-27\n",
        "Copy each prompt block into a **new chat** for each judge.\n",
        "Do NOT share one judge's output with another judge before evaluation.\n\n",
        "---\n",
    ]

    for info in samples_info:
        sid = info["sample_id"]
        dom = info["domain"]
        func = info["pred_function"]
        is_cross = info["is_cross"]
        nexroute_pos = rand_key[info["raw_sid"]]
        nex_d = nexroute[info["raw_sid"]]
        base_d = baseline[info["raw_sid"]]

        prompt = build_prompt(sid, nex_d, base_d, nexroute_pos, is_cross)

        cross_tag = " [CROSS]" if is_cross else ""
        header = f"## {sid} | Domain: {dom} | Function: {func}{cross_tag}\n"
        note = f"_NexRoute is Response {nexroute_pos} — for your records only, do not share with judge._\n\n"

        block = f"{header}{note}```\n{prompt}\n```\n\n---\n"
        all_md_lines.append(block)

        # per-sample file
        with open(os.path.join(prompt_dir, f"prompt_{sid}.md"), "w") as f:
            f.write(f"{header}\n{prompt}\n")

    with open(os.path.join(prompt_dir, "prompts_all.md"), "w") as f:
        f.write("\n".join(all_md_lines))

    print(f"Prompts written to {prompt_dir}/prompts_all.md and {prompt_dir}/prompt_S*.md")


def write_score_sheet(samples_info, rand_key):
    """
    Columns: one row per (sample, judge_provider).
    Analyst fills in A_* and B_* scores after collecting judge outputs.
    system_A / system_B are filled by analyst from randomization_key.json.
    """
    fieldnames = [
        "sample_id", "domain", "function", "is_cross",
        "judge_provider", "judge_model", "judge_date",
        "system_A", "system_B",
        "A_factual", "A_synthesis", "A_task", "A_reasoning",
        "A_completeness", "A_relevance", "A_overall",
        "B_factual", "B_synthesis", "B_task", "B_reasoning",
        "B_completeness", "B_relevance", "B_overall",
        "preference", "confidence", "rationale",
    ]

    rows = []
    providers = ["OpenAI", "Anthropic", "Google", "Judge4"]

    for info in samples_info:
        raw_sid = info["raw_sid"]
        nexroute_pos = rand_key[raw_sid]
        if nexroute_pos == "A":
            sys_A, sys_B = "NexRoute", "NaiveRAG"
        else:
            sys_A, sys_B = "NaiveRAG", "NexRoute"

        for prov in providers:
            rows.append({
                "sample_id":    info["sample_id"],
                "domain":       info["domain"],
                "function":     info["pred_function"],
                "is_cross":     1 if info["is_cross"] else 0,
                "judge_provider": prov,
                "judge_model":  "",
                "judge_date":   "",
                "system_A":     sys_A,
                "system_B":     sys_B,
                "A_factual": "", "A_synthesis": "", "A_task": "",
                "A_reasoning": "", "A_completeness": "", "A_relevance": "",
                "A_overall": "",
                "B_factual": "", "B_synthesis": "", "B_task": "",
                "B_reasoning": "", "B_completeness": "", "B_relevance": "",
                "B_overall": "",
                "preference": "", "confidence": "", "rationale": "",
            })

    out_path = os.path.join(OUT_DIR, "score_sheet.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Score sheet written to {out_path}  ({len(rows)} rows = {len(samples_info)} samples × {len(providers)} judges)")


def write_sample_manifest(samples_info, rand_key):
    fieldnames = ["sample_id", "raw_sample_id", "domain", "function",
                  "is_cross", "nexroute_position", "query_preview"]
    rows = []
    for info in samples_info:
        rows.append({
            "sample_id":        info["sample_id"],
            "raw_sample_id":    info["raw_sid"],
            "domain":           info["domain"],
            "function":         info["pred_function"],
            "is_cross":         1 if info["is_cross"] else 0,
            "nexroute_position": rand_key[info["raw_sid"]],
            "query_preview":    info["query"][:80],
        })
    out_path = os.path.join(OUT_DIR, "sample_manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Sample manifest written to {out_path}")


def write_evaluation_manifest():
    fieldnames = ["judge_provider", "judge_model", "evaluation_date",
                  "interface", "temperature", "notes"]
    rows = [
        {"judge_provider": "OpenAI",     "judge_model": "", "evaluation_date": "",
         "interface": "web", "temperature": "N/A", "notes": "new independent chat"},
        {"judge_provider": "Anthropic",  "judge_model": "", "evaluation_date": "",
         "interface": "web", "temperature": "N/A", "notes": "new independent chat"},
        {"judge_provider": "Google",     "judge_model": "", "evaluation_date": "",
         "interface": "web", "temperature": "N/A", "notes": "new independent chat"},
        {"judge_provider": "Judge4",     "judge_model": "", "evaluation_date": "",
         "interface": "web", "temperature": "N/A", "notes": "new independent chat"},
    ]
    out_path = os.path.join(OUT_DIR, "evaluation_manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluation manifest written to {out_path}  (fill in model names + dates after eval)")


def setup_raw_dirs():
    for provider in ["openai", "claude", "gemini", "judge4"]:
        os.makedirs(os.path.join(OUT_DIR, "raw", provider), exist_ok=True)
    print(f"Raw output dirs created under {OUT_DIR}/raw/")


def main():
    print(f"Seed: {RANDOM_SEED}")
    os.makedirs(OUT_DIR, exist_ok=True)

    nexroute, baseline = load_data()
    selected_sids = select_samples(nexroute)
    rand_key = assign_positions(selected_sids)

    # Save randomization key
    rand_key_path = os.path.join(OUT_DIR, "randomization_key.json")
    with open(rand_key_path, "w") as f:
        json.dump(rand_key, f, indent=2)
    print(f"Randomization key saved to {rand_key_path}")
    n_A = sum(1 for v in rand_key.values() if v == "A")
    print(f"  NexRoute=A: {n_A} samples, NexRoute=B: {len(rand_key)-n_A} samples")

    # Build samples_info list (stable order: domain-by-domain, sorted within domain)
    samples_info = []
    i = 0
    for dom in DOMAINS:
        dom_sids = sorted(
            [sid for sid in selected_sids if nexroute[sid]["domain"] == dom],
            key=lambda x: int(x)
        )
        for raw_sid in dom_sids:
            d = nexroute[raw_sid]
            samples_info.append({
                "sample_id":    make_sample_id(i, dom),
                "raw_sid":      raw_sid,
                "domain":       d["domain"],
                "pred_function": d.get("pred_function", "N/A"),
                "is_cross":     str(d.get("is_cross", "False")).lower() == "true",
                "query":        d["query"],
            })
            i += 1

    print(f"\nSelected {len(samples_info)} samples:")
    from collections import Counter
    dom_counts = Counter(s["domain"] for s in samples_info)
    for dom in DOMAINS:
        print(f"  {dom}: {dom_counts[dom]}")

    write_prompts(samples_info, nexroute, baseline, rand_key)
    write_score_sheet(samples_info, rand_key)
    write_sample_manifest(samples_info, rand_key)
    write_evaluation_manifest()
    setup_raw_dirs()

    print("\nDone. Next steps:")
    print("  1. Open results/llm_judge/prompts/prompts_all.md")
    print("  2. For each judge: new chat → paste prompt → copy JSON output to results/llm_judge/raw/<provider>/S*.json")
    print("  3. Fill in score_sheet.csv from the raw JSON outputs")
    print("  4. Fill in evaluation_manifest.csv with exact model names and dates")
    print("  5. python scripts/aggregate_llm_judge.py")


if __name__ == "__main__":
    main()
