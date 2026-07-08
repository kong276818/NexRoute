"""
load_datasets.py v3 — DualMoE 데이터셋 구축 (도메인당 5,000개+)

전략:
  MED  : PubMedQA(1k) + MedQA(10k) + MMLU medical(all splits)
  LAW  : MMLU law(all splits, 모든 subject) + lex_glue
  STEM : SciQ(12k) + MMLU STEM(all splits)
  SOC  : MMLU soc/psych/geo/poli(all splits) + commonsense_qa
  CROSS: ARC-Challenge + ARC-Easy + MMLU cross(all splits)

저장: /home/sclab/moe_model/processed_v2/
"""

import json
import os
import re
import random
from collections import Counter, defaultdict
from datasets import load_dataset

OUTPUT_DIR = "/home/sclab/moe_model/processed_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PER_DOMAIN = 5000
MAX_PER_FUNC      = 1500


# ────────────────────────────────────────────
# func_type 라벨링
# ────────────────────────────────────────────
def label_func_type(text: str) -> str:
    t = text.lower().strip()
    if re.search(r'^what (is|are|was|were|do|does|did)\b|^which (of|is|are)\b|\bdefin|\bmeaning of\b|\bwhat does .+ mean\b', t):
        return "WHAT"
    if re.search(r'^how (do|does|did|can|to|is|are|would)\b|\bsteps? (to|for)\b|\bmechanism of\b|\bprocedure\b|\bprocess of\b', t):
        return "HOW"
    if re.search(r'^why (is|are|was|do|does|did|would|can)\b|\bcause(s)? of\b|\breason(s)? for\b|\bwhat causes\b|\bwhat led to\b', t):
        return "WHY"
    if re.search(r'\bcompare\b|\bdifference(s)? between\b|\bversus\b|\bvs\.?\b|\bcontrast\b|\bdistinguish\b|\bhow .+ differ\b', t):
        return "COMPARE"
    return "OTHER"


# ────────────────────────────────────────────
# instruction 품질 필터
# ────────────────────────────────────────────
def is_valid(text: str) -> bool:
    text = text.strip()
    if len(text) < 20 or len(text.split()) < 5:
        return False
    if re.match(r'^§|^\d+\.\d+|^[\(\[]\d', text):
        return False
    if text[0].islower() and text[:2].lower() not in ['a ', 'i ', 'an']:
        return False
    return True


# ────────────────────────────────────────────
# MMLU MCQ → output 변환
# ────────────────────────────────────────────
def mmlu_output(question, choices, answer_idx, subject=""):
    labels = ["A", "B", "C", "D"]
    if answer_idx >= len(choices):
        return None, None
    ans_label = labels[answer_idx]
    ans_text  = choices[answer_idx]
    choices_str = "\n".join([f"{labels[i]}: {c}" for i, c in enumerate(choices)])
    output = (
        f"The correct answer is {ans_label}: {ans_text}\n\n"
        f"Subject: {subject}\n\n"
        f"Choices:\n{choices_str}"
    )
    return question.strip(), output


# ────────────────────────────────────────────
# MMLU 로더 (train+validation+test 전부)
# ────────────────────────────────────────────
def load_mmlu_subjects(subjects, domain):
    results = []
    for subject in subjects:
        for split in ["train", "validation", "test"]:
            try:
                ds = load_dataset("cais/mmlu", subject, split=split)
            except Exception as e:
                continue
            for item in ds:
                q       = item.get("question", "").strip()
                choices = item.get("choices", [])
                ans_idx = item.get("answer", 0)
                if not is_valid(q):
                    continue
                instruction, output = mmlu_output(q, choices, ans_idx, subject)
                if not instruction:
                    continue
                results.append({
                    "domain":      domain,
                    "func_type":   label_func_type(instruction),
                    "instruction": instruction,
                    "context":     f"[Subject: {subject}]",
                    "output":      output,
                })
        print(f"  MMLU [{subject}] 완료: 누적 {len(results)}개")
    return results


# ────────────────────────────────────────────
# 균형 샘플링
# ────────────────────────────────────────────
def balance(data, max_per_func=MAX_PER_FUNC, max_total=TARGET_PER_DOMAIN):
    groups = defaultdict(list)
    for item in data:
        groups[item["func_type"]].append(item)
    balanced = []
    for ft, items in groups.items():
        random.seed(42)
        balanced.extend(random.sample(items, min(len(items), max_per_func)))
    if len(balanced) > max_total:
        random.shuffle(balanced)
        balanced = balanced[:max_total]
    return balanced


# ────────────────────────────────────────────
# 1. MED
# ────────────────────────────────────────────
def load_med():
    print("\n[MED] Loading...")
    results = []

    # PubMedQA
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        for item in ds:
            q   = item.get("question", "").strip()
            ans = item.get("long_answer", "").strip()
            ctx = " ".join(item.get("context", {}).get("contexts", []))
            if not is_valid(q) or len(ans.split()) < 15:
                continue
            results.append({"domain": "MED", "func_type": label_func_type(q),
                            "instruction": q, "context": ctx[:500], "output": ans})
        print(f"  PubMedQA: {len(results)}개")
    except Exception as e:
        print(f"  ⚠ PubMedQA: {e}")

    # MedQA (USMLE)
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        cnt = 0
        for item in ds:
            q       = item.get("question", "").strip()
            options = item.get("options", {})
            ans_key = item.get("answer_idx", "")
            ans_txt = options.get(ans_key, "")
            if not is_valid(q) or not ans_txt:
                continue
            opts_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
            output = f"The correct answer is {ans_key}: {ans_txt}\n\nOptions:\n{opts_str}"
            results.append({"domain": "MED", "func_type": label_func_type(q),
                            "instruction": q, "context": "", "output": output})
            cnt += 1
        print(f"  MedQA: {cnt}개")
    except Exception as e:
        print(f"  ⚠ MedQA: {e}")

    # MMLU medical subjects
    med_subjects = [
        "clinical_knowledge", "medical_genetics", "anatomy",
        "professional_medicine", "college_medicine", "virology",
        "college_biology", "high_school_biology",
    ]
    results += load_mmlu_subjects(med_subjects, "MED")

    results = balance(results)
    print(f"  → MED 최종: {len(results)}개 | func: {dict(Counter(d['func_type'] for d in results))}")
    return results


# ────────────────────────────────────────────
# 2. LAW
# ────────────────────────────────────────────
def load_law():
    print("\n[LAW] Loading...")

    law_subjects = [
        "jurisprudence", "international_law", "professional_law",
    ]
    results = load_mmlu_subjects(law_subjects, "LAW")

    # lex_glue case_hold
    try:
        ds = load_dataset("lex_glue", "case_hold", split="train")
        cnt = 0
        for item in ds:
            q       = item.get("context", "").strip()
            holdings = item.get("endings", [])
            ans_idx  = item.get("label", 0)
            if not q or not holdings or ans_idx < 0:
                continue
            ans_txt = holdings[ans_idx] if ans_idx < len(holdings) else ""
            h_str   = "\n".join([f"{i}: {h}" for i, h in enumerate(holdings)])
            output  = f"The correct holding is option {ans_idx}: {ans_txt}\n\nAll holdings:\n{h_str}"
            results.append({"domain": "LAW", "func_type": label_func_type(q),
                            "instruction": q[:300], "context": "", "output": output})
            cnt += 1
        print(f"  lex_glue case_hold: {cnt}개")
    except Exception as e:
        print(f"  ⚠ lex_glue case_hold: {e}")

    # lex_glue ledgar
    try:
        ds = load_dataset("lex_glue", "ledgar", split="train")
        label_names = ds.features["label"].names
        cnt = 0
        for item in ds:
            text      = item.get("text", "").strip()
            label_idx = item.get("label", -1)
            if not text or label_idx < 0:
                continue
            label_name = label_names[label_idx]
            q      = f"What type of contract provision is this? {text[:200]}"
            output = f"This provision is classified as: {label_name}.\n\nFull text: {text[:400]}"
            results.append({"domain": "LAW", "func_type": "WHAT",
                            "instruction": q, "context": "", "output": output})
            cnt += 1
        print(f"  lex_glue ledgar: {cnt}개")
    except Exception as e:
        print(f"  ⚠ lex_glue ledgar: {e}")

    results = balance(results)
    print(f"  → LAW 최종: {len(results)}개 | func: {dict(Counter(d['func_type'] for d in results))}")
    return results


# ────────────────────────────────────────────
# 3. STEM
# ────────────────────────────────────────────
def load_stem():
    print("\n[STEM] Loading...")
    results = []

    # SciQ
    try:
        for split in ["train", "validation", "test"]:
            ds = load_dataset("allenai/sciq", split=split)
            for item in ds:
                q       = item.get("question", "").strip()
                support = item.get("support", "").strip()
                answer  = item.get("correct_answer", "").strip()
                if not is_valid(q) or not answer:
                    continue
                output = f"{support} Therefore, the answer is: {answer}." if support else f"The answer is: {answer}."
                if len(output.split()) < 10:
                    continue
                results.append({"domain": "STEM", "func_type": label_func_type(q),
                                "instruction": q, "context": support[:500], "output": output})
        print(f"  SciQ: {len(results)}개")
    except Exception as e:
        print(f"  ⚠ SciQ: {e}")

    # MMLU STEM
    stem_subjects = [
        "high_school_physics", "college_physics",
        "high_school_chemistry", "college_chemistry",
        "high_school_mathematics", "college_mathematics",
        "electrical_engineering", "computer_science",
        "abstract_algebra", "conceptual_physics",
        "astronomy", "college_computer_science",
    ]
    results += load_mmlu_subjects(stem_subjects, "STEM")

    results = balance(results)
    print(f"  → STEM 최종: {len(results)}개 | func: {dict(Counter(d['func_type'] for d in results))}")
    return results


# ────────────────────────────────────────────
# 4. SOC
# ────────────────────────────────────────────
def load_soc():
    print("\n[SOC] Loading...")

    soc_subjects = [
        "sociology", "high_school_psychology", "professional_psychology",
        "high_school_geography", "political_science", "econometrics",
        "public_relations", "high_school_government_and_politics",
        "us_foreign_policy", "security_studies",
        "human_sexuality", "moral_disputes",
    ]
    results = load_mmlu_subjects(soc_subjects, "SOC")

    # CommonsenseQA
    try:
        ds = load_dataset("commonsense_qa", split="train")
        cnt = 0
        for item in ds:
            q       = item.get("question", "").strip()
            choices = item.get("choices", {})
            ans_key = item.get("answerKey", "").strip()
            labels  = choices.get("label", [])
            texts   = choices.get("text", [])
            choice_map = dict(zip(labels, texts))
            ans_txt    = choice_map.get(ans_key, "")
            if not is_valid(q) or not ans_txt:
                continue
            choices_str = "\n".join([f"{l}: {t}" for l, t in zip(labels, texts)])
            output = f"The correct answer is {ans_key}: {ans_txt}\n\nChoices:\n{choices_str}"
            results.append({"domain": "SOC", "func_type": label_func_type(q),
                            "instruction": q, "context": "", "output": output})
            cnt += 1
        print(f"  CommonsenseQA: {cnt}개")
    except Exception as e:
        print(f"  ⚠ CommonsenseQA: {e}")

    results = balance(results)
    print(f"  → SOC 최종: {len(results)}개 | func: {dict(Counter(d['func_type'] for d in results))}")
    return results


# ────────────────────────────────────────────
# 5. CROSS
# ────────────────────────────────────────────
GPQA_MAP = {"biology": "MED+STEM", "chemistry": "STEM+MED", "physics": "STEM+SOC"}

def load_cross():
    print("\n[CROSS] Loading...")
    results = []

    # ARC-Challenge + ARC-Easy
    for arc_type in ["ARC-Challenge", "ARC-Easy"]:
        try:
            for split in ["train", "validation", "test"]:
                ds = load_dataset("allenai/ai2_arc", arc_type, split=split)
                for item in ds:
                    q       = item.get("question", "").strip()
                    choices = item.get("choices", {})
                    ans_key = item.get("answerKey", "").strip()
                    labels  = choices.get("label", [])
                    texts   = choices.get("text", [])
                    choice_map = dict(zip(labels, texts))
                    ans_txt    = choice_map.get(ans_key, "")
                    if not is_valid(q) or not ans_txt:
                        continue
                    choices_str = "\n".join([f"{l}: {t}" for l, t in zip(labels, texts)])
                    output = f"The correct answer is {ans_key}: {ans_txt}\n\nChoices:\n{choices_str}"
                    cross_pair = "STEM+SOC" if arc_type == "ARC-Challenge" else "CROSS"
                    results.append({"domain": "CROSS", "cross_pair": cross_pair,
                                    "func_type": label_func_type(q),
                                    "instruction": q, "context": "", "output": output})
            print(f"  {arc_type}: 누적 {len(results)}개")
        except Exception as e:
            print(f"  ⚠ {arc_type}: {e}")

    # MMLU cross subjects
    cross_subjects = [
        "global_facts", "formal_logic", "philosophy",
        "logical_fallacies", "miscellaneous", "world_religions",
    ]
    mmlu_cross = load_mmlu_subjects(cross_subjects, "CROSS")
    for item in mmlu_cross:
        item["cross_pair"] = "CROSS"
    results += mmlu_cross

    results = balance(results, max_per_func=1000, max_total=TARGET_PER_DOMAIN)
    print(f"  → CROSS 최종: {len(results)}개 | func: {dict(Counter(d['func_type'] for d in results))}")
    return results


# ────────────────────────────────────────────
# 저장
# ────────────────────────────────────────────
def save(data, domain):
    path = os.path.join(OUTPUT_DIR, f"{domain.lower()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 저장: {path} ({len(data)}개)\n")


# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
if __name__ == "__main__":
    print(f"목표: 도메인당 {TARGET_PER_DOMAIN}개+")
    print(f"저장: {OUTPUT_DIR}\n")

    total = 0

    med_data   = load_med();   save(med_data,   "MED");   total += len(med_data)
    law_data   = load_law();   save(law_data,   "LAW");   total += len(law_data)
    stem_data  = load_stem();  save(stem_data,  "STEM");  total += len(stem_data)
    soc_data   = load_soc();   save(soc_data,   "SOC");   total += len(soc_data)
    cross_data = load_cross(); save(cross_data, "CROSS"); total += len(cross_data)

    print("=" * 50)
    print("✅ 완료!")
    print(f"  MED   : {len(med_data):>5}개")
    print(f"  LAW   : {len(law_data):>5}개")
    print(f"  STEM  : {len(stem_data):>5}개")
    print(f"  SOC   : {len(soc_data):>5}개")
    print(f"  CROSS : {len(cross_data):>5}개")
    print(f"  TOTAL : {total:>5}개")
    print(f"\n저장: {OUTPUT_DIR}")
    print("=" * 50)
