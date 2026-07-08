"""
STEP 1: OOD 데이터 로더
- MedQuAD  → MED 도메인  → medquad_med.jsonl
- Law-StackExchange → LAW 도메인 → lawse_law.jsonl

출력 스키마 (각 라인):
  {"id":..., "question":..., "gold_answer":..., "gold_domain":"MED"/"LAW"}
"""

import json
import random
import re
import os
from datasets import load_dataset

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED    = 42

# ── 유틸 ──────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;",  " ", text)
    text = re.sub(r"&amp;",   "&", text)
    text = re.sub(r"&lt;",    "<", text)
    text = re.sub(r"&gt;",    ">", text)
    text = re.sub(r"&quot;",  '"', text)
    text = re.sub(r"&#\d+;",  " ", text)
    text = re.sub(r"\s+",     " ", text)
    return text.strip()

def word_count(text: str) -> int:
    return len(text.split())

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  저장: {path}  ({len(records)}개)")

# ── 1. MedQuAD (MED) ──────────────────────────────────────────────────────

def load_medquad(n=100):
    print("\n[MedQuAD] 로드 중...")
    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    print(f"  전체: {len(ds)}개")

    records = []
    for row in ds:
        q = str(row.get("Question", "") or "").strip()
        a = str(row.get("Answer",   "") or "").strip()
        if not q or not a:
            continue
        if word_count(a) < 20:
            continue
        records.append({"question": q, "gold_answer": a})

    print(f"  필터 후: {len(records)}개 (answer ≥ 20 words)")

    random.seed(SEED)
    sample = random.sample(records, min(n, len(records)))

    result = []
    for i, rec in enumerate(sample):
        result.append({
            "id":          i,
            "question":    rec["question"],
            "gold_answer": rec["gold_answer"],
            "gold_domain": "MED",
        })

    print(f"  최종 샘플: {len(result)}개")
    print("\n  [샘플 3개]")
    for r in result[:3]:
        print(f"    id={r['id']}  Q: {r['question'][:80]}...")
        print(f"           A: {r['gold_answer'][:100]}...")
    return result

# ── 2. Law-StackExchange (LAW) ────────────────────────────────────────────

def pick_best_answer(answers: list) -> str | None:
    """score 최대 답변 선택 (is_accepted 필드 없음)."""
    if not answers:
        return None
    best = max(answers, key=lambda a: a.get("score", 0))
    body = str(best.get("body", "") or "").strip()
    return strip_html(body) if body else None

def load_lawse(n=100):
    print("\n[Law-StackExchange] 로드 중...")
    ds = load_dataset("ymoslem/Law-StackExchange", split="train")
    print(f"  전체: {len(ds)}개")
    print(f"  컬럼: {ds.column_names}")

    records = []
    for row in ds:
        # 질문: question_title 우선, 없으면 question_body (HTML 제거)
        title = str(row.get("question_title", "") or "").strip()
        qbody = str(row.get("question_body",  "") or "").strip()
        q = title if title else strip_html(qbody)
        if not q:
            continue

        # 최고 답변 선택
        answers = row.get("answers", []) or []
        a = pick_best_answer(answers)
        if not a:
            continue

        wc = word_count(a)
        if wc < 20 or wc > 400:
            continue

        records.append({"question": q, "gold_answer": a})

    print(f"  필터 후: {len(records)}개 (answer 20~400 words)")

    random.seed(SEED)
    sample = random.sample(records, min(n, len(records)))

    result = []
    for i, rec in enumerate(sample):
        result.append({
            "id":          i,
            "question":    rec["question"],
            "gold_answer": rec["gold_answer"],
            "gold_domain": "LAW",
        })

    print(f"  최종 샘플: {len(result)}개")
    print("\n  [샘플 3개]")
    for r in result[:3]:
        print(f"    id={r['id']}  Q: {r['question'][:80]}...")
        print(f"           A: {r['gold_answer'][:100]}...")
    return result

# ── 메인 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    med_records = load_medquad(n=100)
    save_jsonl(med_records, os.path.join(OUT_DIR, "medquad_med.jsonl"))

    law_records = load_lawse(n=100)
    save_jsonl(law_records, os.path.join(OUT_DIR, "lawse_law.jsonl"))

    print("\n[완료]")
    print(f"  MedQuAD   : {len(med_records)}개 → medquad_med.jsonl")
    print(f"  LawSE     : {len(law_records)}개 → lawse_law.jsonl")
