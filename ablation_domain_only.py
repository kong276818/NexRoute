"""
Ablation: Domain-Only (Function Axis masked)
비교군 A: BERT router + lora_weights_7b_v3 + scaffold 제거
비교군 B: 기존 Table 3 결과 (ROUGE-L 29.51 / BERTScore 89.44, CROSS n=100)

차이: Function scaffold (Few-shot + domain instruction) 유무만 다름.
      라우터, LoRA, 검색, 생성 파라미터 모두 동일.

출력: ablation_function_axis_v2.json
"""
import json, os, re, sys, torch, numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from pipeline import DualMoEPipeline

# ── 설정 ──────────────────────────────────────────────────────────────
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
RESULT_PATH = "/home/sclab/moe_model/eval_results_7b_v3_500.json"
OUT_PATH    = "/home/sclab/moe_model/ablation_function_axis_v2.json"

# 비교군 B: Table 3 고정값 (재실행 불필요)
TABLE3_CROSS = {"rouge_l": 29.51, "bertscore": 89.44, "n": 100}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── CROSS 도메인 쿼리 + gold answer 로드 ──────────────────────────────
with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

with open(RESULT_PATH) as f:
    existing = json.load(f)

cross_items = []
for item in existing:
    q = item.get("query", "").strip()
    gt = gt_map.get(q, {})
    if gt.get("expected_domain") == "CROSS":
        gold = item.get("gold_answer", gt.get("gold_answer", "")).strip()
        if gold:
            cross_items.append({"query": q, "gold_answer": gold})

print(f"CROSS 도메인 쿼리: {len(cross_items)}개")
assert len(cross_items) == 100, f"예상 100개, 실제 {len(cross_items)}개"


# ── 파이프라인 로드 (pipeline.py 그대로) ──────────────────────────────
print("\n파이프라인 초기화 (lora_weights_7b_v3 + BERT router)...")
pipe = DualMoEPipeline()
pipe.load_router()
pipe.load_embed_model()
pipe.load_base_model()
pipe.build_db()

# 모든 어댑터 미리 로드
from peft import PeftModel
import os as _os
LORA_DIR = "/home/sclab/moe_model/lora_weights_7b_v3"
LORA_MAP = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc",
            "STEM":"lora_stem","CROSS":"lora_cross"}
pipe.base_model = PeftModel.from_pretrained(
    pipe.base_model,
    _os.path.join(LORA_DIR, "lora_cross"),
    adapter_name="CROSS"
)
for dom, lname in LORA_MAP.items():
    if dom == "CROSS":
        continue
    pipe.base_model.load_adapter(_os.path.join(LORA_DIR, lname), adapter_name=dom)
pipe.current_lora = "CROSS"
pipe.base_model.set_adapter("CROSS")
print("모든 어댑터 로드 완료")


def generate_domain_only(pipe, query: str, context: list,
                         is_cross: bool, domains: list) -> str:
    """
    Function scaffold 제거 버전:
    Few-shot 예시, system role, domain instruction 모두 제거.
    단순 RAG 템플릿만 사용.
    """
    ctx_text = "\n".join(
        [f"[{r['domain']}] {r['text'][:200]}" for r in context]
    )
    # ★ scaffold 없는 단순 프롬프트
    prompt = (
        f"Based on the following evidence, answer the question:\n\n"
        f"{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = pipe.base_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = pipe.base_tokenizer(text, return_tensors="pt").to(pipe.base_model.device)

    with torch.no_grad():
        outputs = pipe.base_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=pipe.base_tokenizer.eos_token_id,
        )
    answer = pipe.base_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # MCQ 잔여 패턴 제거 (pipeline.py와 동일)
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
    for kw in ["Choices:", "All answers:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    return answer if answer else "[GENERATION_FAILED]"


# ── 추론 루프 ──────────────────────────────────────────────────────────
print(f"\n추론 시작: {len(cross_items)}개 CROSS 쿼리...")
preds, golds = [], []
raw_results = []
errors = 0

for i, item in enumerate(cross_items):
    query = item["query"]
    gold  = item["gold_answer"]
    try:
        routing  = pipe.classify_domain(query)
        is_cross = routing["is_cross"]
        domains  = routing["domains"]
        context  = pipe.retrieve(query, domains, is_cross)

        lora_target = "CROSS" if is_cross else domains[0]
        if pipe.current_lora != lora_target:
            pipe.base_model.set_adapter(lora_target)
            pipe.current_lora = lora_target

        pred = generate_domain_only(pipe, query, context, is_cross, domains)
        preds.append(pred)
        golds.append(gold)
        raw_results.append({
            "id": i,
            "query": query,
            "gold_answer": gold,
            "pred_answer": pred,
            "domains": domains,
            "is_cross": is_cross,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/100] 완료  (최근 pred: {pred[:60]}...)")
    except Exception as e:
        errors += 1
        print(f"  [{i+1}] 오류: {e}", file=sys.stderr)
        preds.append("")
        golds.append(gold)

print(f"\n추론 완료. 오류: {errors}개")


# ── 메트릭 계산 ───────────────────────────────────────────────────────
print("\nROUGE-L + BERTScore 계산 중...")
valid = [(g, p) for g, p in zip(golds, preds) if p and p != "[GENERATION_FAILED]"]
v_golds, v_preds = zip(*valid) if valid else ([], [])

rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = np.mean([
    rscorer.score(g, p)["rougeL"].fmeasure
    for g, p in zip(v_golds, v_preds)
]) * 100

_, _, F1 = bert_score_fn(
    list(v_preds), list(v_golds),
    lang="en",
    model_type="roberta-large",
    rescale_with_baseline=False,
    verbose=False,
    device=device,
)
bertscore = F1.mean().item() * 100

print(f"  ROUGE-L  = {rouge_l:.2f}")
print(f"  BERTScore = {bertscore:.2f}")
print(f"  유효 샘플 = {len(valid)}/100")


# ── 결과 저장 ─────────────────────────────────────────────────────────
output = {
    "_meta": {
        "lora_dir":     "/home/sclab/moe_model/lora_weights_7b_v3",
        "lora_law_md5": "76dd9971745d626f8d44466e4120808f",
        "router":       "BERT (router_multilabel.pt)",
        "base_model":   "Qwen/Qwen2.5-7B-Instruct",
        "domain":       "CROSS",
        "n":            100,
        "scaffold":     "removed (Domain-Only baseline)",
    },
    "Domain-Only (Function masked)": {
        "rouge_l":    round(float(rouge_l), 2),
        "bertscore":  round(float(bertscore), 2),
        "n":          len(valid),
        "errors":     errors,
    },
    "NexRoute 7B (Full Dual-Axis)": {
        "rouge_l":   TABLE3_CROSS["rouge_l"],
        "bertscore": TABLE3_CROSS["bertscore"],
        "n":         TABLE3_CROSS["n"],
        "source":    "Table 3 (eval_results_7b_v3_500.json, before_law_fix)",
    },
    "delta": {
        "rouge_l":   round(TABLE3_CROSS["rouge_l"]   - float(rouge_l), 2),
        "bertscore": round(TABLE3_CROSS["bertscore"] - float(bertscore), 2),
    },
    "raw_predictions": raw_results,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n저장 완료: {OUT_PATH}")

# ── 최종 테이블 출력 ──────────────────────────────────────────────────
print()
print("=" * 58)
print(f"{'Model':<35} {'ROUGE-L':>8} {'BERTScore':>10}")
print("-" * 58)
print(f"{'Domain-Only (Function masked)':<35} {rouge_l:>8.2f} {bertscore:>10.2f}")
print(f"{'NexRoute 7B (Full Dual-Axis)':<35} {TABLE3_CROSS['rouge_l']:>8.2f} {TABLE3_CROSS['bertscore']:>10.2f}")
print("-" * 58)
delta_r = TABLE3_CROSS["rouge_l"]   - rouge_l
delta_b = TABLE3_CROSS["bertscore"] - bertscore
print(f"{'Δ (Function Axis 기여분)':<35} {delta_r:>+8.2f} {delta_b:>+10.2f}")
print("=" * 58)
