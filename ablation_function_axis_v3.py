"""
Ablation v3: Function Axis Isolation — 진짜 단일 변인 통제

비교군 A (Scaffold-Generic):
  - FEW_SHOT["CROSS"]            유지 (pipeline.py 그대로)
  - system role                  유지 (pipeline.py 그대로)
  - MCQ-prevention 지시문        유지 (pipeline.py 그대로)
  - max_new_tokens / temperature 유지 (pipeline.py 그대로)
  - domain_instruction만 교체:
      "provide a comprehensive analytical answer with specific mechanisms and implications"
      → "Provide a comprehensive answer based on the evidence above."

비교군 B (NexRoute Full — Baseline):
  - pipeline.py 출력 그대로 사용 (eval_results_7b_v3_500.json stored pred_answer)
  - 동일 100 GT-CROSS 쿼리셋에서 ROUGE-L/BERTScore 재계산
    (기존 ablation_domain_only.py 의 hardcoded n=119 수치와 다름 — 수정된 값)

평가 집합: GT expected_domain=CROSS (gt_dataset_500.json 기준) 100개
LoRA: lora_weights_7b_v3  /  Router: router_multilabel.pt (BERT)
저장: ablation_function_axis_v3.json
"""
import json, os, re, sys, datetime, torch, hashlib, numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from peft import PeftModel
from pipeline import DualMoEPipeline, FEW_SHOT, DOMAIN_SYSTEM

# ── 경로 상수 ──────────────────────────────────────────────────────────
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
RESULT_PATH = "/home/sclab/moe_model/eval_results_7b_v3_500.json"
LORA_DIR    = "/home/sclab/moe_model/lora_weights_7b_v3"
ROUTER_PATH = "/home/sclab/moe_model/router_multilabel.pt"
OUT_PATH    = "/home/sclab/moe_model/ablation_function_axis_v3.json"
SCRIPT_NAME = "ablation_function_axis_v3.py"
TIMESTAMP   = datetime.datetime.now().isoformat()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── lora_law MD5 ────────────────────────────────────────────────────────
def md5_dir(path):
    h = hashlib.md5()
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                h.update(f.read())
    return h.hexdigest()

lora_law_md5 = md5_dir(os.path.join(LORA_DIR, "lora_law"))

# ── GT 로드 ──────────────────────────────────────────────────────────────
with open(GT_PATH) as f:
    gt_list = json.load(f)
gt_map = {d["instruction"].strip(): d for d in gt_list}

# ── eval_results_7b_v3_500.json 로드 → GT-CROSS 100개 추출 ─────────────
with open(RESULT_PATH) as f:
    existing = json.load(f)
exist_map = {item["query"].strip(): item for item in existing}

cross_items = []
for item in existing:
    q = item.get("query", "").strip()
    if gt_map.get(q, {}).get("expected_domain") == "CROSS":
        gold = item.get("gold_answer", gt_map[q].get("gold_answer", "")).strip()
        pred_b = item.get("pred_answer", "").strip()
        if gold and pred_b:
            cross_items.append({
                "query":      q,
                "gold_answer": gold,
                "pred_b":     pred_b,          # B: stored pipeline.py output
                "domains":    item.get("domain_pred", []),
                "is_cross":   item.get("is_cross", False),
            })

print(f"GT-CROSS 쿼리: {len(cross_items)}개")
assert len(cross_items) == 100, f"예상 100개, 실제 {len(cross_items)}개"


# ── 비교군 B: eval_results_7b_v3_500.json stored pred 재계산 ────────────
print("\n비교군 B (NexRoute Full): 메트릭 재계산 (GT-CROSS n=100)...")
b_golds = [item["gold_answer"] for item in cross_items]
b_preds = [item["pred_b"]      for item in cross_items]

rscorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
b_rouge  = np.mean([rscorer.score(g, p)["rougeL"].fmeasure
                    for g, p in zip(b_golds, b_preds)]) * 100
_, _, F1 = bert_score_fn(b_preds, b_golds, lang="en",
                         model_type="roberta-large",
                         rescale_with_baseline=False,
                         verbose=False, device=device)
b_bert = F1.mean().item() * 100
print(f"  ROUGE-L = {b_rouge:.2f}  BERTScore = {b_bert:.2f}")


# ── 파이프라인 초기화 (A 추론용) ──────────────────────────────────────────
print("\n파이프라인 초기화 (lora_weights_7b_v3 + BERT router)...")
pipe = DualMoEPipeline()
pipe.load_router()
pipe.load_embed_model()
pipe.load_base_model()
pipe.build_db()

LORA_MAP = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc",
            "STEM":"lora_stem","CROSS":"lora_cross"}
pipe.base_model = PeftModel.from_pretrained(
    pipe.base_model,
    os.path.join(LORA_DIR, "lora_cross"),
    adapter_name="CROSS"
)
for dom, lname in LORA_MAP.items():
    if dom == "CROSS":
        continue
    pipe.base_model.load_adapter(os.path.join(LORA_DIR, lname), adapter_name=dom)
pipe.current_lora = "CROSS"
pipe.base_model.set_adapter("CROSS")
print("모든 어댑터 로드 완료")


# ── Ablation A 프롬프트 빌더 ───────────────────────────────────────────
# pipeline.py generate()와 100% 동일, 단 domain_instruction만 고정 일반문으로 교체
GENERIC_INSTRUCTION = "Provide a comprehensive answer based on the evidence above."

def build_prompt_a(query: str, context: list, is_cross: bool, domains: list) -> str:
    ctx_text = "\n".join(
        [f"[{r['domain']}] {r['text'][:200]}" for r in context]
    )
    # system / few_shot: pipeline.py CROSS 분기와 동일
    if is_cross:
        system   = (
            f"You are a multidisciplinary expert specializing in "
            f"{' and '.join(domains)}. "
            f"Integrate knowledge from both domains to answer accurately."
        )
        few_shot = FEW_SHOT["CROSS"]
    else:
        system   = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")
        few_shot = FEW_SHOT.get(domains[0], FEW_SHOT["STEM"])

    # ★ 유일한 변경점: domain_instruction → GENERIC_INSTRUCTION (고정)
    prompt = (
        f"{system}\n\n"
        f"Here is an example of a high-quality answer:\n"
        f"{few_shot}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"{GENERIC_INSTRUCTION} "
        f"Do NOT use multiple choice format. "
        f"Do NOT list choices A, B, C, D. "
        f"Do NOT begin with 'The correct answer is' or 'The correct holding is option'. "
        f"Write in flowing paragraph form only.\n\n"
        f"Answer:"
    )
    return prompt


def build_prompt_b_template(query: str, context: list, is_cross: bool, domains: list) -> str:
    """pipeline.py generate()와 동일 — B 비교 확인용 (실제 B는 stored pred 사용)"""
    ctx_text = "\n".join(
        [f"[{r['domain']}] {r['text'][:200]}" for r in context]
    )
    if is_cross:
        system   = (
            f"You are a multidisciplinary expert specializing in "
            f"{' and '.join(domains)}. "
            f"Integrate knowledge from both domains to answer accurately."
        )
        few_shot = FEW_SHOT["CROSS"]
    else:
        system   = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")
        few_shot = FEW_SHOT.get(domains[0], FEW_SHOT["STEM"])

    primary = domains[0] if not is_cross else "CROSS"
    if primary == "LAW":
        domain_instruction = (
            "Provide a comprehensive explanation of the legal holding, "
            "principle, or rule. Describe the court's reasoning, the "
            "applicable legal standard, any relevant exceptions, and the "
            "practical implications. Write as a detailed legal analysis "
            "in full paragraph form."
        )
    else:
        domain_instruction = (
            "provide a comprehensive analytical answer with specific "
            "mechanisms and implications"
        )
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
    return prompt


def generate_a(pipe, query: str, context: list, is_cross: bool, domains: list) -> tuple:
    """prompt_a 반환 포함"""
    prompt = build_prompt_a(query, context, is_cross, domains)
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
    # MCQ 잔여 패턴 제거 (pipeline.py 동일)
    answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
    answer = re.sub(r'The correct holding is option\s*\d+\s*:\s*', '', answer).strip()
    answer = re.sub(r'The correct holding is\s*:\s*', '', answer).strip()
    for kw in ["Choices:", "All answers:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in answer:
            answer = answer[:answer.index(kw)].strip()
    answer = re.sub(r'\n\d+:.*', '', answer).strip()
    return (answer or "[GENERATION_FAILED]"), prompt


# ── 비교군 A 추론 루프 ──────────────────────────────────────────────────
print(f"\n비교군 A 추론 시작: 100개 CROSS 쿼리...")
a_preds, a_golds = [], []
raw_results_a = []
sample_prompts = []   # 처음 3개 raw prompt 저장
errors = 0

for i, item in enumerate(cross_items):
    query    = item["query"]
    gold     = item["gold_answer"]
    domains  = item["domains"]
    is_cross = item["is_cross"]

    try:
        # 동일 쿼리에 대해 retrieval 재실행 (B와 동일 조건)
        context = pipe.retrieve(query, domains, is_cross)

        lora_target = "CROSS" if is_cross else (domains[0] if domains else "CROSS")
        if pipe.current_lora != lora_target:
            pipe.base_model.set_adapter(lora_target)
            pipe.current_lora = lora_target

        pred_a, prompt_a = generate_a(pipe, query, context, is_cross, domains)
        a_preds.append(pred_a)
        a_golds.append(gold)
        raw_results_a.append({
            "id":         i,
            "query":      query,
            "gold_answer": gold,
            "pred_answer_a": pred_a,
            "pred_answer_b": item["pred_b"],
            "domains":    domains,
            "is_cross":   is_cross,
        })
        # 처음 3개 sample prompt 저장 (raw text 전문)
        if len(sample_prompts) < 3:
            ctx_text = "\n".join([f"[{r['domain']}] {r['text'][:200]}" for r in context])
            prompt_b = build_prompt_b_template(query, ctx_text if False else context,
                                               is_cross, domains)
            sample_prompts.append({
                "sample_id": i,
                "query":     query,
                "domains":   domains,
                "is_cross":  is_cross,
                "prompt_A":  prompt_a,
                "prompt_B":  prompt_b,
            })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>3}/100] 완료  (pred: {pred_a[:60]}...)")
    except Exception as e:
        errors += 1
        print(f"  [{i+1}] 오류: {e}", file=sys.stderr)
        a_preds.append("")
        a_golds.append(gold)

print(f"\n추론 완료. 오류: {errors}개")


# ── 비교군 A 메트릭 ─────────────────────────────────────────────────────
print("\n비교군 A 메트릭 계산 중...")
valid_a = [(g, p) for g, p in zip(a_golds, a_preds) if p and p != "[GENERATION_FAILED]"]
vg_a, vp_a = zip(*valid_a) if valid_a else ([], [])

a_rouge = np.mean([rscorer.score(g, p)["rougeL"].fmeasure
                   for g, p in zip(vg_a, vp_a)]) * 100
_, _, F1_a = bert_score_fn(list(vp_a), list(vg_a), lang="en",
                            model_type="roberta-large",
                            rescale_with_baseline=False,
                            verbose=False, device=device)
a_bert = F1_a.mean().item() * 100
print(f"  ROUGE-L = {a_rouge:.2f}  BERTScore = {a_bert:.2f}  유효 샘플 = {len(valid_a)}/100")


# ── 결과 저장 ─────────────────────────────────────────────────────────────
output = {
    "_meta": {
        "script":           SCRIPT_NAME,
        "timestamp":        TIMESTAMP,
        "lora_dir":         LORA_DIR,
        "lora_law_md5":     lora_law_md5,
        "router_path":      ROUTER_PATH,
        "base_model":       "Qwen/Qwen2.5-7B-Instruct",
        "eval_set":         "GT expected_domain=CROSS (gt_dataset_500.json)",
        "n":                100,
        "ablation_axis":    "Function Axis (domain_instruction only)",
        "change_A_vs_B":    (
            "A: domain_instruction → fixed 'Provide a comprehensive answer based on the evidence above.' "
            "B: domain_instruction = 'provide a comprehensive analytical answer with specific mechanisms and implications' "
            "All other prompt components (FEW_SHOT, system role, MCQ-prevention) identical."
        ),
        "baseline_B_source": (
            "eval_results_7b_v3_500.json (before_law_fix) — stored pred_answer, "
            "recomputed on GT-CROSS n=100 (NOT n=119 pipeline-predicted CROSS)"
        ),
        "prompt_template_A": (
            "{system}\n\nHere is an example of a high-quality answer:\n{few_shot}\n\n"
            "Context:\n{ctx_text}\n\nQuestion: {query}\n\n"
            "Provide a comprehensive answer based on the evidence above. "
            "Do NOT use multiple choice format. Do NOT list choices A, B, C, D. "
            "Do NOT begin with 'The correct answer is' or 'The correct holding is option'. "
            "Write in flowing paragraph form only.\n\nAnswer:"
        ),
        "prompt_template_B": (
            "{system}\n\nHere is an example of a high-quality answer:\n{few_shot}\n\n"
            "Context:\n{ctx_text}\n\nQuestion: {query}\n\n"
            "Following the example style above, {domain_instruction}. "
            "Do NOT use multiple choice format. Do NOT list choices A, B, C, D. "
            "Do NOT begin with 'The correct answer is' or 'The correct holding is option'. "
            "Write in flowing paragraph form only.\n\nAnswer:"
        ),
        "sample_prompts_raw": sample_prompts,
    },
    "Scaffold-Generic (A)": {
        "rouge_l":    round(float(a_rouge), 2),
        "bertscore":  round(float(a_bert),  2),
        "n":          len(valid_a),
        "errors":     errors,
        "description": "FEW_SHOT + system role 유지, domain_instruction → generic",
    },
    "NexRoute Full (B)": {
        "rouge_l":    round(float(b_rouge), 2),
        "bertscore":  round(float(b_bert),  2),
        "n":          100,
        "description": "pipeline.py 출력 그대로, GT-CROSS n=100 재계산",
    },
    "delta_B_minus_A": {
        "rouge_l":    round(float(b_rouge) - float(a_rouge), 2),
        "bertscore":  round(float(b_bert)  - float(a_bert),  2),
        "interpretation": "양수 = Function Axis(domain_instruction)가 기여, 음수 = 해가 됨",
    },
    "raw_predictions": raw_results_a,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n저장 완료: {OUT_PATH}")

# ── 최종 테이블 ────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"{'Model':<40} {'ROUGE-L':>8} {'BERTScore':>10}")
print("-" * 70)
print(f"{'NexRoute Full — pipeline.py (B)':<40} {b_rouge:>8.2f} {b_bert:>10.2f}")
print(f"{'Scaffold-Generic — domain_inst 제거 (A)':<40} {a_rouge:>8.2f} {a_bert:>10.2f}")
print("-" * 70)
dr = b_rouge - a_rouge
db = b_bert  - a_bert
print(f"{'Δ (B − A, Function Axis 기여분)':<40} {dr:>+8.2f} {db:>+10.2f}")
print("=" * 70)
print(f"\n주의: 기존 ablation_domain_only.py의 B 수치(29.51/89.44)는 n=119 기반 — 이 결과와 다를 수 있음")
