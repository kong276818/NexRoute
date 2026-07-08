"""
LAW 어댑터 v1 vs v2 품질 비교
- 모델: Qwen2.5-7B-Instruct + lora_law (v1) vs lora_law_v2 (v2)
- 데이터: gt_dataset_500.json 에서 LAW 100개
- 메트릭: ROUGE-L, BERTScore (RoBERTa-large), BLEU-4
"""
import json, os, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)

BASE_MODEL  = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
ADAPTER_V1  = "/home/sclab/moe_model/lora_weights_7b_v3/lora_law"
ADAPTER_V2  = "/home/sclab/moe_model/lora_weights_7b_v3/lora_law_v2"
OUT_PATH    = "/home/sclab/moe_model/eval_results/eval_law_v2.json"
os.makedirs("/home/sclab/moe_model/eval_results", exist_ok=True)

SYSTEM = "You are a legal expert. Answer accurately based on legal principles, statutes, and case law."
MAX_NEW_TOKENS = 256

# ── 데이터 로드 ───────────────────────────────────
print("[*] 데이터 로드...")
with open(GT_PATH) as f:
    all_data = json.load(f)
law_data = [d for d in all_data if d.get("expected_domain") == "LAW"]
print(f"  LAW 샘플: {len(law_data)}개")

# ── 모델 로드 (한 번만) ──────────────────────────
print("[*] 베이스 모델 로드...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
base_model.eval()

def make_prompt(instruction):
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def generate_batch(model, samples, batch_size=8):
    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = [make_prompt(s["instruction"]) for s in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, ids in enumerate(out):
            in_len = inputs["input_ids"].shape[1]
            gen = tokenizer.decode(ids[in_len:], skip_special_tokens=True).strip()
            results.append(gen)
        print(f"  {min(i+batch_size, len(samples))}/{len(samples)} 생성 완료", end="\r")
    print()
    return results

def compute_metrics(preds, refs):
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rs.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]

    smooth = SmoothingFunction().method1
    bleu_scores = []
    for pred, ref in zip(preds, refs):
        ref_tok  = ref.split()
        pred_tok = pred.split()
        bleu_scores.append(sentence_bleu([ref_tok], pred_tok,
                                         weights=(0.25,0.25,0.25,0.25),
                                         smoothing_function=smooth))

    _, _, bert_f1 = bert_score(preds, refs, lang="en",
                                model_type="roberta-large",
                                rescale_with_baseline=True,
                                verbose=False)

    return {
        "rouge_l": float(np.mean(rouge_scores) * 100),
        "bleu4":   float(np.mean(bleu_scores)  * 100),
        "bertscore": float(bert_f1.mean().item() * 100),
    }

refs = [d.get("gold_answer", d.get("output", "")) for d in law_data]
results = {}

for tag, adapter_path in [("v1_original", ADAPTER_V1), ("v2_retrained", ADAPTER_V2)]:
    print(f"\n{'='*50}")
    print(f"[*] 어댑터 로드: {tag}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print(f"[*] 생성 중...")
    preds = generate_batch(model, law_data, batch_size=4)

    print(f"[*] 메트릭 계산...")
    metrics = compute_metrics(preds, refs)
    results[tag] = {"metrics": metrics, "predictions": preds[:5]}

    print(f"  ROUGE-L:   {metrics['rouge_l']:.2f}")
    print(f"  BLEU-4:    {metrics['bleu4']:.2f}")
    print(f"  BERTScore: {metrics['bertscore']:.2f}")

    # 어댑터 언로드
    base_model = model.unload()
    del model
    torch.cuda.empty_cache()

print(f"\n{'='*50}")
print("결과 비교:")
for tag in ["v1_original", "v2_retrained"]:
    m = results[tag]["metrics"]
    print(f"  {tag}: ROUGE-L={m['rouge_l']:.2f} / BLEU-4={m['bleu4']:.2f} / BERTScore={m['bertscore']:.2f}")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n[*] 저장 완료: {OUT_PATH}")
