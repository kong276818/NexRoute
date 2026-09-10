"""
regen_law_v4.py
===============
LAW 도메인 100샘플을 lora_law_v4로 재생성.
기존 preds_500_nexroute.json에서 LAW 샘플만 교체 →
preds_500_nexroute_lawv4.json 저장.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/regen_law_v4.py
"""
import os, json, re, gc, torch
import numpy as np
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_7B    = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_V4    = "/home/sclab/moe_model/lora_weights_7b_v3/lora_law_v4"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
EMBED_NAME = "intfloat/multilingual-e5-large"

IN_PATH    = "/home/sclab/moe_model/reinference_results/preds_500_nexroute.json"
OUT_PATH   = "/home/sclab/moe_model/reinference_results/preds_500_nexroute_lawv4.json"
CKPT_PATH  = "/home/sclab/moe_model/reinference_results/preds_500_nexroute_lawv4_ckpt.json"

TOP_K = 10

# v4 훈련에 사용된 시스템 프롬프트
SYSTEM_LAW_V4 = (
    "You are a legal expert. Answer accurately based on legal principles, "
    "statutes, and case law. Provide detailed analytical explanations in "
    "flowing paragraph form. Do NOT use multiple choice format. "
    "Do NOT begin with 'The correct holding is option'."
)


def clean_answer(text: str) -> str:
    text = re.sub(r'The correct (holding|answer) is option\s*\d+\s*:\s*', '', text).strip()
    text = re.sub(r'The correct (holding|answer) is\s*:\s*', '', text).strip()
    text = re.sub(r'All holdings?:\s*', '', text).strip()
    text = re.sub(r'All answers?:\s*', '', text).strip()
    for kw in ["Choices:", "Options:", "\nA:", "\nB:", "\nC:", "\nD:"]:
        if kw in text:
            text = text[:text.index(kw)].strip()
    text = re.sub(r'\n\d+:.*', '', text).strip()
    return text if text.strip() else "[FAILED]"


# ── 기존 결과 로드 ─────────────────────────────────────────────────────────────
print("[*] 기존 preds_500_nexroute.json 로드...")
with open(IN_PATH) as f:
    all_preds = json.load(f)

law_samples = [(i, p) for i, p in enumerate(all_preds) if p["expected_domain"] == "LAW"]
print(f"  LAW 샘플: {len(law_samples)}개")

# 체크포인트 로드 (재개 시)
done_idxs = set()
if os.path.exists(CKPT_PATH):
    with open(CKPT_PATH) as f:
        ckpt = json.load(f)
    done_idxs = {item["idx"] for item in ckpt}
    print(f"  체크포인트: {len(done_idxs)}개 이미 완료")
else:
    ckpt = []

# ── 임베딩 모델 + LAW FAISS 구축 ─────────────────────────────────────────────
print("\n[*] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_NAME, device="cuda")

print("[*] LAW 코퍼스 로드 및 FAISS 인덱스 구축...")
with open(os.path.join(DATA_DIR, "law_rewritten.json")) as f:
    law_corpus = json.load(f)

law_texts = []
for d in law_corpus:
    out = d.get("output", d.get("gold_answer", ""))
    if any(k in out for k in ["The correct answer", "All answers:", "All holdings:"]):
        continue
    law_texts.append(f"{d['instruction']} {out}")

embs = embed_model.encode(law_texts, batch_size=128, show_progress_bar=True)
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
law_index = faiss.IndexFlatIP(embs.shape[1])
law_index.add(embs.astype(np.float32))
print(f"  LAW 인덱스: {len(law_texts)}개 문서")


def retrieve_law(query: str) -> list:
    emb = embed_model.encode([query])
    emb = emb / np.linalg.norm(emb)
    D, I = law_index.search(emb.astype(np.float32), TOP_K)
    return [law_texts[i] for i in I[0]]


# ── 언어 모델 로드 ─────────────────────────────────────────────────────────────
print("\n[*] Qwen2.5-7B + lora_law_v4 로드...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_7B,
    quantization_config=bnb,
    device_map="cuda:0",
    trust_remote_code=True,
    local_files_only=True,
    attn_implementation="eager",
)
model = PeftModel.from_pretrained(base_model, LORA_V4)
model.eval()
print("[*] 로드 완료!")


def generate(query: str, context: list) -> str:
    ctx = "\n".join([c[:250] for c in context])
    prompt = (
        f"{SYSTEM_LAW_V4}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n\n"
        f"Provide a detailed legal analysis in flowing paragraph form.\n\nAnswer:"
    )
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=350,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return clean_answer(answer)


# ── 재생성 루프 ────────────────────────────────────────────────────────────────
print(f"\n[*] LAW 재생성 시작 (v4 adapter)...")
for pos, (arr_idx, sample) in enumerate(law_samples):
    global_idx = sample["idx"]
    if global_idx in done_idxs:
        continue

    query = sample["instruction"]
    ctx   = retrieve_law(query)
    ans   = generate(query, ctx)

    result = {
        "idx":              global_idx,
        "instruction":      query,
        "gold_answer":      sample["gold_answer"],
        "expected_domain":  "LAW",
        "is_cross":         False,
        "pred_domain":      sample.get("pred_domain", "LAW"),
        "pred_is_cross":    sample.get("pred_is_cross", False),
        "pred_domains":     sample.get("pred_domains", ["LAW"]),
        "pred_answer":      ans,
        "adapter_version":  "lora_law_v4",
    }
    ckpt.append(result)
    done_idxs.add(global_idx)

    # 진행상황 출력
    has_citation = "All holdings" in ans or "holding that" in ans.lower()[:60]
    print(f"  [{pos+1:3d}/{len(law_samples)}] idx={global_idx:3d}  "
          f"citation={'Y' if has_citation else 'N'}  "
          f"len={len(ans.split())}w  {ans[:80]!r}")

    # 50개마다 체크포인트
    if len(done_idxs) % 50 == 0:
        with open(CKPT_PATH, "w") as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2)
        print(f"  [체크포인트 저장] {len(done_idxs)}개 완료")

# ── 최종 병합 저장 ─────────────────────────────────────────────────────────────
print(f"\n[*] 최종 병합: LAW v4 결과 → preds_500_nexroute_lawv4.json")

# ckpt를 idx로 인덱싱
v4_map = {item["idx"]: item for item in ckpt}

merged = []
for sample in all_preds:
    global_idx = sample["idx"]
    if sample["expected_domain"] == "LAW" and global_idx in v4_map:
        merged.append(v4_map[global_idx])
    else:
        merged.append(sample)

merged.sort(key=lambda x: x["idx"])

with open(OUT_PATH, "w") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

# 체크포인트도 최종 저장
with open(CKPT_PATH, "w") as f:
    json.dump(ckpt, f, ensure_ascii=False, indent=2)

# 결과 요약
v4_law = [m for m in merged if m["expected_domain"] == "LAW"]
citation_n = sum(1 for x in v4_law
                 if "All holdings" in x["pred_answer"] or
                    "holding that" in x["pred_answer"].lower()[:60])
print(f"\n[+] 완료!")
print(f"  총 샘플: {len(merged)}")
print(f"  LAW citation 잔존: {citation_n}/100")
print(f"  출력: {OUT_PATH}")

gc.collect()
torch.cuda.empty_cache()
