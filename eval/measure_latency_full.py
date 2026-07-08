"""
NexRoute 500-sample Latency Benchmark
논문 Efficiency 섹션용 3가지 지표:
  1. Routing & Retrieval (ms): classify_domain + retrieve + set_adapter 완료까지
  2. TTFT (ms): 질문 입력 시점 ~ 첫 번째 토큰 출력까지 (라우팅 포함 전체)
  3. Throughput (tok/s): 총 생성 토큰 수 / 첫 토큰 이후 완료까지 시간

TextIteratorStreamer로 TTFT를 정확하게 측정하며,
torch.cuda.synchronize()로 GPU 비동기 실행 오차를 방지합니다.
"""
import json
import time
import threading
import os
import re
import sys

import torch
import numpy as np
import faiss
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModel,
    TextIteratorStreamer,
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ── 설정 ──────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
LORA_DIR    = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR    = "/home/sclab/moe_model/processed_v2"
ROUTER_PATH = "/home/sclab/moe_model/router_multilabel.pt"
EMBED_MODEL = "intfloat/multilingual-e5-large"
GT_PATH     = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
OUT_PATH    = "/home/sclab/moe_model/latency_full_results.json"

DOMAINS     = ["LAW", "MED", "SOC", "STEM"]
THRESHOLD   = 0.30
GAP         = 0.40
TOP_K       = 10
MAX_NEW_TOKENS = 512

N_WARMUP    = 3    # 측정 제외 워밍업 샘플 수

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":  "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":  "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM": "You are a science and engineering expert. Answer accurately based on scientific principles.",
}
FEW_SHOT_LAW = (
    "Q: What is the doctrine of promissory estoppel?\n"
    "A: Promissory estoppel is a legal doctrine that prevents a party from "
    "withdrawing a promise when another party has reasonably relied on it to "
    "their detriment. Courts apply this principle to enforce promises even without "
    "formal consideration, thereby ensuring fairness and preventing injustice."
)
FEW_SHOT_DEFAULT = (
    "Q: How does CRISPR-Cas9 technology work?\n"
    "A: CRISPR-Cas9 uses a guide RNA to direct the Cas9 endonuclease to specific "
    "DNA sequences, creating a precise double-strand break that the cell then repairs."
)
FEW_SHOT_CROSS = (
    "Q: How might advances in neuroscience impact criminal justice policy?\n"
    "A: Advances in neuroscience are reshaping criminal justice by providing insights "
    "into the biological underpinnings of behavior, raising questions about moral "
    "responsibility and the balance between punishment and rehabilitation."
)

# ── 유틸 ──────────────────────────────────────────────────────────────
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ── Router 정의 ───────────────────────────────────────────────────────
class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 4)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


# ── 모델 로드 ─────────────────────────────────────────────────────────
print("=" * 65)
print("  NexRoute Full-Pipeline Latency Benchmark (500 samples)")
print("=" * 65)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)

print("\n[1/5] 베이스 모델 로드 (4bit)...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL, trust_remote_code=True, local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_cfg, device_map="auto",
    trust_remote_code=True, local_files_only=True, attn_implementation="eager",
)
model.eval()

print("[2/5] BERT 라우터 로드...")
router_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
router = DomainRouter()
router.load_state_dict(torch.load(ROUTER_PATH, map_location="cpu"))
router.eval().to("cuda")

print("[3/5] 임베딩 모델 로드...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("[4/5] FAISS 인덱스 구축...")
db = {}
domain_files = {
    "LAW": "law_rewritten.json", "MED": "med_rewritten.json",
    "SOC": "soc_rewritten.json", "STEM": "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}
for dom, fname in domain_files.items():
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    filtered = [d for d in data
                if not any(k in d.get("output", d.get("gold_answer", ""))
                           for k in ["The correct answer", "All answers:"])]
    texts = [f"{d['instruction']} {d.get('output', d.get('gold_answer', ''))}"
             for d in filtered]
    embs = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    db[dom] = (texts, idx)
    print(f"  {dom}: {len(texts)}개")

print("[5/5] LoRA 초기 어댑터(LAW) 로드...")
LORA_MAP = {"LAW": "lora_law", "MED": "lora_med",
            "SOC": "lora_soc", "STEM": "lora_stem", "CROSS": "lora_cross"}
model = PeftModel.from_pretrained(
    model, os.path.join(LORA_DIR, "lora_law"), adapter_name="LAW"
)
# 나머지 어댑터 미리 메모리에 로드 (set_adapter 전환만 측정하기 위해)
for dom, lname in LORA_MAP.items():
    if dom == "LAW":
        continue
    model.load_adapter(os.path.join(LORA_DIR, lname), adapter_name=dom)
model.set_adapter("LAW")
current_lora = "LAW"
print("  모든 어댑터 사전 로드 완료 (set_adapter 전환만 측정)")


# ── 핵심 함수 ─────────────────────────────────────────────────────────
def classify_domain(query: str):
    enc = router_tok(query, max_length=256, truncation=True,
                     padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = router(enc["input_ids"], enc["attention_mask"])
        scores = torch.sigmoid(logits).squeeze().cpu().tolist()
    domain_scores = dict(zip(DOMAINS, scores))
    sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
    top1, s1 = sorted_d[0]
    top2, s2 = sorted_d[1]
    is_cross = (s2 >= THRESHOLD) and ((s1 - s2) <= GAP)
    return {
        "is_cross": is_cross,
        "domains": [top1, top2] if is_cross else [top1],
    }


def retrieve(query: str, domains: list, is_cross: bool) -> list:
    emb_q = embed_model.encode([query])
    emb_q = emb_q / np.linalg.norm(emb_q)
    results = []
    if is_cross:
        per_domain = max(1, TOP_K // len(domains))
        for dom in domains:
            if dom not in db:
                continue
            texts, index = db[dom]
            D, I = index.search(emb_q.astype(np.float32), per_domain)
            for score, idx in zip(D[0], I[0]):
                results.append({"domain": dom, "text": texts[idx], "score": float(score)})
    else:
        texts, index = db[domains[0]]
        D, I = index.search(emb_q.astype(np.float32), TOP_K)
        for score, idx in zip(D[0], I[0]):
            results.append({"domain": domains[0], "text": texts[idx], "score": float(score)})
    results.sort(key=lambda x: -x["score"])
    return results[:TOP_K]


def build_prompt(query: str, context: list, domains: list, is_cross: bool) -> str:
    ctx_text = "\n".join([f"[{r['domain']}] {r['text'][:200]}" for r in context])
    primary = domains[0] if not is_cross else "CROSS"
    if is_cross:
        system = (f"You are a multidisciplinary expert specializing in "
                  f"{' and '.join(domains)}. Integrate knowledge from both domains.")
        few_shot = FEW_SHOT_CROSS
    else:
        system = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")
        few_shot = FEW_SHOT_LAW if domains[0] == "LAW" else FEW_SHOT_DEFAULT

    domain_instr = (
        "Provide a comprehensive explanation of the legal holding, principle, or rule. "
        "Write as a detailed legal analysis in full paragraph form."
        if primary == "LAW" else
        "provide a comprehensive analytical answer with specific mechanisms and implications"
    )
    prompt = (
        f"{system}\n\nHere is an example of a high-quality answer:\n{few_shot}\n\n"
        f"Context:\n{ctx_text}\n\nQuestion: {query}\n\n"
        f"Following the example style above, {domain_instr}. "
        f"Do NOT use multiple choice format. Write in flowing paragraph form only.\n\nAnswer:"
    )
    return prompt


def measure_one(query: str):
    """
    Returns (rr_ms, ttft_ms, throughput_toks_per_sec) or None values on failure.
    """
    global current_lora

    # ── [1] Routing & Retrieval 타이머 시작 ────────────────────────────
    cuda_sync()
    t_query_start = time.perf_counter()

    routing  = classify_domain(query)
    is_cross = routing["is_cross"]
    domains  = routing["domains"]

    context = retrieve(query, domains, is_cross)

    lora_target = "CROSS" if is_cross else domains[0]
    if current_lora != lora_target:
        model.set_adapter(lora_target)
        current_lora = lora_target

    cuda_sync()
    t_rr_end = time.perf_counter()
    rr_ms = (t_rr_end - t_query_start) * 1000

    # ── [2] 프롬프트 조합 + Tokenization ──────────────────────────────
    prompt   = build_prompt(query, context, domains, is_cross)
    messages = [{"role": "user", "content": prompt}]
    text_in  = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_in, return_tensors="pt").to(model.device)
    n_prompt_tokens = inputs["input_ids"].shape[1]

    max_tok = 800 if (not is_cross and domains[0] == "LAW") else MAX_NEW_TOKENS

    # ── [3] TextIteratorStreamer로 생성 + TTFT/Throughput 측정 ─────────
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = {
        **inputs,
        "streamer":       streamer,
        "max_new_tokens": max_tok,
        "temperature":    0.7,
        "do_sample":      True,
        "pad_token_id":   tokenizer.eos_token_id,
    }

    output_holder = {}

    def _generate():
        with torch.no_grad():
            output_holder["out"] = model.generate(**gen_kwargs)

    cuda_sync()
    # TTFT 타이머: 질문 입력 시점(t_query_start) 기준 — routing 포함
    gen_thread = threading.Thread(target=_generate, daemon=True)
    gen_thread.start()

    t_first_token = None
    for chunk in streamer:
        if chunk and t_first_token is None:
            # 첫 번째 실제 텍스트 청크 수신 = 첫 토큰 생성 완료
            t_first_token = time.perf_counter()

    cuda_sync()
    t_gen_end = time.perf_counter()
    gen_thread.join()

    # TTFT: 질문 입력 시점 ~ 첫 토큰까지 (라우팅+검색+스위칭+첫토큰 포함)
    ttft_ms = (t_first_token - t_query_start) * 1000 if t_first_token else None

    # Throughput: 총 생성 토큰 / 첫 토큰 이후 완료까지 시간
    out_tensor = output_holder.get("out")
    if out_tensor is not None:
        total_new_tokens = out_tensor.shape[1] - n_prompt_tokens
    else:
        total_new_tokens = 0

    if t_first_token and total_new_tokens > 0:
        elapsed_after_first = t_gen_end - t_first_token
        throughput = total_new_tokens / elapsed_after_first if elapsed_after_first > 0 else None
    else:
        throughput = None

    return rr_ms, ttft_ms, throughput


# ── 평가 데이터 로드 ──────────────────────────────────────────────────
print(f"\n평가 데이터 로드: {GT_PATH}")
with open(GT_PATH) as f:
    gt_list = json.load(f)
queries = [d["instruction"].strip() for d in gt_list]
print(f"  총 {len(queries)}개 쿼리 로드")

# ── 워밍업 ────────────────────────────────────────────────────────────
print(f"\n워밍업 ({N_WARMUP}개)...")
for i in range(min(N_WARMUP, len(queries))):
    try:
        measure_one(queries[i])
        print(f"  warmup [{i+1}/{N_WARMUP}] 완료")
    except Exception as e:
        print(f"  warmup [{i+1}] 오류: {e}")

# ── 본 측정 ───────────────────────────────────────────────────────────
print(f"\n본 측정 시작 ({len(queries)}개)...")
rr_list, ttft_list, tput_list = [], [], []
errors = 0

for i, query in enumerate(queries):
    try:
        rr_ms, ttft_ms, tput = measure_one(query)

        rr_list.append(rr_ms)
        if ttft_ms is not None:
            ttft_list.append(ttft_ms)
        if tput is not None:
            tput_list.append(tput)

        if (i + 1) % 50 == 0:
            print(
                f"  [{i+1:>3}/{len(queries)}]  "
                f"RR={np.mean(rr_list):.1f}ms  "
                f"TTFT={np.mean(ttft_list):.1f}ms  "
                f"TPS={np.mean(tput_list):.1f}tok/s"
            )
    except Exception as e:
        errors += 1
        print(f"  [{i+1}] 오류: {e}", file=sys.stderr)
        continue

# ── 요약 리포트 ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Summary Report — NexRoute 7B Latency (N={})".format(len(queries)))
print("=" * 65)

avg_rr   = np.mean(rr_list)   if rr_list   else float("nan")
avg_ttft = np.mean(ttft_list) if ttft_list else float("nan")
avg_tput = np.mean(tput_list) if tput_list else float("nan")
std_rr   = np.std(rr_list)    if rr_list   else float("nan")
std_ttft = np.std(ttft_list)  if ttft_list else float("nan")
std_tput = np.std(tput_list)  if tput_list else float("nan")

print(f"\n  {'Metric':<32} {'Avg':>10}  {'Std':>10}  {'N':>6}")
print("  " + "-" * 62)
print(f"  {'Routing & Retrieval (ms)':<32} {avg_rr:>10.1f}  {std_rr:>10.1f}  {len(rr_list):>6}")
print(f"  {'TTFT (ms)':<32} {avg_ttft:>10.1f}  {std_ttft:>10.1f}  {len(ttft_list):>6}")
print(f"  {'Throughput (tok/s)':<32} {avg_tput:>10.1f}  {std_tput:>10.1f}  {len(tput_list):>6}")
print(f"\n  오류 샘플: {errors}개")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.max_memory_allocated() / 1e9
    print(f"  GPU: {gpu_name}  |  Peak VRAM: {vram_gb:.1f} GB")

# ── 저장 ──────────────────────────────────────────────────────────────
results = {
    "n_samples": len(queries),
    "n_measured": len(rr_list),
    "n_errors": errors,
    "routing_retrieval_ms": {
        "avg": round(float(avg_rr),   1),
        "std": round(float(std_rr),   1),
        "min": round(float(np.min(rr_list)),  1) if rr_list else None,
        "max": round(float(np.max(rr_list)),  1) if rr_list else None,
    },
    "ttft_ms": {
        "avg": round(float(avg_ttft), 1),
        "std": round(float(std_ttft), 1),
        "min": round(float(np.min(ttft_list)), 1) if ttft_list else None,
        "max": round(float(np.max(ttft_list)), 1) if ttft_list else None,
    },
    "throughput_toks_per_sec": {
        "avg": round(float(avg_tput), 1),
        "std": round(float(std_tput), 1),
        "min": round(float(np.min(tput_list)), 1) if tput_list else None,
        "max": round(float(np.max(tput_list)), 1) if tput_list else None,
    },
    "raw": {
        "rr_ms":   [round(x, 2) for x in rr_list],
        "ttft_ms": [round(x, 2) for x in ttft_list],
        "tput":    [round(x, 2) for x in tput_list],
    },
    "hardware": {
        "gpu":  torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_peak_gb": round(vram_gb, 1) if torch.cuda.is_available() else 0,
    },
}

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ 결과 저장: {OUT_PATH}")
