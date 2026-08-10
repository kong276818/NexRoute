"""
inference.py — DualMoE 인퍼런스 + RAG + Ablation Study

WARNING: This module is a development prototype and is NOT used for any result
reported in the paper. The DualMoE class uses HybridRouter
(routers/domain_router/pipeline.pkl), a TF-IDF + keyword + threshold-based
CROSS detector that predates the DeBERTa dual-axis router. The class name
"DualMoE" refers to this prototype, not the paper's dual-axis architecture.

For paper experiments, see:
  - Routing accuracy (Table 4):   train/train_deberta_router.py  +  preds_500.csv
  - Generation quality (Table 5): run_reinference_500.py  +  eval/eval_all_metrics_paper.py
  - Hallucination (Table 6):      eval/eval_factscore_hallucination.py

모드:
  base        : 베이스 모델만 (라우팅 없음, LoRA 없음, RAG 없음)
  lora_only   : 도메인 고정 LoRA (라우팅 없음, 정답 도메인 LoRA + RAG)
  domain_only : Domain Router만 사용 = Partitioned RAG Baseline (기능 스캐폴딩 없음)
  dualmoe     : 전체 NexRoute 제안 모델 (Domain + Func Router + LoRA + RAG + 기능 스캐폴딩)

사용법:
  # 기본 평가 (processed_v2 데이터)
  python inference.py --eval --domain MED --n 100 --mode dualmoe
"""

import os, gc, re, json, pickle, random, argparse
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── 경로 ──────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
LORA_DIR   = "/home/sclab/moe_model/lora_weights_v2"
ROUTER_DIR = "/home/sclab/moe_model/routers"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"   # 기본값
RESULT_DIR = "/home/sclab/moe_model/results"

# [추가] FAISS 및 코퍼스 경로 (실제 구축하신 경로로 맞춰주세요)
FAISS_INDEX_DIR = "/home/sclab/moe_model/faiss_indices"
CORPUS_DIR      = "/home/sclab/moe_model/corpora"

os.makedirs(RESULT_DIR, exist_ok=True)

CROSS_THRESHOLD = 0.50 # 논문에 맞춰 0.50으로 수정 권장

DOMAIN_SYSTEM = {
    "MED":   "You are a medical expert. Answer accurately based on medical facts.",
    "LAW":   "You are a legal expert. Answer accurately based on legal principles.",
    "STEM":  "You are a science expert. Answer accurately based on scientific principles.",
    "SOC":   "You are a social science expert. Answer based on social context and research.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains.",
}

# ── Router Classes (pickle 복원용) ─────────────────────────────
class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, keyword_dict):
        self.keyword_dict = keyword_dict
        self.domains = list(keyword_dict.keys())
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.array([
            [sum(1 for kw in self.keyword_dict[d] if kw in t.lower()) for d in self.domains]
            for t in X
        ], dtype=np.float64)

class HybridRouter:
    def __init__(self, tfidf, kw_extractor, clf):
        self.tfidf = tfidf
        self.kw_extractor = kw_extractor
        self.clf = clf
        self.classes_ = clf.classes_
    def _feat(self, texts):
        return hstack([self.tfidf.transform(texts), self.kw_extractor.transform(texts) * 5])
    def predict(self, texts): return self.clf.predict(self._feat(texts))
    def predict_proba(self, texts): return self.clf.predict_proba(self._feat(texts))


# ── RAG 모듈 (Domain-Partitioned Retriever) ────────────────────
class DomainRetriever:
    def __init__(self, encoder_name="intfloat/multilingual-e5-large"):
        print("  [RAG] 검색용 인코더 로드 중...")
        try:
            self.encoder = SentenceTransformer(encoder_name, device="cuda")
        except Exception as e:
            print(f"  [RAG] 인코더 로드 실패 (RAG 없이 진행됨): {e}")
            self.encoder = None
            
        self.indices = {}
        self.corpora = {}
        
        # FAISS 인덱스와 코퍼스 로드
        domains_to_load = ["MED", "LAW", "STEM", "SOC"]
        for dom in domains_to_load:
            idx_path = os.path.join(FAISS_INDEX_DIR, f"{dom.lower()}.index")
            corp_path = os.path.join(CORPUS_DIR, f"{dom.lower()}_corpus.json")
            if os.path.exists(idx_path) and os.path.exists(corp_path):
                self.indices[dom] = faiss.read_index(idx_path)
                with open(corp_path, "r") as f:
                    self.corpora[dom] = json.load(f)
                print(f"  [RAG] {dom} 도메인 FAISS 인덱스 로드 완료")
            else:
                pass # 테스트 환경을 위해 파일이 없으면 패스

    def retrieve(self, query, route_info, top_k=10):
        if not self.encoder or not self.indices:
            return "No external context available. (RAG DB missing)"

        domain, top1_d, top2_d, top1_p, top2_p = route_info
        
        # 1) 단일 도메인 검색
        if domain != "CROSS":
            if domain not in self.indices: return "No context available."
            q_emb = self.encoder.encode([query], normalize_embeddings=True)
            distances, idxs = self.indices[domain].search(q_emb, top_k)
            return "\n".join([self.corpora[domain][str(i)] for i in idxs[0] if str(i) in self.corpora[domain]])
            
        # 2) CROSS 도메인 검색 (논문 Eq 6 반영: 예산 비례 할당)
        else:
            total_p = top1_p + top2_p
            k1 = int(top_k * (top1_p / total_p))
            k2 = top_k - k1
            
            context_parts = []
            q_emb = self.encoder.encode([query], normalize_embeddings=True)
            
            if top1_d in self.indices and k1 > 0:
                d1, idx1 = self.indices[top1_d].search(q_emb, k1)
                context_parts.extend([self.corpora[top1_d][str(i)] for i in idx1[0] if str(i) in self.corpora[top1_d]])
            if top2_d in self.indices and k2 > 0:
                d2, idx2 = self.indices[top2_d].search(q_emb, k2)
                context_parts.extend([self.corpora[top2_d][str(i)] for i in idx2[0] if str(i) in self.corpora[top2_d]])
                
            return "\n".join(context_parts) if context_parts else "No cross-domain context available."


# ── MCQ 유틸 ──────────────────────────────────────────────────
def parse_mcq(output: str):
    ans_m = re.search(r"correct answer is ([A-D])", output)
    if not ans_m: return None, {}
    answer = ans_m.group(1)
    choices = {}
    for line in output.split('\n'):
        m = re.match(r'^([A-D]):\s*(.+)', line.strip())
        if m: choices[m.group(1)] = m.group(2).strip()
    return answer, choices

def extract_pred(response: str) -> str:
    for pat in [r'^([A-D])\b', r'\bAnswer[:\s]+([A-D])\b',
                r'correct answer is ([A-D])', r'\b([A-D])[)\.]', r'\*\*([A-D])\*\*']:
        m = re.search(pat, response.strip(), re.IGNORECASE)
        if m: return m.group(1).upper()
    m = re.search(r'\b([A-D])\b', response)
    return m.group(1).upper() if m else None


# ── 모델 로드 ──────────────────────────────────────────────────
def load_model():
    print("  베이스 모델 로드 중 (4bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        local_files_only=True, attn_implementation="eager",
    )
    model.config.use_cache = True
    print("  ✅ 모델 로드 완료")
    return model, tok

def load_lora(model, domain: str):
    if domain == "CROSS": domain = "cross" # 폴더명 소문자 매칭
    lora_path = os.path.join(LORA_DIR, f"lora_{domain.lower()}")
    if not os.path.exists(lora_path):
        print(f"  ⚠ LoRA 없음: {lora_path}")
        return model
    if hasattr(model, 'peft_config'):
        model = model.merge_and_unload()
        gc.collect(); torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(model, lora_path)
    print(f"  LoRA [{domain}] 활성화")
    return model

def generate(model, tok, prompt, system_msg, max_new_tokens=64):
    messages = [{"role":"system","content":system_msg},
                {"role":"user","content":prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


# ── DualMoE 엔진 (프롬프트 빌더 포함) ──────────────────────────
class DualMoE:
    def __init__(self, use_func_router=True):
        self.use_func_router = use_func_router
        print("  라우터 로드...")
        with open(f"{ROUTER_DIR}/domain_router/pipeline.pkl","rb") as f:
            self.d_router = pickle.load(f)
        if use_func_router:
            with open(f"{ROUTER_DIR}/func_router/pipeline.pkl","rb") as f:
                self.f_router = pickle.load(f)
        
        self.retriever = DomainRetriever()
        self.model, self.tok = load_model()
        self.cur_domain = None

    def route_domain(self, query):
        proba = self.d_router.predict_proba([query])[0]
        classes = self.d_router.classes_
        
        # CROSS 판단 로직
        sorted_idx = np.argsort(proba)[::-1]
        top1_d, top2_d = classes[sorted_idx[0]], classes[sorted_idx[1]]
        top1_p, top2_p = proba[sorted_idx[0]], proba[sorted_idx[1]]
        
        domain = top1_d if top1_p >= CROSS_THRESHOLD else "CROSS"
        func = self.f_router.predict([query])[0] if self.use_func_router else "N/A"
        
        return domain, func, (domain, top1_d, top2_d, top1_p, top2_p)

    def load_expert(self, domain):
        if self.cur_domain == domain: return
        self.model = load_lora(self.model, domain)
        self.cur_domain = domain

    # 제안 모델 (NexRoute) 스캐폴드
    def build_function_aware_prompt(self, instruction, choices, context, func):
        scaffolds = {
            "What": "Based on the following evidence, identify and define the key facts. Do not infer beyond the provided context:",
            "How": "Using the following evidence, describe the mechanism or procedure in a step-by-step logical flow:",
            "Why": "Based on the following evidence, construct a definitive causal chain. Explicitly explain the underlying reasons connecting premise to outcome:",
            "Compare": "Using the following evidence, contrast the entities in the query. Highlight structural similarities and key differences in a comparative framework:"
        }
        scaffold_text = scaffolds.get(func, "Answer the following based on evidence:")
        choices_str = "\n".join([f"{k}: {v}" for k, v in sorted(choices.items())])
        return f"{scaffold_text}\n\n[Evidence]\n{context}\n\n[Query]\n{instruction}\n\nOptions:\n{choices_str}\n\nAnswer with only the letter (A, B, C, or D):"

    # Ablation (Partitioned RAG 베이스라인) 범용 프롬프트
    def build_generic_prompt(self, instruction, choices, context):
        choices_str = "\n".join([f"{k}: {v}" for k, v in sorted(choices.items())])
        if context.strip():
            return f"Answer the following question based on the provided evidence:\n\n[Evidence]\n{context}\n\n[Query]\n{instruction}\n\nOptions:\n{choices_str}\n\nAnswer with only the letter (A, B, C, or D):"
        else:
            return f"{instruction}\n\nOptions:\n{choices_str}\n\nAnswer with only the letter (A, B, C, or D):"

    def predict(self, instruction, choices):
        # 1. 라우팅
        domain, func, route_info = self.route_domain(instruction)
        
        # 2. 로라 로드
        self.load_expert(domain)
        
        # 3. 도메인 파티션 RAG
        context = self.retriever.retrieve(instruction, route_info, top_k=10)
        
        # 4. 프롬프트 생성 (Ablation 핵심 분기)
        if self.use_func_router and func != "N/A":
            prompt = self.build_function_aware_prompt(instruction, choices, context, func)
        else:
            prompt = self.build_generic_prompt(instruction, choices, context)
            
        sys_msg = DOMAIN_SYSTEM.get(domain, DOMAIN_SYSTEM["CROSS"])
        resp = generate(self.model, self.tok, prompt, sys_msg)
        return extract_pred(resp), domain, func, route_info[3] # prob


# ── 평가 ──────────────────────────────────────────────────────
def evaluate(domain: str, n: int, mode: str, data_dir: str, suffix: str = ""):
    fname = f"{domain.lower()}{suffix}.json"
    fpath = os.path.join(data_dir, fname)

    if not os.path.exists(fpath):
        fpath = os.path.join(data_dir, f"{domain.lower()}.json")

    with open(fpath) as f:
        data = json.load(f)

    mcq = []
    for item in data:
        ans, choices = parse_mcq(item.get("output",""))
        if ans and len(choices) >= 2:
            mcq.append({"instruction": item["instruction"],
                        "answer": ans, "choices": choices,
                        "true_domain": item.get("domain", domain)})

    random.seed(42)
    samples = random.sample(mcq, min(n, len(mcq)))

    mode_desc = {
        "base":        "베이스 모델 (라우팅 X, LoRA X, RAG X)",
        "lora_only":   "정답 도메인 LoRA 고정 (라우팅 X, RAG O)",
        "domain_only": "Domain Router만 사용 = Partitioned RAG Ablation (Func 스캐폴드 X)",
        "dualmoe":     "전체 NexRoute 제안 모델 (Domain + Func + LoRA + RAG + 스캐폴드)",
    }
    data_tag = "MMLU" if suffix else "processed_v2"
    print(f"\n{'='*55}")
    print(f"  [{mode.upper()}] {domain} 평가 ({len(samples)}개) [{data_tag}]")
    print(f"  → {mode_desc.get(mode,'?')}")
    print(f"{'='*55}")

    # 모델 준비
    if mode == "base":
        model, tok = load_model()
        sys_msg = "You are a helpful expert. Answer with only the letter (A, B, C, or D)."
        engine = DualMoE(use_func_router=False) # RAG 뼈대만 활용
    elif mode == "lora_only":
        model, tok = load_model()
        model = load_lora(model, domain)
        sys_msg = DOMAIN_SYSTEM.get(domain, DOMAIN_SYSTEM["CROSS"])
        engine = DualMoE(use_func_router=False)
    elif mode == "domain_only":
        engine = DualMoE(use_func_router=False)
    elif mode == "dualmoe":
        engine = DualMoE(use_func_router=True)

    correct = 0
    routing_correct = 0
    results = []

    for i, item in enumerate(samples):
        if mode in ["base", "lora_only"]:
            # RAG 적용 (lora_only의 경우 정답 도메인에서 가져옴)
            route_info = (item["true_domain"], item["true_domain"], "", 1.0, 0.0)
            context = engine.retriever.retrieve(item["instruction"], route_info, top_k=10) if mode == "lora_only" else ""
            prompt = engine.build_generic_prompt(item["instruction"], item["choices"], context)
            
            resp = generate(model, tok, prompt, sys_msg)
            pred = extract_pred(resp)
            routed = domain if mode == "lora_only" else "N/A"
            func = "N/A"

        elif mode in ["domain_only", "dualmoe"]:
            pred, routed, func, prob = engine.predict(item["instruction"], item["choices"])
            if routed == item["true_domain"]:
                routing_correct += 1

        ok = (pred == item["answer"]) if pred else False
        if ok: correct += 1

        results.append({
            "q": item["instruction"][:80],
            "answer": item["answer"], "predicted": pred,
            "correct": ok, "routed_domain": routed,
            "true_domain": item["true_domain"], "func": func,
        })

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] 정확도: {correct/(i+1):.3f}")

    acc = correct / len(samples)
    routing_acc = routing_correct / len(samples) if mode in ["domain_only","dualmoe"] else None

    print(f"\n  [{mode.upper()} {domain}] 정확도: {acc:.4f} ({acc*100:.1f}%)")
    if routing_acc is not None:
        print(f"  라우팅 정확도: {routing_acc:.4f} ({routing_acc*100:.1f}%)")

    save_name = f"eval_{mode}_{domain.lower()}{suffix}.json"
    save_path = os.path.join(RESULT_DIR, save_name)
    with open(save_path, "w") as f:
        json.dump({"mode": mode, "domain": domain, "data": data_tag,
                   "n": len(samples), "accuracy": acc,
                   "routing_accuracy": routing_acc,
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"  저장: {save_path}")

    if mode in ["base","lora_only"]:
        del model; gc.collect(); torch.cuda.empty_cache()

    return acc

# ── 메인 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",    type=str)
    parser.add_argument("--eval",     action="store_true")
    parser.add_argument("--domain",   type=str, default="MED",
                        choices=["MED","LAW","STEM","SOC","CROSS","ALL"])
    parser.add_argument("--n",        type=int, default=100)
    parser.add_argument("--mode",     type=str, default="dualmoe",
                        choices=["base","lora_only","domain_only","dualmoe"])
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--suffix",   type=str, default="")
    args = parser.parse_args()

    if args.query:
        engine = DualMoE(use_func_router=True)
        domain, func, prob = engine.route_domain(args.query)
        engine.load_expert(domain)
        sys_msg = DOMAIN_SYSTEM.get(domain, DOMAIN_SYSTEM["CROSS"])
        resp = generate(engine.model, engine.tok, args.query, sys_msg, max_new_tokens=512)
        print(f"\n{'='*60}")
        print(f"Q: {args.query}")
        print(f"Domain: {domain} / Func: {func}")
        print(f"\nA: {resp}")
        print("="*60)

    elif args.eval:
        domains = ["MED","LAW","STEM","SOC","CROSS"] if args.domain=="ALL" else [args.domain]
        scores = {}
        for d in domains:
            scores[d] = evaluate(d, args.n, args.mode, args.data_dir, args.suffix)

        if len(domains) > 1:
            avg = sum(scores.values()) / len(scores)
            print(f"\n{'='*55}")
            print(f"[{args.mode.upper()}] 전체 결과 [{args.suffix or 'default'}]:")
            for d, s in scores.items():
                print(f"  {d:6s}: {s*100:.1f}%")
            print(f"  AVG  : {avg*100:.1f}%")
            
            save_name = f"eval_{args.mode}_all{args.suffix}.json"
            with open(os.path.join(RESULT_DIR, save_name),"w") as f:
                json.dump({"mode":args.mode,"scores":scores,"avg":avg}, f, indent=2)
    else:
        parser.print_help()