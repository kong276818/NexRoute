import os
import re
import json
import torch
import faiss
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModel,
)
import torch.nn as nn
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ── 설정 ────────────────────────────────────────
BASE_MODEL       = "Qwen/Qwen2.5-7B-Instruct"
LORA_DIR         = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR         = "/home/sclab/moe_model/processed_v2"
EMBED_MODEL      = "intfloat/multilingual-e5-large"
ROUTER_PATH      = "/home/sclab/moe_model/router_multilabel.pt"

DOMAINS          = ["LAW", "MED", "SOC", "STEM"]
THRESHOLD        = 0.30
GAP              = 0.40
TOP_K            = 10

# 키워드 기반 멀티도메인 감지 (라우터 보완용)
_CROSS_KW = {
    "LAW":  {"law","legal","contract","clause","provision","liability","court",
             "attorney","agreement","statute","plaintiff","consent","judicial"},
    "MED":  {"medical","patient","clinical","treatment","drug","physician",
             "health","hospital","surgery","diagnosis","toxicologist",
             "cognitive","therapeutic","pharmacological","symptom","disease",
             "protein","gene","rna","dna","cell","receptor","enzyme","mutation",
             "genome","pathway","trial","randomised","controlled","inhibitor",
             "syndrome","cancer","tumor","virus","bacteria","immune","antibody",
             "chromosome","expression","sequence","methylation","kinase"},
    "STEM": {"scientific","physics","chemical","biology","engineering",
             "technology","algorithm","quantum","molecular","genetic",
             "protein","gene","rna","dna","cell","biochemical","genome",
             "enzyme","receptor","pathway","chromosome","sequence","methylation",
             "kinase","polymerase","nucleosome","subunit","replication",
             "transcription","binding","mechanism","signal"},
    "SOC":  {"social","society","culture","economic","political","community",
             "psychological","demographic","sociological"},
}

def _keyword_domains(query: str) -> list:
    """쿼리에 키워드가 있는 도메인 목록 반환 (2개 이상이면 cross 후보)."""
    q = query.lower()
    return [d for d, kws in _CROSS_KW.items() if any(k in q for k in kws)]

DOMAIN_SYSTEM = {
    "LAW":  "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":  "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":  "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM": "You are a science and engineering expert. Answer accurately based on scientific principles.",
}

# ★ 도메인별 Few-shot 예시
FEW_SHOT = {
    "LAW": (
        "Q: What is the doctrine of promissory estoppel?\n"
        "A: Promissory estoppel is a legal doctrine that prevents a party from "
        "withdrawing a promise when another party has reasonably relied on it to "
        "their detriment. The doctrine requires a clear and definite promise, "
        "reasonable reliance by the promisee, and actual detriment suffered as a "
        "result of that reliance. Courts apply this principle to enforce promises "
        "even without formal consideration, thereby ensuring fairness and preventing "
        "injustice in contractual relationships."
    ),
    "MED": (
        "Q: How does the renin-angiotensin system regulate blood pressure?\n"
        "A: The renin-angiotensin system regulates blood pressure through a cascade "
        "of enzymatic reactions initiated when the kidneys detect low blood pressure "
        "or reduced sodium levels. Renin converts angiotensinogen to angiotensin I, "
        "which is subsequently converted to angiotensin II by ACE in the lungs. "
        "Angiotensin II causes vasoconstriction and stimulates aldosterone secretion, "
        "increasing sodium retention and blood volume, ultimately elevating blood pressure."
    ),
    "SOC": (
        "Q: How does social stratification affect educational outcomes?\n"
        "A: Social stratification significantly influences educational outcomes by "
        "creating unequal access to resources, opportunities, and cultural capital. "
        "Children from higher socioeconomic backgrounds benefit from better-funded "
        "schools, private tutoring, and enriched home environments that foster "
        "academic achievement. Conversely, those from lower strata face systemic "
        "barriers including underfunded schools and economic pressures that hinder "
        "educational attainment, perpetuating intergenerational inequality."
    ),
    "STEM": (
        "Q: How does CRISPR-Cas9 technology work?\n"
        "A: CRISPR-Cas9 is a revolutionary gene-editing technology that uses a "
        "guide RNA to direct the Cas9 endonuclease to specific DNA sequences. "
        "The guide RNA binds complementarily to the target DNA, enabling Cas9 to "
        "create a precise double-strand break at the desired location. The cell "
        "then repairs this break through either non-homologous end joining, which "
        "can disrupt gene function, or homology-directed repair, which can introduce "
        "specific genetic modifications with high precision."
    ),
    "CROSS": (
        "Q: How might advances in neuroscience impact criminal justice policy?\n"
        "A: Advances in neuroscience are profoundly reshaping criminal justice policy "
        "by providing empirical insights into the biological underpinnings of behavior, "
        "decision-making, and culpability. Neuroimaging studies reveal that certain "
        "criminal behaviors may be associated with structural or functional brain "
        "abnormalities, raising questions about moral responsibility and the "
        "appropriate balance between punishment and rehabilitation. Policymakers must "
        "therefore integrate neuroscientific evidence with established legal principles "
        "to develop more nuanced, evidence-based approaches to sentencing and "
        "treatment that reflect both individual accountability and neurological realities."
    ),
}

class DomainRouter(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


class DualMoEPipeline:
    def __init__(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.base_model       = None
        self.base_tokenizer   = None
        self.current_lora     = None
        self.domain_router    = None
        self.router_tokenizer = None
        self.embed_model      = None
        self.db               = {}

    def load_router(self):
        print("[*] Loading BERT Router...")
        self.router_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.domain_router = DomainRouter()
        self.domain_router.load_state_dict(
            torch.load(ROUTER_PATH, map_location="cpu")
        )
        self.domain_router.eval()
        self.domain_router.to("cuda")
        print("[*] BERT Router loaded!")

    def classify_domain(self, query: str) -> dict:
        enc = self.router_tokenizer(
            query,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            logits = self.domain_router(enc["input_ids"], enc["attention_mask"])
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()

        domain_scores = dict(zip(DOMAINS, scores))
        sorted_d      = sorted(domain_scores.items(), key=lambda x: -x[1])
        top1, s1      = sorted_d[0]
        top2, s2      = sorted_d[1]

        is_cross = (s2 >= THRESHOLD) and ((s1 - s2) <= GAP)

        # 라우터가 single-domain으로 판정해도 키워드로 2개 이상 도메인 감지되면 cross 보완
        if not is_cross:
            kw_domains = _keyword_domains(query)
            if len(kw_domains) >= 2:
                # 키워드 도메인 중 score 높은 순으로 top-2 선택
                kw_ranked = sorted(
                    [(d, domain_scores.get(d, 0.0)) for d in kw_domains],
                    key=lambda x: -x[1]
                )
                is_cross = True
                top1, top2 = kw_ranked[0][0], kw_ranked[1][0]

        return {
            "is_cross": is_cross,
            "domains":  [top1, top2] if is_cross else [top1],
            "scores":   domain_scores
        }

    def load_lora(self, domain: str):
        if self.current_lora == domain:
            return
        print(f"[*] Switching LoRA → {domain}")
        lora_map = {
            "LAW":   "lora_law",
            "MED":   "lora_med",
            "STEM":  "lora_stem",
            "SOC":   "lora_soc",
            "CROSS": "lora_cross",
        }
        lora_path = os.path.join(LORA_DIR, lora_map[domain])
        if self.current_lora is None:
            self.base_model = PeftModel.from_pretrained(
                self.base_model, lora_path, adapter_name=domain
            )
        else:
            self.base_model.load_adapter(lora_path, adapter_name=domain)
            self.base_model.set_adapter(domain)
        self.current_lora = domain
        print(f"[*] LoRA {domain} active!")

    def load_base_model(self):
        if self.base_model is not None:
            return
        print("[*] Loading base model (4bit)...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, local_files_only=True
        )
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager"
        )
        print("[*] Base model loaded!")

    def load_embed_model(self):
        if self.embed_model is not None:
            return
        print("[*] Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print("[*] Embedding model loaded!")

    def build_db(self):
        print("[*] Building vector DB...")
        domain_files = {
            "LAW":   "law_rewritten.json",
            "MED":   "med_rewritten.json",
            "SOC":   "soc_rewritten.json",
            "STEM":  "stem_rewritten.json",
            "CROSS": "cross_rewritten.json",
        }

        for domain, fname in domain_files.items():
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                print(f"  [Skip] {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            filtered = []
            for d in data:
                out = d.get("output", d.get("gold_answer", ""))
                if any(k in out for k in ["The correct answer", "All answers:"]):
                    continue
                filtered.append(d)

            texts = [
                f"{d['instruction']} {d.get('output', d.get('gold_answer',''))}"
                for d in filtered
            ]
            embs  = self.embed_model.encode(
                texts, show_progress_bar=False, batch_size=64
            )
            embs  = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs.astype(np.float32))
            self.db[domain] = (texts, index)
            print(f"  {domain}: {len(texts)}개")

        print("[*] Vector DB ready!")

    def retrieve(self, query: str, domains: list,
                 is_cross: bool, top_k: int = TOP_K) -> list:
        emb_q = self.embed_model.encode([query])
        emb_q = emb_q / np.linalg.norm(emb_q)
        results = []

        if is_cross:
            per_domain = max(1, top_k // len(domains))
            for domain in domains:
                if domain not in self.db:
                    continue
                texts, index = self.db[domain]
                D, I = index.search(emb_q.astype(np.float32), per_domain)
                for score, idx in zip(D[0], I[0]):
                    results.append({
                        "domain": domain,
                        "text":   texts[idx],
                        "score":  float(score)
                    })
        else:
            texts, index = self.db[domains[0]]
            D, I = index.search(emb_q.astype(np.float32), top_k)
            for score, idx in zip(D[0], I[0]):
                results.append({
                    "domain": domains[0],
                    "text":   texts[idx],
                    "score":  float(score)
                })

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    def generate(self, query: str, context: list,
                 is_cross: bool, domains: list) -> str:

        ctx_text = "\n".join(
            [f"[{r['domain']}] {r['text'][:200]}" for r in context]
        )

        primary_domain = domains[0] if not is_cross else "CROSS"

        if is_cross:
            system = (
                f"You are a multidisciplinary expert specializing in "
                f"{' and '.join(domains)}. "
                f"Integrate knowledge from both domains to answer accurately."
            )
            few_shot = FEW_SHOT["CROSS"]
        else:
            system    = DOMAIN_SYSTEM.get(domains[0], "You are a helpful expert.")
            few_shot  = FEW_SHOT.get(domains[0], FEW_SHOT["STEM"])

        # LAW 도메인 전용 추가 지시 (판결 holding의 서술형 설명 유도)
        if primary_domain == "LAW":
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

        # ★ Few-shot 프롬프트
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

        messages = [{"role": "user", "content": prompt}]
        text = self.base_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.base_tokenizer(
            text, return_tensors="pt"
        ).to(self.base_model.device)

        # LAW는 gold_answer가 평균 95단어로 길기 때문에 더 많은 토큰 허용
        max_new_tokens = 800 if primary_domain == "LAW" else 512

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.base_tokenizer.eos_token_id
            )

        answer = self.base_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        answer = re.sub(r'The correct answer is[^\n]*\n?', '', answer).strip()
        answer = re.sub(r'Therefore,? the answer is[^\n]*\n?', '', answer).strip()
        # LAW 특유의 "The correct holding is option X:" 패턴 제거 후 내용 보존
        answer = re.sub(r'The correct holding is option\s*\d+\s*:\s*', '', answer).strip()
        answer = re.sub(r'The correct holding is\s*:\s*', '', answer).strip()
        for kw in ["Choices:", "All answers:", "Options:",
                   "\nA:", "\nB:", "\nC:", "\nD:"]:
            if kw in answer:
                answer = answer[:answer.index(kw)].strip()
        answer = re.sub(r'\n\d+:.*', '', answer).strip()

        if not answer.strip():
            answer = "[GENERATION_FAILED]"

        return answer

    def run(self, query: str) -> dict:
        routing  = self.classify_domain(query)
        is_cross = routing["is_cross"]
        domains  = routing["domains"]

        lora_target = "CROSS" if is_cross else domains[0]
        self.load_lora(lora_target)

        context = self.retrieve(query, domains, is_cross)
        answer  = self.generate(query, context, is_cross, domains)

        return {
            "query":         query,
            "is_cross":      is_cross,
            "domains":       domains,
            "domain_scores": routing["scores"],
            "context":       context,
            "answer":        answer
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",       type=str, default=None)
    parser.add_argument("--input_file",  type=str,
        default="/home/sclab/moe_model/processed_v2/gt_dataset_rewritten.json")
    parser.add_argument("--output_file", type=str,
        default="/home/sclab/moe_model/eval_results.json")
    args = parser.parse_args()

    pipe = DualMoEPipeline()
    pipe.load_router()
    pipe.load_embed_model()
    pipe.load_base_model()
    pipe.build_db()

    if args.query:
        result = pipe.run(args.query)
        print(f"\n도메인:  {result['domains']} (CROSS: {result['is_cross']})")
        print(f"스코어:  { {k: f'{v:.3f}' for k, v in result['domain_scores'].items()} }")
        print(f"답변:\n{result['answer']}")

    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        print(f"[*] 평가 시작: {len(eval_data)}개")
        results = []

        for idx, item in enumerate(eval_data):
            query       = item.get("instruction", "")
            gold_answer = item.get("gold_answer", "")
            domain_gt   = item.get("expected_func", "UNKNOWN")

            print(f"[{idx+1}/{len(eval_data)}] {query[:50]}...")

            try:
                out = pipe.run(query)
                results.append({
                    "id":            idx,
                    "query":         query,
                    "gold_answer":   gold_answer,
                    "domain_gt":     domain_gt,
                    "domain_pred":   out["domains"],
                    "is_cross":      out["is_cross"],
                    "domain_scores": out["domain_scores"],
                    "pred_answer":   out["answer"],
                })
            except Exception as e:
                print(f"  [Error] {e}")
                continue

            if (idx + 1) % 50 == 0:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"  [저장] {idx+1}개 중간 저장")

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n[+] 완료! 저장: {args.output_file}")
        print(f"총 {len(results)}개 평가 완료")