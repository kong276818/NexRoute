"""
Adaptive-RAG Baseline (Qwen2.5-72B)
--------------------------------------
Jeong et al., NAACL 2024
쿼리 복잡도에 따라 검색 전략을 다르게 적용
- Simple   → No retrieval (LLM 자체 지식)
- Medium   → Single-step retrieval
- Complex  → Multi-step retrieval

Usage:
    python adaptive_rag.py --query "What is the treatment for diabetes?"
"""
import os
import json
import argparse
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-72B-Instruct"
DATA_DIR    = "/home/sclab/moe_model/processed"
EMBED_MODEL = "intfloat/multilingual-e5-large"
TOP_K       = 10
DOMAINS     = ["med", "stem", "soc", "law"]

COMPLEXITY_PROMPT = """Classify the complexity of this question into one of three levels:
- SIMPLE: Can be answered with basic knowledge, no retrieval needed
- MEDIUM: Requires some domain knowledge, single retrieval sufficient
- COMPLEX: Requires deep expertise or multiple pieces of information, multi-step retrieval needed

Question: {query}
Answer only one word (SIMPLE, MEDIUM, or COMPLEX):"""

FOLLOWUP_PROMPT = """Based on the initial answer below, generate a follow-up search query to find additional relevant information.

Original question: {query}
Initial answer: {initial_answer}

Generate a short follow-up search query (one sentence):"""


class AdaptiveRAG:
    def __init__(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.model = None
        self.tokenizer = None
        self.embed_model = None
        self.texts = []
        self.index = None

    def load_model(self):
        print("[1/3] Loading Qwen 72B...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager"
        )
        self.model.eval()
        print("[1/3] Model loaded!")

    def load_embed_model(self):
        print("[2/3] Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print("[2/3] Embedding model loaded!")

    def build_db(self):
        print("[3/3] Building vector DB...")
        all_texts = []
        for domain in DOMAINS:
            path = os.path.join(DATA_DIR, f"{domain}.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                all_texts.append(f"{d['instruction']} {d.get('output', '')}")
        self.texts = all_texts
        print(f"  총 {len(all_texts)}개 문서 인덱싱 중...")
        embeddings = self.embed_model.encode(all_texts, show_progress_bar=True, batch_size=64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        print("[3/3] Vector DB ready!")

    def classify_complexity(self, query: str) -> str:
        """쿼리 복잡도 분류: SIMPLE / MEDIUM / COMPLEX"""
        prompt = COMPLEXITY_PROMPT.format(query=query)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        result = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().upper()
        if "COMPLEX" in result:
            return "COMPLEX"
        elif "MEDIUM" in result:
            return "MEDIUM"
        return "SIMPLE"

    def retrieve(self, query: str, top_k: int = None) -> list:
        k = top_k or TOP_K
        q_emb = self.embed_model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb)
        D, I = self.index.search(q_emb.astype(np.float32), k)
        return [{"text": self.texts[i], "score": float(D[0][j])} for j, i in enumerate(I[0])]

    def generate_followup_query(self, query: str, initial_answer: str) -> str:
        """Multi-step retrieval용 후속 쿼리 생성"""
        prompt = FOLLOWUP_PROMPT.format(query=query, initial_answer=initial_answer[:300])
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def generate(self, query: str, context: list = None) -> str:
        if context:
            ctx_text = "\n".join([f"- {d['text'][:200]}" for d in context])
            prompt = f"""Based on the following context, answer the question concisely in 3-5 sentences.

Context:
{ctx_text}

Question: {query}
Answer:"""
        else:
            prompt = f"Answer the question concisely in 3-5 sentences.\n\nQuestion: {query}\nAnswer:"

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.7,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def run(self, query: str) -> dict:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Step 1: 복잡도 분류
        complexity = self.classify_complexity(query)
        print(f"[1] Complexity: {complexity}")

        context = []
        steps = 1

        if complexity == "SIMPLE":
            # 검색 없이 바로 생성
            print("[2] No retrieval (SIMPLE)")
            answer = self.generate(query)

        elif complexity == "MEDIUM":
            # Single-step retrieval
            context = self.retrieve(query)
            print(f"[2] Single retrieval: {len(context)} docs")
            answer = self.generate(query, context)

        else:  # COMPLEX
            # Multi-step retrieval
            # Step 1: 첫 번째 검색
            context = self.retrieve(query, top_k=5)
            print(f"[2] Multi-step retrieval step 1: {len(context)} docs")

            # Step 2: 초기 답변으로 후속 쿼리 생성
            initial_answer = self.generate(query, context)
            followup_query = self.generate_followup_query(query, initial_answer)
            print(f"[2] Follow-up query: {followup_query}")

            # Step 3: 후속 검색
            followup_context = self.retrieve(followup_query, top_k=5)
            context = context + followup_context
            print(f"[2] Multi-step retrieval step 2: {len(context)} docs total")
            steps = 2

            # 최종 답변 생성
            answer = self.generate(query, context)

        print(f"[3] Answer generated (steps={steps})")

        return {
            "query": query,
            "complexity": complexity,
            "retrieval_steps": steps,
            "context": context,
            "answer": answer
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    rag = AdaptiveRAG()
    rag.load_embed_model()
    rag.load_model()
    rag.build_db()

    result = rag.run(args.query)

    print(f"\n{'='*60}")
    print(f"Query      : {result['query']}")
    print(f"Complexity : {result['complexity']}")
    print(f"Steps      : {result['retrieval_steps']}")
    print(f"Answer     :\n{result['answer']}")
    print(f"{'='*60}")