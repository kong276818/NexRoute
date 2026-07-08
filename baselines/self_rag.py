"""
Self-RAG Baseline
------------------
Original Self-RAG (selfrag/selfrag_llama2_7b) on our dataset
Usage:
    pip install vllm
    python self_rag.py --query "What is the treatment for diabetes?"
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
SELFRAG_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DATA_DIR      = "/home/sclab/moe_model/processed"
EMBED_MODEL   = "intfloat/multilingual-e5-large"
TOP_K         = 10
DOMAINS       = ["med", "stem", "soc", "law"]
THRESHOLD     = 0.2  # Self-RAG 기본값

# Self-RAG special tokens
RETRIEVE_TOKEN    = "[Retrieval]"
NO_RETRIEVE_TOKEN = "[No Retrieval]"
RELEVANT_TOKEN    = "[Relevant]"
IRRELEVANT_TOKEN  = "[Irrelevant]"
SUPPORTED_TOKEN   = "[Fully supported]"
UTILITY_TOKENS    = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


class SelfRAG:
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
        print("[1/3] Loading Self-RAG model (Llama2-7B)...")
        self.tokenizer = AutoTokenizer.from_pretrained(SELFRAG_MODEL, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            SELFRAG_MODEL,
            quantization_config=self.bnb_config,
            device_map="auto",
        )
        self.model.eval()
        print("[1/3] Self-RAG model loaded!")

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

    def retrieve(self, query: str) -> list:
        q_emb = self.embed_model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb)
        D, I = self.index.search(q_emb.astype(np.float32), TOP_K)
        return [{"text": self.texts[i], "score": float(D[0][j])} for j, i in enumerate(I[0])]

    def should_retrieve(self, query: str) -> bool:
        """Self-RAG: 검색 필요 여부 스스로 판단"""
        prompt = f"### Instruction:\n{query}\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        generated = self.tokenizer.decode(
            outputs.sequences[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
        # [Retrieval] 토큰이 나오면 검색 필요
        return RETRIEVE_TOKEN in generated

    def score_passage(self, query: str, passage: str) -> float:
        """Self-RAG: 검색된 문서의 관련성 평가"""
        prompt = (
            f"### Instruction:\n{query}\n"
            f"### Retrieved passage:\n{passage[:300]}\n"
            f"### Is this passage relevant?\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        result = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )
        if RELEVANT_TOKEN in result:
            return 1.0
        elif IRRELEVANT_TOKEN in result:
            return 0.0
        return 0.5

    def generate(self, query: str, context: list = None) -> str:
        """Self-RAG: 컨텍스트 포함 답변 생성"""
        if context:
            # 관련성 높은 문서만 필터링
            scored = []
            for doc in context[:5]:
                score = self.score_passage(query, doc["text"])
                if score > 0.3:
                    scored.append(doc["text"][:200])

            if scored:
                ctx_text = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(scored)])
                prompt = (
                    f"### Instruction:\n{query}\n"
                    f"### Retrieved passages:\n{ctx_text}\n"
                    f"### Response:\n"
                )
            else:
                prompt = f"### Instruction:\n{query}\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{query}\n### Response:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Self-RAG 특수 토큰 제거
        for token in [RETRIEVE_TOKEN, NO_RETRIEVE_TOKEN, RELEVANT_TOKEN,
                      IRRELEVANT_TOKEN, SUPPORTED_TOKEN] + UTILITY_TOKENS:
            answer = answer.replace(token, "")
        return answer.strip()

    def run(self, query: str) -> dict:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Step 1: 검색 필요 여부 판단
        need_retrieve = self.should_retrieve(query)
        print(f"[1] Need retrieval: {need_retrieve}")

        # Step 2: 검색 (필요한 경우)
        context = None
        if need_retrieve:
            context = self.retrieve(query)
            print(f"[2] Retrieved {len(context)} docs")
        else:
            print("[2] Skipped retrieval")

        # Step 3: 답변 생성
        answer = self.generate(query, context)
        print(f"[3] Answer generated")

        return {
            "query": query,
            "need_retrieve": need_retrieve,
            "context": context,
            "answer": answer
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    rag = SelfRAG()
    rag.load_embed_model()
    rag.load_model()
    rag.build_db()

    result = rag.run(args.query)

    print(f"\n{'='*60}")
    print(f"Query          : {result['query']}")
    print(f"Need Retrieval : {result['need_retrieve']}")
    print(f"Answer         :\n{result['answer']}")
    print(f"{'='*60}")