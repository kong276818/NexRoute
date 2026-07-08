"""
Single LoRA RAG Baseline (Ablation)
--------------------------------------
도메인 구분 없이 단일 LoRA만 사용하는 버전
Dual MoE RAG와의 ablation study용

Usage:
    python single_lora_rag.py --query "What is the treatment for diabetes?"
"""
import os
import json
import argparse
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-72B-Instruct"
# 단일 LoRA: MED LoRA를 대표로 사용 (또는 가장 많이 학습된 것)
LORA_PATH   = "/home/sclab/moe_model/lora_weights/lora_med"
DATA_DIR    = "/home/sclab/moe_model/processed"
EMBED_MODEL = "intfloat/multilingual-e5-large"
TOP_K       = 10
DOMAINS     = ["med", "stem", "soc", "law"]


class SingleLoRARAG:
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
        print("[1/3] Loading base model + Single LoRA...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager"
        )
        # 단일 LoRA 로드 (도메인 구분 없이)
        self.model = PeftModel.from_pretrained(base, LORA_PATH)
        self.model.eval()
        print("[1/3] Model loaded!")

    def load_embed_model(self):
        print("[2/3] Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print("[2/3] Embedding model loaded!")

    def build_db(self):
        print("[3/3] Building vector DB (all domains, no prefix)...")
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
        # prefix 없이 단순 검색
        q_emb = self.embed_model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb)
        D, I = self.index.search(q_emb.astype(np.float32), TOP_K)
        return [{"text": self.texts[i], "score": float(D[0][j])} for j, i in enumerate(I[0])]

    def generate(self, query: str, context: list) -> str:
        ctx_text = "\n".join([f"- {d['text'][:200]}" for d in context])
        prompt = f"""Based on the following context, answer the question concisely in 3-5 sentences.

Context:
{ctx_text}

Question: {query}
Answer:"""
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

        context = self.retrieve(query)
        print(f"[1] Retrieved {len(context)} docs (no domain separation)")
        print(f"[2] Using single LoRA (lora_med)")

        answer = self.generate(query, context)
        print(f"[3] Answer generated")

        return {
            "query": query,
            "context": context,
            "answer": answer
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    rag = SingleLoRARAG()
    rag.load_embed_model()
    rag.load_model()
    rag.build_db()

    result = rag.run(args.query)

    print(f"\n{'='*60}")
    print(f"Query  : {result['query']}")
    print(f"Answer :\n{result['answer']}")
    print(f"{'='*60}")