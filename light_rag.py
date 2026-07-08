"""
LightRAG Baseline (Qwen2.5-72B)
---------------------------------
Guo et al., EMNLP 2025
그래프 기반 듀얼 레벨 검색 (Local + Global)

Usage:
    pip install lightrag-hku
    python light_rag.py --build_db   # 최초 1회 인덱싱
    python light_rag.py --query "What is the treatment for diabetes?"
"""
import os
import json
import argparse
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.utils import EmbeddingFunc
import numpy as np

# ── Config ────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-72B-Instruct"
EMBED_MODEL  = "intfloat/multilingual-e5-large"
DATA_DIR     = "/home/sclab/moe_model/processed"
WORKING_DIR  = "/home/sclab/moe_model/lightrag_db"
DOMAINS      = ["med", "stem", "soc", "law"]

os.makedirs(WORKING_DIR, exist_ok=True)

# ── Global model references ───────────────────────────────────
_llm_model = None
_llm_tokenizer = None
_embed_model = None


def load_models():
    global _llm_model, _llm_tokenizer, _embed_model

    print("[1/2] Loading Qwen 72B...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    _llm_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    _llm_tokenizer.pad_token = _llm_tokenizer.eos_token
    _llm_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager"
    )
    _llm_model.eval()

    print("[2/2] Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(EMBED_MODEL)
    print("Models loaded!")


async def llm_func(prompt, system_prompt=None, **kwargs) -> str:
    """LightRAG용 LLM 함수"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    text = _llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _llm_tokenizer(text, return_tensors="pt").to(_llm_model.device)
    with torch.no_grad():
        outputs = _llm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=_llm_tokenizer.eos_token_id
        )
    return _llm_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


async def embed_func(texts: list[str]) -> np.ndarray:
    """LightRAG용 임베딩 함수"""
    embeddings = _embed_model.encode(texts, batch_size=32)
    return np.array(embeddings)


def build_lightrag() -> LightRAG:
    """LightRAG 인스턴스 생성"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,  # multilingual-e5-large 차원
            max_token_size=512,
            func=embed_func,
        ),
    )
    return rag


async def build_db(rag: LightRAG):
    """데이터셋 인덱싱"""
    print("[*] Indexing documents into LightRAG...")
    for domain in DOMAINS:
        path = os.path.join(DATA_DIR, f"{domain}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 도메인별 텍스트 합치기
        texts = []
        for item in data:
            text = f"Domain: {domain.upper()}\nQ: {item['instruction']}\nA: {item.get('output', '')}"
            texts.append(text)

        # 배치로 인덱싱
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            combined = "\n\n".join(batch)
            await rag.ainsert(combined)
            print(f"  {domain.upper()}: {min(i+batch_size, len(texts))}/{len(texts)} 완료")

    print("[*] Indexing complete!")


async def query_lightrag(rag: LightRAG, query: str, mode: str = "hybrid") -> dict:
    """LightRAG 쿼리 실행"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    # hybrid 모드: local + global 동시 검색
    answer = await rag.aquery(
        query,
        param=QueryParam(mode=mode)
    )

    return {
        "query": query,
        "mode": mode,
        "answer": answer
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--build_db", action="store_true")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["local", "global", "hybrid", "naive"])
    args = parser.parse_args()

    load_models()
    rag = build_lightrag()

    if args.build_db:
        await build_db(rag)
        print("DB 구축 완료! 이제 --query로 질문하세요.")
        return

    if args.query:
        result = await query_lightrag(rag, args.query, mode=args.mode)
        print(f"\n{'='*60}")
        print(f"Query  : {result['query']}")
        print(f"Mode   : {result['mode']}")
        print(f"Answer :\n{result['answer']}")
        print(f"{'='*60}")
    else:
        print("--query 또는 --build_db 옵션을 사용하세요.")


if __name__ == "__main__":
    asyncio.run(main())