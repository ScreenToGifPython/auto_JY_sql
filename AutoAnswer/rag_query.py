#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_sql_generator.py
使用 RAG 检索表结构 → 调用 LLM 生成 SQL
Author: KevinChen
"""

import os, json, argparse
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI, OpenAIError
import backoff

# ----------------- 默认参数 -----------------
EMBED_MODEL = "BAAI/bge-m3"
INDEX_PATH = "faiss_index.bin"
MAP_PATH = "table_mapping.json"
TABLE_JSON = "table.json"
TOP_K = 10
MAX_LEN = 512
BATCH_SIZE = 16
LLM_MODEL = "deepseek-chat"  # 可换其他
API_KEY = "sk-43a58ad2bbd740b095f5f61671ed9fae"
LLM_URL = "https://api.deepseek.com"
QUESTION = "怎么查找周频私募基金的日涨跌数据?"
# ----------------------------------------

# --------- 禁用并行 tokenizer 线程，避免脚本不退出 ----------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- LLM 调用 ----------------
SYSTEM_PROMPT = """
你是一名资深数据工程师，精通 SQL 编写。请严格遵循以下原则：
1. **值映射：** 若字段备注已给中文↔代码映射，请先把用户描述转换为对应代码再过滤。
2. **表选择：** 只选择与需求直接相关的表，避免冗余 JOIN。
3. **JOIN 条件：** 当确有关系时使用表间关联字段。
4. **注释：** SQL 加上中文注释解释字段含义、过滤条件及 JOIN 逻辑。
"""


@backoff.on_exception(backoff.expo, OpenAIError, max_tries=3)
def call_llm(prompt: str,
             api_key: str,
             base_url: str,
             model: str = LLM_MODEL,
             temperature: float = 0.1) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
    return resp.choices[0].message.content.strip()


# ------------- 构造 prompt ----------------
def build_prompt(user_q: str, ctx_blocks: List[str]) -> str:
    ctx_txt = "\n\n--- 相关表结构 ---\n" + "\n\n".join(ctx_blocks)
    return f"""{ctx_txt}

--- 用户需求 ---
{user_q}

请在 ```sql ``` 块中给出最终 SQL, 并在代码块中用注释解释所用表、字段、JOIN 逻辑。
"""


# ------------- 主函数 --------------------
def rag_sql(question: str,
            api_key: str,
            base_url: str,
            top_k: int = TOP_K) -> str:
    # 1. 加载资源（全局懒加载）
    global faiss_index, id2text, embedder
    if "faiss_index" not in globals():
        faiss_index = faiss.read_index(INDEX_PATH)
        id2text = json.load(open(MAP_PATH, encoding="utf-8"))
        embedder = SentenceTransformer(EMBED_MODEL)
        embedder.max_seq_length = MAX_LEN

    # 2. 编码问题向量
    q_vec = embedder.encode([question], normalize_embeddings=True).astype("float32")

    # 3. 检索
    _, I = faiss_index.search(q_vec, top_k)
    ctx_blocks = [id2text[str(idx)] if isinstance(id2text, dict) else id2text[idx]
                  for idx in I[0]]
    print(f"\n🎯 Top-{top_k} 命中表索引: {I[0]}")
    for i, block in enumerate(ctx_blocks):
        print(f"\n—— 表结构 Top-{i + 1} ——\n{block}")

    # 4. 构造 Prompt
    prompt = build_prompt(question, ctx_blocks)
    print(f"\n📤 发送给大模型的Prompt如下：\n{prompt}\n")

    # 5. 调用 LLM
    return call_llm(prompt, api_key=api_key, base_url=base_url)


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 生成 SQL demo")
    parser.add_argument("--question", default=QUESTION, help="用户自然语言问题")
    parser.add_argument("--k", type=int, default=TOP_K, help="检索 top-k")
    parser.add_argument("--key", default=API_KEY, help="OpenAI API Key")
    parser.add_argument("--url", default=LLM_URL, help="OpenAI Base URL")
    args = parser.parse_args()

    answer = rag_sql(args.question, api_key=args.key, base_url=args.url, top_k=args.k)
    print("\n=== LLM 回复 ===\n")
    print(answer)
    os._exit(0)
