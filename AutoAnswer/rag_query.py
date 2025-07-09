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
INDEX_PATH = "faiss_index.bin"
MAP_PATH = "table_mapping.json"
TABLE_JSON = "table.json"
SQL_TYPE = "mysql"
TOP_K = 10
MAX_LEN = 512
BATCH_SIZE = 16
LLM_MODEL = "deepseek-chat"  # 可换其他
API_KEY = "sk-xxxxx"
API_KEY = "sk-43a58ad2bbd740b095f5f61671ed9fae"
LLM_URL = "https://api.deepseek.com"
QUESTION = "公募基金的二级分类基金类型是股票型的最近1年净值和收益率数据,只要交易日的数据,用上海市场交易日?"
DEFAULT_EMBED_MODEL_PATH = "/Users/chenjunming/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"  # 默认嵌入模型路径

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
             sql_type: str,
             model: str = LLM_MODEL,
             temperature: float = 0.1) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    sys_prompt = f"""
你是一名资深数据工程师，精通{sql_type}数据库的 SQL 编写。请严格遵循以下原则：
1. **值映射：** 若字段备注已给中文↔代码映射，请先把用户描述转换为对应代码再过滤。
2. **表选择：** 只选择与需求直接相关的表，避免冗余 JOIN。
3. **JOIN 条件：** 当确有关系时使用表间关联字段。
4. **注释：** SQL 加上中文注释解释字段含义、过滤条件及 JOIN 逻辑。
5. **执行效率：** 你写的 SQL 一定是执行效率最高的SQL代码, 绝对符合{sql_type}数据库的特性,语法,执行效率。
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ])
    return resp.choices[0].message.content.strip()


# ------------- 构造 prompt ----------------
def build_prompt(user_q: str, sql_type: str, ctx_blocks: List[str]) -> str:
    ctx_txt = "\n\n--- 相关表结构 ---\n" + "\n\n".join(ctx_blocks)
    return f"""{ctx_txt}

--- 用户需求 ---
{user_q}

请在 ```sql ``` 块中给出最终符合{sql_type}语法的 SQL, 要有详细的字段注释,关联注释,以及条件注释。
并在代码块中用注释解释所用表、字段、JOIN 逻辑。对于你不知道的字段值映射关系,你要进行说明。
请勿返回其他内容，只返回 ```sql ``` 块。请注意SQL的执行效率,写出最佳性能的sql代码。
    """


# 全局变量用于缓存嵌入模型实例和当前加载的模型路径
_embedder_instance = None
_current_embed_model_path = None


# ------------- 主函数 --------------------
def rag_sql(question: str,
            api_key: str,
            base_url: str,
            sql_type: str,
            embed_model_path: str,  # 新增参数
            top_k: int = TOP_K):
    # 1. 加载资源（全局懒加载）
    global faiss_index, id2text, _embedder_instance, _current_embed_model_path

    if "faiss_index" not in globals():
        faiss_index = faiss.read_index(INDEX_PATH)
        id2text = json.load(open(MAP_PATH, encoding="utf-8"))

    # 检查是否需要重新加载嵌入模型
    if _embedder_instance is None or _current_embed_model_path != embed_model_path:
        yield f"正在加载嵌入模型: {embed_model_path}..."
        _embedder_instance = SentenceTransformer(embed_model_path)
        _embedder_instance.max_seq_length = MAX_LEN
        _current_embed_model_path = embed_model_path
        yield "嵌入模型加载完成."

    embedder = _embedder_instance  # 使用缓存的实例

    # 2. 编码问题向量
    q_vec = embedder.encode([question], normalize_embeddings=True).astype("float32")

    # 3. 检索
    _, I = faiss_index.search(q_vec, top_k)
    ctx_blocks = [id2text[str(idx)] if isinstance(id2text, dict) else id2text[idx]
                  for idx in I[0]]
    yield f"🎯 Top-{top_k} 命中表索引: {I[0]}"
    for i, block in enumerate(ctx_blocks):
        yield f"—— 表结构 Top-{i + 1} ——{block}"

    # 4. 构造 Prompt
    prompt = build_prompt(question, sql_type, ctx_blocks)
    yield f"📤 发送给大模型的Prompt如下：{prompt}"

    # 5. 调用 LLM
    final_answer = call_llm(prompt, api_key=api_key, base_url=base_url, sql_type=sql_type)
    yield "=== LLM 回复 ==="
    yield final_answer


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 生成 SQL demo")
    parser.add_argument("--question", default=QUESTION, help="用户自然语言问题")
    parser.add_argument("--k", type=int, default=TOP_K, help="检索 top-k")
    parser.add_argument("--key", type=str, default=API_KEY, help="OpenAI API Key")
    parser.add_argument("--url", type=str, default=LLM_URL, help="OpenAI Base URL")
    parser.add_argument("--sql_type", type=str, default=SQL_TYPE, help="OpenAI Base URL")
    parser.add_argument("--embed_model_path", type=str, default=DEFAULT_EMBED_MODEL_PATH, help="嵌入模型路径")
    args = parser.parse_args()

    # Iterate over the generator and print each yielded message
    for message in rag_sql(question=args.question, api_key=args.key, base_url=args.url, sql_type=args.sql_type,
                           embed_model_path=args.embed_model_path, top_k=args.k):
        print(message)
    os._exit(0)
