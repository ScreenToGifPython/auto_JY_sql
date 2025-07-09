#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_sql_generator.py
使用 RAG 检索表结构 
调用 LLM 生成 SQL 或直接进行表检索
Author: KevinChen
"""

import faiss
import backoff
import warnings
import requests
from typing import List
import os, json, argparse, sys
from openai import OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer

# 忽略警告
warnings.filterwarnings("ignore")

# ----------------- 默认参数 -----------------
INDEX_PATH = "faiss_index.bin"
MAP_PATH = "table_mapping.json"
SQL_TYPE = "mysql"
TOP_K = 10
MAX_LEN = 512
LLM_MODEL = "deepseek-chat"
API_KEY = "sk-xxxxxxxx"
LLM_URL = "https://api.deepseek.com"
QUESTION = "公募基金的二级分类基金类型是股票型的最近1年净值和收益率数据,只要交易日的数据,用上海市场交易日?"
DEFAULT_EMBED_MODEL_PATH = "BAAI/bge-m3"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 全局变量用于缓存，避免重复加载
_embedder_instance = None
_current_embed_model_path = None
_faiss_index = None
_id2text = None


def load_retrieval_resources(embed_model_path: str):
    """懒加载或重新加载 FAISS 索引、文本映射和嵌入模型。"""
    global _faiss_index, _id2text, _embedder_instance, _current_embed_model_path

    if _faiss_index is None:
        print("首次加载 FAISS 索引和映射...", file=sys.stderr)
        _faiss_index = faiss.read_index(INDEX_PATH)
        _id2text = json.load(open(MAP_PATH, encoding="utf-8"))

    if _embedder_instance is None or _current_embed_model_path != embed_model_path:
        print(f"正在加载嵌入模型: {embed_model_path}...", file=sys.stderr)
        _embedder_instance = SentenceTransformer(embed_model_path)
        _embedder_instance.max_seq_length = MAX_LEN
        _current_embed_model_path = embed_model_path
        print("嵌入模型加载完成.", file=sys.stderr)

    return _embedder_instance, _faiss_index, _id2text


@backoff.on_exception(backoff.expo, (OpenAIError, requests.exceptions.RequestException), max_tries=3)
def call_llm(prompt: str, api_key: str, base_url: str, sql_type: str, model: str = LLM_MODEL,
             temperature: float = 0.1) -> str:
    sys_prompt = f"""
你是一名资深数据工程师，精通{sql_type}数据库的 SQL 编写。请严格遵循以下原则：
1. **值映射：** 若字段备注已给中文代码映射，请先把用户描述转换为对应代码再过滤。
2. **表选择：** 只选择与需求直接相关的表，避免冗余 JOIN。
3. **JOIN 条件：** 当确有关系时使用表间关联字段。
4. **注释：** SQL 加上中文注释解释字段含义、过滤条件及 JOIN 逻辑。
5. **执行效率：** 你写的 SQL 一定是执行效率最高的SQL代码, 绝对符合{sql_type}数据库的特性,语法,执行效率。
6. **提问：** 如果你认为用户的问题太模糊,或者你需要更多的信息,请在生成SQL之后加再以注释形式说明。可以要求用户提供更多信息并且 "提高检索Top-K"。说明你需要什么信息。
6. **解释模拟：** 如果生成SQL的表和字段数据不在提供的表结构中 (是你模拟的), 请在SQL代码块中用注释强调说明。
    """
    if "lightcode-ui" in base_url:
        headers = {'Accept': "*/*", 'Authorization': f"Bearer {api_key}", 'Content-Type': "application/json"}
        payload = {"model": "gpt-4o",
                   "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
                   "stream": False}
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        if response.ok:
            result = response.json()
            if 'choices' in result and result.get('choices'):
                message = result['choices'][0].get('message', {})
                if 'content' in message:
                    return message['content'].strip()
            raise KeyError(f"无法从响应中提取内容。收到的响应: {json.dumps(result, ensure_ascii=False)}")
        else:
            response.raise_for_status()
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(model=model, temperature=temperature,
                                              messages=[{"role": "system", "content": sys_prompt},
                                                        {"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()


def build_prompt(user_q: str, sql_type: str, ctx_blocks: List[str]) -> str:
    ctx_txt = "\n\n--- 相关表结构 ---\n" + "\n\n".join(ctx_blocks)
    return f"""{ctx_txt}

--- 用户需求 ---
{user_q}

请在 ```sql ``` 块中给出最终符合{sql_type}语法的 SQL, 要有详细的字段注释,关联注释,表含义数组, 以及条件注释。
并在代码块中用注释解释所用表、字段、JOIN 逻辑。对于你不知道的字段值映射关系,你要进行说明。
请勿返回其他内容，只返回 ```sql ``` 块。请注意SQL的执行效率,写出最佳性能的sql代码。
如果需求过于模糊, 在返回SQL后用注释说明你需要什么信息。 如果你需要更多的表结构, 在返回SQL后用注释告诉用户 '提高检索Top-K'。
如果生成SQL的表和字段数据不在提供的表结构中 (是你模拟的), 请在SQL代码块中用注释强调说明。
    """


def rag_sql_generator(question: str, api_key: str, base_url: str, sql_type: str, embed_model_path: str, top_k: int):
    embedder, faiss_index, id2text = load_retrieval_resources(embed_model_path)
    q_vec = embedder.encode([question], normalize_embeddings=True).astype("float32")
    _, I = faiss_index.search(q_vec, top_k)
    ctx_blocks = [id2text[idx] for idx in I[0]]
    yield f"🎯 Top-{top_k} 命中表索引: {I[0]}"
    for i, block in enumerate(ctx_blocks):
        yield f"—— 表结构 Top-{i + 1} ——\n{block}"
    prompt = build_prompt(question, sql_type, ctx_blocks)
    yield f"📤 发送给大模型的Prompt如下：\n{prompt}"
    final_answer = call_llm(prompt, api_key=api_key, base_url=base_url, sql_type=sql_type)
    yield "=== LLM 回复 ==="
    yield final_answer


def find_similar_tables(question: str, embed_model_path: str, top_k: int):
    embedder, faiss_index, id2text = load_retrieval_resources(embed_model_path)
    q_vec = embedder.encode([question], normalize_embeddings=True).astype("float32")
    distances, indices = faiss_index.search(q_vec, top_k)
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        similarity = max(0, 1 - (dist ** 2) / 2)
        details = id2text[idx]
        table_name = "未知表"
        for line in details.split('\n'):
            if line.startswith("表名:"):
                table_name = line.replace("表名:", "").strip()
                break
        results.append({
            "table_name": table_name,
            "similarity_percentage": round(similarity * 100, 2),
            "details": details
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG SQL Agent")
    parser.add_argument("--question", default=QUESTION, help="用户自然语言问题")
    parser.add_argument("--k", type=int, default=TOP_K, help="检索 top-k")
    parser.add_argument("--embed_model_path", type=str, default=DEFAULT_EMBED_MODEL_PATH, help="嵌入模型路径")
    parser.add_argument("--mode", type=str, default="sql", choices=["sql", "search"],
                        help="执行模式: 'sql' (生成SQL) 或 'search' (表检索)")
    # 'sql' mode specific arguments
    parser.add_argument("--key", type=str, default=API_KEY, help="LLM API Key")
    parser.add_argument("--url", type=str, default=LLM_URL, help="LLM Base URL")
    parser.add_argument("--sql_type", type=str, default=SQL_TYPE, help="目标SQL方言")
    args = parser.parse_args()

    if args.mode == 'sql':
        for message in rag_sql_generator(question=args.question, api_key=args.key, base_url=args.url,
                                         sql_type=args.sql_type, embed_model_path=args.embed_model_path, top_k=args.k):
            print(message)
    elif args.mode == 'search':
        search_results = find_similar_tables(question=args.question, embed_model_path=args.embed_model_path,
                                             top_k=args.k)
        print(json.dumps(search_results, ensure_ascii=False, indent=2))

    os._exit(0)
