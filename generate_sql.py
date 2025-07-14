"""
@File: app.py
@Modify Time: 2025/7/7
@Author: KevinChen
@Descriptions: 用户用自然语言提供他们的查询需求（例如：“查询2023年以来所有股票的最高价”），
选择目标SQL方言（MySQL或Oracle），并选择一种生成模式 (RAG或完整上下文)。
"""
import faiss, json, os, numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from llm_utils import call_llm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_resources(index_path, mapping_path, embedding_model):
    index = faiss.read_index(index_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        table_names = json.load(f)
    embedder = SentenceTransformer(embedding_model)
    return index, table_names, embedder


def search_tables(query, index, embedder, table_names, top_k=4):
    q_emb = embedder.encode([query])  # shape (1, dim)
    D, I = index.search(np.asarray(q_emb, dtype="float32"), top_k)
    hits = [(table_names[i], float(D[0][rank])) for rank, i in enumerate(I[0])]
    return hits  # 返回 (表名, 距离) 列表


def build_prompt(query, hits, table_definitions, db_dialect):
    context_blocks = []
    for tbl, _ in hits:
        info = table_definitions[tbl]
        # 只放关键信息，避免 token 过大
        block = (
                f"表名: {tbl}（{info['tableChiName']}）\n"
                f"说明: {info['description']}（{info.get('description_en', '无')}）"
                f"主键: {info['key']}\n"
                f"字段:\n"
                "\n".join([
                              f"  - 列名: {c.get('列名', 'N/A')}, 数据类型: {c.get('数据类型', 'N/A')}, 备注: {c.get('备注', '无')}"
                              for c in info['columns'][:15]]) + "\n")
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)

    prompt = f"""
你是一名资深数据工程师，请根据以下数据库元数据，为用户编写高质量 SQL：

### 数据库元数据
{context}

### 用户需求
{query}

### 要求
- 生成的SQL必须兼容 {db_dialect} 语法。
- 请在SQL中添加注释，清晰地解释关键部分的逻辑，例如JOIN条件、WHERE子句的目的或复杂的函数用法。
- 必要时请合理 JOIN，JOIN 条件优先使用主键/外键 InnerCode 等。
- 只返回 SQL 代码（包含注释），不要附带任何额外的解释性文字。
"""
    return prompt.strip()


def build_full_prompt(query, table_definitions, db_dialect):
    """将所有表定义作为上下文构建prompt。"""
    context_blocks = []
    for tbl, info in table_definitions.items():
        block = (
                f"表名: {tbl}（{info.get('tableChiName', 'N/A')}）\n"
                f"说明: {info.get('description', 'N/A')}（{info.get('description_en', 'N/A')}）\n"
                f"主键: {info.get('key', 'N/A')}\n"
                f"字段:\n" + "\n".join([
            f"  - 列名: {c.get('列名', 'N/A')}, 数据类型: {c.get('数据类型', 'N/A')}, 备注: {c.get('备注', '无')}"
            for c in info.get('columns', [])
        ]) + "\n"
        )
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)

    prompt = f"""
你是一名资深数据工程师，请根据以下完整的数据库元数据，为用户编写高质量 SQL：

### 数据库元数据
{context}

### 用户需求
{query}

### 要求
- 生成的SQL必须兼容 {db_dialect} 语法。
- 请在SQL中添加注释，清晰地解释关键部分的逻辑，例如JOIN条件、WHERE子句的目的或复杂的函数用法。
- 必要时请合理 JOIN，JOIN 条件优先使用主键/外键 InnerCode 等。
- 只返回 SQL 代码（包含注释），不要附带任何额外的解释性文字。
"""
    return prompt.strip()


def generate_sql_with_llm(prompt, api_key, base_url, llm_model):
    system_prompt = """
你是一名资深数据工程师，精通 SQL 编写。请严格遵循以下原则：
1.  **数据类型和值映射：** 如果字段备注中提到与 `CT_SystemConst` 表关联，或明确给出值到描述的映射（例如：`7-指数型, 8-优化指数型, 16-非指数型`），请务必将用户查询中的中文描述转换为对应的数字代码或英文缩写进行过滤。例如，如果用户查询“QDII类型”，而备注中说明 `InvestmentType` 字段 `7-QDII`，则应使用 `InvestmentType = 7`。
2.  **表选择：** 优先选择包含用户所需信息的表。对于日期相关的查询（如清盘日期），请优先考虑 `MF_FundArchives` 表中的 `ExpireDate` 或 `LastOperationDate` 字段，而不是 `MF_Transformation` 等不包含此类信息的表。
3.  **JOIN 条件：** 必要时请合理 JOIN，JOIN 条件优先使用主键/外键 InnerCode 等。
4.  **仔细遵循用户在 ### 要求 部分提供的所有格式化和内容指令。"""

    sql_code = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        api_key=api_key,
        base_url=base_url,
        model_name=llm_model,
        temperature=0.1
    )
    return sql_code


def main(args):
    json_path = "table_definitions.json"
    index_path = "faiss_index.bin"
    mapping_path = "table_mapping.json"

    with open(json_path, "r", encoding="utf-8") as f:
        table_defs = json.load(f)

    if args.mode == 'RAG':
        print("🚀 正在使用 RAG 模式...")
        index, names, embedder = load_resources(index_path, mapping_path, args.embedding_model)
        hits = search_tables(args.query, index, embedder, names, args.top_k)
        print("🔍 Top-k 检索结果:", hits)
        prompt = build_prompt(args.query, hits, table_defs, args.sql_type)
    elif args.mode == 'FULL':
        print("🚀 正在使用完整上下文模式...")
        prompt = build_full_prompt(args.query, table_defs, args.sql_type)
    else:
        raise ValueError("无效的模式，请选择 'RAG' 或 'FULL'")

    sql = generate_sql_with_llm(prompt, args.api_key, args.base_url, args.model_name)
    print("\n=== 生成的 SQL ===\n", sql)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据用户需求和数据库元数据生成SQL查询。")

    # LLM 参数
    parser.add_argument("--api_key", type=str, required=True, help="LLM API Key")
    parser.add_argument("--base_url", type=str, required=True, help="LLM API Base URL")
    parser.add_argument("--model_name", type=str, required=True, help="LLM 模型名称")

    # 模式选择
    parser.add_argument("--mode", type=str, required=True, choices=['RAG', 'FULL'],
                        help="运行模式: RAG 或 FULL (完整上下文)")

    # RAG特定参数
    parser.add_argument("--embedding_model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="RAG模式下使用的嵌入模型")
    parser.add_argument("--top_k", type=int, default=4, help="RAG模式下检索的Top-K表定义")

    # SQL生成参数
    parser.add_argument("--sql_type", type=str, required=True, choices=['MySQL', 'Oracle'], help="要生成的SQL方言")
    parser.add_argument("--query", type=str, required=True, help="用户的自然语言查询需求")

    args = parser.parse_args()

    if args.query:
        main(args)
    else:
        print("❌ 查询不能为空。")
