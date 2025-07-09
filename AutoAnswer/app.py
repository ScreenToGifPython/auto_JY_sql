import os
import sys
import json
import csv
import subprocess
import traceback
from typing import Tuple, List
from collections import defaultdict

import gradio as gr
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import sqlglot
    from sqlglot.expressions import Table, Column, EQ
except ImportError:
    print("警告: 未安装 'sqlglot' 库，SQL关系分析功能将不可用。")
    print("请通过 'pip install sqlglot' 命令安装。")
    sqlglot = None

# 配置文件路径
CONFIG_FILE = "llm_config.json"

# 定义脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义默认值
QUESTION = "公募基金的二级分类基金类型是股票型的最近1年净值和收益率数据,只要交易日的数据,用上海市场交易日?"
DEFAULT_EMBED_MODEL_PATH = "BAAI/bge-m3"

# 表信息文件路径
TABLE_JSON_PATH = os.path.join(SCRIPT_DIR, "table.json")
_TABLE_DATA = None


# --- Data Preprocessing Functions (Integrated) ---

def parse_csv_to_json(csv_file_path, json_file_path):
    tables_dict = {}
    current_table_name = None
    current_table_data = None
    in_fields_section = False
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not any(field.strip() for field in row):
                if in_fields_section:
                    in_fields_section = False
                    if current_table_name and current_table_data:
                        tables_dict[current_table_name] = current_table_data
                        current_table_name = None
                        current_table_data = None
                continue
            if row and row[0].strip() == '表名':
                if current_table_name and current_table_data:
                    tables_dict[current_table_name] = current_table_data
                table_name = row[1].strip() if len(row) > 1 else ''
                if table_name:
                    current_table_name = table_name
                    current_table_data = {'description': '', 'fields': []}
                in_fields_section = False
                continue
            if current_table_data:
                if row and row[0].strip() == '描述':
                    current_table_data['description'] = row[1].strip() if len(row) > 1 else ''
                    continue
                if row and row[0].strip() == '字段明细':
                    in_fields_section = True
                    continue
                if in_fields_section and len(row) > 5 and row[1].strip():
                    field_info = {
                        'field_name': row[1].strip(), 'field_type': row[2].strip(),
                        'is_nullable': row[3].strip(), 'default_value': row[4].strip(),
                        'field_description': row[5].strip()
                    }
                    current_table_data['fields'].append(field_info)
    if current_table_name and current_table_data:
        tables_dict[current_table_name] = current_table_data
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(tables_dict, jsonfile, ensure_ascii=False, indent=4)


def extract_data_to_json(csv_file_path, output_json_path):
    results_dict = {}
    chunk_iter = pd.read_csv(csv_file_path, chunksize=10000, on_bad_lines='skip', low_memory=False)
    required_columns = ['id', 'req_url', 'db_sql']
    for i, chunk in enumerate(chunk_iter):
        if not all(col in chunk.columns for col in required_columns):
            raise ValueError("错误: CSV文件中缺少必需的列 (id, req_url, db_sql)。")
        chunk.dropna(subset=['id'], inplace=True)
        chunk['id'] = chunk['id'].astype(str)
        for index, row in chunk.iterrows():
            results_dict[row['id']] = {'req_url': row.get('req_url'), 'db_sql': row.get('db_sql')}
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(results_dict, jsonfile, ensure_ascii=False, indent=4)


def analyze_sql_relationships(input_json_path, output_json_path, dialect="oracle"):
    if not sqlglot:
        raise ImportError("sqlglot库未安装，无法分析SQL关系。")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    table_relationships = defaultdict(lambda: defaultdict(set))
    for key, value in data.items():
        sql_query = value.get('db_sql')
        if not sql_query or not isinstance(sql_query, str): continue
        try:
            expr = sqlglot.parse_one(sql_query, read=dialect)
            if not expr: continue
            alias_map = {(t.alias_or_name or t.this.sql(dialect=dialect)).upper(): t.this.sql(dialect=dialect).upper()
                         for t in expr.find_all(Table)}
            for eq in expr.find_all(EQ):
                if not (isinstance(eq.left, Column) and isinstance(eq.right, Column)): continue
                l_table, r_table = alias_map.get((eq.left.table or "").upper()), alias_map.get(
                    (eq.right.table or "").upper())
                if not (l_table and r_table and l_table != r_table): continue
                l_col, r_col = eq.left.name.upper(), eq.right.name.upper()
                if (l_table, l_col) > (r_table, r_col):
                    l_table, r_table, l_col, r_col = r_table, l_table, r_col, l_col
                canonical = f"{l_table}.{l_col} = {r_table}.{r_col}"
                table_relationships[l_table][r_table].add(canonical)
        except Exception:
            pass
    final_relationships = defaultdict(dict)
    for table1, related_tables in table_relationships.items():
        for table2, conditions in related_tables.items():
            if table1 != table2:
                final_relationships[table1][table2] = sorted(list(conditions))
    with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(final_relationships, jsonfile, ensure_ascii=False, indent=4)


def merge_and_govern_relations(info_path, relation_path, output_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        table_info = json.load(f)
    with open(relation_path, 'r', encoding='utf-8') as f:
        table_relations = json.load(f)
    merged_data = {name: {"description": info.get("description", ""), "fields": info.get("fields", []),
                          "relations": table_relations.get(name, {})} for name, info in table_info.items()}
    for name, rel in table_relations.items():
        if name not in merged_data:
            merged_data[name] = {"description": "", "fields": [], "relations": rel}
    canonical_relations = {tuple(sorted([p.strip() for p in c.split('=')])) for d in merged_data.values() for cs in
                           d.get("relations", {}).values() for c in cs if len(c.split('=')) == 2}
    governed_relations = defaultdict(lambda: defaultdict(list))
    for rel_tuple in canonical_relations:
        p1, p2 = rel_tuple
        t1, t2 = p1.split('.')[0], p2.split('.')[0]
        if t1 in merged_data and t2 in merged_data:
            governed_relations[t1][t2].append(f"{p1} = {p2}")
            governed_relations[t2][t1].append(f"{p2} = {p1}")
    for name, relations in governed_relations.items():
        if name in merged_data:
            merged_data[name]["relations"] = dict(relations)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)


def create_text_chunks_from_json(src_json: dict) -> Tuple[List[str], List[str]]:
    table_names, text_chunks = [], []
    for tbl, info in src_json.items():
        if not isinstance(info, dict): continue
        desc = info.get("description") or info.get("tableChiName", "")
        fields, relations = info.get("fields", []), info.get("relations", {})
        block = [f"表名: {tbl}", f"表含义: {desc or 'N/A'}", "字段:"]
        block.extend([
                         f"  - 字段名:{f.get('field_name', '')}, 类型:{f.get('field_type', '')}, 含义:{f.get('field_description', '')}"
                         for f in fields] if fields else ["  (无字段信息)"])
        block.append("与其他表的关联关系:")
        block.extend([f"  - {c}" for tgt, conds in relations.items() for c in conds] if relations else [
            "与其他表的关联关系: (无关联信息)"])
        table_names.append(tbl)
        text_chunks.append("\n".join(block))
    return table_names, text_chunks


def run_vectorization(model_path, json_path, index_path, map_path, index_type, batch, max_len, progress=gr.Progress()):
    progress(0, desc="读取JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        table_defs = json.load(f)
    progress(0.1, desc="生成文本块...")
    _, chunks = create_text_chunks_from_json(table_defs)
    if not chunks: raise ValueError("无文本块生成，退出")
    progress(0.2, desc="加载嵌入模型...")
    model = SentenceTransformer(model_path, trust_remote_code=True)
    model.max_seq_length = max_len
    progress(0.4, desc="编码向量...")
    vecs = model.encode(chunks, batch_size=batch, show_progress_bar=True, normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")
    progress(0.8, desc=f"构建 {index_type} 索引...")
    dim = vecs.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
        index.hnsw.efConstruction, index.hnsw.efSearch = 200, 64
        index.add(vecs)
    else:
        raise ValueError("索引类型仅支持 flat / hnsw")
    faiss.write_index(index, index_path)
    progress(0.9, desc="保存FAISS索引...")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    progress(1, desc="保存表结构映射...")


def run_preprocessing_pipeline(table_struct_csv, segment_sql_csv, sql_dialect, embed_model, faiss_type, batch_size,
                               max_len, progress=gr.Progress()):
    log_output = ""

    def log(msg):
        nonlocal log_output; log_output += msg + "\n"; return log_output

    try:
        yield log("【步骤 1/5】 正在转换表结构 CSV -> JSON...")
        table_info_json = os.path.join(SCRIPT_DIR, "table_info.json")
        parse_csv_to_json(table_struct_csv, table_info_json)
        yield log("【步骤 1/5】 ✅ 成功生成 table_info.json")

        yield log("\n【步骤 2/5】 正在从 SQL 日志提取数据...")
        extracted_data_json = os.path.join(SCRIPT_DIR, "extracted_data.json")
        extract_data_to_json(segment_sql_csv, extracted_data_json)
        yield log("【步骤 2/5】 ✅ 成功生成 extracted_data.json")

        yield log(f"\n【步骤 3/5】 正在使用 {sql_dialect} 方言分析 SQL 关系...")
        table_relation_json = os.path.join(SCRIPT_DIR, "table_relation.json")
        analyze_sql_relationships(extracted_data_json, table_relation_json, dialect=sql_dialect)
        yield log("【步骤 3/5】 ✅ 成功生成 table_relation.json")

        yield log("\n【步骤 4/5】 正在合并表信息与关系...")
        final_table_json = os.path.join(SCRIPT_DIR, "table.json")
        merge_and_govern_relations(table_info_json, table_relation_json, final_table_json)
        yield log("【步骤 4/5】 ✅ 成功生成 table.json")

        yield log("\n【步骤 5/5】 正在向量化表结构...")
        faiss_index_bin = os.path.join(SCRIPT_DIR, "faiss_index.bin")
        table_mapping_json = os.path.join(SCRIPT_DIR, "table_mapping.json")
        run_vectorization(embed_model, final_table_json, faiss_index_bin, table_mapping_json, faiss_type,
                          int(batch_size), int(max_len), progress)
        yield log("【步骤 5/5】 ✅ 成功生成 FAISS 索引和映射文件。")

        yield log("\n🎉 所有预处理步骤完成！")
    except Exception as e:
        yield log(f"❌ 处理失败: {e}\n{traceback.format_exc()}")


# --- Core App Functions ---

def load_table_data():
    global _TABLE_DATA
    if _TABLE_DATA is None:
        try:
            with open(TABLE_JSON_PATH, 'r', encoding='utf-8') as f:
                _TABLE_DATA = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _TABLE_DATA = {}
    return _TABLE_DATA


def save_config(embed_model_path, top_k, llm_model, api_key, llm_url, sql_type):
    config = {"EMBED_MODEL": embed_model_path, "TOP_K": top_k, "LLM_MODEL": llm_model, "API_KEY": api_key,
              "LLM_URL": llm_url, "SQL_TYPE": sql_type}
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
    return f"✅ 配置成功保存到 {CONFIG_FILE}"


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def format_table_details(table_name: str):
    table_info = load_table_data().get(table_name.strip().upper())
    if not table_info: return f"❌ 未找到表: **{table_name}**"
    md = f"## 表名: {table_name}\n**表含义**: {table_info.get('description', '无')}\n\n### 字段信息:\n"
    if table_info.get('fields'):
        md += "| 字段名 | 字段含义 | 字段格式 | 可否为空 | 默认值 |\n|---|---|---|---|---|\n"
        md += "\n".join([
                            f"| {f.get('field_name', '')} | {f.get('field_description', '')} | {f.get('field_type', '')} | {f.get('is_nullable', '')} | {f.get('default_value', '')} |"
                            for f in table_info['fields']])
    else:
        md += "无字段信息。\n"
    md += "\n### 关联信息:\n"
    if table_info.get('relations'):
        md += "\n".join(
            [f"- **{rel_table}**: ` {cond} `" for rel_table, conds in table_info['relations'].items() for cond in
             conds])
    else:
        md += "无关联信息。\n"
    return md


def generate_sql_and_log(question, embed_model_path, top_k, llm_model, api_key, llm_url, sql_type):
    if not all([question, embed_model_path, llm_model, api_key, llm_url, sql_type]):
        yield "❌ 错误: 所有字段均不能为空.", ""
        return
    command = [sys.executable, os.path.join(SCRIPT_DIR, "rag_query.py"), "--question", question, "--embed_model_path",
               embed_model_path, "--k", str(int(top_k)), "--key", api_key, "--url", llm_url, "--sql_type", sql_type]
    log_content, sql_content = f"▶️ 执行命令: {' '.join(command)}\n" + "-" * 20 + "\n", "⏳ 等待脚本执行..."
    yield log_content, sql_content
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                               bufsize=1, universal_newlines=True, cwd=SCRIPT_DIR)
    for line in iter(process.stdout.readline, ''):
        log_content += line
        yield log_content, sql_content
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        log_content += f"\n--- 脚本执行出错, 返回码: {return_code} ---\n"
        sql_content = "❌ 执行失败"
    else:
        log_content += "\n--- 脚本执行完毕 ---\n"
        try:
            start_tag, end_tag = "```sql", "```"
            start_index = log_content.rfind(start_tag)
            if start_index != -1:
                end_index = log_content.find(end_tag, start_index + len(start_tag))
                sql_content = log_content[
                              start_index + len(start_tag):end_index].strip() if end_index != -1 else "❌ 未找到结束标记"
            else:
                sql_content = "❌ 未找到开始标记"
        except Exception as e:
            sql_content = f"❌ 解析SQL时出错: {e}"
    yield log_content, sql_content


# --- Gradio UI ---
def create_ui():
    loaded_config = load_config()
    with gr.Blocks(title="RAG SQL Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 📝 RAG-based SQL Generator")
        gr.Markdown("通过输入自然语言问题, 检索相关库表结构, 并由大模型生成相应的 SQL 查询.")

        with gr.Tab("大模型SQL生成"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(lines=8, label="用户问题", placeholder=f"例如: '{QUESTION}'")
                    sql_result_output = gr.Code(label="SQL 生成结果", language="sql", lines=10, interactive=False)
                    log_output = gr.Textbox(lines=10, label="详细日志", interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("#### ⚙️ 参数配置")
                    embed_model_input = gr.Textbox(value=loaded_config.get("EMBED_MODEL", ""), label="嵌入模型路径",
                                                   interactive=True, placeholder="例如: BAAI/bge-m3")
                    top_k_input = gr.Slider(minimum=1, maximum=50, value=loaded_config.get("TOP_K", 10), step=1,
                                            label="检索 Top-K")
                    sql_type_input = gr.Dropdown(["MYSQL", "PostgreSQL", "SparkSQL", "Oracle", "SQLServer"],
                                                 value=loaded_config.get("SQL_TYPE", "MYSQL"), label="目标 SQL 类型")
                    gr.Markdown("---")
                    gr.Markdown("#### 🧠 大模型配置")
                    llm_model_input = gr.Textbox(value=loaded_config.get("LLM_MODEL", ""), label="大模型名称")
                    api_key_input = gr.Textbox(value=loaded_config.get("API_KEY", ""), label="API 密钥",
                                               type="password")
                    llm_url_input = gr.Textbox(value=loaded_config.get("LLM_URL", ""), label="大模型服务地址")
                    save_button = gr.Button("保存配置")
                    generate_button = gr.Button("🚀 生成SQL", variant="primary")
            save_button.click(fn=save_config,
                              inputs=[embed_model_input, top_k_input, llm_model_input, api_key_input, llm_url_input,
                                      sql_type_input], outputs=[log_output])
            generate_button.click(fn=generate_sql_and_log,
                                  inputs=[question_input, embed_model_input, top_k_input, llm_model_input,
                                          api_key_input, llm_url_input, sql_type_input],
                                  outputs=[log_output, sql_result_output])

        with gr.Tab("数据预处理"):
            gr.Markdown("## ⚙️ 数据预处理与向量化")
            gr.Markdown("一键完成从原始CSV到FAISS向量索引的完整流程。")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### 输入文件路径")
                    table_struct_csv_input = gr.Textbox(label="表结构CSV文件路径",
                                                        value=os.path.join(SCRIPT_DIR, "表结构.csv"))
                    segment_sql_csv_input = gr.Textbox(label="SQL日志CSV文件路径",
                                                       value=os.path.join(SCRIPT_DIR, "segment_sql.csv"))
                    gr.Markdown("#### 参数配置")
                    sql_dialect_input = gr.Dropdown(label="SQL方言 (用于关系分析)",
                                                    choices=["oracle", "mysql", "postgres", "spark"], value="mysql")
                    preprocess_embed_model_input = gr.Textbox(label="嵌入模型 (路径或HuggingFace名称)",
                                                              value=loaded_config.get("EMBED_MODEL",
                                                                                      DEFAULT_EMBED_MODEL_PATH))
                    faiss_type_input = gr.Dropdown(label="FAISS索引类型", choices=["hnsw", "flat"], value="hnsw")
                    with gr.Row():
                        batch_size_input = gr.Number(label="批处理大小", value=16, minimum=1, precision=0)
                        max_len_input = gr.Number(label="最大序列长度", value=512, minimum=1, precision=0)
                    preprocess_button = gr.Button("🚀 开始一键预处理", variant="primary")
                with gr.Column(scale=3):
                    preprocess_log_output = gr.Textbox(label="处理日志", lines=22, interactive=False)
            preprocess_button.click(fn=run_preprocessing_pipeline,
                                    inputs=[table_struct_csv_input, segment_sql_csv_input, sql_dialect_input,
                                            preprocess_embed_model_input, faiss_type_input, batch_size_input,
                                            max_len_input], outputs=[preprocess_log_output])

        with gr.Tab("表信息查询"):
            gr.Markdown("## 🔍 表信息查询")
            gr.Markdown("输入表名，查询其详细结构、含义及关联信息。")
            with gr.Row():
                with gr.Column(scale=1):
                    table_name_query_input = gr.Textbox(label="输入表名", placeholder="例如: ADS_CODE_MAPPING")
                    query_table_button = gr.Button("查询表信息")
                with gr.Column(scale=2):
                    table_details_output = gr.Markdown(label="表详细信息")
            query_table_button.click(fn=format_table_details, inputs=[table_name_query_input],
                                     outputs=[table_details_output])
            table_name_query_input.submit(fn=format_table_details, inputs=[table_name_query_input],
                                          outputs=[table_details_output])

        with gr.Tab("说明文档"):
            gr.Markdown("""
                ## 📖 使用说明
                ### 大模型SQL生成
                在此页面，您可以输入自然语言问题，并配置相关参数，调用大模型生成SQL。
                ### 数据预处理
                在此页面，您可以一键完成数据预处理和向量化，为“大模型SQL生成”做数据准备。
                ### 表信息查询
                在此页面，您可以查询已处理好的表的详细信息。
                """)
    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7861)
