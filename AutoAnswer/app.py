import os
import sys
import json
import subprocess
import gradio as gr

# 配置文件路径
CONFIG_FILE = "llm_config.json"

# 定义脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义默认值 (不再从 rag_query 导入)
QUESTION = "公募基金的二级分类基金类型是股票型的最近1年净值和收益率数据,只要交易日的数据,用上海市场交易日?"
DEFAULT_EMBED_MODEL_PATH = "BAAI/bge-m3"  # 默认嵌入模型路径，这里使用Hugging Face模型名称

# 表信息文件路径
TABLE_JSON_PATH = os.path.join(SCRIPT_DIR, "table.json")
_TABLE_DATA = None  # 用于缓存加载的表信息


def load_table_data():
    """加载 table.json 文件中的表信息"""
    global _TABLE_DATA
    if _TABLE_DATA is None:
        try:
            with open(TABLE_JSON_PATH, 'r', encoding='utf-8') as f:
                _TABLE_DATA = json.load(f)
            print(f"✅ 成功加载表信息文件: {TABLE_JSON_PATH}")
        except FileNotFoundError:
            print(f"❌ 错误: 未找到表信息文件: {TABLE_JSON_PATH}")
            _TABLE_DATA = {}
        except json.JSONDecodeError as e:
            print(f"❌ 错误: 解析表信息文件 {TABLE_JSON_PATH} 失败: {e}")
            _TABLE_DATA = {}
    return _TABLE_DATA


# --- 配置加载与保存 ---
def save_config(embed_model_path, top_k, llm_model, api_key, llm_url, sql_type):
    """保存所有配置到 JSON 文件"""
    config = {
        "EMBED_MODEL": embed_model_path,
        "TOP_K": top_k,
        "LLM_MODEL": llm_model,
        "API_KEY": api_key,
        "LLM_URL": llm_url,
        "SQL_TYPE": sql_type
    }
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        return f"✅ 配置成功保存到 {CONFIG_FILE}"
    except Exception as e:
        return f"❌ 保存配置失败: {e}"


def load_config():
    """加载本地配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 读取配置文件 {CONFIG_FILE} 失败: {e}")
            return {}
    return {}


def format_table_details(table_name: str):
    """根据表名格式化并返回表详细信息"""
    load_table_data()  # 确保数据已加载
    if not _TABLE_DATA:
        return "❌ 错误: 表信息数据未加载或为空。"

    table_name = table_name.strip().upper()  # 统一转大写处理
    table_info = _TABLE_DATA.get(table_name)

    if not table_info:
        return f"❌ 未找到表: **{table_name}**"

    markdown_output = f"## 表名: {table_name}\n"
    markdown_output += f"**表含义**: {table_info.get('description', '无')}\n\n"

    markdown_output += "### 字段信息:\n"
    if 'fields' in table_info and table_info['fields']:
        markdown_output += "| 字段名 | 字段含义 | 字段格式 | 可否为空 | 默认值 |\n"
        markdown_output += "|---|---|---|---|---|\n"
        for field in table_info['fields']:
            markdown_output += (
                f"| {field.get('field_name', '无')} "
                f"| {field.get('field_description', '无')} "
                f"| {field.get('field_type', '无')} "
                f"| {field.get('is_nullable', '无')} "
                f"| {field.get('default_value', '无')} |\n"
            )
    else:
        markdown_output += "无字段信息。\n"

    markdown_output += "\n### 关联信息:\n"
    if 'relations' in table_info and table_info['relations']:
        for related_table, conditions in table_info['relations'].items():
            markdown_output += f"- **{related_table}**:\n"
            for condition in conditions:
                markdown_output += f"  - `{condition}`\n"
    else:
        markdown_output += "无关联信息。\n"

    return markdown_output


# --- 主逻辑 ---
def generate_sql_and_log(question, embed_model_path, top_k, llm_model, api_key, llm_url, sql_type):
    """执行RAG-SQL生成, 并实时捕获日志和最终SQL"""
    if not all([question, embed_model_path, llm_model, api_key, llm_url, sql_type]):
        error_msg = "❌ 错误: 用户问题、嵌入模型路径、LLM 模型、API 密钥、URL 和 SQL 类型不能为空."
        yield error_msg, ""
        return

    command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "rag_query.py"),
        "--question", question,
        "--embed_model_path", embed_model_path,
        "--k", str(int(top_k)),
        "--key", api_key,
        "--url", llm_url,
        "--sql_type", sql_type
    ]

    log_content = f"▶️ 执行命令: {' '.join(command)}\n" + "-" * 20 + "\n"
    sql_content = "⏳ 等待脚本执行..."
    yield log_content, sql_content

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1,
        universal_newlines=True,
        cwd=SCRIPT_DIR
    )

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
            start_tag = "```sql"
            end_tag = "```"
            start_index = log_content.rfind(start_tag)
            if start_index != -1:
                end_index = log_content.find(end_tag, start_index + len(start_tag))
                if end_index != -1:
                    sql_content = log_content[start_index + len(start_tag):end_index].strip()
                else:
                    sql_content = "❌ 未找到SQL代码块的结束标记 '```'。"
            else:
                sql_content = "❌ 未在日志中找到SQL代码块的开始标记 '```sql'。"
        except Exception as e:
            sql_content = f"❌ 解析SQL时出错: {e}"

    yield log_content, sql_content


# --- Gradio 界面 ---
def create_ui():
    """创建并返回 Gradio UI 实例"""
    loaded_config = load_config()

    with gr.Blocks(title="RAG SQL Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 📝 RAG-based SQL Generator")
        gr.Markdown("通过输入自然语言问题, 检索相关库表结构, 并由大模型生成相应的 SQL 查询.")

        with gr.Tab("大模型SQL生成"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        lines=8,
                        label="用户问题",
                        placeholder=f"请输入你的数据查询需求, 例如: '{QUESTION}'"
                    )
                    sql_result_output = gr.Code(
                        label="SQL 生成结果",
                        language="sql",
                        lines=10,
                        interactive=False
                    )
                    log_output = gr.Textbox(
                        lines=10,
                        label="详细日志",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    gr.Markdown("#### ⚙️ 参数配置")
                    embed_model_input = gr.Textbox(
                        value=loaded_config.get("EMBED_MODEL", ""),
                        label="嵌入模型路径",
                        interactive=True,
                        placeholder="例如: BAAI/bge-m3 或本地模型路径"
                    )
                    top_k_input = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=loaded_config.get("TOP_K", 10),
                        step=1,
                        label="检索 Top-K"
                    )
                    sql_type_input = gr.Dropdown(
                        ["MYSQL", "PostgreSQL", "SparkSQL", "Oracle", "SQLServer"],
                        value=loaded_config.get("SQL_TYPE", "MYSQL"),
                        label="目标 SQL 类型"
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### 🧠 大模型配置")
                    llm_model_input = gr.Textbox(
                        value=loaded_config.get("LLM_MODEL", ""),
                        label="大模型名称"
                    )
                    api_key_input = gr.Textbox(
                        value=loaded_config.get("API_KEY", ""),
                        label="API 密钥",
                        type="password"
                    )
                    llm_url_input = gr.Textbox(
                        value=loaded_config.get("LLM_URL", ""),
                        label="大模型服务地址"
                    )

                    save_button = gr.Button("保存配置")
                    generate_button = gr.Button("🚀 生成SQL", variant="primary")

            # --- 事件绑定 ---
            save_button.click(
                fn=save_config,
                inputs=[
                    embed_model_input,
                    top_k_input,
                    llm_model_input,
                    api_key_input,
                    llm_url_input,
                    sql_type_input
                ],
                outputs=[log_output]
            )

            generate_button.click(
                fn=generate_sql_and_log,
                inputs=[
                    question_input,
                    embed_model_input,
                    top_k_input,
                    llm_model_input,
                    api_key_input,
                    llm_url_input,
                    sql_type_input
                ],
                outputs=[log_output, sql_result_output]
            )

        with gr.Tab("表信息查询"):
            gr.Markdown("## 🔍 表信息查询")
            gr.Markdown("输入表名，查询其详细结构、含义及关联信息。")
            with gr.Row():
                with gr.Column(scale=1):
                    table_name_query_input = gr.Textbox(
                        label="输入表名",
                        placeholder="例如: ADS_CODE_MAPPING"
                    )
                    query_table_button = gr.Button("查询表信息")
                with gr.Column(scale=2):
                    table_details_output = gr.Markdown(
                        label="表详细信息"
                    )
            query_table_button.click(
                fn=format_table_details,
                inputs=[table_name_query_input],
                outputs=[table_details_output]
            )
            table_name_query_input.submit(
                fn=format_table_details,
                inputs=[table_name_query_input],
                outputs=[table_details_output]
            )

        with gr.Tab("说明文档"):
            gr.Markdown(
                """
                ## 📖 使用说明

                本工具旨在帮助用户通过自然语言快速生成 SQL 查询语句。

                ### 工作流程
                1.  **参数配置**: 
                    - `嵌入模型路径`: 用于将表结构和用户问题转换为向量的文本嵌入模型路径。可以是 Hugging Face 模型名称（如 `BAAI/bge-m3`）或本地模型路径。
                    - `检索 Top-K`: 控制在向量数据库中检索与用户问题最相关的表结构的数量。数量越多, 提供给大模型的上下文越丰富, 但也可能引入噪音。
                    - `目标 SQL 类型`: 选择你希望生成的 SQL 方言, 如 `MYSQL`, `PostgreSQL` 等。
                    - `大模型名称`: 填入你所使用的大语言模型的名称。
                    - `API 密钥`: 填入你所使用大语言模型的 API Key。
                    - `大模型服务地址`: 填入你所使用大语言模型的服务地址（Base URL）。
                        - **保存配置**: 填入所有配置信息后, 点击 "保存配置" 按钮, 信息将被存储在本地的 `llm_config.json` 文件中, 下次启动时会自动加载。

                2.  **输入问题**: 
                    - 在 "用户问题" 文本框中, 用清晰的中文描述你的数据查询需求。

                3.  **生成 SQL**: 
                    - 点击 "生成SQL" 按钮。
                    - 程序将启动一个独立的进程来执行 `rag_query.py` 脚本, 并将所有配置参数通过命令行传递给它。
                    - `rag_query.py` 会执行以下步骤:
                        - 使用 `嵌入模型` 将你的问题向量化。
                        - 在预先构建好的表结构向量索引中, 查找最相似的 `检索 Top-K` 个表结构。
                        - 将你的问题和检索到的表结构一起发送给 `大模型`, 指示它生成相应的 SQL。

                4.  **查看结果**: 
                    - "日志和结果" 区域会实时显示 `rag_query.py` 脚本的输出, 包括检索到的表, 以及大模型最终返回的 SQL 语句。
                """
            )

    return demo


if __name__ == "__main__":
    ui = create_ui()
    # 允许局域网访问，绑定到所有网络接口，并指定端口
    ui.launch(
        server_name="0.0.0.0",
        server_port=7861
    )
