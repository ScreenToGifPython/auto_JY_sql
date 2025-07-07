# -*- encoding: utf-8 -*-
"""
@File: app.py
@Modify Time: 2025/7/7
@Author: KevinChen
@Descriptions: 本脚本使用Gradio构建了用户友好型Web界面, 以便用户可以简单的使用本框架的三大功能: 数据爬取, 信息向量化, SQL生成
"""
import gradio as gr
import subprocess
import os
import json
import re

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "gradio_config.json")


# --- 核心功能 ---

def load_config():
    """加载保存的Gradio配置"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
    # 为嵌入模型设置默认值
    if "embedding_model" not in config:
        config["embedding_model"] = "paraphrase-multilingual-MiniLM-L12-v2"
    return config


def save_config(config):
    """保存Gradio配置"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def run_script_and_stream_output(full_command):
    """
    执行一个命令并实时流式传输其输出。
    """
    log_content = ""
    yield f"▶️ 执行命令: {' '.join(full_command)}\n" + "-" * 20 + "\n"

    process = subprocess.Popen(
        full_command,
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
        yield log_content

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        log_content += f"\n--- 脚本执行出错, 返回码: {return_code} ---\n"
    else:
        log_content += "\n--- 脚本执行完毕 ---\n"

    yield log_content


# --- Gradio 界面逻辑 ---

def spider_tab_logic(table_names_str, api_key, base_url, model_name, login_wait_time):
    """表信息爬虫页面的逻辑"""
    required_fields = {
        "表名": table_names_str,
        "API Key": api_key,
        "Base URL": base_url,
        "Model Name": model_name
    }
    for field_name, value in required_fields.items():
        if not value:
            yield f"错误: {field_name} 不能为空。"
            return

    # 支持中英文逗号, 并去除空白
    table_list = [name.strip() for name in re.split('[,，]', table_names_str) if name.strip()]
    tables_arg = ",".join(table_list)

    command = [
        "python", "-u", os.path.join(SCRIPT_DIR, "table_info_spider.py"),
        "--tables", tables_arg,
        "--api_key", api_key,
        "--base_url", base_url,
        "--model_name", model_name,
        "--login_wait_time", str(login_wait_time)
    ]

    yield from run_script_and_stream_output(command)


def vectorize_tab_logic(model_name):
    """表信息矢量化页面的逻辑"""
    if not model_name:
        yield "错误: 必须选择一个模型。"
        return

    command = [
        "python", "-u", os.path.join(SCRIPT_DIR, "vectorize_tables.py"),
        "--model_name_or_path", model_name
    ]
    yield from run_script_and_stream_output(command)


def sql_gen_tab_logic(api_key, base_url, model_name, mode, rag_model, top_k, sql_type, query):
    """大模型生成SQL页面的逻辑"""
    required_fields = {
        "API Key": api_key,
        "Base URL": base_url,
        "Model Name": model_name,
        "查询内容": query
    }
    if mode == "RAG模式":
        required_fields["RAG嵌入模型"] = rag_model

    for field_name, value in required_fields.items():
        if not value:
            yield f"错误: {field_name} 不能为空。"
            return

    command = [
        "python", "-u", os.path.join(SCRIPT_DIR, "generate_sql.py"),
        "--api_key", api_key,
        "--base_url", base_url,
        "--model_name", model_name,
        "--sql_type", sql_type,
        "--query", query
    ]

    if mode == "RAG模式":
        command.extend(["--mode", "RAG", "--embedding_model", rag_model, "--top_k", str(top_k)])
    else:
        command.extend(["--mode", "FULL"])

    yield from run_script_and_stream_output(command)


# --- 构建Gradio界面 ---

def save_and_run_spider(table_names_str, api_key, base_url, model_name, login_wait_time):
    """先保存配置，然后执行爬虫脚本"""
    # 保存通用配置
    config = {
        "llm_api_key": api_key,
        "llm_base_url": base_url,
        "llm_model_name": model_name
    }
    save_config(config)

    # 执行逻辑
    yield from spider_tab_logic(table_names_str, api_key, base_url, model_name, login_wait_time)


def save_and_run_sql_gen(api_key, base_url, model_name, mode, rag_model, top_k, sql_type, query):
    """先保存配置，然后执行SQL生成脚本"""
    # 保存通用配置
    config = {
        "llm_api_key": api_key,
        "llm_base_url": base_url,
        "llm_model_name": model_name,
        "embedding_model": rag_model  # 保存RAG模式下的嵌入模型
    }
    save_config(config)

    # 执行逻辑
    yield from sql_gen_tab_logic(api_key, base_url, model_name, mode, rag_model, top_k, sql_type, query)


def save_and_run_vectorize(model_name):
    """先保存配置，然后执行矢量化脚本"""
    config = load_config()  # 加载现有配置
    config["embedding_model"] = model_name  # 更新嵌入模型
    save_config(config)

    yield from vectorize_tab_logic(model_name)


initial_config = load_config()

with gr.Blocks(title="恒生聚源 AutoSQLTools", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 恒生聚源 自动化SQL生成工具")

    with gr.Tab("表信息爬虫"):
        with gr.Row():
            with gr.Column(scale=3):
                spider_table_names = gr.Textbox(label="输入表名 (多个请用逗号隔开)",
                                                placeholder="例如: a_stock_eod_price, a_trade_cal")
                gr.Markdown("**登录设置**")
                spider_login_wait_time = gr.Number(label="登录超时时间 (秒)", value=30, minimum=10, step=10)
                gr.Markdown("**大模型参数**")
                llm_api_key_spider = gr.Textbox(label="API Key", value=initial_config.get("llm_api_key", ""),
                                                type="password")
                llm_base_url_spider = gr.Textbox(label="Base URL", value=initial_config.get("llm_base_url", ""))
                llm_model_name_spider = gr.Textbox(label="Model Name", value=initial_config.get("llm_model_name", ""))
                spider_button = gr.Button("开始爬取", variant="primary")
            with gr.Column(scale=7):
                spider_log = gr.Textbox(label="执行日志", lines=20, interactive=False, autoscroll=True)

    with gr.Tab("表信息矢量化"):
        with gr.Row():
            with gr.Column(scale=3):
                vectorize_model = gr.Dropdown(
                    label="选择嵌入模型",
                    choices=["paraphrase-multilingual-MiniLM-L12-v2"],
                    value=initial_config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"),
                    interactive=True
                )
                vectorize_button = gr.Button("开始矢量化", variant="primary")
            with gr.Column(scale=7):
                vectorize_log = gr.Textbox(label="执行日志", lines=40, interactive=False, autoscroll=True)

    with gr.Tab("大模型生成SQL"):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("**大模型参数**")
                llm_api_key_sql_gen = gr.Textbox(label="API Key", value=initial_config.get("llm_api_key", ""),
                                                 type="password")
                llm_base_url_sql_gen = gr.Textbox(label="Base URL", value=initial_config.get("llm_base_url", ""))
                llm_model_name_sql_gen = gr.Textbox(label="Model Name", value=initial_config.get("llm_model_name", ""))

                gr.Markdown("**生成设置**")
                sql_gen_mode = gr.Radio(
                    label="选择模式",
                    choices=["RAG模式", "完整上下文模式"],
                    value="RAG模式"
                )

                with gr.Group(visible=True) as rag_options:
                    sql_gen_rag_model = gr.Dropdown(
                        label="选择RAG嵌入模型",
                        choices=["paraphrase-multilingual-MiniLM-L12-v2"],
                        value=initial_config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"),
                        interactive=True
                    )
                    sql_gen_top_k = gr.Slider(label="Top-K", minimum=1, maximum=20, value=4, step=1)

                sql_gen_type = gr.Dropdown(label="选择SQL类型", choices=["MySQL", "Oracle"], value="MySQL")
                sql_gen_query = gr.Textbox(label="输入你的查询内容",
                                           placeholder="例如: '查询2023年以来所有股票的最高价'", lines=3)

                sql_gen_button = gr.Button("生成SQL", variant="primary")

            with gr.Column(scale=7):
                sql_gen_log = gr.Textbox(label="执行日志", lines=25, interactive=False, autoscroll=True)

    # --- 事件绑定 ---

    # 爬虫页面
    spider_button.click(
        fn=save_and_run_spider,
        inputs=[spider_table_names, llm_api_key_spider, llm_base_url_spider, llm_model_name_spider,
                spider_login_wait_time],
        outputs=spider_log
    )

    # 矢量化页面
    vectorize_button.click(
        fn=save_and_run_vectorize,
        inputs=[vectorize_model],
        outputs=vectorize_log
    )

    # SQL生成页面
    def toggle_rag_options(mode):
        return gr.update(visible=mode == "RAG模式")


    sql_gen_mode.change(
        fn=toggle_rag_options,
        inputs=sql_gen_mode,
        outputs=rag_options
    )

    sql_gen_button.click(
        fn=save_and_run_sql_gen,
        inputs=[llm_api_key_sql_gen, llm_base_url_sql_gen, llm_model_name_sql_gen, sql_gen_mode, sql_gen_rag_model,
                sql_gen_top_k,
                sql_gen_type, sql_gen_query],
        outputs=sql_gen_log
    )

if __name__ == "__main__":
    demo.launch()
