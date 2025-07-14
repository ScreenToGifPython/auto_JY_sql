# -*- encoding: utf-8 -*-
"""
@File: llm_utils.py
@Modify Time: 2025/7/14
@Author: KevinChen
@Descriptions: 通用大模型调用工具函数
"""
import traceback
import json
import re
import requests
from openai import OpenAI


def call_llm(prompt: str,
             sys_prompt: str,
             api_key: str,
             base_url: str,
             model: str,
             temperature: float = 0.1):
    """
    通用大模型调用函数, 支持 OpenAI 和兼容的 API (如 lightcode-ui).

    :param prompt: 用户输入的主要提示
    :param system_prompt: 系统提示
    :param api_key: API Key
    :param base_url: API Base URL
    :param model_name: 模型名称
    :param temperature: 温度参数
    :param stream: 是否流式输出
    :param json_output: 是否要求返回JSON对象
    :param kwargs: 其他传递给 acreate 的参数
    :return: 模型返回的文本内容, 或在 stream=True 时返回 stream 对象, 或在出错时返回 None
    """
    if not all([model, api_key, base_url]):
        raise KeyError("Missing one or more required keys (LLM_MODEL, API_KEY, LLM_URL) in config.json")

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    if 'lightcode-uis' in base_url:
        # 通过lightcode-ui接口调用大模型API
        print("Using lightcode-ui API...")
        headers = {'Accept': "*/*", 'Authorization': f"Bearer {api_key}", 'Content-Type': "application/json"}
        payload = {"model": model, "messages": messages, "stream": False}
        resp = None
        try:
            resp = requests.post(base_url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()  # 检查HTTP响应状态码
            response_data = resp.json()
            # 解析 返回结果
            json_inner = response_data["choices"][0]["message"]["content"]
            return re.sub(r"^```json\n|```$", "", json_inner.strip(), flags=re.MULTILINE)
        except requests.exceptions.RequestException as e:
            raise Exception(f"lightcode-ui API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected response format from lightcode-ui API: {e} Response: {resp.text}")

    else:
        # 通过OpenAI接口调用大模型API
        print("Using OpenAI API...")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
            llm_response_content = resp.choices[0].message.content
            # 移除Markdown代码块标记
            if llm_response_content.startswith("```json") and llm_response_content.endswith("```"):
                llm_response_content = llm_response_content[len("```json"): -len("```")].strip()
            return llm_response_content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")


if __name__ == '__main__':
    print((call_llm(prompt="请生成一个10个单词的英文句子",
                    sys_prompt="你是一个 generates English sentences with 10 words. ",
                    api_key="eyJhbGciOiJIUzI1NiJ9",
                    base_url="http://lightcode-uis.hundsun.com:8080/uis/v1/chat/completions",
                    model="gpt-4o",
                    temperature=0.1,
                    )))
