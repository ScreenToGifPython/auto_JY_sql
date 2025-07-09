# -*- encoding: utf-8 -*-
"""
@File: merge_table_json.py
@Modify Time: 2025/7/9 08:07       
@Author: Kevin-Chen
@Descriptions: 
"""
import json
import argparse
from pathlib import Path
import re
import json
import argparse
from pathlib import Path
from textwrap import indent

NULL_DESC = "（无描述）"
NULL_FIELD = "（无字段信息）"
NULL_REL = "（无关联信息）"


def parse_eq(eq: str) -> str:
    """
    将 'A.FIELD = B.FIELD' 转成中文句子：
    → 通过字段 FIELD 与 B 表的 FIELD 字段关联
    """
    match = re.match(r"\s*([\w$]+)\.([\w$]+)\s*=\s*([\w$]+)\.([\w$]+)", eq, re.I)
    if not match:
        return eq  # 原样返回
    left_tbl, left_col, right_tbl, right_col = match.groups()
    if left_tbl == right_tbl:  # 自连接
        return f"表 {left_tbl} 的 {left_col} 字段与同表的 {right_col} 字段关联"
    return f"通过字段 {left_col} 与 {right_tbl} 表的 {right_col} 字段关联"


def build_markdown(table_name: str, meta: dict) -> str:
    md_lines = [f"### {table_name}"]

    # 描述
    desc = meta.get("description") or NULL_DESC
    md_lines.append(f"**描述**：{desc}")

    # 字段
    fields = meta.get("fields", [])
    if fields:
        md_lines.append("**字段**")
        for f in fields:
            name = f.get("field_name", "")
            dtype = f.get("field_type", "")
            null = "NOT NULL" if f.get("is_nullable", "Y") == "N" else "NULL"
            comment = f.get("field_description", "")
            md_lines.append(f"- `{name}` ({dtype}, {null}) – {comment}")
    else:
        md_lines.append(f"**字段**：{NULL_FIELD}")

    # 关系
    relations = meta.get("relations", {})
    if relations:
        md_lines.append("**关联**")
        for target_tbl, conds in relations.items():
            for cond in conds:
                md_lines.append(f"- 与 `{target_tbl}` 表 {parse_eq(cond)}")
    else:
        md_lines.append(f"**关联**：{NULL_REL}")

    return "\n".join(md_lines) + "\n\n"


def json_to_markdown(json_path, md_path):
    json_path = Path(json_path)
    md_path = Path(md_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    md_chunks = [build_markdown(tbl, meta) for tbl, meta in data.items()]
    md_path.write_text("".join(md_chunks), encoding="utf-8")
    print(f"✅ 已生成 Markdown：{md_path}")


def merge_table_info_and_relations(info_path, relation_path, output_path):
    # 读取两个 JSON 文件
    with open(info_path, 'r', encoding='utf-8') as f:
        table_info = json.load(f)

    with open(relation_path, 'r', encoding='utf-8') as f:
        table_relations = json.load(f)

    # 合并逻辑
    merged_data = {}
    for table_name, info in table_info.items():
        merged_data[table_name] = {
            "description": info.get("description", ""),
            "fields": info.get("fields", []),
            "relations": table_relations.get(table_name, {})
        }

    # 对 relation 中存在但 info 中不存在的补充空字段结构
    for table_name, rel in table_relations.items():
        if table_name not in merged_data:
            merged_data[table_name] = {
                "description": "",
                "fields": [],
                "relations": rel
            }

    # 保存为 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 合并完成，结果已保存到: {output_path}")


if __name__ == "__main__":
    # merge_table_info_and_relations("/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer/table_info.json",
    #                                "/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer/table_relation.json",
    #                                "/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer/table.json"
    #                                )

    json_to_markdown("/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer/table.json",
                     "/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer/table.md"
                     )
