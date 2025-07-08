# -*- encoding: utf-8 -*-
"""
@File: extract_pck_lineage.py
@Modify Time: 2025/7/8 11:39
@Author: Kevin-Chen
@Descriptions: 提取 Oracle PL/SQL .pck 文件中的 Procedure 逻辑，生成
{目标表: {sql: "...", description: "...", procedure: "..."} } 的 JSON。
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional


def read_text(path: Path) -> str:
    "读取文件文本（自动使用 utf-8 或系统默认编码回退）"
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text()


# ------- 需要忽略的日志表 -------
LOG_TABLES = {"procedure_log"}

# ---------- 包头里的注释（描述） ----------
HEAD_DESC_PATTERN = re.compile(
    r"--\s*(?P<comment>[^\n]+)\s*\n\s*procedure\s+(?P<proc>\w+)\s*\(",
    re.IGNORECASE,
)

# ---------- 包体里的 procedure ----------
BODY_PROC_PATTERN = re.compile(
    r"""
    procedure\s+(?P<name>\w+)\s*\([^\)]*\)\s*(?:is|as)   # 头
    (?P<body>.*?)                                        # 主体
    \bend\s*;                                            # end;
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

INSERT_SELECT_PATTERN = re.compile(
    r"""
    insert\s+into\s+\w+\s*\([^)]*\)\s*          # insert 目标表 (…)
    select\s+.*?;                               # select … ;
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

# ---------- regex: 下游(target) & 上游(source) ---------- #
TARGET_PATTERN = re.compile(
    r"""
    (?!execute\s+immediate)      # 排除动态 SQL
    (?:
        insert\s+into\s+(?P<tbl_ins>[a-zA-Z0-9_$.]+) |
        merge\s+into\s+(?P<tbl_merge>[a-zA-Z0-9_$.]+) |
        delete\s+from\s+(?P<tbl_del>[a-zA-Z0-9_$.]+) |
        update\s+(?P<tbl_upd>[a-zA-Z0-9_$.]+)\s+set
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

SOURCE_PATTERN = re.compile(
    r"(?:from|join|using)\s+(?P<table>[a-zA-Z0-9_$.]+)", re.IGNORECASE
)

# ------- 匹配整条针对 procedure_log 的 DML，用于删除 -------
LOG_DML_PATTERN = re.compile(
    r"""
    (?:insert\s+into|merge\s+into|update|delete\s+from)\s+procedure_log\b.*?;
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def clean_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return name.strip(" ,\n\t()").lower()


def nearest_comment_block(text: str, idx: int) -> str:
    buf: List[str] = []
    for ln in reversed(text[:idx].splitlines()):
        s = ln.strip()
        if s.startswith("--"):
            buf.append(s.lstrip("-").strip())
        elif s == "":
            continue
        else:
            break
    return "；".join(reversed(buf)).strip()


def parse_pck(content: str) -> Dict[str, Dict]:
    head_desc = {m.group("proc").lower(): m.group("comment").strip()
                 for m in HEAD_DESC_PATTERN.finditer(content)}
    res = {}
    for m in BODY_PROC_PATTERN.finditer(content):
        proc = m.group("name").lower()
        full_sql = LOG_DML_PATTERN.sub("", m.group(0).strip())  # 去掉日志

        # ---- 捕获 target 表 ----
        targets: Set[str] = set()
        for mt in TARGET_PATTERN.finditer(full_sql):
            tbl = (
                    mt.group("tbl_ins") or mt.group("tbl_merge") or
                    mt.group("tbl_del") or mt.group("tbl_upd")
            )
            tbl = clean_name(tbl)
            if tbl and tbl not in LOG_TABLES:
                targets.add(tbl)

        # ---- 捕获 source 表 ----
        sources = {clean_name(s.group("table")) for s in SOURCE_PATTERN.finditer(full_sql)}

        res[proc] = {
            "description": head_desc.get(proc) or nearest_comment_block(content, m.start()) or "",
            "sql": full_sql,
            "sources": sorted(s for s in sources if s),
            "targets": sorted(targets),
        }
    return res


# ----------------------------  CLI 入口  ---------------------------- #
def main(pck_path: str, out_path: str):
    pck_path = Path(pck_path).expanduser()
    out_path = Path(out_path).expanduser()

    # 读取文件内容（字符串）
    content = read_text(pck_path)

    # 解析字符串内容，返回 lineage 字典
    lineage = parse_pck(content)

    # 写出 JSON
    out_path.write_text(json.dumps(lineage, indent=2, ensure_ascii=False))
    print(f"解析完成，结果已写入: {out_path}")


if __name__ == "__main__":
    main(
        "/Users/chenjunming/Desktop/auto_JY_sql/pkg/pkg_dw_transdata.pck",
        "/Users/chenjunming/Desktop/auto_JY_sql/lineage.json",
    )
