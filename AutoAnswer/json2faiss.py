# -*- coding: utf-8 -*-
"""
vectorize_tables.py
转换表结构 JSON → 文本块 → Embedding → FAISS 索引
author: KevinChen
"""

import os, json, argparse, uuid
from typing import Tuple, List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ---------- 将 json 转为文本块 ----------
def create_text_chunks_from_json(src_json: dict) -> Tuple[List[str], List[str]]:
    table_names, text_chunks = [], []

    for tbl, info in src_json.items():
        if not isinstance(info, dict):
            continue

        desc = info.get("description") or info.get("tableChiName", "")
        fields = info.get("fields", [])
        relations = info.get("relations", {})

        block = [f"表名: {tbl}",
                 f"表含义: {desc or 'N/A'}",
                 "字段:"]
        if fields:
            for f in fields:
                block.append(f"  - 字段名:{f.get('field_name', '')}, "
                             f"类型:{f.get('field_type', '')}, "
                             f"含义:{f.get('field_description', '')}")
        else:
            block.append("  (无字段信息)")

        if relations:
            block.append("与其他表的关联关系:")
            for tgt, conds in relations.items():
                for c in conds:
                    block.append(f"  - {c}")
        else:
            block.append("与其他表的关联关系: (无关联信息)")

        table_names.append(tbl)
        text_chunks.append("\n".join(block))

    print(f"✅ 生成 {len(text_chunks)} 个文本块")
    return table_names, text_chunks


# ---------- FAISS 索引构建 ----------
def build_flat(vecs: np.ndarray) -> faiss.Index:
    idx = faiss.IndexFlatL2(vecs.shape[1]);
    idx.add(vecs);
    return idx


def build_hnsw(vecs: np.ndarray, M=32, efC=200, efS=64) -> faiss.Index:
    dim = vecs.shape[1]
    idx = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    idx.hnsw.efConstruction, idx.hnsw.efSearch = efC, efS
    idx.add(vecs);
    return idx


def build_ivfpq(vecs: np.ndarray, nlist=4096, M=16, nbits=8) -> faiss.Index:
    dim = vecs.shape[1];
    quant = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFPQ(quant, dim, nlist, M, nbits)
    idx.train(vecs);
    idx.nprobe = 16;
    idx.add(vecs);
    return idx


# ---------- 主流程 ----------
def main(args):
    # 1. 读取 JSON
    with open(args.json, 'r', encoding='utf-8') as f:
        table_defs = json.load(f)

    # 2. 文本块
    table_names, chunks = create_text_chunks_from_json(table_defs)
    if not chunks:
        print("❗ 无文本块生成，退出");
        return

    # 3. 加载模型
    print("加载模型…")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    model.max_seq_length = args.max_len  # 避免超长 OOM

    # 4. 生成向量
    print("编码向量…")
    vecs = model.encode(
        chunks,
        batch_size=args.batch,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # 5. 构建索引
    if args.type == "flat":
        index = build_flat(vecs)
    elif args.type == "hnsw":
        index = build_hnsw(vecs)
    elif args.type == "ivfpq":
        index = build_ivfpq(vecs)
    else:
        raise ValueError("index_type 仅支持 flat / hnsw / ivfpq")

    faiss.write_index(index, args.index)
    print(f"✅ FAISS 索引保存 → {args.index}")

    # 6. 保存完整文本块（表结构描述）
    with open(args.map, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ 表结构描述保存 → {args.map}")
    os._exit(0)


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="表结构向量化并写入 FAISS")
    ap.add_argument("--model", default="BAAI/bge-m3", help="SentenceTransformer 模型或本地路径")
    ap.add_argument("--json", default="table.json", help="输入表结构 JSON")
    ap.add_argument("--index", default="faiss_index.bin", help="输出索引文件")
    ap.add_argument("--map", default="table_mapping.json", help="输出表名映射")
    ap.add_argument("--type", default="hnsw", choices=["flat", "hnsw", "ivfpq"], help="索引类型")
    ap.add_argument("--batch", type=int, default=16, help="encode 批大小")
    ap.add_argument("--max_len", type=int, default=512, help="模型 max_seq_length")
    args = ap.parse_args()

    main(args)
