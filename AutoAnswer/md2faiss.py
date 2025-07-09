# -*- encoding: utf-8 -*-
"""
@File: md2faiss.py
@Modify Time: 2025/7/9 08:26       
@Author: Kevin-Chen
@Descriptions: Markdown → chunk → bge-m3 embedding → 存 FAISS (本地磁盘)
"""
import re
import json
import glob
import uuid
from pathlib import Path
from tqdm import tqdm
from markdown_it import MarkdownIt

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --------- 超参数 ---------
CHUNK_SIZE = 350  # token 左右
CHUNK_OVERLAP = 50
EMB_MODEL = "BAAI/bge-m3"
DEVICE = "cpu"  # "cpu" / "cuda"
# --------------------------

encoder = SentenceTransformer(EMB_MODEL, device=DEVICE)


def md_to_plain(md_text: str) -> str:
    """
    把 markdown 渲染成纯文本（去掉格式符号）。
    """
    md = MarkdownIt()
    tokens = md.parse(md_text)
    return "\n".join(t.content for t in tokens if t.type == "inline")


def split_markdown(md_text: str, chunk=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    简单按行分块，然后滑窗。
    """
    lines = md_text.splitlines()
    chunks = []
    cur = []
    cur_len = 0
    for ln in lines:
        if ln.strip() == "":  # 空行 → 自然分段
            ln = "\n"
        cur.append(ln)
        cur_len += len(ln.split())  # 粗略用单词数做 token 近似
        if cur_len >= chunk:
            chunks.append("\n".join(cur))
            # overlap 保留结尾
            cur = cur[-overlap:]
            cur_len = sum(len(l.split()) for l in cur)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def embed_texts(text_list):
    return encoder.encode(text_list, normalize_embeddings=True, convert_to_numpy=True)


def build_faiss_index(dim):
    index = faiss.IndexHNSWFlat(dim, 32)  # HNSW, ef=32
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = 64
    return index


def main(single_md_path, index_path, meta_path):
    print(f"开始向量化：{single_md_path} → {index_path}，元数据 {meta_path}")
    md_text = Path(single_md_path).read_text(encoding="utf-8")
    print(f"读取 Markdown 文件成功：{single_md_path}，大小 {len(md_text)} 字符")
    plain = md_to_plain(md_text)

    # ① 先按 '### 表名' 进行粗切：一张表 ≈ 一段
    big_chunks = plain.split("\n### ")
    big_chunks = ["### " + c if i > 0 else c for i, c in enumerate(big_chunks)]

    all_vectors, all_meta = [], []
    for chunk in tqdm(big_chunks, desc="表级切分"):
        # 再进行 token 级滑窗细切
        sub_chunks = split_markdown(chunk)

        vecs = embed_texts(sub_chunks)
        all_vectors.append(vecs)

        for txt in sub_chunks:
            all_meta.append({
                "id": str(uuid.uuid4()),
                "table": re.search(r"###\s+(\S+)", chunk).group(1),
                "text": txt[:200]
            })

    all_vectors = np.vstack(all_vectors).astype("float32")
    index = build_faiss_index(all_vectors.shape[1])
    index.add(all_vectors)

    faiss.write_index(index, index_path)
    Path(meta_path).write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 完成向量化：{len(all_meta)} 个 chunk → {index_path}")


if __name__ == "__main__":
    main("/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer",
         "/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer/index.faiss",
         "/Users/chenjunming/Desktop/auto_JY_sql/AutoAnswer/meta.json"
         )
