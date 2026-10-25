# -*- coding: utf-8 -*-
"""
文档解析与切片模块。
支持 PDF / TXT / MD，按字符切分（chunk_size=500, chunk_overlap=100）。
"""
import io
from typing import List, Dict

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {"pdf", "txt", "md"}


def parse_document(file_bytes: bytes, filename: str) -> str:
    """
    根据文件扩展名把上传的字节流解析为纯文本。

    :param file_bytes: 上传文件的原始字节
    :param filename:   文件名（用于判断扩展名）
    :return: 解析后的纯文本字符串
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        # pypdf 接收类文件对象，用 BytesIO 包装字节流
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text)

    if ext in ("txt", "md"):
        # 优先 utf-8，失败回退 gbk（兼容 Windows 中文 txt）
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("gbk", errors="ignore")

    raise ValueError(f"不支持的文件格式：{ext}，仅支持 {sorted(SUPPORTED_EXTENSIONS)}")


def split_text(text: str, filename: str) -> List[Dict]:
    """
    按字符切分文本，返回带元数据的 chunk 列表。

    固定参数：chunk_size=500, chunk_overlap=100。
    每个 chunk 的元数据包含：filename、chunk_index、doc_id（全局唯一，用于向量/BM25 两路对齐）。

    :param text:     文档纯文本
    :param filename: 文件名
    :return: [{"content": str, "metadata": {...}}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 按字符切分：优先在换行/标点/空格处断句，都没有时按单字切
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " ", ""],
    )
    pieces = splitter.split_text(text)

    chunks: List[Dict] = []
    for idx, content in enumerate(pieces):
        chunks.append({
            "content": content,
            "metadata": {
                "filename": filename,
                "chunk_index": idx,
                # 用 文件名::序号 作为全局唯一 id，供 RRF 对齐两路召回结果
                "doc_id": f"{filename}::{idx}",
            },
        })
    return chunks


def process_uploaded_file(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    上传文件处理入口：解析 -> 切片。
    """
    text = parse_document(file_bytes, filename)
    return split_text(text, filename)
