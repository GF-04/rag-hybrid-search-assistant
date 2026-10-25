# -*- coding: utf-8 -*-
"""
检索模块：BM25 稀疏索引 + RRF 融合算法。
- BM25Index：纯内存索引，不支持持久化；每次上传追加 chunk 后整体重建。
- rrf_fusion：独立的 RRF 融合函数，k 固定取 60。
"""
from typing import List, Dict
from collections import Counter

import re

import jieba
from rank_bm25 import BM25Okapi


# 标识符正则：连续的字母/数字/下划线/点/横杠作为单个 token，
# 保证 ERR_404_A、V2.3.1-hotfix、HNSW-M16 等不被 jieba 拆散。
_IDENT_RE = re.compile(r"[A-Za-z0-9_./-]+")

# 标识符命中加分：每出现一次标识符 token 追加的分数。
# 目的是压过 BM25 长度归一化带来的偏差 —— 手册概述段虽然"什么错误码都提到一次"，
# 但真正讲某个错误码的章节会出现多次，加分后正确 chunk 能排到第一。
IDENTIFIER_BOOST = 1.2


def _is_identifier(tok: str) -> bool:
    """
    判断是否为"硬标识符"token：纯 ASCII 且含数字或分隔符（_ - . /）。
    例：ERR_404_A、V2.3.1-hotfix、HNSW-M16、404 判定为标识符；
        "怎么"、"处理"、"efSearch"（纯字母无数字）不判别为标识符。
    """
    if not tok or not _IDENT_RE.fullmatch(tok):
        return False
    return bool(re.search(r"[0-9_./-]", tok))


def tokenize(text: str) -> List[str]:
    """
    中文 + 标识符混合分词，用于 BM25 构建与查询。
    - 中文片段走 jieba 精确模式；
    - 字母/数字/下划线/点/横杠的连续串（如 ERR_404_A、V2.3.1-hotfix、HNSW-M16）
      作为单个 token 保留，确保 BM25 能精确匹配标识符。
    """
    tokens: List[str] = []
    # 按"中文块"和"标识符块"切分
    parts = re.split(r"([\u4e00-\u9fff]+)", text)
    for part in parts:
        if not part.strip():
            continue
        if re.match(r"[\u4e00-\u9fff]", part):
            tokens.extend(tok for tok in jieba.cut(part) if tok.strip())
        else:
            tokens.extend(tok for tok in _IDENT_RE.findall(part) if tok.strip())
    return tokens


class BM25Index:
    """
    纯内存 BM25 稀疏索引。

    重要限制（必须写入 README）：
    - 索引完全驻留内存，程序重启后丢失；
    - 不支持增量更新，每次 add_chunks 后基于全量 chunk 重建 BM25Okapi 对象；
    - 因此"多文件增量构建"指的是 chunk 池只追加不覆盖，而 BM25 对象每次重建。
    """

    def __init__(self):
        self.chunks: List[Dict] = []      # 累积的全量 chunk
        self.bm25: BM25Okapi | None = None
        self._token_counts: List[Counter] = []  # 每个 chunk 的 token 计数，供标识符加分使用
        self._rebuild()

    def add_chunks(self, new_chunks: List[Dict]) -> None:
        """追加新 chunk 并重建索引；已存在相同 doc_id 的 chunk 会被跳过（去重）。"""
        existing_ids = {c["metadata"].get("doc_id") for c in self.chunks}
        fresh = [
            c for c in new_chunks
            if c["metadata"].get("doc_id") not in existing_ids
        ]
        if not fresh:
            return
        self.chunks.extend(fresh)
        self._rebuild()

    def _rebuild(self) -> None:
        """基于当前全量 chunk 重建 BM25Okapi，并缓存每个 chunk 的 token 计数。"""
        if not self.chunks:
            self.bm25 = None
            self._token_counts = []
            return
        corpus = [tokenize(c["content"]) for c in self.chunks]
        self._token_counts = [Counter(toks) for toks in corpus]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        BM25 检索，返回按分数降序的 top_k 结果。

        查询里含"硬标识符"（如 ERR_404_A / V2.3.1-hotfix）时，对命中该标识符的 chunk
        按出现次数追加 IDENTIFIER_BOOST 分，避免概述类短 chunk 靠长度归一化抢占第一名。

        :return: [{"content", "metadata", "rank", "bm25_score"}, ...]
                 rank 从 1 开始，用于 RRF 融合时的原始名次标注。
        """
        if not self.bm25:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # 标识符加权：统计每个 chunk 中命中的标识符出现次数
        id_tokens = {t for t in tokens if _is_identifier(t)}
        if id_tokens:
            for i, counter in enumerate(self._token_counts):
                hits = sum(counter.get(t, 0) for t in id_tokens)
                if hits:
                    scores[i] += IDENTIFIER_BOOST * hits

        # 按分数降序取前 top_k
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results: List[Dict] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "rank": rank,
                "bm25_score": float(scores[idx]),
            })
        return results

    @property
    def size(self) -> int:
        return len(self.chunks)


def rrf_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
    top_k: int = 5,
) -> List[Dict]:
    """
    RRF（Reciprocal Rank Fusion，倒数排名融合）算法。

    公式： score(d) = Σ  1 / (k + rank_i(d))
    - k：平滑常数，本项目固定取 60；
    - rank_i(d)：文档 d 在第 i 路召回结果中的排名（从 1 开始）；
    - 对每一路召回中的文档，按其排名累加 1/(k+rank)，未命中则不计分；
    - 最终按融合总分降序，取 top_k。

    RRF 的优点：无需对两路分数做归一化，仅依赖排名，实现简单且效果稳定。

    :param vector_results: 向量召回结果，每项含 content/metadata/rank
    :param bm25_results:   BM25 召回结果，每项含 content/metadata/rank
    :param k:              RRF 常数，默认 60
    :param top_k:          最终返回数量
    :return: 融合结果列表，每项额外包含：
             - vector_rank：在向量召回中的原始名次（None 表示未命中）
             - bm25_rank：  在 BM25 召回中的原始名次（None 表示未命中）
             - source_tag： 【仅向量】/【仅BM25】/【两路都命中】
             - score：      融合总分
    """
    fused: Dict[str, Dict] = {}

    def _ensure(doc_id: str, item: Dict) -> Dict:
        if doc_id not in fused:
            fused[doc_id] = {
                "content": item["content"],
                "metadata": item["metadata"],
                "vector_rank": None,
                "bm25_rank": None,
                "score": 0.0,
            }
        return fused[doc_id]

    # 第一路：向量召回
    for item in vector_results:
        doc_id = item["metadata"]["doc_id"]
        entry = _ensure(doc_id, item)
        entry["vector_rank"] = item["rank"]
        entry["score"] += 1.0 / (k + item["rank"])

    # 第二路：BM25 召回
    for item in bm25_results:
        doc_id = item["metadata"]["doc_id"]
        entry = _ensure(doc_id, item)
        entry["bm25_rank"] = item["rank"]
        entry["score"] += 1.0 / (k + item["rank"])

    # 打标：根据两路命中情况标记来源
    for entry in fused.values():
        if entry["vector_rank"] is not None and entry["bm25_rank"] is not None:
            entry["source_tag"] = "两路都命中"
        elif entry["vector_rank"] is not None:
            entry["source_tag"] = "仅向量"
        else:
            entry["source_tag"] = "仅BM25"

    # 按融合总分降序，取 top_k
    sorted_items = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return sorted_items[:top_k]
