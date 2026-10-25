# -*- coding: utf-8 -*-
"""
全局配置与常量。
System Prompt、模型参数、切片参数、检索参数均集中在此，业务代码不硬编码。
"""
import os
from dotenv import load_dotenv

# 加载 .env 中的 API_KEY
load_dotenv()

# ===================== 模型配置（通义千问 OpenAI 兼容接口）=====================
# 通义千问 DashScope 的 OpenAI 兼容模式地址
OPENAI_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 从 .env 读取 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 对话模型
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen-plus")
# Embedding 模型（通义千问兼容模式支持的向量模型）
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v3")

# ===================== 切片固定参数（按字符切分）=====================
CHUNK_SIZE = 500        # 每个 chunk 的最大字符数
CHUNK_OVERLAP = 100     # 相邻 chunk 的重叠字符数

# ===================== 检索参数 =====================
VECTOR_TOP_K = 5          # 模式A：纯向量召回数量
HYBRID_CANDIDATE_K = 10   # 模式B：向量 / BM25 各自召回候选数量
RRF_K = 60                # RRF 融合算法的常数 k（固定取 60）
FINAL_TOP_K = 5           # 混合检索最终返回数量

# ===================== 存储路径 =====================
CHROMA_PERSIST_DIR = "./chroma_db"   # Chroma 向量库持久化目录

# ===================== System Prompt（文字严格按需求）=====================
# 注意：该常量单独抽出，不埋在业务代码里；文字与需求保持一致。
SYSTEM_PROMPT = (
    "仅基于下面给出的上下文回答。如果上下文没有对应信息，"
    "只输出固定话术：文档中未查询到相关内容，严禁编造外部知识，禁止额外发挥"
)
