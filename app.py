# -*- coding: utf-8 -*-
"""
Streamlit RAG 知识库问答助手（应届生简历 Demo）。

功能：
- 文档上传（PDF/TXT/MD），同时构建 Chroma 稠密向量索引 + BM25 稀疏索引；
- 检索模式切换：纯向量（Top5） / 混合检索（向量+BM25 各 Top10，RRF 融合取 Top5）；
- 左右双栏：左聊天问答（多轮历史），右召回预览；
- 幻觉控制：System Prompt 单独抽出，仅基于上下文回答。
"""
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from config import (
    DASHSCOPE_API_KEY,
    OPENAI_COMPATIBLE_BASE_URL,
    CHAT_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    SYSTEM_PROMPT,
    VECTOR_TOP_K,
    HYBRID_CANDIDATE_K,
    RRF_K,
    FINAL_TOP_K,
)
from document_processor import process_uploaded_file, SUPPORTED_EXTENSIONS
from retrievers import BM25Index, rrf_fusion


# =====================================================================
# 页面基础配置
# =====================================================================
st.set_page_config(page_title="RAG 知识库问答助手", page_icon="📚", layout="wide")

# 左右两栏各自独立滚动，高度固定，避免内容把整页无限拉长
CHAT_AREA_HEIGHT = 430      # 左侧消息区高度（px）
PREVIEW_AREA_HEIGHT = 520   # 右侧召回预览区高度（px，内部再分三栏）
PREVIEW_SNIPPET_CHARS = 26  # 折叠标题里展示的片段长度（三栏较窄，故取短）
PREVIEW_BODY_HEIGHT = 150   # 折叠展开后正文的滚动区高度（px）

st.markdown(
    """
    <style>
      /* 页面留白收紧，让双栏尽量在首屏内完整呈现 */
      .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 1600px; }
      footer { visibility: hidden; }

      /* ---- 召回条目：卡片化折叠样式，头部更紧凑 ---- */
      [data-testid="stExpander"] {
        border: 1px solid #e5eaf2; border-radius: 10px;
        background: #ffffff; margin-bottom: 2px; overflow: hidden;
      }
      [data-testid="stExpander"] summary {
        padding: 0.3rem 0.5rem !important; min-height: 0 !important;
      }
      [data-testid="stExpander"] summary p {
        font-size: 0.76rem !important; margin: 0 !important;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      [data-testid="stExpander"] summary:hover { background: rgba(37,99,235,0.05); }
      [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0.2rem 0.5rem 0.35rem !important;
      }
      /* 折叠区内的正文：小字号、行距收紧 */
      [data-testid="stExpander"] .stMarkdown p { font-size: 0.78rem; line-height: 1.5; }

      /* ---- 底部输入条：描边留白，视觉上更聚焦 ---- */
      [data-testid="stBottomBlockContainer"] { padding-bottom: 0.9rem; }

      /* ---- 强制横向排列：防止窄视口下 Streamlit 把 columns 压成竖排 ---- */
      [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
      @media (max-width: 1100px) {
        [data-testid="stHorizontalBlock"] { flex-direction: row !important; }
        [data-testid="stColumn"] { width: auto !important; min-width: 0 !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Session State 初始化（只执行一次）
# =====================================================================
def init_state() -> None:
    """初始化向量库、BM25 索引、LLM、对话历史与统计计数。"""
    if "embeddings" not in st.session_state:
        # 通义千问 DashScope 兼容端点只接受 str / List[str]，
        # 而 LangChain 默认 check_embedding_ctx_length=True 会用 tiktoken 分词后
        # 把 token-id（int 列表）作为 input 发送，触发 400 InvalidParameter。
        # 本项目 chunk 仅 500 字符，关闭上下文长度检查是安全的。
        st.session_state.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            check_embedding_ctx_length=False,
        )

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=st.session_state.embeddings,
        )

    if "bm25_index" not in st.session_state:
        # BM25 为纯内存索引，每次启动为空；仅在本次会话内上传后才会构建
        st.session_state.bm25_index = BM25Index()

    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            base_url=OPENAI_COMPATIBLE_BASE_URL,
            temperature=0,  # 关闭随机性，抑制幻觉
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_preview" not in st.session_state:
        # 保存最近一次检索的预览数据，供右栏渲染
        # {"mode": str, "vector": [...], "bm25": [...], "rrf": [...]}
        st.session_state.last_preview = None

    # 统计信息：从 Chroma 持久化数据中推导（BM25 重启丢失，但向量库持久化）
    if "doc_count" not in st.session_state or "chunk_count" not in st.session_state:
        _refresh_stats_from_chroma()


def _refresh_stats_from_chroma() -> None:
    """从 Chroma 向量库重新计算文档总数与 chunk 总数。"""
    try:
        collection = st.session_state.vector_store._collection
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        filenames = {m.get("filename") for m in metadatas if m}
        st.session_state.doc_count = len(filenames)
        st.session_state.chunk_count = len(data.get("ids") or [])
    except Exception:
        st.session_state.doc_count = 0
        st.session_state.chunk_count = 0


def _rebuild_bm25_from_chroma() -> None:
    """
    从 Chroma 的持久化数据重建 BM25 内存索引。

    BM25 本身不持久化，页面刷新或服务重启都会清空；而 Chroma 的 chunk 与写入
    BM25 的 chunk 同源（同样含 filename / chunk_index / doc_id），因此可直接读回重建，
    免去每次刷新后手动重传文档。已有索引时直接跳过，避免每次 rerun 都重建。
    """
    if st.session_state.bm25_index.size:
        return
    try:
        collection = st.session_state.vector_store._collection
        data = collection.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        chunks = [
            {"content": doc, "metadata": meta or {}}
            for doc, meta in zip(documents, metadatas)
            if doc
        ]
        if chunks:
            st.session_state.bm25_index.add_chunks(chunks)
    except Exception:
        # 重建失败不阻塞主流程：混合检索会退化为"BM25 无结果"，仍可正常问答
        pass


init_state()
# 刷新页面 / 重启服务后自动从 Chroma 恢复 BM25 索引，无需手动重传文档
_rebuild_bm25_from_chroma()


# =====================================================================
# 工具函数
# =====================================================================
def chroma_to_result_docs(docs) -> list:
    """把 Chroma 返回的 Document 列表转成统一的 dict 结构（含 rank）。"""
    results = []
    for rank, doc in enumerate(docs, start=1):
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "rank": rank,
        })
    return results


def _snippet(text: str, n: int) -> str:
    """把 chunk 正文压成一行短摘要，用于折叠标题。"""
    return (text or "").replace("\n", " ").strip()[:n]


def render_preview_item(item: dict, order: int, show_rank_meta: bool = False) -> None:
    """
    渲染单条召回结果：
    - 折叠标题只放名次 + chunk 序号 + 短摘要，保证单行不换行；
    - 展开后正文放进固定高度的滚动容器，长文档不会把折叠区撑高。
    show_rank_meta=True 时折叠标题改为「#名次 · 向量#x / BM25#y · 命中来源」，
    让 RRF 融合依据无需展开即可见（仅 RRF 融合结果需要）。
    """
    meta = item.get("metadata") or {}
    filename = meta.get("filename", "?")
    chunk_index = meta.get("chunk_index", "?")

    if show_rank_meta:
        # RRF 结果把两路原始名次直接写进折叠标题：不展开也能看出融合依据，
        # 也不放过窄栏时被省略号截断，因此不再拼接正文摘要。
        vr = item.get("vector_rank")
        br = item.get("bm25_rank")
        vec_txt = f"向量#{vr}" if vr is not None else "向量×"
        bm25_txt = f"BM25#{br}" if br is not None else "BM25×"
        label = f"#{order} · {vec_txt} / {bm25_txt} · {item.get('source_tag', '')}"
    else:
        headline = _snippet(item.get("content"), PREVIEW_SNIPPET_CHARS)
        label = f"#{order} · chunk {chunk_index} · {headline}"

    with st.expander(label, expanded=False):
        st.caption(f"📄 {filename} · chunk {chunk_index}")

        with st.container(height=PREVIEW_BODY_HEIGHT, border=False):
            st.markdown(item.get("content") or "")


def build_context_text(chunks: list) -> str:
    """把召回的 chunk 拼成给 LLM 的上下文字符串。"""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[片段{i}] {c['content']}")
    return "\n\n".join(parts)


def retrieve(mode: str, query: str):
    """
    根据模式执行检索，返回 (用于回答的chunks, 预览数据)。

    模式A 纯向量：Chroma 召回 Top5
    模式B 混合：  向量 Top10 + BM25 Top10 -> RRF(k=60) -> Top5
    """
    vector_store = st.session_state.vector_store
    bm25_index = st.session_state.bm25_index

    if mode == "A":
        docs = vector_store.similarity_search(query, k=VECTOR_TOP_K)
        vector_results = chroma_to_result_docs(docs)
        answer_chunks = vector_results
        preview = {"mode": "A", "vector": vector_results, "bm25": [], "rrf": []}
        return answer_chunks, preview

    # 模式B：混合检索
    vec_docs = vector_store.similarity_search(query, k=HYBRID_CANDIDATE_K)
    vector_results = chroma_to_result_docs(vec_docs)
    bm25_results = bm25_index.search(query, top_k=HYBRID_CANDIDATE_K)
    rrf_results = rrf_fusion(vector_results, bm25_results, k=RRF_K, top_k=FINAL_TOP_K)

    answer_chunks = rrf_results
    preview = {
        "mode": "B",
        "vector": vector_results,
        "bm25": bm25_results,
        "rrf": rrf_results,
    }
    return answer_chunks, preview


def generate_answer(question: str, context_chunks: list) -> str:
    """
    调用 LLM 生成回答。
    - System Prompt 使用全局常量 SYSTEM_PROMPT；
    - 保留多轮对话历史；
    - 把当前问题的检索上下文拼到最新一条 HumanMessage 中。
    """
    llm = st.session_state.llm
    context_text = build_context_text(context_chunks)

    # 构造消息序列：System + 历史 + 当前(问题+上下文)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # 当前问题与上下文
    current_input = f"上下文：\n{context_text}\n\n问题：{question}"
    messages.append(HumanMessage(content=current_input))

    response = llm.invoke(messages)
    answer = response.content
    # 拒答场景强制收敛为固定话术：模型偶尔会把 System Prompt 里的约束语句
    # （"严禁编造外部知识，禁止额外发挥"）一起复述出来，暴露提示词且显得啰嗦
    if "文档中未查询到相关内容" in answer:
        answer = "文档中未查询到相关内容"
    return answer


# =====================================================================
# 上传处理（增量构建两套索引）
# =====================================================================
def handle_upload(uploaded_files) -> None:
    vector_store = st.session_state.vector_store
    bm25_index = st.session_state.bm25_index

    total_new_chunks = 0
    for uf in uploaded_files:
        file_bytes = uf.read()
        filename = uf.name
        chunks = process_uploaded_file(file_bytes, filename)
        if not chunks:
            continue

        # 1) 写入 Chroma 稠密向量索引（支持增量 add）
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [c["metadata"]["doc_id"] for c in chunks]
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        # 2) 追加到 BM25 内存索引（内部整体重建）
        bm25_index.add_chunks(chunks)

        total_new_chunks += len(chunks)

    # 刷新统计
    _refresh_stats_from_chroma()
    st.success(
        f"✅ 成功上传 {len(uploaded_files)} 个文件，新增 {total_new_chunks} 个 chunk。"
        f"当前文档总数：{st.session_state.doc_count}，总 chunk 数：{st.session_state.chunk_count}"
    )


def _upload_callback() -> None:
    """
    上传按钮的 on_click 回调。
    关键：on_click 回调在脚本重渲染之前执行，因此 handle_upload 写入 Chroma 并刷新统计后，
    顶部指标卡能在同一次渲染中显示最新值，不会出现"提示正确但指标卡为0"的时序问题。
    """
    files = st.session_state.get("uploader") or []
    if files:
        handle_upload(files)


# =====================================================================
# 页面渲染
# =====================================================================
st.title("📚 RAG 知识库问答助手")
st.caption("双路召回架构 · 纯向量 VS 混合检索（BM25+RRF）对照实验")

# ---------- 顶部：统计 + 检索模式（紧凑横排） ----------
# 每次渲染前从 Chroma 重新拉取统计，避免上传后顶部指标卡显示旧值
_refresh_stats_from_chroma()
head_stat1, head_stat2, head_mode = st.columns([1, 1, 2.2], gap="medium")
head_stat1.metric("文档总数", st.session_state.doc_count)
head_stat2.metric("总 chunk 数量", st.session_state.chunk_count)
with head_mode:
    mode = st.radio(
        "检索模式",
        options=["A", "B"],
        format_func=lambda x: "模式A：纯向量 Top5" if x == "A"
        else "模式B：混合检索（BM25+RRF）",
        index=0,  # 默认选中模式A
        horizontal=True,
        key="mode_radio",
    )

with st.expander("📤 文档上传（PDF / TXT / MD）", expanded=False):
    uploaded = st.file_uploader(
        "选择文件（支持多选，增量构建）",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key="uploader",
    )
    if uploaded:
        st.button("开始上传并构建索引", type="primary", on_click=_upload_callback)

st.divider()

# ---------- 左右双栏：各自固定高度，独立滚动 ----------
left_col, right_col = st.columns([1, 1.3], gap="medium")

# ---------------- 左栏：聊天问答区 ----------------
# 注意：输入框不放在这里，必须放在主区域（见文件末尾），
#       Streamlit 才会把它钉在视口最底部，像微信一样不随内容滚动。
with left_col:
    title_col, action_col = st.columns([4, 1], vertical_alignment="center")
    title_col.markdown("#### 💬 问答区")
    if st.session_state.messages and action_col.button(
        "🗑 清空", key="clear_chat", use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.last_preview = None
        st.rerun()

    # 固定高度的消息容器：新消息自动滚到底部，不撑高整页
    with st.container(height=CHAT_AREA_HEIGHT, border=True, autoscroll=True):
        if not st.session_state.messages:
            st.caption("👋 在下方输入问题，我会严格基于知识库作答；右侧会并列展示三路召回结果。")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# ---------------- 右栏：召回预览区（三栏并列，容器独立滚动） ----------------
with right_col:
    st.markdown("#### 🔍 召回预览")
    preview = st.session_state.last_preview

    with st.container(height=PREVIEW_AREA_HEIGHT, border=True):
        if not preview:
            st.info("提出问题后，这里会并列展示：向量召回 / BM25 召回 / RRF 融合 三路结果。")
        elif preview["mode"] == "A":
            st.caption(f"模式A · 纯向量检索 Top{len(preview['vector'])}")
            for i, item in enumerate(preview["vector"], start=1):
                render_preview_item(item, i)
        else:
            # 三栏并列：向量召回 / BM25 召回 / RRF 融合，同时可见，无需切换
            c_vec, c_bm25, c_rrf = st.columns(3, gap="small")
            with c_vec:
                st.caption(f"向量召回 · {len(preview['vector'])}")
                for i, item in enumerate(preview["vector"], start=1):
                    render_preview_item(item, i)
            with c_bm25:
                st.caption(f"BM25 召回 · {len(preview['bm25'])}")
                if not preview["bm25"]:
                    st.caption("（索引为空，请先上传文档）")
                for i, item in enumerate(preview["bm25"], start=1):
                    render_preview_item(item, i)
            with c_rrf:
                st.caption(f"RRF 融合 · Top{len(preview['rrf'])}")
                for i, item in enumerate(preview["rrf"], start=1):
                    render_preview_item(item, i, show_rank_meta=True)

# ---------------- 底部输入框：主区域调用 → 钉在视口最底部 ----------------
question = st.chat_input("输入你的问题…", key="chat_input")

if question:
    if not DASHSCOPE_API_KEY:
        st.error("未检测到 DASHSCOPE_API_KEY，请在 .env 中配置后重启。")
    else:
        with st.spinner("正在检索并生成回答…"):
            answer_chunks, preview = retrieve(mode, question)
            answer = generate_answer(question, answer_chunks)

        # 先落库再重渲染：消息进入左侧滚动容器，右侧同步刷新召回预览
        st.session_state.last_preview = preview
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
