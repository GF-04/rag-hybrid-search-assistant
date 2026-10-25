# -*- coding: utf-8 -*-
"""
对抗性对照实验 v2：加入语义-标识符冲突的对抗性查询。
核心判据：正确 chunk 是否进入 Top5（召回率@5）及其名次。
"""
import json, os, shutil
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

from config import (DASHSCOPE_API_KEY, OPENAI_COMPATIBLE_BASE_URL, CHAT_MODEL_NAME,
    EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR, SYSTEM_PROMPT,
    VECTOR_TOP_K, HYBRID_CANDIDATE_K, RRF_K, FINAL_TOP_K)
from document_processor import process_uploaded_file
from retrievers import BM25Index, rrf_fusion

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=DASHSCOPE_API_KEY,
    base_url=OPENAI_COMPATIBLE_BASE_URL, check_embedding_ctx_length=False)
if os.path.exists(CHROMA_PERSIST_DIR):
    shutil.rmtree(CHROMA_PERSIST_DIR)
vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
bm25_index = BM25Index()
llm = ChatOpenAI(model=CHAT_MODEL_NAME, api_key=DASHSCOPE_API_KEY,
    base_url=OPENAI_COMPATIBLE_BASE_URL, temperature=0)

DOC_PATH = r"c:\Users\xwr04\Desktop\混合检索 RAG 知识库问答助手\测试文档_运维手册.md"
with open(DOC_PATH, "rb") as f:
    chunks = process_uploaded_file(f.read(), "测试文档_运维手册.md")
print(f"[INFO] {len(chunks)} chunks")
vector_store.add_texts([c["content"] for c in chunks],
    metadatas=[c["metadata"] for c in chunks], ids=[c["metadata"]["doc_id"] for c in chunks])
bm25_index.add_chunks(chunks)

def retrieve_a(q):
    return [{"content":d.page_content,"metadata":d.metadata,"rank":i+1}
            for i,d in enumerate(vector_store.similarity_search(q, k=VECTOR_TOP_K))]

def retrieve_b(q):
    vec=[{"content":d.page_content,"metadata":d.metadata,"rank":i+1}
         for i,d in enumerate(vector_store.similarity_search(q, k=HYBRID_CANDIDATE_K))]
    bm=bm25_index.search(q, top_k=HYBRID_CANDIDATE_K)
    rrf=rrf_fusion(vec, bm, k=RRF_K, top_k=FINAL_TOP_K)
    return vec, bm, rrf

def answer(q, ctx):
    c="\n\n".join(f"[片段{i+1}] {x['content']}" for i,x in enumerate(ctx))
    return llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"上下文：\n{c}\n\n问题：{q}")]).content

def chunk_rank(results, target):
    if target is None: return None
    ts=target if isinstance(target,list) else [target]
    for r in results:
        if r["metadata"].get("chunk_index") in ts:
            return r.get("rank") or r.get("vector_rank") or 999
    return None

# 16 个用例：4 普通语义 / 4 同义改写 / 3 精确编号 / 3 对抗性混合 / 2 无答案
CASES = [
    # 普通语义
    ("S1","这个运维手册主要讲了什么内容？","普通语义",None,"概述"),
    ("S2","系统报错时应该先查哪部分？","普通语义",None,"错误码章节"),
    ("S3","支付遇到问题怎么排查？","普通语义",1,"ERR_404_A"),
    ("S4","服务挂了怎么自动恢复？","普通语义",7,"Watchdog"),
    # 同义改写
    ("S5","付钱的时候卡住不动是啥问题？","同义改写",1,"ERR_404_A"),
    ("S6","秒杀时库存扣不下来怎么办？","同义改写",2,"ERR_404_B"),
    ("S7","网关转发请求失败怎么搞？","同义改写",4,"ERR_502_GATEWAY"),
    ("S8","近似最近邻索引参数怎么调？","同义改写",5,"HNSW-M16"),
    # 精确编号/缩写
    ("S9","ERR_404_A 是什么错误？","精确编号",1,"支付网关超时"),
    ("S10","HNSW-M16 的 efSearch 建议值？","精确编号",5,"200"),
    ("S11","V2.3.1-hotfix 修复了什么？","精确编号",6,"死锁"),
    # 对抗性混合（3-4个强语义词指向错误chunk，1个精确标识符指向正确chunk）
    ("S12","Panic Watchdog 看门狗机制 ERR_404_A 是什么错误？","对抗性混合",1,"ERR_404_A"),
    ("S13","efSearch efConstruction HNSW 索引 ERR_404_B 怎么处理？","对抗性混合",2,"ERR_404_B"),
    ("S14","微信支付宝支付回调超时 V2.3.1-hotfix 修复了啥？","对抗性混合",6,"V2.3.1-hotfix"),
    # 无答案
    ("S15","ERR_999 是什么意思？","无答案",None,"未查询到"),
    ("S16","怎么配置 Redis 集群？","无答案",None,"未查询到"),
]

results=[]
for cid,q,qtype,exp,kw in CASES:
    print(f"\n[{cid}] {q} ({qtype})")
    a=retrieve_a(q); ans_a=answer(q,a)
    vec,bm,rrf=retrieve_b(q); ans_b=answer(q,rrf)
    ra=chunk_rank(a,exp); rb=chunk_rank(rrf,exp)
    rec={"id":cid,"query":q,"type":qtype,"expect_chunk":exp,"keyword":kw,
        "modeA":{"top1":a[0]["metadata"]["chunk_index"] if a else None,
                 "top5":[r["metadata"]["chunk_index"] for r in a],
                 "correct_rank":ra,"in_top5":ra is not None and ra<=5,
                 "answer":ans_a},
        "modeB":{"top1":rrf[0]["metadata"]["chunk_index"] if rrf else None,
                 "top5":[r["metadata"]["chunk_index"] for r in rrf],
                 "correct_rank":rb,"in_top5":rb is not None and rb<=5,
                 "rrf_top1_vrank":rrf[0].get("vector_rank") if rrf else None,
                 "rrf_top1_brank":rrf[0].get("bm25_rank") if rrf else None,
                 "rrf_top1_tag":rrf[0].get("source_tag") if rrf else None,
                 "bm25_top5":[r["metadata"]["chunk_index"] for r in bm[:5]],
                 "answer":ans_b}}
    # 判分：无答案看是否拒答；有答案看关键词是否出现
    if qtype=="无答案":
        rec["modeA"]["ok"]="未查询到" in ans_a
        rec["modeB"]["ok"]="未查询到" in ans_b
    else:
        rec["modeA"]["ok"]=kw in ans_a
        rec["modeB"]["ok"]=kw in ans_b
    results.append(rec)
    print(f"  A: top1={rec['modeA']['top1']} 正确chunk名次={ra} top5={rec['modeA']['top5']} ok={rec['modeA']['ok']}")
    print(f"  B: top1={rec['modeB']['top1']} 正确chunk名次={rb} top5={rec['modeB']['top5']} bm25top5={rec['modeB']['bm25_top5']} tag={rec['modeB']['rrf_top1_tag']} ok={rec['modeB']['ok']}")

out=r"c:\Users\xwr04\Desktop\混合检索 RAG 知识库问答助手\adversarial_test_results.json"
with open(out,"w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

print("\n===== 汇总 =====")
print(f"模式A 回答正确: {sum(r['modeA']['ok'] for r in results)}/{len(results)}")
print(f"模式B 回答正确: {sum(r['modeB']['ok'] for r in results)}/{len(results)}")
print(f"模式A 正确chunk进Top5: {sum(r['modeA']['in_top5'] for r in results if r['expect_chunk'] is not None)}/{sum(1 for r in results if r['expect_chunk'] is not None)}")
print(f"模式B 正确chunk进Top5: {sum(r['modeB']['in_top5'] for r in results if r['expect_chunk'] is not None)}/{sum(1 for r in results if r['expect_chunk'] is not None)}")
print("\n★ 纯向量漏召/错召，混合检索修复的 Bad Case：")
for r in results:
    if r["expect_chunk"] is not None and (not r["modeA"]["in_top5"] or r["modeA"]["correct_rank"] and r["modeA"]["correct_rank"]>2) and r["modeB"]["in_top5"] and (r["modeB"]["correct_rank"] or 99)<=2:
        print(f"  [{r['id']}] {r['query']}  | A名次={r['modeA']['correct_rank']} B名次={r['modeB']['correct_rank']}")
print(f"\n结果已保存: {out}")
