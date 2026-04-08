"""
知识库问答助手 - Streamlit 应用
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from rag_engine import KnowledgeBaseRAG
from document_loader import DocumentLoader

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="知识库问答助手",
    page_icon="📚",
    layout="wide"
)

# 初始化 session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = 0


def init_rag_engine():
    """初始化 RAG 引擎"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key :
        st.error("⚠️ 请先配置 OPENAI_API_KEY 环境变量")
        return None

    return KnowledgeBaseRAG(
        persist_directory="./faiss_index",
        chunk_size=500,
        chunk_overlap=50,
        api_key=api_key
    )


# 侧边栏
with st.sidebar:
    st.title("📚 知识库管理")

    # 状态显示
    if st.session_state.rag_engine:
        stats = st.session_state.rag_engine.get_stats()
        st.metric("文档块数量", stats.get("total_chunks", "N/A"))
        st.success("✅ RAG 引擎已初始化")
    else:
        st.warning("RAG 引擎未初始化")

    st.divider()

    # 初始化按钮
    if st.button("🔄 初始化/重置 RAG 引擎", use_container_width=True):
        with st.spinner("初始化中..."):
            st.session_state.rag_engine = init_rag_engine()
            st.session_state.messages = []
            st.session_state.docs_loaded = 0
            st.rerun()

    st.divider()

    # 文档上传
    st.subheader("📄 上传文档")
    uploaded_files = st.file_uploader(
        "上传 PDF/TXT/MD 文档",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("处理上传的文档", use_container_width=True):
            if not st.session_state.rag_engine:
                st.error("请先初始化 RAG 引擎")
            else:
                with st.spinner("处理文档中..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        # 保存临时文件
                        temp_path = f"./temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())

                        # 加载文档
                        try:
                            docs = DocumentLoader.load_file(temp_path)
                            chunks = st.session_state.rag_engine.add_documents(docs)
                            total_chunks += chunks
                            st.session_state.docs_loaded += len(docs)
                            os.remove(temp_path)
                        except Exception as e:
                            st.error(f"处理失败 {uploaded_file.name}: {e}")
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                    st.success(f"✅ 处理完成！新增 {total_chunks} 个文档块")
                    st.rerun()

    st.divider()

    # 加载示例文档
    st.subheader("📂 示例文档")
    docs_dir = Path("./docs")
    if docs_dir.exists():
        doc_files = list(docs_dir.glob("*"))
        if doc_files:
            if st.button("加载 docs 目录所有文档", use_container_width=True):
                if st.session_state.rag_engine:
                    with st.spinner("加载文档中..."):
                        docs = DocumentLoader.load_directory(str(docs_dir))
                        st.session_state.rag_engine.add_documents(docs)
                        st.session_state.docs_loaded += len(docs)
                        st.success(f"✅ 加载 {len(docs)} 个文档")
                        st.rerun()
                else:
                    st.error("请先初始化 RAG 引擎")

    st.divider()

    # 清除历史
    if st.button("🗑️ 清除对话历史", use_container_width=True):
        if st.session_state.rag_engine:
            st.session_state.rag_engine.clear_history()
            st.session_state.messages = []
            st.success("对话历史已清除")
            st.rerun()


# 主界面
st.title("📚 知识库问答助手")
st.markdown("基于 RAG 技术的智能知识库问答系统")

if not st.session_state.rag_engine:
    st.info("👈 请先在侧边栏点击「初始化 RAG 引擎」")
    st.markdown("""
    ### 使用步骤：
    1. 配置 `.env` 文件，填入你的 Anthropic API Key
    2. 点击侧边栏的「初始化 RAG 引擎」
    3. 上传文档或加载 docs 目录的文档
    4. 开始提问！
    """)
else:
    # 显示已有对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📖 查看来源"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**来源 {i}:**")
                        st.markdown(f"- 文件：{source.get('metadata', {}).get('source', '未知')}")
                        st.markdown(f"- 内容：{source.get('content', 'N/A')}")

    # 聊天输入
    if prompt := st.chat_input("请输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    result = st.session_state.rag_engine.query(prompt)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📖 查看来源"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**来源 {i}:**")
                                st.markdown(f"- 内容：{source.get('content', 'N/A')}")
                                st.markdown(f"- 元数据：{source.get('metadata', {})}")

                    # 保存助手消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"❌ 错误：{e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"抱歉，发生错误：{e}"
                    })

# 页脚
st.divider()
st.caption("💡 提示：上传更多文档可以增强知识库的回答能力")
