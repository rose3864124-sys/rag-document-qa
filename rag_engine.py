import os
from typing import List, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# 基础组件
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

class KnowledgeBaseRAG:
    def __init__(
        self,
        index_path: str = "./faiss_index",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        api_key: Optional[str] = None,
    ):
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化通义千问大模型
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请配置 OPENAI_API_KEY 环境变量")

        self.llm = ChatOpenAI(
            model="qwen-max",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3
        )

        # 文档分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # ✅ 通义千问官方嵌入模型（已替换Fake）
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        # 向量库
        self.vectorstore = self._init_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.chat_history: List[Any] = []

        # ✅ LCEL原生链（无chains模块，彻底解决导入问题）
        self.chain = self._build_lcel_chain()

    def _init_vectorstore(self) -> FAISS:
        if os.path.exists(self.index_path):
            return FAISS.load_local(
                self.index_path, self.embeddings, allow_dangerous_deserialization=True
            )
        initial_doc = [Document(page_content="init", metadata={"source": "init"})]
        return FAISS.from_documents(initial_doc, self.embeddings)

    def _build_lcel_chain(self):
        """LCEL原生实现对话历史+检索+问答，无chains依赖"""
        # 1. 对话历史重写Prompt
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "结合聊天历史，将用户问题重写为独立的查询语句"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 2. 问答Prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "仅根据上下文回答问题，禁止编造。上下文：{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 3. 上下文格式化
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 4. LCEL核心链
        contextualize_chain = contextualize_prompt | self.llm | StrOutputParser()

        def retrieve_chain(inputs):
            if inputs["chat_history"]:
                query = contextualize_chain.invoke(inputs)
            else:
                query = inputs["input"]
            return self.retriever.invoke(query)

        return (
            RunnablePassthrough.assign(context=retrieve_chain)
            | RunnablePassthrough.assign(answer=qa_prompt | self.llm | StrOutputParser())
        )

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        chunks = self.text_splitter.split_documents(documents)
        if not chunks:
            return 0
        new_db = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.merge_from(new_db)
        self.vectorstore.save_local(self.index_path)
        return len(chunks)

    def query(self, question: str) -> Dict[str, Any]:
        if not question.strip():
            return {"answer": "请输入有效问题", "sources": []}

        result = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history
        })

        answer = result["answer"]
        docs = result["context"]

        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])

        sources = [{"content": doc.page_content[:200]+"...", "metadata": doc.metadata} for doc in docs]
        return {"answer": answer, "sources": sources}

    def clear_history(self):
        self.chat_history = []

    def get_stats(self) -> Dict[str, Any]:
        try:
            return {"total_chunks": max(0, self.vectorstore.index.ntotal - 1)}
        except:
            return {"total_chunks": 0}