"""
文档加载器
支持多种文档格式：PDF, Markdown, TXT, Word
"""

import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    DirectoryLoader
)


class DocumentLoader:
    """文档加载器 - 支持多种格式"""

    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.txt': 'txt',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.docx': 'word'
    }

    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            文档列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext not in DocumentLoader.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式：{ext}\n"
                f"支持的格式：{', '.join(DocumentLoader.SUPPORTED_EXTENSIONS.keys())}"
            )

        loader_type = DocumentLoader.SUPPORTED_EXTENSIONS[ext]

        # 根据类型选择加载器
        if loader_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif loader_type == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif loader_type == 'markdown':
            loader = UnstructuredMarkdownLoader(file_path)
        elif loader_type == 'word':
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')

        print(f"加载文件：{file_path} (类型：{loader_type})")
        return loader.load()

    @staticmethod
    def load_directory(
        dir_path: str,
        glob_pattern: str = "**/*",
        recursive: bool = True
    ) -> List[Document]:
        """
        加载目录下所有支持的文档

        Args:
            dir_path: 目录路径
            glob_pattern: 文件匹配模式
            recursive: 是否递归子目录

        Returns:
            文档列表
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录不存在：{dir_path}")

        all_docs = []

        # 遍历目录
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                if ext in DocumentLoader.SUPPORTED_EXTENSIONS:
                    try:
                        docs = DocumentLoader.load_file(file_path)
                        # 添加文件路径到元数据
                        for doc in docs:
                            doc.metadata['source_file'] = file_path
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"加载文件失败 {file_path}: {e}")

            if not recursive:
                break

        print(f"从目录加载 {len(all_docs)} 个文档块：{dir_path}")
        return all_docs

    @staticmethod
    def load_multiple_files(file_paths: List[str]) -> List[Document]:
        """
        加载多个文件

        Args:
            file_paths: 文件路径列表

        Returns:
            文档列表
        """
        all_docs = []
        for file_path in file_paths:
            try:
                docs = DocumentLoader.load_file(file_path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
        return all_docs
