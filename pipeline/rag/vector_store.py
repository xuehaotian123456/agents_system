"""ChromaDB 向量库 + 文章加载"""
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from model.factory import embed_model
from utils.config_handler import get_chroma_config, get_hybrid_config, get_data_config
from utils.file_handler import get_file_md5, load_markdown, list_files_by_type
from utils.logger_handler import logger

chroma_cfg = get_chroma_config()
data_cfg = get_data_config()

class VectorStore:
    def __init__(self):
        persist = str(Path(__file__).parent.parent / chroma_cfg["persist_directory"])
        os.makedirs(persist, exist_ok=True)

        self.store = Chroma(
            collection_name=chroma_cfg["collection_name"],
            embedding_function=embed_model,
            persist_directory=persist,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_cfg["chunk_size"],
            chunk_overlap=chroma_cfg["chunk_overlap"],
            separators=["\n\n", "\n", "。", ".", "！", "？", "，", ","],
        )
        self.all_chunks = []
        self.all_docs = []
        self.hybrid_retriever = None

    def load_articles(self):
        data_path = str(Path(__file__).parent.parent / data_cfg.get("data_path", "data/articles"))
        allowed = tuple(data_cfg.get("allowed_types", ["md", "txt"]))
        md5_store = str(Path(__file__).parent.parent / data_cfg["md5_store"])

        # 加载 md5 缓存
        md5_cache = {}
        if os.path.exists(md5_store):
            try:
                with open(md5_store, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) == 2:
                            md5_cache[parts[0]] = parts[1]
            except (UnicodeDecodeError, Exception):
                # 文件损坏，跳过缓存
                md5_cache = {}

        files = list_files_by_type(data_path, allowed)
        new_md5 = {}

        for fpath in files:
            md5 = get_file_md5(fpath)
            if not md5:
                continue
            new_md5[fpath] = md5

            if fpath in md5_cache and md5_cache[fpath] == md5:
                self._recover_from_store(fpath)
                continue

            try:
                docs = load_markdown(fpath)
                chunks = self.splitter.split_documents(docs)
                self.store.add_documents(chunks)
                for d in chunks:
                    self.all_chunks.append(d.page_content)
                    self.all_docs.append(d)
                logger.info(f"文章已加载: {os.path.basename(fpath)} ({len(chunks)} chunks)")
            except Exception as e:
                logger.warning(f"文章加载失败: {fpath}: {e}")

        # 更新 md5
        with open(md5_store, "w") as f:
            for path, md5 in new_md5.items():
                f.write(f"{path}\t{md5}\n")

        self._init_hybrid()

    def _recover_from_store(self, fpath: str):
        """从已有向量库恢复文档块（用于混合检索）"""
        try:
            fname = os.path.basename(fpath)
            results = self.store.get(where={"source": fname}, include=["documents", "metadatas"])
            if not results or not results.get("documents"):
                results = self.store.get(include=["documents", "metadatas"])
                if results and results.get("documents"):
                    filtered_docs, filtered_meta = [], []
                    for doc, meta in zip(results["documents"], results["metadatas"]):
                        if fname in meta.get("source", ""):
                            filtered_docs.append(doc)
                            filtered_meta.append(meta)
                    results = {"documents": filtered_docs, "metadatas": filtered_meta}

            if results and results.get("documents"):
                for doc, meta in zip(results["documents"], results["metadatas"]):
                    self.all_chunks.append(doc)
                    self.all_docs.append(Document(page_content=doc, metadata=meta))
        except Exception as e:
            logger.warning(f"恢复文档块失败: {fname}: {e}")

    def _init_hybrid(self):
        hybrid_cfg = get_hybrid_config()
        if hybrid_cfg.get("enabled") and self.all_chunks:
            from rag.hybrid_retriever import HybridRetriever
            self.hybrid_retriever = HybridRetriever(
                self.store, self.all_chunks, self.all_docs,
                top_k_retrieve=hybrid_cfg["top_k_retrieve"],
                top_k_rerank=hybrid_cfg["top_k_rerank"],
                alpha=hybrid_cfg["alpha"],
            )
            logger.info(f"混合检索器初始化完成: {len(self.all_chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        if self.hybrid_retriever:
            return self.hybrid_retriever.search(query)
        return self.store.similarity_search(query, k=top_k)
