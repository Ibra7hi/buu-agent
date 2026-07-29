from typing import Optional, Dict, Any, List
from langchain_core.documents import Document

# Reranker imports
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from src.config.settings import RERANKER_TOP_N

class RAGPipeline:
    """
    Core RAG Pipeline logic. 
    It combines the base retriever (Hybrid) with a Re-ranker (Flashrank)
    to return highly relevant documents.
    """
    def __init__(self, retriever):
        self.retriever = retriever
        print("Loading Reranker Model (Flashrank - tiny, lightweight)...")
        try:
            self.compressor = FlashrankRerank(top_n=RERANKER_TOP_N)
            print("Reranker Model loaded successfully.")
        except ImportError:
            print("Warning: flashrank not installed. Reranking will be disabled. Run: pip install flashrank")
            self.compressor = None

    def retrieve(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Executes the full pipeline:
        1. Hybrid Search (BM25 + Semantic)
        2. Reranking (Flashrank Cross-Encoder)
        """
        # 1. Base Retrieval
        retrieved_docs = self.retriever.invoke(query, metadata_filter=metadata_filter)

        if not retrieved_docs:
            return []

        # 2. Re-ranking
        if self.compressor:
            print(f"Reranking {len(retrieved_docs)} documents...")
            final_docs = self.compressor.compress_documents(retrieved_docs, query)
            print(f"Kept top {len(final_docs)} documents after reranking.")
        else:
            final_docs = retrieved_docs

        return final_docs
