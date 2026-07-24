from langchain_core.tools import tool
from typing import Optional, Dict, Any

# Reranker imports
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from src.config.settings import RERANKER_TOP_N

def create_retrieval_tool(retriever):
    """
    Creates a retrieval tool that the RAG agent can use to search for relevant documents.

    Args:
        retriever: Any LangChain retriever (hybrid, semantic-only, etc.)
                   Must support .invoke(query) and return List[Document]
    """
    print("Loading Reranker Model (Flashrank - tiny, lightweight)...")
    # Initialize the cross-encoder model for re-ranking
    # FlashRank uses highly optimized ONNX models and is ~30MB in size total.
    try:
        # Set top_n to the number of documents you want to keep after reranking
        compressor = FlashrankRerank(top_n=RERANKER_TOP_N)
        print("Reranker Model loaded successfully.")
    except ImportError:
        print("Warning: flashrank not installed. Reranking will be disabled. Run: pip install flashrank")
        compressor = None

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str, metadata_filter: Optional[Dict[str, Any]] = None):
        """Retrieve information to help answer a query.
        
        Args:
            query: The search query to find relevant context.
            metadata_filter: Optional dictionary of metadata to filter by (e.g. {"company": "FakeCorp", "doc_type": "handbook"}). Use this when the user asks about a specific company, document, or category.
        """
        # Run the retriever (hybrid: BM25 + semantic search with RRF fusion)
        # We pass the metadata filter down to the DynamicHybridRetriever for strict pre-filtering!
        retrieved_docs = retriever.invoke(query, metadata_filter=metadata_filter)

        if not retrieved_docs:
            return "No matching documents found after filtering.", []

        # Apply Re-ranking if the compressor is available
        if compressor:
            print(f"Reranking {len(retrieved_docs)} documents...")
            final_docs = compressor.compress_documents(retrieved_docs, query)
            print(f"Kept top {len(final_docs)} documents after reranking.")
        else:
            final_docs = retrieved_docs

        # Serialize results for the agent to read
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in final_docs
        )
        return serialized, final_docs

    return retrieve_context
