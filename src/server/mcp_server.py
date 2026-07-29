"""
MCP Server for the RAG system.

Exposes the RAG retrieval pipeline as a discoverable MCP tool.
Any MCP-compatible client (LangChain agent, Claude Desktop, etc.)
can connect and use these tools dynamically — no hardcoding needed.

This server owns ALL the heavy RAG dependencies:
  - Embedding model (Ollama)
  - Vector database connection (PGVector)
  - Hybrid retriever (BM25 + Semantic)

The orchestrator (app.py) knows NOTHING about any of this.
"""

import sys
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP
from langchain_ollama import OllamaEmbeddings

from src.rag.db_connection import get_vector_store
from src.rag.hybrid_retriever import create_hybrid_retriever
from src.rag.query_rewriter import create_query_rewriter
from src.rag.retrieval import RAGPipeline
from src.config.settings import EMBEDDING_MODEL

# ── Initialize the MCP Server ──────────────────────────────────────
mcp = FastMCP("RAG-Tools-Server")

# ── Initialize the RAG backend (runs once at startup) ──────────────
# NOTE: We print to stderr because stdout is reserved for the MCP
#       stdio protocol when launched as a subprocess.
print("🔧 MCP Server: Initializing RAG backend...", file=sys.stderr)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = get_vector_store(embedding_function=embeddings)
hybrid_retriever = create_hybrid_retriever(vector_store)

# Create the full pipeline (Hybrid + Reranker)
rag_pipeline = RAGPipeline(hybrid_retriever)

# Initialize query rewriter (rewrites user queries → optimized search queries)
rewrite_query = create_query_rewriter()

print("✅ MCP Server: RAG backend ready.\n", file=sys.stderr)


# ── MCP Tools ──────────────────────────────────────────────────────
# Each @mcp.tool() becomes a capability that the agent discovers
# automatically. Add more functions here to expand the agent's skills.

@mcp.tool()
def retrieve_context(query: str, metadata_filter: Optional[dict] = None) -> str:
    """Retrieve information from indexed documents to help answer a query.

    Uses a hybrid search (BM25 keyword + semantic vector) followed by 
    a cross-encoder reranker for best results. The query is automatically 
    rewritten to optimize retrieval quality.

    Args:
        query: The search query to find relevant context.
        metadata_filter: Optional dictionary of metadata to filter by,
            e.g. {"company": "FakeCorp", "doc_type": "handbook"}.
            Use this when the user asks about a specific company,
            document, or category.

    Returns:
        A formatted string of the most relevant document chunks,
        or a message saying no matches were found.
    """
    filter_dict = metadata_filter

    # Rewrite the query for better retrieval
    optimized_query = rewrite_query(query)

    # Run the full RAG pipeline (Hybrid + Reranking)
    retrieved_docs = rag_pipeline.retrieve(optimized_query, metadata_filter=filter_dict)

    if not retrieved_docs:
        return "No matching documents found after filtering."

    # Serialize results for the agent to read
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
