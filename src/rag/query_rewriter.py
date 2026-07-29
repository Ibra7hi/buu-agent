"""
Query Rewriter for RAG Retrieval.

Rewrites user queries into optimized search queries before
they hit the retriever. This improves retrieval by:
  - Removing conversational filler ("hey", "can you", etc.)
  - Expanding abbreviations and clarifying ambiguous terms
  - Reformulating questions into search-oriented keyword phrases

Uses OpenRouter (same provider as the main agent) to keep
the setup simple — no local model required.
Falls back gracefully to the original query if rewriting fails.
"""

import sys
from langchain_openai import ChatOpenAI
from src.config.settings import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    REWRITER_MODEL, REWRITE_TIMEOUT_SECONDS,
)

REWRITE_PROMPT = """You are a search query optimizer for a document retrieval system.

Your job: rewrite the user's question into an optimized search query that will find the most relevant documents.

Rules:
- Output ONLY the rewritten query, nothing else
- Remove conversational filler (greetings, "can you tell me", "I want to know", etc.)
- Keep the core intent and key terms
- Expand abbreviations if obvious
- If the query is already a good search query, return it as-is
- Do NOT add information that wasn't in the original query
- Keep it concise (under 30 words)

User question: {query}

Optimized search query:"""


def create_query_rewriter():
    """Create a query rewriter backed by OpenRouter.

    Returns a callable that takes a query string and returns a rewritten query.
    If the API is unavailable or too slow, the rewriter falls back to the original query.
    """
    api_key = OPENROUTER_API_KEY

    if not api_key:
        print("⚠️  Query rewriter: No OPENROUTER_API_KEY found. Falling back to raw queries.", file=sys.stderr)
        return lambda query: query

    try:
        llm = ChatOpenAI(
            model=REWRITER_MODEL,
            openai_api_key=api_key,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,      # Deterministic rewrites
            max_tokens=64,      # Short output — we just need the rewritten query
            timeout=REWRITE_TIMEOUT_SECONDS,        # HTTP timeout
            max_retries=0,      # Don't retry — just fall back to original query
        )
        print(f"✅ Query rewriter ready (OpenRouter: {REWRITER_MODEL})", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  Query rewriter init failed ({e}). Falling back to raw queries.", file=sys.stderr)
        return lambda query: query

    def rewrite(query: str) -> str:
        """Rewrite a user query into an optimized search query.

        Args:
            query: The raw user query.

        Returns:
            The rewritten (optimized) query, or the original if rewriting fails.
        """
        try:
            prompt = REWRITE_PROMPT.format(query=query)
            response = llm.invoke(prompt)
            rewritten = response.content.strip().strip('"').strip("'")

            # Sanity checks: if the rewrite is empty or way too long, skip it
            if not rewritten or len(rewritten) > 300:
                return query

            print(f"🔄 Query rewrite: '{query}' → '{rewritten}'", file=sys.stderr)
            return rewritten

        except Exception as e:
            print(f"⚠️  Rewrite failed ({e}), using original query.", file=sys.stderr)
            return query

    return rewrite
