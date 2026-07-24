from langchain_ollama import OllamaEmbeddings
from src.core.db_connection import get_vector_store
from src.config.settings import EMBEDDING_MODEL, CACHE_COLLECTION, CACHE_THRESHOLD
from langchain_core.documents import Document

class SemanticCache:
    """
    A PostgreSQL-backed Semantic Cache.
    It embeds the incoming query and searches the vector database for a highly similar past query.
    If a match is found (score < threshold), it returns the cached response instantly, 
    saving LLM generation time and costs.
    """
    def __init__(self):
        print("Initializing Semantic Cache...")
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        # Create a separate collection specifically for caching
        self.vector_store = get_vector_store(
            embedding_function=self.embeddings,
            collection_name=CACHE_COLLECTION
        )
        # Cosine distance threshold. Lower is more strict. 
        # 0.05 means ~95% similarity
        self.threshold = CACHE_THRESHOLD

    async def check(self, query: str) -> str | None:
        try:
            # We use synchronous similarity search wrapped in an async function for compatibility
            results = self.vector_store.similarity_search_with_score(query, k=1)
            if results:
                doc, score = results[0]
                # If distance is less than threshold, it's a semantic match!
                if score < self.threshold:
                    print(f"\n⚡ Semantic Cache HIT! (Distance: {score:.4f})")
                    return doc.metadata.get("response")
        except Exception as e:
            print(f"⚠️ Cache check error: {e}")
        return None

    async def store(self, query: str, response: str):
        try:
            # We store the QUERY as the embedded content, and the RESPONSE as metadata
            doc = Document(page_content=query, metadata={"response": response})
            self.vector_store.add_documents([doc])
            print("💾 Saved new response to Semantic Cache.")
        except Exception as e:
            print(f"⚠️ Cache store error: {e}")

# Global instance
semantic_cache = SemanticCache()
