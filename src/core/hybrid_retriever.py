from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from src.config.settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


class DynamicHybridRetriever:
    def __init__(self, vector_store, k=5, bm25_weight=0.4, semantic_weight=0.6):
        self.vector_store = vector_store
        self.k = k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        
        print("📚 Loading documents from PostgreSQL for BM25 index...")
        import psycopg2
        self.documents = []
        try:
            conn = psycopg2.connect(
                host=DB_HOST, 
                port=DB_PORT, 
                user=DB_USER, 
                password=DB_PASSWORD, 
                dbname=DB_NAME
            )
            cur = conn.cursor()
            
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (vector_store.collection_name,))
            row = cur.fetchone()
            
            if row:
                collection_id = row[0]
                cur.execute("SELECT document, cmetadata FROM langchain_pg_embedding WHERE collection_id = %s;", (collection_id,))
                rows = cur.fetchall()
                
                for content, metadata in rows:
                    if content:
                        self.documents.append(Document(page_content=content, metadata=metadata or {}))
            
            print(f"   -> Loaded {len(self.documents)} chunks.")
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error fetching documents from PostgreSQL: {e}")

        if not self.documents:
            raise ValueError("No documents found in PostgreSQL!")

        # Build the baseline BM25 retriever for queries without filters
        self.base_bm25_retriever = BM25Retriever.from_documents(self.documents, k=self.k)
        print(f"✅ Hybrid retriever ready! (BM25 weight={bm25_weight}, Semantic weight={semantic_weight})")

    def invoke(self, query: str, metadata_filter: dict = None):
        """Executes the search with PRE-FILTERING applied directly to the vector store and BM25."""
        if not metadata_filter:
            # No filters: use standard vector store and base BM25
            semantic_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.k})
            hybrid = EnsembleRetriever(
                retrievers=[self.base_bm25_retriever, semantic_retriever],
                weights=[self.bm25_weight, self.semantic_weight],
            )
            return hybrid.invoke(query)
        else:
            # 1. Pre-filter BM25 documents in memory BEFORE keyword search
            filtered_docs = []
            for doc in self.documents:
                match = all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
                if match:
                    filtered_docs.append(doc)
            
            if not filtered_docs:
                return []
                
            # Build dynamic BM25 index exclusively on the filtered subset
            # We adjust k to not exceed the number of filtered docs
            dynamic_k = min(self.k, len(filtered_docs))
            dynamic_bm25 = BM25Retriever.from_documents(filtered_docs, k=dynamic_k)
            
            # 2. Pre-filter Vector Store natively using PostgreSQL's JSONB queries BEFORE vector search
            semantic_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.k, "filter": metadata_filter}
            )
            
            # 3. Combine both pre-filtered results
            hybrid = EnsembleRetriever(
                retrievers=[dynamic_bm25, semantic_retriever],
                weights=[self.bm25_weight, self.semantic_weight],
            )
            return hybrid.invoke(query)

def create_hybrid_retriever(vector_store, k=8, bm25_weight=0.4, semantic_weight=0.6):
    """Factory function for backward compatibility with app.py"""
    return DynamicHybridRetriever(vector_store, k, bm25_weight, semantic_weight)
