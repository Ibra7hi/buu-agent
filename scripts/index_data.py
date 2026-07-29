"""Index documents into the vector database."""

from langchain_ollama import OllamaEmbeddings
from src.rag.document_loader import load_pdf_document
from src.rag.db_connection import get_vector_store
from src.rag.indexing import split_and_index
from src.config.settings import EMBEDDING_MODEL

def index_my_data():
    print("Step 1: Initialize Database Connection")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = get_vector_store(embedding_function=embeddings)

    print("Step 2: Load Target Documents")
    pdf_path = "docs/hima.pdf" 
    try:
        docs = load_pdf_document(pdf_path)
        print(f"   -> Successfully loaded {len(docs)} pages.")
    except Exception as e:
        print(f"   -> Error loading PDF: {e}")
        return

    print("Step 3: Chunk and Index Data")
    split_and_index(vector_store, docs, embeddings=embeddings)
    
    print("\n✅ Indexing Complete.")

if __name__ == "__main__":
    index_my_data()
