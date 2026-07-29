from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_and_index(vector_store, docs, embeddings=None):
    # Extract embeddings function from the vector store if not explicitly provided
    if not embeddings:
        embeddings = getattr(vector_store, 'embeddings', getattr(vector_store, 'embedding_function', None))

    if embeddings:
        print("🤖 Using Semantic Chunking... (Splitting by topic instead of character count)")
        # SemanticChunker uses the embedding model to find changes in topic/meaning
        text_split = SemanticChunker(
            embeddings, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=70
        )
    else:
        print("⚠️ Embeddings not found. Falling back to RecursiveCharacterTextSplitter.")
        text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)

    print("✂️ Splitting documents. This might take a moment...")
    all_splites = text_split.split_documents(docs)
    print(f"✅ Generated {len(all_splites)} semantic chunks.")
    
    _ = vector_store.add_documents(documents=all_splites)
    return all_splites
