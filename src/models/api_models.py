from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from src.config.settings import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL,
)

# -------------------------------------------------------------
# OPENROUTER API CONFIGURATION
# Use this file if you want fast, cloud-based LLM generation.
# -------------------------------------------------------------

def get_fast_llm(model_name="meta-llama/llama-3.1-8b-instruct"):
    """
    Connects to OpenRouter to use powerful, fast cloud models 
    without needing a heavy local GPU.
    """
    api_key = OPENROUTER_API_KEY
    
    if not api_key:
        print("⚠️ WARNING: OPENROUTER_API_KEY is not set!")
        
    return ChatOpenAI(
        model=model_name,
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        # optional: OpenRouter headers for routing stats
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Local RAG Assistant",
        }
    )

def get_embeddings():
    """
    We still use Ollama for embeddings because embedding models 
    are very small, extremely fast locally, and completely free.
    """
    return OllamaEmbeddings(model=EMBEDDING_MODEL)
