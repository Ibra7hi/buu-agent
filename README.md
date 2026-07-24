# RAG Intelligence: Enterprise AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FastAPI**, **PGVector**, and **Next.js**. This project allows you to index your own PDF/Web documents and query them using either entirely local models or lightning-fast cloud APIs.

---

## 🏗 Tech Stack

*   **LLM Providers:** [Ollama](https://ollama.com/) (Local) OR [OpenRouter](https://openrouter.ai/) (Cloud API)
*   **Embeddings:** Nomic Embed Text v2 (Local via Ollama)
*   **AI Logic:** [LangChain](https://www.langchain.com/) & LangGraph (ReAct Agent)
*   **Tool Protocol:** [MCP](https://modelcontextprotocol.io/) (Model Context Protocol)
*   **Vector Database:** [PGVector](https://github.com/pgvector/pgvector) (Running via Docker)
*   **Backend API:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
*   **Frontend UI:** [Next.js](https://nextjs.org/) (React, Tailwind CSS, Framer Motion)

---

## 📂 Project Structure

```text
RAG/
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # PostgreSQL + pgvector container
├── README.md
│
├── docs/                        # 📄 Documents for indexing
│   ├── fake_company.pdf
│   ├── hima.pdf
│   └── math.pdf
│
├── scripts/                     # 🔧 CLI utilities
│   ├── index_data.py            # Index PDFs into the vector database
│   ├── reset_db.py              # Wipe the database clean
│   └── clean_cache.py           # Clear the semantic cache only
│
├── src/                         # 🐍 Application source code
│   ├── server/                  # Entry points
│   │   ├── app.py               # FastAPI orchestrator (the main backend)
│   │   └── mcp_server.py        # MCP tool server (RAG tools)
│   │
│   ├── core/                    # Core RAG pipeline
│   │   ├── db_connection.py     # PGVector connection factory
│   │   ├── document_loader.py   # PDF/Web document ingestion
│   │   ├── indexing.py          # Semantic chunking & vector indexing
│   │   ├── retrieval.py         # Retrieval tool (with FlashRank reranker)
│   │   ├── hybrid_retriever.py  # BM25 + Semantic hybrid search
│   │   ├── query_rewriter.py    # LLM-based query optimization
│   │   └── cache.py             # PostgreSQL-backed semantic cache
│   │
│   ├── models/                  # LLM & embedding configuration
│   │   └── api_models.py        # OpenRouter / Ollama model factories
│   │
│   └── config/                  # Centralized settings
│       └── settings.py          # All constants & env vars in one place
│
└── frontend/                    # 🌐 Next.js UI application
    ├── src/app/page.tsx         # Main chat interface
    └── src/app/globals.css      # UI styling
```

---

## 🛠 Prerequisites

Before running the project, ensure you have:
1.  **Python 3.10+** (with a virtual environment set up)
2.  **Node.js 18+** (for the Next.js frontend)
3.  **Docker** (to run PGVector)
4.  **Ollama** (installed locally for embeddings & optional local generation)

**Pull the required Ollama models:**
```bash
ollama pull nomic-embed-text-v2-moe
```

**Set up your environment:**
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

---

## 🚀 Running the Project

### Step 1: Start the Database
```bash
docker compose up -d
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start the Backend API
```bash
python -m src.server.app
```

### Step 4: Start the Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Access the interface at **http://localhost:3000**.

---

## 📚 Data Indexing

To populate the vector database with your documents:

1.  Place your PDF in the `docs/` folder.
2.  Edit `scripts/index_data.py` and update the `pdf_path` variable.
3.  Run the indexer:
```bash
python -m scripts.index_data
```

To **reset** the database (wipe everything):
```bash
python -m scripts.reset_db
```

To **clear only the semantic cache** (keep document index):
```bash
python -m scripts.clean_cache
```
