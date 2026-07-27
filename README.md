# ⚡ chroma-rerank-rag

> **A High-Precision Dual-Stage RAG Pipeline featuring Hybrid Retrieval (ChromaDB + BM25), Cross-Encoder Re-Ranking, Real-Time NDJSON Token Streaming, and Multi-Turn Conversational Memory.**

---

## 📐 Architecture Overview

```
                        ┌───────────────────────────────────┐
                        │      Uploaded PDF Document        │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │    Document Processor (PyMuPDF)   │
                        │    Text Extraction & Chunking     │
                        └─────────────────┬─────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
                    ▼                                           ▼
  ┌──────────────────────────────────┐        ┌──────────────────────────────────┐
  │   Dense Vector Store (ChromaDB)  │        │   Sparse Keyword Index (BM25)    │
  │   Model: BAAI/bge-base-en-v1.5   │        │   Tokenizer: Alphanumeric        │
  └─────────────────┬────────────────┘        └─────────────────┬────────────────┘
                    │ (Top 15 Dense)                            │ (Top 15 Sparse)
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │  Reciprocal Rank Fusion (RRF)     │
                        │  Deduplication & Rank Merging     │
                        └─────────────────┬─────────────────┘
                                          │ (15 Fused Candidates)
                                          ▼
                        ┌───────────────────────────────────┐
                        │    Cross-Encoder Re-Ranker        │
                        │ Model: ms-marco-MiniLM-L-6-v2     │
                        └─────────────────┬─────────────────┘
                                          │ (Top K Filtered Chunks)
                                          ▼
                        ┌───────────────────────────────────┐
                        │    Groq LLM (Llama 3.1 8B)        │
                        │ Multi-Turn Chat History Injected  │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │  NDJSON Real-Time Token Stream    │
                        │  Streamlit UI Typewriter Display  │
                        └───────────────────────────────────┘
```

---

## ✨ Key Technical Features

- **Dual-Stage Retrieval Architecture**:
  - **Stage 1a (Dense Vector Search)**: Semantic retrieval powered by ChromaDB & HuggingFace `BAAI/bge-base-en-v1.5` embeddings.
  - **Stage 1b (Sparse Lexical Search)**: Keyword precision powered by `rank-bm25`.
  - **Stage 1c (Reciprocal Rank Fusion)**: Combines dense & sparse candidate rankings into a unified score list.
  - **Stage 2 (Cross-Encoder Re-Ranking)**: Re-scores top candidate chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2` to maximize factual precision.
- **Real-Time NDJSON Streaming**: Asynchronous token-by-token HTTP streaming for real-time typewriter UI rendering.
- **Multi-Turn Conversational Memory**: Preserves context history across queries using LangChain message objects (`SystemMessage`, `HumanMessage`, `AIMessage`).
- **PDF Document Processing**: Fast page text extraction with PyMuPDF (`fitz`).
- **Smart Citation Suppression**: Automatically suppresses citation footers when questions cannot be answered from context.

---

## 📁 Repository Structure

```text
chroma-rerank-rag/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # FastAPI endpoints (/upload, /query with NDJSON stream)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Environment configuration & Pydantic settings
│   │   ├── hybrid_search.py       # BM25 Retriever & Reciprocal Rank Fusion (RRF)
│   │   ├── ingestion.py           # DocumentProcessor (PyMuPDF text extraction & chunking)
│   │   ├── llm_service.py         # ChatGroq integration, prompt engineering & streaming
│   │   ├── reranker.py            # CrossEncoder re-ranking module
│   │   └── vector_store.py        # VectorDBManager (ChromaDB persistent client)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── pydantic_models.py     # Request/Response schemas (QueryRequest, UploadResponse)
│   ├── main.py                    # FastAPI app initialization & CORS middleware
│   └── requirements.txt           # Backend dependencies
├── frontend/
│   ├── app.py                     # Streamlit application UI & streaming logic
│   └── requirements.txt           # Frontend dependencies
├── .gitignore                     # Git exclusion rules
├── README.md                      # Project documentation
└── sample_portfolio.pdf           # Sample PDF document
```

---

## 🛠️ Tech Stack

* **Backend**: FastAPI, Uvicorn, Python 3.11
* **Vector Database**: ChromaDB
* **Embeddings**: HuggingFace (`BAAI/bge-base-en-v1.5`)
* **Keyword Search**: `rank-bm25` (BM25Okapi)
* **Re-Ranker**: SentenceTransformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **LLM Engine**: Groq (`llama-3.1-8b-instant`) via LangChain
* **Frontend**: Streamlit
* **PDF Parser**: PyMuPDF (`fitz`)

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/ksrinivasarao2012/chroma-rerank-rag.git
cd chroma-rerank-rag
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\activate

# Install Dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Running the Project

#### **Terminal 1: Start FastAPI Backend**
```powershell
$env:PYTHONPATH="backend"
uvicorn backend.main:app --reload
```
*The backend API will run at `http://127.0.0.1:8000`.*

#### **Terminal 2: Start Streamlit Frontend**
```powershell
streamlit run frontend/app.py
```
*The web interface will open at `http://localhost:8501`.*

---

## 🔌 API Endpoints

### `POST /api/v1/upload`
Uploads a PDF document, extracts text chunks, inserts vectors into ChromaDB, and updates the BM25 index.

* **Request**: `multipart/form-data` with `file: PDF`
* **Response**: `UploadResponse` JSON

### `POST /api/v1/query`
Executes dual-stage retrieval (Vector Search + BM25 -> RRF Fusion -> Cross-Encoder Re-Ranking) and streams NDJSON tokens.

* **Media Type**: `application/x-ndjson`
* **Request Body**:
```json
{
  "query": "What is the standard tax deduction threshold?",
  "top_k": 3,
  "chat_history": [
    {"role": "user", "content": "Tell me about personal tax deadlines."},
    {"role": "assistant", "content": "The federal filing deadline is April 15th."}
  ]
}
```

---

## 📜 License
MIT License. Created by [K. Srinivasa Rao](https://github.com/ksrinivasarao2012).
