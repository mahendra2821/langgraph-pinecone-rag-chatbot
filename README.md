# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a PDF document (e.g. an "Agentic AI" ebook), using only the content of that document — no outside knowledge, no hallucinated answers.

Built entirely with **free-tier tools**: local HuggingFace embeddings, Groq for the LLM, and Pinecone's free serverless tier for vector storage.

---

## How It Works

1. **Ingestion (`ingest.py`)** — Loads a PDF, splits it into overlapping text chunks, generates embeddings locally using a HuggingFace sentence-transformer model, and uploads the chunks + embeddings to a Pinecone vector index.
2. **Retrieval (`rag.py`)** — Given a user question, embeds the question and searches Pinecone for the most relevant chunks from the document.
3. **Generation (`llm.py`)** — Passes the retrieved chunks as context to a Groq-hosted LLM, which answers strictly using that context. If the answer isn't in the document, it says so instead of guessing.
4. **Orchestration (`graph.py`)** — Wires retrieval and generation together as a two-node LangGraph pipeline: `retrieve` → `generate`.
5. **API (`api.py`)** — Exposes the pipeline over HTTP via FastAPI, so any frontend or client can send a question and get back an answer with sources and a confidence score.

---

## Project Structure

```
agentic-ai/
├── .env                  # API keys and config (not committed to version control)
├── requirements.txt
├── README.md
├── data/
│   └── your_document.pdf
└── app/
    ├── __init__.py
    ├── config.py         # Loads environment variables
    ├── utils.py          # PDF loading helper (PyMuPDF)
    ├── ingest.py          # Chunking + embedding + Pinecone upload
    ├── rag.py            # Vector search / retrieval
    ├── llm.py             # Groq LLM + prompt template
    ├── graph.py           # LangGraph pipeline (retrieve -> generate)
    ├── test_graph.py      # Standalone test script
    └── api.py             # FastAPI app exposing /chat endpoint
```

---

## Setup

### 1. Clone and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file in the project root

```
OPENAI_API_KEY=            # not required anymore (kept optional for reference)
GROQ_API_KEY=your_groq_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=your_index_name
PDF_PATH=data/your_document.pdf
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

- Get a free Groq API key: https://console.groq.com/keys
- Get a free Pinecone API key: https://app.pinecone.io

### 3. Create your Pinecone index

Important: the index must be created as a **Dense**, custom vector index (not Pinecone's integrated/hosted embedding option), with:

- **Dimension:** `384` (matches `sentence-transformers/all-MiniLM-L6-v2`)
- **Metric:** `cosine`
- **Capacity mode:** Serverless → On-demand

You can create it via the Pinecone console, or with a short script:

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your_key")
pc.create_index(
    name="your_index_name",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

### 4. Ingest your PDF

```bash
python -m app.ingest
```

This loads the PDF specified in `PDF_PATH`, chunks it, embeds it, and uploads it to your Pinecone index.

### 5. Test the pipeline directly (optional)

```bash
python -m app.test_graph
```

### 6. Run the API

```bash
uvicorn app.api:app --reload
```

Then open **http://127.0.0.1:8000/docs** to test the `/chat` endpoint interactively.

---

## API Reference

### `GET /`
Health check.

**Response:**
```json
{ "message": "Agentic AI RAG chatbot running" }
```

### `POST /chat`
Ask a question about the ingested document.

**Request body:**
```json
{ "question": "What is Agentic AI?" }
```

**Response:**
```json
{
  "answer": "Agentic AI refers to ...",
  "contexts": ["...chunk 1...", "...chunk 2..."],
  "confidence": 0.82
}
```

If the answer isn't found in the document, `answer` will read:
> "I could not find this information in the provided Agentic AI ebook."

---

## Using a Different PDF

1. Update `PDF_PATH` in `.env` to point to the new file.
2. If you want to fully replace the old document (rather than mix both), clear the Pinecone index first:
   ```python
   from pinecone import Pinecone
   pc = Pinecone(api_key="your_key")
   index = pc.Index("your_index_name")
   index.delete(delete_all=True)
   ```
3. Re-run ingestion:
   ```bash
   python -m app.ingest
   ```

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF parsing | PyMuPDF |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| Vector database | Pinecone (serverless, free tier) |
| LLM | Groq `llama-3.3-70b-versatile` (free tier) |
| Orchestration | LangGraph |
| API | FastAPI + Uvicorn |

---

## Notes

- All components run on free tiers — no billing required.
- Embeddings run locally on your machine (first run downloads the ~80MB model from HuggingFace, then it's cached).
- The system is designed to avoid hallucination: if the answer isn't in the retrieved context, it explicitly says so rather than guessing.
