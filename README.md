# simple-rag-python

A Python Retrieval-Augmented Generation (RAG) microservice for querying PDF documents using LLMs (Ollama or OpenAI).  
It provides endpoints for PDF ingestion and question answering over the ingested content.

---

## Features

- **PDF Ingestion:** Upload and embed PDF documents for semantic search.
- **RAG Query API:** Ask questions and get answers grounded in your documents, with source citations.
- **Supports Ollama and OpenAI:** Choose your LLM and embedding provider.
- **Metrics:** Prometheus-compatible metrics endpoint.
- **Observability:** Integrates with Langfuse for tracing and monitoring.

---

## Quickstart

### First time setup

```bash
docker compose up -d rag-api-ollama
docker compose exec rag-api-ollama ollama pull nomic-embed-text
docker compose exec rag-api-ollama ollama pull llama3:8b
```

### Run services

```bash
docker compose up
```

The API will run on port `8008`.  
Interactive docs: [http://localhost:8008/docs](http://localhost:8008/docs)

Verify Ollama is running:
```bash
curl http://localhost:11535/api/tags
```

---

## API Endpoints

### Health Check

- `GET /health`  
  Returns `"OK"` if the service is running.

### Ingest PDF

- `POST /ingest`  
  Repeat ingestion for default file or (optionally) Upload a PDF file for ingestion.  
  **Body:** `multipart/form-data` with a `.pdf` file.  
  **Response:** `"OK"`  
  Only one ingestion can run at a time.

### Query

- `POST /query`  
  Ask a question about the ingested PDF. (optional) specify the number of most similar results/documents to use, defaults to 5.
  
  **Body:**  
  ```json
  {
    "question": "What is Python?",
    "k": 5
  }
  ```
  **Response:**  
  ```json
  {
    "answer": "...",
    "sources": [
      {"text": "...", "page": "5"},
      ...
    ]
  }
  ```

---

## Configuration

Settings are managed via environment variables and `.env` file.  
See [`app/settings.py`](app/settings.py) for all options.

Key settings:
- **Provider:** `ollama` (default) or `openai`
- **PDF path:** Default PDF at `pdfs/thinkpython2.pdf`
- **Ollama/OpenAI models:** Set model names and API keys as needed
- **Langfuse:** Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in `.env`

You can copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

**Key variables:**

- `PROVIDER` — Choose `ollama` or `openai` for LLM and embedding provider.
- `ENABLE_METRICS` — Set to `true` to enable Prometheus metrics endpoint.
- `TEXT_SPLITTER__CHUNK_SIZE` — Number of characters per text chunk (default: 500).
- `TEXT_SPLITTER__CHUNK_OVERLAP` — Overlap between chunks (default: 50).
- `LANGFUSE__PUBLIC_KEY`, `LANGFUSE__SECRET_KEY`, `LANGFUSE__HOST` — [Langfuse](https://langfuse.com/) tracing credentials.
- `OLLAMA__HOST`, `OLLAMA__EMBEDDING_MODEL`, `OLLAMA__CHAT_MODEL`, `OLLAMA__TEMPERATURE` — Ollama settings.
- `OPENAI__API_KEY`, `OPENAI__EMBEDDING_MODEL`, `OPENAI__CHAT_MODEL`, `OPENAI__TEMPERATURE` — OpenAI settings.

**Example:**

```env
PROVIDER=ollama
ENABLE_METRICS=true

TEXT_SPLITTER__CHUNK_SIZE=500
TEXT_SPLITTER__CHUNK_OVERLAP=50

LANGFUSE__PUBLIC_KEY=your-public-key
LANGFUSE__SECRET_KEY=your-secret-key
LANGFUSE__HOST=https://us.cloud.langfuse.com

OLLAMA__HOST=http://localhost:11535
OLLAMA__EMBEDDING_MODEL=nomic-embed-text
OLLAMA__CHAT_MODEL=llama3:8b
OLLAMA__TEMPERATURE=0.5

OPENAI__API_KEY=your-openai-key
OPENAI__EMBEDDING_MODEL=text-embedding-ada-002
OPENAI__CHAT_MODEL=gpt-4
OPENAI__TEMPERATURE=0.5
```

See [`.env.example`](.env.example) for all available options.

---

## Limitations

- Multi-page tables may not be embedded correctly
- Images inside pages are ignored
- Extremely large PDFs (>1GB) may cause issues
- Redundant text (headers, footers, references) may affect results

---

## Project Structure

- [`app/api.py`](app/api.py): FastAPI endpoints
- [`app/ingestion.py`](app/ingestion.py): PDF loading, chunking, and embedding
- [`app/query.py`](app/query.py): Query logic and LLM integration
- [`app/settings.py`](app/settings.py): Configuration

---

## License

MIT