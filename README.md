# RAG App — PDF Question Answering

A Retrieval-Augmented Generation (RAG) system that lets you ask questions
about the content of a PDF document and get accurate, source-grounded
answers.

## Features

- **PDF ingestion** — upload a PDF, it gets chunked and embedded automatically
- **Semantic search** — retrieves the most relevant chunks from a vector database before answering
- **Grounded answers** — the LLM answers strictly from retrieved context and returns its sources
- **Durable background processing** — ingestion and querying run as retryable, observable background steps (not blocking HTTP requests)
- **Rate/throttle control** — protects against upstream provider rate limits

## Tech Stack

- Python
- FastAPI
- LlamaIndex
- Qdrant
- Inngest
- OpenRouter
- Pydantic
- uv

## Prerequisites

- Python 3.13+
- uv
- Docker(for running Qdrant locally)
- Node.js (for the Inngest dev server via `npx`)
- An OpenRouter API key

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/FaezehGharari/RAG_App.git
cd RAG_App
uv sync
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
```

### 3. Start Qdrant (vector database)

```bash
docker run -d --name qdrantRAGDb -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 4. Start the Inngest dev server

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### 5. Run the app

```bash
uv run uvicorn main:app
```

- API: `http://127.0.0.1:8000`
- Inngest dashboard: `http://127.0.0.1:8288`
- Qdrant dashboard: `http://127.0.0.1:6333/dashboard`