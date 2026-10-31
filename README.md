# ai-chat-with-your-files

Chat with your own documents (PDF, DOCX, HTML) over a Retrieval-Augmented Generation (RAG) pipeline: upload files, they get chunked and embedded locally, stored in Pinecone, and a Groq-hosted LLM answers questions grounded in that content with page citations.

## How it works

1. **Upload** — `doc_chat.py` serves a Gradio UI where you upload one or more `.pdf`, `.docx`, or `.html`/`.htm` files.
2. **Chunk & embed** — `chunk_buildIndex.py` loads each file with the matching LangChain loader, splits it into overlapping chunks, embeds them locally with a HuggingFace sentence-transformer (`all-MiniLM-L6-v2`, 384 dimensions), and upserts them into a Pinecone index.
3. **Ask** — `ask_llm_with_query.py` retrieves the most relevant chunks for your question and sends them to a Groq-hosted LLM (`openai/gpt-oss-120b`) to produce an answer with page-number sources.

| File | Responsibility |
|---|---|
| `InitializePinecone.py` | One-time Pinecone index creation (384-dim, cosine, AWS serverless) |
| `chunk_buildIndex.py` | File loading, chunking, embedding, and upsert into Pinecone (`ingest_file`) |
| `ask_llm_with_query.py` | Similarity search + LLM answer generation (`ask`) |
| `doc_chat.py` | Gradio app wiring upload and chat together |

## Prerequisites

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management
- A [Pinecone](https://www.pinecone.io/) account and API key
- A [Groq](https://console.groq.com/) account and API key

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with your own keys:

```
PINE_CONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
```

`.env` is gitignored — never commit real keys. If a real key ever ends up in a tracked file (including `.env.example`), rotate it immediately.

Create the Pinecone index (only needed once, or after deleting it):

```bash
uv run python InitializePinecone.py
```

## Run

```bash
uv run python doc_chat.py
```

This launches a Gradio app (with a public share link, since `demo.launch(share=True)`) where you can upload documents and chat about them.

## Notes

- Pinecone index dimension must match the embedding model's output dimension. This project uses `all-MiniLM-L6-v2` (384 dims) — if you switch embedding models, delete and recreate the index with the matching `dimension` in `InitializePinecone.py`.
- Each uploaded file gets a unique `doc_id` derived from its filename, so uploading multiple files indexes them independently instead of overwriting each other.
- Page citations are only meaningful for paginated formats (PDF); DOCX and HTML sources default to "page 1" since they aren't paginated.
