# WorkMind

A personal knowledge-base assistant built with Python, Streamlit, OpenRouter API, ChromaDB, and OpenAI Text Embeddings.

WorkMind indexes your code and text documents into a vector store, then answers questions using retrieved context from your own files.

## Tech Stack

| Layer | Technology |
| --- | --- |
| **App & UI** | Python, Streamlit |
| **LLM** | OpenRouter API (`openai/gpt-4o-mini`) |
| **Embeddings** | OpenRouter API (`openai/text-embedding-3-small`) |
| **Vector Store** | ChromaDB |
| **Metadata DB** | SQLite |
| **Chunking Engine** | Custom Semantic Chunker (`chunking.py`) |

## Setup

1. Create a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install lightweight dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment variables.

```bash
copy .env.example .env
```

Edit `.env` and set your OpenRouter API key:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
```

4. Run the app.

```bash
streamlit run app.py
```

Open the local URL Streamlit prints in your terminal.

## Usage

1. **Dashboard**: Create projects, upload files or index local folders.
2. **Multi-File Deletion**: Select individual or multiple files with checkboxes to batch delete from DB & Vector Store.
3. **Assistant**: Ask questions about your indexed codebase & documents.
4. **Search**: Run direct semantic searches to inspect retrieved vector matches.

## Configuration

All settings live in `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | empty | Required API key for LLM and embeddings |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | OpenRouter chat model ID |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | OpenRouter embedding model ID |
| `DB_PATH` | `data/workmind.db` | SQLite metadata database path |
| `VECTOR_DB_DIR` | `data/vector-store` | Persistent ChromaDB directory |
| `CHUNK_SIZE` | `800` | Hard character limit for each semantic chunk |
| `CHUNK_OVERLAP` | `150` | Context carried over from previous chunk |
| `SEMANTIC_MIN_CHUNK_SIZE` | `150` | Minimum size before semantic boundary split |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | `80` | Distance percentile for topic splits |
| `SEMANTIC_MIN_BREAKPOINT_DISTANCE` | `0.10` | Minimum distance for semantic split |
| `SEMANTIC_CONTEXT_WINDOW` | `1` | Neighboring units for context comparison |
| `TOP_K` | `4` | Retrieved context chunks per question |

## Semantic Chunking

WorkMind uses custom sentence and line-aware semantic chunking (`chunking.py`). It embeds candidate text units, measures adjacent vector cosine distances, and starts new chunks where meaning shifts significantly.

## Project Architecture

- **`app.py`**: Streamlit web dashboard & multi-view UI.
- **`chunking.py`**: Pure semantic text chunking algorithm.
- **`embeddings.py`**: OpenRouter embeddings client with auto-retries and timeout safety.
- **`vector_store.py`**: ChromaDB vector store manager.
- **`database.py`**: SQLite metadata & project file database.
- **`text_utils.py`**: Text file IO & timestamp utilities.
