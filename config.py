import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

APP_NAME = "WorkMind"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "workmind.db"))
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", DATA_DIR / "vector-store"))
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "openai/text-embedding-3-small"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SEMANTIC_MIN_CHUNK_SIZE = int(os.getenv("SEMANTIC_MIN_CHUNK_SIZE", "250"))
SEMANTIC_BREAKPOINT_PERCENTILE = float(
    os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", "90")
)
SEMANTIC_MIN_BREAKPOINT_DISTANCE = float(
    os.getenv("SEMANTIC_MIN_BREAKPOINT_DISTANCE", "0.15")
)
SEMANTIC_CONTEXT_WINDOW = int(os.getenv("SEMANTIC_CONTEXT_WINDOW", "1"))
TOP_K = int(os.getenv("TOP_K", "6"))
