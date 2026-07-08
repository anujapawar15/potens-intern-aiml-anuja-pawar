"""
Central configuration, loaded from environment variables / .env.
Keeping every tunable in one place makes the chunking/retrieval trade-offs
easy to find and change without touching pipeline code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
RERANK_MULTIPLIER = int(os.getenv("RERANK_MULTIPLIER", 4))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.45))


def _resolve(raw: str) -> str:
    p = Path(raw)
    return str(p if p.is_absolute() else (BASE_DIR / p).resolve())


CHROMA_DIR = _resolve(os.getenv("CHROMA_DIR", "./chroma_db"))
PDF_DIR = _resolve(os.getenv("PDF_DIR", "./data/pdfs"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
