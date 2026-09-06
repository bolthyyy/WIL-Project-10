from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MANUAL_DIR = DATA_DIR / "manuals"
MANUAL_METADATA_FILE = DATA_DIR / "manuals.json"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Vector database
COLLECTION_NAME = "rental_fleet_manuals"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"

# Baseline RAG settings
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
TOP_K = 5