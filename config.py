import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstores"

# Create directories
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Default settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
DEFAULT_TOP_K = 5

# Supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx']

# Chunking strategies
CHUNKING_STRATEGIES = {
    "Fixed Length": "fixed_length",
    "Overlapping (Sliding Window)": "overlapping", 
    "Recursive Text Splitter": "recursive",
    "Semantic Chunking": "semantic",
    "Sentence Based": "sentence",
    "Paragraph Based": "paragraph",
    "Heading Based": "heading",
    "Topic Based": "topic",
    "Dynamic AI Chunking": "dynamic_ai"
}

# Retriever types
RETRIEVER_TYPES = {
    "BM25 (Keyword)": "bm25",
    "FAISS (Dense)": "faiss",
    "Chroma (Dense)": "chroma",
    "Hybrid (BM25 + Dense)": "hybrid",
    "MMR (Diverse)": "mmr",
    "Reranking": "reranking",
    "Multi-Vector": "multi_vector"
}

# API Keys (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
