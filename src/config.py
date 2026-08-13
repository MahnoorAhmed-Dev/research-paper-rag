from pathlib import Path

# Project root (one level up from this file), so paths work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing the source PDF files to ingest
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"

# Folder where the persisted Chroma vector store lives
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Target size (in tokens) of each text chunk fed to the splitter
CHUNK_SIZE = 600

# Number of tokens of overlap between consecutive chunks
CHUNK_OVERLAP = 100

# HuggingFace model used to embed chunks and queries
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq-hosted LLM used to generate answers
GROQ_MODEL = "llama-3.3-70b-versatile"

# Number of chunks the retriever returns per query
RETRIEVER_K = 5

# Name of the Chroma collection storing the paper embeddings
COLLECTION_NAME = "research_papers"
