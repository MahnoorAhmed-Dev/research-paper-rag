from pathlib import Path

# Project root (one level up from this file), so paths work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing the source PDF files to ingest
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"

# Folder where the persisted Chroma vector store lives
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Target size (in characters) of each text chunk fed to the splitter.
# Larger chunks keep more surrounding context around equations and
# multi-sentence claims, at the cost of fewer, coarser retrieval hits.
CHUNK_SIZE = 1000

# Number of characters of overlap between consecutive chunks. Larger overlap
# lowers the odds that a key sentence gets orphaned right at a chunk boundary.
CHUNK_OVERLAP = 200

# HuggingFace model used to embed chunks and queries
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq-hosted LLM used to generate answers
GROQ_MODEL = "openai/gpt-oss-120b"

# Number of chunks the retriever returns per query
RETRIEVER_K = 5

# Name of the Chroma collection storing the paper embeddings
COLLECTION_NAME = "research_papers"
