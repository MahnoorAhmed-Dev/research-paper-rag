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

# HuggingFace model used to embed chunks and queries. bge-base-en-v1.5 was
# chosen over the earlier all-MiniLM-L6-v2 because it performs noticeably
# better on technical/academic text (it's trained with retrieval-focused
# contrastive objectives on a broader, more formal text mix) while still
# being free and runnable locally. IMPORTANT: embeddings from different
# models live in different vector spaces and are not comparable to each
# other, so changing this value requires a full rebuild_index() -- an
# incremental ingest_pdfs() call would mix old and new embeddings in the
# same collection and silently corrupt retrieval.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# BGE models are trained asymmetrically: they expect queries (but not the
# passages/chunks being searched) to be prefixed with this instruction text
# for best retrieval accuracy. Applied only in chain.py's retrieval step.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Groq-hosted LLM used to generate answers
GROQ_MODEL = "openai/gpt-oss-120b"

# Number of chunks the retriever returns per query
RETRIEVER_K = 5

# Name of the Chroma collection storing the paper embeddings
COLLECTION_NAME = "research_papers"
