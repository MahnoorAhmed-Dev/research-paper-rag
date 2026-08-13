"""
Builds the vector index for the RAG project.

Reads PDFs, splits their text into overlapping chunks, embeds those chunks
locally, and persists them to a Chroma collection on disk. Run this script
directly to fully rebuild the index from every PDF in PDF_DIR. ingest_pdfs()
is also importable for adding new PDFs incrementally (e.g. from the
Streamlit uploader) without touching what's already indexed. This file does
not answer questions — that's chain.py's job.
"""

import sys
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    PDF_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
)


def load_pdfs(pdf_paths):
    """Extract text from the given PDFs, one Document per page."""
    documents = []
    for pdf_path in pdf_paths:
        pdf = fitz.open(pdf_path)
        for page_number, page in enumerate(pdf, start=1):
            # Keep filename + page number as metadata so later answers can
            # cite exactly where a chunk came from.
            documents.append(
                Document(
                    page_content=page.get_text(),
                    metadata={"source": pdf_path.name, "page": page_number},
                )
            )
        pdf.close()

    return documents


def _load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_pdfs(pdf_paths: list[Path]) -> int:
    """
    Load, chunk, embed, and add the given PDFs to the existing Chroma
    collection at CHROMA_DIR, leaving whatever is already indexed in place.

    Use this for incremental additions, e.g. new PDFs uploaded through the
    Streamlit app. For a full rebuild from PDF_DIR, use rebuild_index().
    Returns the number of chunks added.
    """
    documents = load_pdfs(pdf_paths)
    print(f"Loaded {len(pdf_paths)} PDFs, {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    # split_documents copies each source Document's metadata onto every
    # chunk it produces, so filename/page tracking survives the split.
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding and storing in Chroma...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_load_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.add_documents(chunks)

    print(f"Done. Persisted to {CHROMA_DIR}/")
    return len(chunks)


def rebuild_index() -> int:
    """
    Wipe the existing Chroma collection and rebuild it from scratch using
    every PDF currently in PDF_DIR.

    Used by the CLI entry point below for full rebuilds; incremental
    additions should go through ingest_pdfs() instead. Returns the number
    of chunks the rebuilt index contains.
    """
    # Drop any existing collection with this name first so the rebuild
    # doesn't append duplicate chunks on top of the old index.
    Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_load_embeddings(),
        persist_directory=str(CHROMA_DIR),
    ).delete_collection()

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    return ingest_pdfs(pdf_paths)


def main():
    if not any(PDF_DIR.glob("*.pdf")):
        print(f"No PDFs found in {PDF_DIR}. Add some PDF files there and re-run this script.")
        sys.exit(1)

    rebuild_index()


if __name__ == "__main__":
    main()
