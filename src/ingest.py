"""
Builds the vector index for the RAG project.

Reads every PDF in PDF_DIR, splits the text into overlapping chunks, embeds
those chunks locally, and persists them to a Chroma collection on disk. Run
this script once up front (and again whenever the PDFs change) to (re)build
the index. It does not answer questions — that's chain.py's job.
"""

import sys

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


def load_pdfs():
    """Extract text from every PDF in PDF_DIR, one Document per page."""
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))

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

    return pdf_paths, documents


def main():
    if not any(PDF_DIR.glob("*.pdf")):
        print(f"No PDFs found in {PDF_DIR}. Add some PDF files there and re-run this script.")
        sys.exit(1)

    pdf_paths, documents = load_pdfs()
    print(f"Loaded {len(pdf_paths)} PDFs, {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    # split_documents copies each source Document's metadata onto every
    # chunk it produces, so filename/page tracking survives the split.
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding and storing in Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Drop any existing collection with this name first so re-running the
    # script rebuilds the index instead of appending duplicate chunks.
    Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    ).delete_collection()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Done. Persisted to {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
