"""
Builds the vector index for the RAG project.

Reads PDFs, splits their text into overlapping chunks, embeds those chunks
locally, and persists them to a Chroma collection on disk. Run this script
directly to fully rebuild the index from every PDF in PDF_DIR. ingest_pdfs()
is also importable for adding new PDFs incrementally (e.g. from the
Streamlit uploader) without touching what's already indexed. This file does
not answer questions — that's chain.py's job.
"""

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
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


# Matches a research-paper section heading sitting alone on its own line,
# optionally preceded by outline numbering ("1.", "3.2"). Deliberately a
# short, specific allowlist rather than a general "looks like a heading"
# detector, since we have no font/layout info to lean on -- just plain text.
SECTION_HEADER_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(Abstract|Introduction|Related\s+Work|Background|Method(?:ology)?|Approach|"
    r"Experiments?|Results?|Discussion|Conclusion|References|Appendix)"
    r"[:.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _tag_sections(documents: list[Document]) -> list[Document]:
    """
    Split each page Document into smaller Documents at detected
    section-header boundaries, tagging each with a "section" metadata field
    (e.g. "Method", "Results"). Text before the first header found in a
    paper is tagged "unknown".

    Why this matters: without a section tag, every chunk from a paper looks
    equally authoritative to the retriever and the LLM -- a speculative
    aside from the Discussion and a description of what was actually done in
    the Method get cited identically. Knowing which section a chunk came
    from lets a reader (or the LLM) weigh a retrieved claim appropriately
    when critiquing a paper's methodology, and lets a citation point at
    *where in the paper* to go verify a claim, not just which PDF it's in.

    This is a best-effort heuristic, not real document parsing: it looks for
    section names sitting alone on a line in PyMuPDF's plain-text
    extraction, which has no font-size or layout information. It will miss
    unusually formatted headers and can't be 100% correct -- but an
    approximate section label is still far more useful for the two reasons
    above than no label at all.
    """
    tagged_documents = []
    current_section = None
    last_source = None

    for doc in documents:
        # A new source PDF starts fresh -- a paper shouldn't inherit
        # whatever section was still open at the end of the previous paper.
        source = doc.metadata.get("source")
        if source != last_source:
            current_section = None
            last_source = source

        matches = list(SECTION_HEADER_RE.finditer(doc.page_content))
        if not matches:
            tagged_documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "section": current_section or "unknown"},
                )
            )
            continue

        boundaries = [0] + [m.start() for m in matches] + [len(doc.page_content)]
        # segment 0 (before the first header on this page) still belongs to
        # whatever section was open when the previous page ended.
        section_labels = [current_section or "unknown"] + [
            m.group(1).title() for m in matches
        ]
        for start, end, section in zip(boundaries, boundaries[1:], section_labels):
            segment = doc.page_content[start:end]
            if not segment.strip():
                continue
            tagged_documents.append(
                Document(
                    page_content=segment,
                    metadata={**doc.metadata, "section": section},
                )
            )
        current_section = section_labels[-1]

    return tagged_documents


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

    # Pre-segment each page at detected section-header boundaries so every
    # chunk the splitter produces below inherits a "section" tag.
    documents = _tag_sections(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ordered from "most structurally meaningful" to "last resort": the
        # splitter always tries the first separator that lets a chunk fit
        # within chunk_size, falling back down the list only when it can't.
        # Paragraphs stay whole when they fit; failing that, sentences;
        # failing that, at least whole words. The final "" entry is a raw
        # character-level cut -- the worst case for technical text, since it
        # can slice an equation or a citation number in half -- so it's only
        # ever reached when nothing higher on the list fits.
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    # split_documents copies each source Document's metadata onto every
    # chunk it produces, so filename/page/section tracking survives the split.
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
    print(f"Rebuilding with {EMBEDDING_MODEL} -- this is a larger model than "
          "all-MiniLM-L6-v2, so embedding will take noticeably longer.")

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
