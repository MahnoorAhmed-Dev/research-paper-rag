"""
Streamlit UI for the RAG project: upload PDFs, index them, and chat with
them. Retrieval/generation logic lives in src/chain.py; indexing logic
lives in src/ingest.py — this file only wires them to the UI.
"""

import streamlit as st

from src.chain import get_answer
from src.ingest import ingest_pdfs
from src.config import PDF_DIR

st.set_page_config(page_title="Research Paper RAG", page_icon="📄")
st.title("📄 Research Paper RAG")
st.caption("Ask questions about your indexed research papers and get answers cited back to the source PDFs.")

with st.sidebar:
    st.header("Indexed Papers")
    indexed_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if indexed_pdfs:
        for pdf_path in indexed_pdfs:
            st.write(f"- {pdf_path.name}")
    else:
        st.write("No papers indexed yet.")

    st.divider()
    st.header("Add Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if st.button("Add to index", disabled=not uploaded_files):
        # Skip files that are already on disk, so re-clicking the button
        # (or a rerun where the uploader still holds old selections) doesn't
        # re-save and re-embed the same PDF.
        new_files = [f for f in uploaded_files if not (PDF_DIR / f.name).exists()]

        if not new_files:
            st.info("No new files to add — these have already been indexed.")
        else:
            saved_paths = []
            for uploaded_file in new_files:
                dest_path = PDF_DIR / uploaded_file.name
                dest_path.write_bytes(uploaded_file.getvalue())
                saved_paths.append(dest_path)

            with st.spinner("Embedding and indexing..."):
                added = ingest_pdfs(saved_paths)

            st.success(f"Added {added} chunks from {len(saved_paths)} file(s).")
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        for source in message.get("sources") or []:
            st.caption(f"{source['filename']} - page {source['page']}")

if not any(PDF_DIR.glob("*.pdf")):
    st.info("No papers indexed yet. Upload a PDF from the sidebar to get started.")
else:
    question = st.chat_input("Ask a question about your papers...")
    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question, "sources": []}
        )
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = get_answer(question)
                except Exception as e:
                    result = None
                    st.error(f"Couldn't get an answer: {e}")

            if result is not None:
                st.write(result["answer"])
                for source in result["sources"]:
                    st.caption(f"{source['filename']} - page {source['page']}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"] if result else "Sorry, something went wrong answering that question.",
                "sources": result["sources"] if result else [],
            }
        )
