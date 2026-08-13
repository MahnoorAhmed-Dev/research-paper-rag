# Research Paper RAG

A local retrieval-augmented generation (RAG) project that answers questions over a small
collection (5–7) of research paper PDFs.

## Architecture

- **PDF parsing:** PyMuPDF
- **Chunking:** LangChain `RecursiveCharacterTextSplitter` (~600 tokens, ~100 overlap)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vector store:** Chroma (local, persisted)
- **LLM:** Groq API (`llama-3.3-70b-versatile`) via `langchain-groq`
- **Orchestration:** LangChain LCEL
- **UI:** Streamlit

## Status

Project scaffolding only — no RAG logic implemented yet.
