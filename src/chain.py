"""
Connects retrieval (Chroma) with generation (Groq) via an LCEL chain.

Loads the vector index that ingest.py already built and exposes get_answer(),
which retrieves relevant chunks for a question and asks the LLM to answer
using only those chunks, citing sources. This file does not build or modify
the index — run ingest.py first.
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    BGE_QUERY_PREFIX,
    GROQ_MODEL,
    RERANK_MODEL,
    RETRIEVAL_CANDIDATES,
    RETRIEVER_K,
    COLLECTION_NAME,
)

# Load GROQ_API_KEY (and any other vars) from .env into the environment.
load_dotenv()

_embeddings = None


def _get_embeddings():
    """
    Lazily load and cache the embedding model.

    Loading it is the expensive part of connecting to Chroma, but unlike
    the vectorstore/retriever it doesn't go stale when the collection is
    rebuilt, so it's safe to reuse across calls.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


_reranker = None


def _get_reranker():
    """
    Lazily load and cache the cross-encoder re-ranking model.

    Same rationale as _get_embeddings(): loading the model is the slow,
    one-time cost, but scoring a handful of (question, chunk) pairs with an
    already-loaded cross-encoder is cheap, so it's safe to reuse across calls.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _rerank(question: str, candidates: list) -> list:
    """
    Re-score the candidate pool with a cross-encoder and keep the top
    RETRIEVER_K.

    Embedding similarity (used for the initial Chroma retrieval) is a fast
    approximation: the query and each chunk are embedded independently, then
    compared by vector distance -- the model never actually looks at the
    query and the chunk together. A cross-encoder scores each pair jointly,
    which is far more accurate, but too slow to run against the whole
    collection. So the pipeline retrieves broad (RETRIEVAL_CANDIDATES chunks
    via cheap vector search), then reranks narrow (down to RETRIEVER_K via
    the more expensive but more precise cross-encoder).
    """
    reranker = _get_reranker()
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:RETRIEVER_K]]


PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. Do not use any
outside knowledge. If the context does not contain enough information to
answer, respond exactly with:
"I don't have enough information in the provided documents to answer that."

Cite sources inline after each claim, using the format (filename, p. page).
If a context label also names a section (e.g. "[paper.pdf, Method section,
p. 4]"), include it: (filename, Method section, p. page).

Context:
{context}

Question: {question}

Answer:"""
)

llm = ChatGroq(model=GROQ_MODEL, temperature=0)


def _format_docs(docs):
    """Render retrieved chunks as labeled context text for the prompt."""
    labeled = []
    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        section = doc.metadata.get("section")
        # "section" is only present for chunks ingested after section-aware
        # chunking was added (see ingest.py's _tag_sections); older chunks
        # and the "unknown" tag both fall back to the plain filename/page label.
        if section and section != "unknown":
            label = f"[{source}, {section} section, p. {page}]"
        else:
            label = f"[{source}, p. {page}]"
        labeled.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(labeled)


def get_answer(question: str) -> dict:
    """Retrieve relevant chunks for `question` and generate a cited answer.

    The vectorstore/retriever/chain are built fresh on every call instead
    of being cached at import time, so this keeps working even if the
    Chroma collection was deleted and rebuilt (e.g. via ingest.py) after
    this module was first imported — a cached retriever would otherwise
    hold a stale reference to the old, now-deleted collection.
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    # langchain_chroma doesn't expose a public "is this populated?" check,
    # so we drop to the underlying chromadb collection to count chunks.
    if vectorstore._collection.count() == 0:
        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR} is missing or "
            "empty. Run `python src/ingest.py` first to build the index."
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_CANDIDATES})

    # RunnablePassthrough.assign carries the retrieved documents forward
    # alongside the generated answer, so we can still report sources after
    # the output parser has reduced the LLM response to plain text.
    rag_chain = (
        RunnablePassthrough.assign(
            # BGE's asymmetric training means the query needs the search
            # instruction prefix to embed into the same "intent space" the
            # document chunks were embedded into, but the chunks themselves
            # were embedded plain (see ingest.py) -- prefixing both sides
            # would cancel the effect the prefix is meant to have. The
            # cross-encoder re-ranking step, in contrast, uses the raw
            # question -- it wasn't trained on that instruction format.
            source_documents=lambda x: _rerank(
                x["question"],
                retriever.invoke(BGE_QUERY_PREFIX + x["question"]),
            )
        )
        | RunnablePassthrough.assign(
            context=lambda x: _format_docs(x["source_documents"])
        )
        | RunnablePassthrough.assign(answer=PROMPT | llm | StrOutputParser())
    )

    result = rag_chain.invoke({"question": question})
    sources = [
        {
            "filename": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "section": doc.metadata.get("section"),
        }
        for doc in result["source_documents"]
    ]
    return {"answer": result["answer"], "sources": sources}
