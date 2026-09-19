"""
Connects retrieval (Chroma) with generation (Groq) via an LCEL chain.

Loads the vector index that ingest.py already built and exposes two entry
points:

- get_answer(question) -- single-pass retrieval: one global top-k search
  across the whole collection, re-ranked down to RETRIEVER_K chunks. Best
  for specific, narrow questions about a particular paper or concept, where
  the most relevant chunks legitimately cluster in one or two papers and
  pulling from the rest would just add noise.

- get_answer_multi_doc(question) -- multi-document retrieval: guarantees
  every indexed paper gets a chance to contribute a few chunks before they're
  all merged and re-ranked down to a wider final set. Best for broad,
  comparative questions -- "how do these papers differ in X", methodology
  gap analysis, novelty checks -- where a single global top-k search could
  easily return results from just the one or two most similar papers and
  silently ignore the rest, defeating the point of a cross-paper comparison.

Both cite sources the same way. This file does not build or modify the
index — run ingest.py first.
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    MULTI_DOC_CHUNKS_PER_PAPER,
    MULTI_DOC_FINAL_K,
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


def _rerank(question: str, candidates: list, top_k: int = RETRIEVER_K) -> list:
    """
    Re-score the candidate pool with a cross-encoder and keep the top top_k.

    Embedding similarity (used for the initial Chroma retrieval) is a fast
    approximation: the query and each chunk are embedded independently, then
    compared by vector distance -- the model never actually looks at the
    query and the chunk together. A cross-encoder scores each pair jointly,
    which is far more accurate, but too slow to run against the whole
    collection. So the pipeline retrieves broad (RETRIEVAL_CANDIDATES chunks
    via cheap vector search, or MULTI_DOC_CHUNKS_PER_PAPER per paper in
    get_answer_multi_doc), then reranks narrow via the more expensive but
    more precise cross-encoder. top_k defaults to RETRIEVER_K for the
    single-document path; get_answer_multi_doc passes MULTI_DOC_FINAL_K.
    """
    reranker = _get_reranker()
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# Citation format is enforced strictly (exact pattern + worked examples,
# not just a description) because this project is meant to support
# methodology critique of research papers -- a claim that can't be traced
# back to a specific chunk is worse than useless there, since it looks
# authoritative while being unverifiable. Earlier, looser phrasing ("cite
# sources inline") let the model improvise its own bracket style
# (observed: 【】 instead of a consistent format) and blur the line between
# what a paper actually says and what the model inferred from it -- both are
# fine to include, but only if they're clearly distinguishable and checkable
# against the source.
#
# Pulled out as its own constant (rather than inlined in PROMPT below) so
# src/critique.py's CRITIQUE_PROMPT can reuse the exact same citation and
# evidence-labeling rules instead of re-describing them and risking drift
# between the two prompts.
CITATION_AND_EVIDENCE_RULES = """CITATION FORMAT (required, exact):
Every sentence that makes a claim must end with a citation in this exact
pattern: (Source: filename, Section section, p. X). If the context label
for that chunk has no section (or it says "unknown"), drop the section
clause instead: (Source: filename, p. X). Do not use any other bracket
style (no [...], no 【...】) and do not invent a section name that wasn't in
the context label.

Example: "GATs assign a different attention weight to each neighbor
(Source: graph_attention_networks.pdf, Method section, p. 4)."

DIRECT EVIDENCE vs. INFERENCE -- label every claim as one or the other:
- Direct evidence: something a paper explicitly states. Paraphrase it
  closely in your own words (quote verbatim for no more than a few words at
  a time) and cite it normally, as above.
  Example: "The authors report a 3.5% accuracy gain over the prior
  state-of-the-art router (Source: MasRouter.pdf, Experiments section, p. 6)."
- Inference: a conclusion that follows from the context but that no single
  chunk states in those words. Prefix the sentence with "Inference:" and
  still cite the chunk(s) it's drawn from.
  Example: "Inference: because both systems rely on a fixed adjacency
  matrix, neither can adapt its communication structure to a specific query
  (Source: comms.pdf, Introduction section, p. 2; Source: MasRouter.pdf,
  Related Work section, p. 3)."
"""

PROMPT = ChatPromptTemplate.from_template(
    f"""You are answering a question about research papers for someone who
needs to verify every claim against its source. Use ONLY the context below
-- no outside knowledge, and no claim without a citation to back it up.

{CITATION_AND_EVIDENCE_RULES}
STRICT RULE -- do not violate this: if a claim is not directly supported by
at least one retrieved chunk, do not make that claim, even if it seems true,
likely, or common knowledge. Every sentence in your answer must trace back
to a citation. If the context as a whole does not contain enough
information to answer the question, do not guess or partially answer --
respond with exactly this and nothing else:
"I don't have enough information in the provided documents to answer that."

Context:
{{context}}

Question: {{question}}

Answer:"""
)

llm = ChatGroq(model=GROQ_MODEL, temperature=0)


def format_docs(docs):
    """Render retrieved chunks as labeled context text for the prompt.

    No leading underscore: this is reused by src/critique.py so the
    critique prompt's context is formatted identically to get_answer()'s.
    """
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


def _connect_vectorstore() -> Chroma:
    """
    Connect to the Chroma collection fresh (not cached at import time) and
    verify it's populated.

    Built fresh on every call rather than cached at import time so this
    keeps working even if the collection was deleted and rebuilt (e.g. via
    ingest.py) after this module was first imported — a cached reference
    would otherwise point at the old, now-deleted collection.
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
    return vectorstore


def _generate_answer(question: str, source_documents: list) -> dict:
    """
    Shared generation step for both retrieval modes: format the given
    chunks as context, ask the LLM using the shared PROMPT, and build the
    sources list. Retrieval strategy (single global search vs. per-paper
    multi-document search) is decided by the caller; this just turns
    whatever chunks it's handed into a cited answer, so get_answer() and
    get_answer_multi_doc() stay identical in prompt/citation behavior.
    """
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {"question": question, "context": format_docs(source_documents)}
    )
    sources = [
        {
            "filename": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "section": doc.metadata.get("section"),
        }
        for doc in source_documents
    ]
    return {"answer": answer, "sources": sources}


def get_answer(question: str) -> dict:
    """Retrieve relevant chunks for `question` and generate a cited answer.

    Single-pass retrieval: one global top-k search across the whole
    collection. See this module's docstring for when to prefer
    get_answer_multi_doc() instead.
    """
    vectorstore = _connect_vectorstore()

    # Retrieve a wide candidate pool from Chroma (cheap, embedding-based);
    # _rerank() below narrows it down to RETRIEVER_K with the more accurate
    # but slower cross-encoder.
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_CANDIDATES})

    # BGE's asymmetric training means the query needs the search instruction
    # prefix to embed into the same "intent space" the document chunks were
    # embedded into, but the chunks themselves were embedded plain (see
    # ingest.py) -- prefixing both sides would cancel the effect the prefix
    # is meant to have.
    candidates = retriever.invoke(BGE_QUERY_PREFIX + question)
    source_documents = _rerank(question, candidates)

    return _generate_answer(question, source_documents)


def _list_indexed_sources(vectorstore: Chroma) -> list[str]:
    """
    Return the distinct source filenames actually present in the Chroma
    collection.

    Reads this from the collection's own metadata rather than PDF_DIR: the
    collection is the source of truth for what's actually searchable right
    now, whereas PDF_DIR can drift out of sync with it (a file dropped into
    PDF_DIR but not yet run through ingest_pdfs(), or removed from disk
    after being indexed). This also follows the same established pattern
    _connect_vectorstore() already uses -- reaching into
    vectorstore._collection directly, since langchain_chroma doesn't expose
    a public method for either of these lookups.
    """
    metadatas = vectorstore._collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metadatas if m.get("source")})


def retrieve_multi_doc(question: str, final_k: int = MULTI_DOC_FINAL_K) -> list:
    """
    Retrieve candidate chunks across every indexed paper for `question`:
    pull MULTI_DOC_CHUNKS_PER_PAPER chunks from each paper individually (so
    every paper gets a chance to contribute, unlike a single global search),
    merge them into one pool, and re-rank down to `final_k`.

    Pure retrieval, no generation -- kept separate from get_answer_multi_doc()
    below so other callers can reuse the same retrieval strategy with a
    different prompt. No leading underscore: this is reused by
    src/critique.py, which needs this module's multi-document evidence
    gathering but generates with its own CRITIQUE_PROMPT instead of PROMPT.
    """
    vectorstore = _connect_vectorstore()
    filenames = _list_indexed_sources(vectorstore)

    candidates = []
    for filename in filenames:
        candidates.extend(
            vectorstore.similarity_search(
                BGE_QUERY_PREFIX + question,
                k=MULTI_DOC_CHUNKS_PER_PAPER,
                filter={"source": filename},
            )
        )

    return _rerank(question, candidates, top_k=final_k)


def get_answer_multi_doc(question: str) -> dict:
    """Retrieve relevant chunks for `question` across every indexed paper
    and generate a cited answer.

    Multi-document retrieval: see retrieve_multi_doc() above for how the
    candidate pool is built. See this module's docstring for when to prefer
    this over get_answer().
    """
    source_documents = retrieve_multi_doc(question)
    return _generate_answer(question, source_documents)
