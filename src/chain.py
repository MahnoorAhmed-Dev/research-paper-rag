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

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    RETRIEVER_K,
    COLLECTION_NAME,
)

# Load GROQ_API_KEY (and any other vars) from .env into the environment.
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

# langchain_chroma doesn't expose a public "is this populated?" check, so we
# drop to the underlying chromadb collection to count stored chunks.
if vectorstore._collection.count() == 0:
    raise RuntimeError(
        f"Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR} is missing or "
        "empty. Run `python src/ingest.py` first to build the index."
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. Do not use any
outside knowledge. If the context does not contain enough information to
answer, respond exactly with:
"I don't have enough information in the provided documents to answer that."

Cite sources inline after each claim, using the format (filename, p. page).

Context:
{context}

Question: {question}

Answer:"""
)

llm = ChatGroq(model=GROQ_MODEL, temperature=0)


def _format_docs(docs):
    """Render retrieved chunks as labeled context text for the prompt."""
    return "\n\n".join(
        f"[{doc.metadata.get('source')}, p. {doc.metadata.get('page')}]\n{doc.page_content}"
        for doc in docs
    )


# RunnablePassthrough.assign carries the retrieved documents forward
# alongside the generated answer, so get_answer() can still report sources
# after the output parser has reduced the LLM response to plain text.
rag_chain = (
    RunnablePassthrough.assign(
        source_documents=(lambda x: x["question"]) | retriever
    )
    | RunnablePassthrough.assign(
        context=lambda x: _format_docs(x["source_documents"])
    )
    | RunnablePassthrough.assign(answer=PROMPT | llm | StrOutputParser())
)


def get_answer(question: str) -> dict:
    """Retrieve relevant chunks for `question` and generate a cited answer."""
    result = rag_chain.invoke({"question": question})
    sources = [
        {"filename": doc.metadata.get("source"), "page": doc.metadata.get("page")}
        for doc in result["source_documents"]
    ]
    return {"answer": result["answer"], "sources": sources}
