"""
Methodology critique mode.

Takes a full methodology document (e.g. a project plan or paper draft,
pasted as plain text), breaks it into sections, and critiques each section
against the paper corpus already indexed in Chroma -- looking for gaps,
contradictions with established approaches, and missed opportunities.

This is separate from src/chain.py's question-answering responsibilities:
chain.py answers a user's question against the corpus, while this module
takes a document as input and generates a structured critique of it,
section by section. Both share the same retrieval discipline
(retrieve_multi_doc(), so every paper in the corpus gets a chance to
contribute evidence) and the same citation/evidence-labeling rules
(CITATION_AND_EVIDENCE_RULES), reused from chain.py rather than
re-implemented here, so a citation means the same thing everywhere in the
project.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.chain import (
    llm,
    format_docs,
    retrieve_multi_doc,
    CITATION_AND_EVIDENCE_RULES,
)
from src.ingest import SECTION_HEADER_RE


def split_into_sections(document_text: str) -> list[dict]:
    """
    Split a pasted methodology document into sections, using the same
    header-detection heuristic ingest.py uses for PDFs (SECTION_HEADER_RE)
    -- imported from there rather than redefined here, so "what counts as a
    section header" stays consistent across ingestion and critique.

    Applied here to plain pasted text rather than a PyMuPDF-extracted PDF
    page, so the matching is inherently a bit more flexible than at
    ingestion time: a pasted document might use different line breaks or
    spacing than a PDF extraction would, but the same "header sits alone on
    its own line" pattern still applies reasonably well to plain text.

    Returns a list of {"section_name": str, "text": str}, in the order
    sections appear in the document. If no headers are detected at all, the
    whole document is returned as a single section named "Full Document"
    rather than failing -- a pasted methodology write-up won't necessarily
    follow academic section-heading conventions.
    """
    if not document_text or not document_text.strip():
        raise ValueError("document_text is empty -- nothing to critique.")

    matches = list(SECTION_HEADER_RE.finditer(document_text))
    if not matches:
        return [{"section_name": "Full Document", "text": document_text.strip()}]

    boundaries = [0] + [m.start() for m in matches] + [len(document_text)]
    # Text before the first detected header (if any) has no header of its
    # own; keep it as a leading "Preamble" section instead of discarding it.
    names = ["Preamble"] + [m.group(1).title() for m in matches]

    sections = []
    for start, end, name in zip(boundaries, boundaries[1:], names):
        text = document_text[start:end].strip()
        if not text:
            # Empty preamble (a header right at the start of the document)
            # or two headers back-to-back with nothing between them.
            continue
        sections.append({"section_name": name, "text": text})

    return sections


# Distinct from chain.py's PROMPT: that one answers a question from
# retrieved evidence, this one critiques a piece of methodology against
# retrieved evidence. They share CITATION_AND_EVIDENCE_RULES so citations
# mean the same thing in both, but the task and required structure differ
# enough (three critique categories, section-by-section) to warrant a
# separate template rather than overloading one prompt for both jobs.
CRITIQUE_PROMPT = ChatPromptTemplate.from_template(
    f"""You are critiquing the methodology of a paper or project plan against
a corpus of published research. You are given one section of the
methodology document and evidence chunks retrieved from that corpus. Use
ONLY the retrieved evidence below -- no outside knowledge -- to identify:

1. GAPS: things this section doesn't address that the retrieved literature
   suggests it should.
2. CONTRADICTIONS: places where this section conflicts with established
   approaches described in the retrieved papers.
3. MISSED OPPORTUNITIES: techniques or considerations from the retrieved
   papers that could strengthen this section.

Organize your critique under exactly these three headings, in this order.
If you have nothing substantive to say under a heading, write "None
identified from the retrieved evidence" under it rather than inventing a
point just to fill it.

{CITATION_AND_EVIDENCE_RULES}
STRICT RULE -- do not violate this: every gap, contradiction, or missed
opportunity you raise must cite the retrieved chunk(s) that support it,
using the citation format above. Do not raise a point "in general" without
tying it to specific evidence.

If the retrieved evidence has nothing clearly relevant to this section, say
so explicitly instead of forcing a critique -- respond with exactly this:
"No clearly relevant evidence was found in the corpus for this section."

If the section itself is too short or thin to meaningfully critique (e.g.
just a heading with no real content), say so explicitly instead of
inventing a critique -- respond with exactly this:
"Insufficient detail in this section to critique meaningfully."

Methodology section ({{section_name}}):
{{section_text}}

Retrieved evidence from the corpus:
{{context}}

Critique:"""
)

_critique_chain = CRITIQUE_PROMPT | llm | StrOutputParser()


def critique_methodology(document_text: str) -> list[dict]:
    """
    Break `document_text` into sections and critique each one against the
    paper corpus.

    For every section, its own text is used as the retrieval query against
    retrieve_multi_doc() (chain.py's multi-document retrieval, reused here
    rather than get_answer_multi_doc() directly, since that function forces
    generation through chain.py's question-answering PROMPT and this module
    needs CRITIQUE_PROMPT instead) -- so every paper gets a chance to
    surface relevant evidence, not just the one or two most similar to the
    section's wording. Raises ValueError via split_into_sections() if
    document_text is empty.

    Returns one {"section_name", "section_text", "critique", "sources"}
    dict per section, in the same order the sections appeared in the
    original document. A section with too little text to critique still
    gets processed -- CRITIQUE_PROMPT is instructed to say so explicitly
    rather than inventing a critique from nothing.
    """
    sections = split_into_sections(document_text)

    results = []
    for section in sections:
        source_documents = retrieve_multi_doc(section["text"])
        critique = _critique_chain.invoke(
            {
                "section_name": section["section_name"],
                "section_text": section["text"],
                "context": format_docs(source_documents),
            }
        )
        sources = [
            {
                "filename": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "section": doc.metadata.get("section"),
            }
            for doc in source_documents
        ]
        results.append(
            {
                "section_name": section["section_name"],
                "section_text": section["text"],
                "critique": critique,
                "sources": sources,
            }
        )

    return results
