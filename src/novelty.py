"""
Novelty check mode.

Takes a single research claim and searches the full paper corpus already
indexed in Chroma for the closest existing work, returning a structured
novelty assessment: what the closest matching paper(s) say, how the claim
compares to them concretely, and a one-line verdict -- explicitly scoped to
this corpus only, not a claim about the broader literature.

This is separate from src/chain.py's question-answering and
src/critique.py's section-by-section methodology critique: this module
takes one claim as input and asks a narrower question ("has this been done
before, in this corpus?") rather than answering a question or critiquing a
whole document. It shares the same retrieval discipline (retrieve_multi_doc(),
so a genuinely novel claim's scattered partial overlaps across several
papers aren't missed just because no single paper dominates the match) and
the same citation/evidence-labeling rules (CITATION_AND_EVIDENCE_RULES) as
the rest of the system, reused from chain.py rather than re-implemented.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.chain import (
    llm,
    format_docs,
    retrieve_multi_doc,
    count_indexed_papers,
    CITATION_AND_EVIDENCE_RULES,
)

# Distinct from chain.py's PROMPT (answers a question) and critique.py's
# CRITIQUE_PROMPT (critiques a methodology section): this one compares a
# single claim against the corpus and must always end with a caveat scoping
# the verdict to this corpus, regardless of what that verdict is.
NOVELTY_PROMPT = ChatPromptTemplate.from_template(
    f"""You are assessing whether a research claim is novel relative to a
corpus of papers. You are given a single claim and evidence chunks
retrieved from across that corpus. Use ONLY the retrieved evidence below --
no outside knowledge -- to produce your assessment.

Structure your response under exactly these headings, in this order:

CLOSEST EXISTING WORK:
Name the paper(s) most similar to the claim, if any were found in the
retrieved evidence. If nothing retrieved is meaningfully related, say so
explicitly instead of naming an unrelated paper just to fill this section.

SIMILARITY ASSESSMENT:
A specific, concrete comparison -- what's the same, what's different --
between the claim and the closest existing work. Avoid vague language like
"somewhat similar" or "related"; name the actual technique, mechanism, or
result being compared.

VERDICT:
Exactly one line, using one of these forms:
- "Already covered by [paper]" -- the claim is substantively the same as
  what a paper already does.
- "Appears novel relative to this corpus" -- no retrieved evidence
  meaningfully overlaps with the claim.
- "Partial overlap with [paper] -- the [specific aspect] differs" -- some
  meaningful similarity exists, but a specific, named aspect sets the claim
  apart.

{CITATION_AND_EVIDENCE_RULES}
STRICT RULE -- do not violate this: every claim you make in CLOSEST EXISTING
WORK and SIMILARITY ASSESSMENT must cite the retrieved chunk(s) it's based
on, using the citation format above. If no retrieved evidence is
meaningfully related to the claim, do not force a comparison -- state that
plainly in both sections above, and give the "Appears novel relative to
this corpus" verdict.

Always end your response with this exact caveat, regardless of verdict:
"Caveat: this assessment is relative to the {{corpus_size}} papers in this
corpus only and is not a claim about the broader literature."

Claim:
{{claim}}

Retrieved evidence from the corpus:
{{context}}

Assessment:"""
)

_novelty_chain = NOVELTY_PROMPT | llm | StrOutputParser()


def check_novelty(claim: str) -> dict:
    """
    Assess whether `claim` is novel relative to the indexed paper corpus.

    Uses retrieve_multi_doc() (the same shared multi-document retrieval
    critique.py uses) with the claim itself as the query, so evidence is
    pulled from across every paper rather than just the top matches from
    one -- a genuinely novel claim might have several small partial
    overlaps scattered across different papers instead of one dominant
    match, and a single global top-k search could miss that pattern.

    Raises ValueError if `claim` is empty. If retrieval finds no evidence
    at all (e.g. an empty index), returns a valid "Appears novel" result
    with an explicit note instead of forcing a comparison against nothing;
    the more common case -- non-empty but genuinely irrelevant retrieval --
    is handled by NOVELTY_PROMPT's own instructions to the LLM, not here.

    Returns {"claim": str, "assessment": str, "sources": list}.
    """
    if not claim or not claim.strip():
        raise ValueError("claim is empty -- nothing to check.")

    source_documents = retrieve_multi_doc(claim)
    corpus_size = count_indexed_papers()

    if not source_documents:
        return {
            "claim": claim,
            "assessment": (
                "CLOSEST EXISTING WORK: None found -- no evidence was retrieved "
                "from the corpus.\n"
                "SIMILARITY ASSESSMENT: No evidence was available to compare "
                "against.\n"
                "VERDICT: Appears novel relative to this corpus\n"
                f"Caveat: this assessment is relative to the {corpus_size} papers "
                "in this corpus only and is not a claim about the broader "
                "literature."
            ),
            "sources": [],
        }

    assessment = _novelty_chain.invoke(
        {
            "claim": claim,
            "context": format_docs(source_documents),
            "corpus_size": corpus_size,
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
    return {"claim": claim, "assessment": assessment, "sources": sources}
