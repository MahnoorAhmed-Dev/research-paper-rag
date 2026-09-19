"""
Streamlit UI for the RAG project: upload PDFs, index them, and chat with
them. Retrieval/generation logic lives in src/chain.py; indexing logic
lives in src/ingest.py; methodology critique and novelty-check logic live
in src/critique.py and src/novelty.py — this file only wires them to the UI.
"""

import streamlit as st

from src.chain import get_answer
from src.critique import critique_methodology
from src.novelty import check_novelty
from src.ingest import ingest_pdfs
from src.config import PDF_DIR


def _format_source(source: dict) -> str:
    """Render a source dict as a citation caption, including the section
    when known (see src/chain.py's prompt citations for the matching style)."""
    section = source.get("section")
    if section and section != "unknown":
        return f"{source['filename']} - {section} section, page {source['page']}"
    return f"{source['filename']} - page {source['page']}"


def _critique_to_markdown(critique_result: list) -> str:
    """Build a downloadable Markdown report from critique_methodology()'s
    structured output, rather than dumping raw Python objects."""
    lines = ["# Methodology Critique Report", ""]
    for section in critique_result:
        lines.append(f"## {section['section_name']}")
        lines.append("")
        lines.append(section["critique"])
        lines.append("")
        if section["sources"]:
            lines.append("**Sources:**")
            for source in section["sources"]:
                lines.append(f"- {_format_source(source)}")
            lines.append("")
    return "\n".join(lines)


def _novelty_to_markdown(novelty_result: dict) -> str:
    """Build a downloadable Markdown report from check_novelty()'s
    structured output, rather than dumping raw Python objects."""
    lines = [
        "# Novelty Check Report",
        "",
        "## Claim",
        novelty_result["claim"],
        "",
        "## Assessment",
        novelty_result["assessment"],
        "",
    ]
    if novelty_result["sources"]:
        lines.append("**Sources:**")
        for source in novelty_result["sources"]:
            lines.append(f"- {_format_source(source)}")
    return "\n".join(lines)


def _render_novelty_assessment(assessment: str) -> None:
    """
    Best-effort split of check_novelty()'s CLOSEST EXISTING WORK /
    SIMILARITY ASSESSMENT / VERDICT / Caveat structure into visually
    distinct pieces, with the verdict colored by its type. This is parsing
    the LLM's own structured text, not a guaranteed schema, so any heading
    it doesn't recognize just falls back to plain markdown for the whole
    assessment instead of dropping content.
    """
    headings = {
        "CLOSEST EXISTING WORK:": "closest_work",
        "SIMILARITY ASSESSMENT:": "similarity",
        "VERDICT:": "verdict",
    }
    sections = {}
    current_key = None
    for line in assessment.splitlines():
        cleaned = line.strip().strip("*").strip()
        matched = next((h for h in headings if cleaned.upper().startswith(h)), None)
        if matched:
            current_key = headings[matched]
            sections[current_key] = cleaned[len(matched):].strip()
        elif cleaned.lower().startswith("caveat:"):
            current_key = "caveat"
            sections[current_key] = cleaned[len("caveat:"):].strip()
        elif current_key:
            sections[current_key] = (sections.get(current_key, "") + "\n" + line).strip()

    if not sections:
        st.markdown(assessment)
        return

    if sections.get("closest_work"):
        st.markdown("**Closest Existing Work**")
        st.write(sections["closest_work"])
    if sections.get("similarity"):
        st.markdown("**Similarity Assessment**")
        st.write(sections["similarity"])
    if sections.get("verdict"):
        st.markdown("**Verdict**")
        verdict_text = sections["verdict"]
        verdict_lower = verdict_text.lower()
        if "already covered" in verdict_lower:
            st.warning(verdict_text)
        elif "appears novel" in verdict_lower:
            st.success(verdict_text)
        elif "partial overlap" in verdict_lower:
            st.info(verdict_text)
        else:
            st.write(verdict_text)
    if sections.get("caveat"):
        st.caption(f"Caveat: {sections['caveat']}")


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

# Three modes share the sidebar above (indexed papers + uploader) but are
# otherwise independent screens. Chat is a running conversation, so its
# history lives in session_state.messages as before. Critique and Novelty
# Check are single-shot reports instead -- one document or claim in, one
# structured result out, not a back-and-forth -- so each keeps its own
# session_state key (last_critique / last_novelty) rather than being folded
# into chat history. That also means switching tabs and coming back doesn't
# lose the last report, since it's re-rendered from session_state on every
# rerun rather than only right after the button click that produced it.
mode = st.radio(
    "Mode",
    ["Chat", "Methodology Critique", "Novelty Check"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            for source in message.get("sources") or []:
                st.caption(_format_source(source))

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
                        st.caption(_format_source(source))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"] if result else "Sorry, something went wrong answering that question.",
                    "sources": result["sources"] if result else [],
                }
            )

elif mode == "Methodology Critique":
    st.subheader("Methodology Critique")
    st.caption(
        "Paste a methodology write-up below. Each detected section is critiqued "
        "separately against the indexed corpus for gaps, contradictions, and "
        "missed opportunities."
    )

    if "last_critique" not in st.session_state:
        st.session_state.last_critique = None

    document_text = st.text_area(
        "Methodology document", height=300, key="critique_input"
    )

    if st.button("Run Critique"):
        if not document_text or not document_text.strip():
            st.warning("Paste a methodology document first.")
        else:
            with st.spinner("Analyzing methodology against the corpus..."):
                try:
                    st.session_state.last_critique = critique_methodology(document_text)
                except Exception as e:
                    st.session_state.last_critique = None
                    st.error(f"Couldn't run the critique: {e}")

    if st.session_state.last_critique:
        for section in st.session_state.last_critique:
            with st.expander(section["section_name"]):
                st.write(section["critique"])
                for source in section["sources"]:
                    st.caption(_format_source(source))

        st.download_button(
            "Download critique as Markdown",
            data=_critique_to_markdown(st.session_state.last_critique),
            file_name="methodology_critique.md",
            mime="text/markdown",
        )

elif mode == "Novelty Check":
    st.subheader("Novelty Check")
    st.caption(
        "Paste a single research claim to check it against the indexed corpus."
    )

    if "last_novelty" not in st.session_state:
        st.session_state.last_novelty = None

    claim_text = st.text_area("Claim", height=100, key="novelty_input")

    if st.button("Check Novelty"):
        if not claim_text or not claim_text.strip():
            st.warning("Paste a claim first.")
        else:
            with st.spinner("Searching corpus for related work..."):
                try:
                    st.session_state.last_novelty = check_novelty(claim_text)
                except Exception as e:
                    st.session_state.last_novelty = None
                    st.error(f"Couldn't check novelty: {e}")

    if st.session_state.last_novelty:
        result = st.session_state.last_novelty
        st.markdown(f"**Claim:** {result['claim']}")
        _render_novelty_assessment(result["assessment"])
        for source in result["sources"]:
            st.caption(_format_source(source))

        st.download_button(
            "Download assessment as Markdown",
            data=_novelty_to_markdown(result),
            file_name="novelty_assessment.md",
            mime="text/markdown",
        )
