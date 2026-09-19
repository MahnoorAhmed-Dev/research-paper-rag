# Research Paper RAG

Research Paper RAG started as a tool I built for myself and a few friends to make sense of the papers piling up during our Final Year Projects; instead of re-reading and cross-referencing a dozen PDFs by hand every time we needed to check whether an idea was actually novel or grounded in the literature, we wanted something that could search, cite, and critique against our own paper collection directly. It grew into something more useful than a one-off script, so I'm sharing it in case it's helpful to anyone else working through the same thing.

It runs entirely locally and free! PDF parsing, chunking, embedding, retrieval, and re-ranking all happen on your own machine, with the only network call going to Groq's API (free tier) for the final generation step. It's not a production system or a substitute for careful peer review; but it's straightforward to run, inspect, and adapt to your own research.

## Getting Started

1. **Clone the repo**
   ```
   git clone https://github.com/MahnoorAhmed-Dev/research-paper-rag.git
   cd research-paper-rag
   ```
2. **Get a free Groq API key** at [console.groq.com](https://console.groq.com/keys).
3. **Set up your environment variables**
   ```
   cp .env.example .env
   ```
   Then open `.env` and paste in your Groq API key.
4. **Install dependencies** (a virtual environment is recommended)
   ```
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
5. **Add your own PDFs** to `data/pdfs/` — or skip this step and upload them
   later from the Streamlit sidebar.
6. **Build the index**
   ```
   python -m src.ingest
   ```
7. **Launch the app**
   ```
   streamlit run app.py
   ```

## Architecture

- **PDF parsing:** PyMuPDF (`fitz`), one page at a time. A regex-based
  section-header detector (`SECTION_HEADER_RE` in `src/ingest.py`) tags each
  chunk with a best-effort `section` field (e.g. `"Method"`, `"Results"`) so
  retrieval and citations can point at *where in a paper* a claim comes from,
  not just which PDF.
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`,
  `CHUNK_SIZE=1000` / `CHUNK_OVERLAP=200` (characters). Separators are
  explicitly ordered (paragraph break → line break → sentence-ending
  punctuation → word break → raw character) so a split prefers the largest
  structurally meaningful boundary that fits, falling back to a mid-word/
  mid-equation cut only as a last resort.
- **Embeddings:** `BAAI/bge-base-en-v1.5` (local, free, via
  `sentence-transformers`/`langchain-huggingface`). BGE is trained
  asymmetrically, so queries are prefixed with an instruction string
  (`BGE_QUERY_PREFIX`) at retrieval time; document chunks are embedded plain.
- **Vector store:** Chroma, persisted to `chroma_db/`, one collection
  (`research_papers`). Supports per-document metadata filtering
  (`filter={"source": filename}`), which multi-document retrieval uses to
  search one paper at a time.
- **Retrieval — two-stage retrieve-then-rerank:**
  - *Single-document mode* (`get_answer`): pull `RETRIEVAL_CANDIDATES` (20)
    chunks globally by embedding similarity, then re-rank with a
    cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) down to
    `RETRIEVER_K` (5). A cross-encoder scores the query and a chunk jointly
    rather than independently — more accurate, but too slow to run against
    the whole collection — so it only re-orders a candidate pool cheap
    vector search already narrowed down.
  - *Multi-document mode* (`get_answer_multi_doc`, and reused by critique
    and novelty checking): pull `MULTI_DOC_CHUNKS_PER_PAPER` (2) chunks from
    **every** indexed paper individually, merge, then re-rank down to
    `MULTI_DOC_FINAL_K` (10). This exists because a single global top-k
    search can return all its results from the one or two most similar
    papers and silently ignore the rest — which defeats the point of a
    cross-paper comparison, a methodology critique, or a novelty check.
- **Prompting:** three distinct prompt templates — question-answering,
  methodology critique, novelty check — all built on one shared
  `CITATION_AND_EVIDENCE_RULES` block (`src/chain.py`), so the exact citation
  format and the Direct-Evidence-vs-Inference labeling rules stay identical
  across all three instead of drifting apart.
- **Orchestration:** plain Python function composition (`retrieve_multi_doc()`,
  `format_docs()`, etc. called directly), not LangChain LCEL. Earlier
  versions used LCEL's `RunnablePassthrough.assign` pipe-chains; that was
  simplified away once retrieval needed to be shared across three different
  callers (Q&A, critique, novelty) with three different prompts — plain
  functions composed that more directly than branching LCEL chains.
- **LLM:** Groq-hosted `openai/gpt-oss-120b` via `langchain-groq`,
  temperature 0. (Originally `llama-3.3-70b-versatile`; switched after Groq
  retired that model from general availability.)
- **UI:** Streamlit (`app.py`), with a mode selector across three screens —
  Chat, Methodology Critique, Novelty Check — sharing one sidebar (indexed
  papers list + PDF uploader). The two report-style modes support Markdown
  export via download buttons.

## File Map

- `src/config.py` — pure constants (paths, chunk size, model names,
  retrieval/re-ranking/multi-doc tuning, collection name). No logic.
- `src/ingest.py` — builds/updates the index: `load_pdfs()` extracts text
  per page; `_tag_sections()` pre-splits pages at detected section headers;
  `ingest_pdfs(pdf_paths)` chunks, embeds, and adds to the existing Chroma
  collection (incremental); `rebuild_index()` wipes and rebuilds from every
  PDF in `PDF_DIR`; `main()` is the CLI entry point.
- `src/chain.py` — question-answering plus the shared retrieval/generation
  building blocks other modules reuse: `format_docs()`, `retrieve_multi_doc()`,
  `count_indexed_papers()`, `CITATION_AND_EVIDENCE_RULES`. Exposes
  `get_answer(question)` (single-document retrieval) and
  `get_answer_multi_doc(question)` (multi-document retrieval).
- `src/critique.py` — methodology critique mode. `split_into_sections()`
  breaks a pasted document into sections (reusing `ingest.py`'s header regex);
  `critique_methodology(document_text)` retrieves evidence per section via
  `retrieve_multi_doc()` and asks a dedicated `CRITIQUE_PROMPT` to identify
  gaps, contradictions, and missed opportunities, one structured result per
  section.
- `src/novelty.py` — novelty check mode. `check_novelty(claim)` retrieves
  evidence for a single claim via `retrieve_multi_doc()` and asks a dedicated
  `NOVELTY_PROMPT` for the closest existing work, a concrete similarity
  comparison, and a one-line verdict, always closing with a corpus-scope
  caveat.
- `app.py` (project root, not in `src/`) — Streamlit UI wiring all of the
  above to a mode selector, chat history, PDF upload, and Markdown export.
  Contains no retrieval/generation logic of its own.
- `tests/test_pipeline.py` — manual smoke script (not pytest): rebuild →
  incremental add → ask a question, printing results.
- `data/pdfs/` — source PDFs (gitignored; see Source Documents below for
  what's currently indexed).
- `chroma_db/` — persisted Chroma vector store (gitignored).

## Modes

- **Chat** — ask a specific, factual question about the corpus. Best when
  the answer legitimately lives in one or two papers; uses single-document
  retrieval.
- **Methodology Critique** — paste a methodology write-up (a project plan,
  a paper draft). It's split into sections, and each section is critiqued
  independently against the corpus for gaps, contradictions, and missed
  opportunities. Section-by-section (rather than critiquing the whole
  document at once) was chosen so each critique can cite evidence specific
  to that section's claims, instead of one long, harder-to-verify response
  mixing feedback on unrelated parts of the document.
- **Novelty Check** — paste a single claim to see whether it's already
  covered by, partially overlaps with, or appears novel relative to the
  indexed papers. **This assessment is scoped to this corpus only** — a
  "novel" verdict means no indexed paper covers it, not that the claim is
  novel against the broader research literature. Scoping novelty to the
  local corpus (rather than attempting a broader literature claim the
  system has no way to actually verify) keeps the verdict honest about what
  it can and can't see.

## Known Limitations

- **Section detection is heuristic, not real document parsing.**
  `SECTION_HEADER_RE` matches a fixed allowlist of common section names
  sitting alone on a line (with optional numbering). A paper that titles a
  section something nonstandard — e.g. "Framework" or "Our Approach"
  instead of "Method" — won't be recognized as a new section, and its
  content stays tagged under whatever section preceded it (or "unknown").
- **Critique and novelty quality is bounded by what's actually indexed.**
  Both modes can only compare against the papers in `data/pdfs/`; they
  cannot identify a gap or claim novelty against work that was never
  ingested. A "no gaps found" or "appears novel" result reflects this
  corpus, not an exhaustive literature review.
- **Free-tier Groq rate limits may apply under heavy use.** Methodology
  critique in particular makes one LLM call per detected section, so a long
  document can trigger several calls in quick succession.

## Running it

```
# add a real GROQ_API_KEY to .env (see .env.example for the placeholder)
python -m src.ingest          # full rebuild from everything in data/pdfs/
streamlit run app.py          # launch the UI (Chat / Methodology Critique / Novelty Check)
python tests/test_pipeline.py # manual smoke test of rebuild/ingest/answer
```

## License

MIT — see [LICENSE](LICENSE).
