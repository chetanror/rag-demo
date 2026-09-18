from pathlib import Path

import streamlit as st

from rag_engine import RagEngine, default_documents, split_into_chunks


st.set_page_config(page_title="Workshop RAG Demo", page_icon="⌘", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600&display=swap');
    :root { --ink: #172126; --muted: #66747a; --mint: #cdeee5; --coral: #e8755e; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
    .hero { padding: 2.5rem 0 1.5rem; border-bottom: 1px solid #d8e2df; }
    .eyebrow { color: var(--coral); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(2.4rem, 6vw, 5.4rem); line-height: .95; margin: .4rem 0 1rem; max-width: 760px; }
    .hero p { color: var(--muted); max-width: 600px; font-size: 1.1rem; }
    .result { background: #f4faf8; border-left: 4px solid var(--coral); padding: 1.25rem 1.4rem; white-space: pre-wrap; line-height: 1.65; }
    .metric { color: var(--muted); font-size: .9rem; }
    </style>
    <div class="hero">
    <div class="eyebrow">AI workshop exercise</div>
    <h1>Ask my documents.<br>Check the answer.</h1>
    <p>A small local RAG demo: upload notes, find the most relevant passages, and see which sources were used for the answer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def build_engine(uploaded_files) -> RagEngine:
    chunks = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            content = uploaded_file.getvalue().decode("utf-8", errors="replace")
            chunks.extend(split_into_chunks(content, Path(uploaded_file.name).name))
    return RagEngine(chunks or default_documents())


with st.sidebar:
    st.markdown("### Knowledge base")
    uploaded_files = st.file_uploader(
        "Add .md or .txt files",
        type=["md", "txt"],
        accept_multiple_files=True,
    )
    st.caption("Files are indexed in memory and are not sent to a third-party service.")
    st.markdown("### Pipeline")
    st.markdown("1. **Chunk** paragraphs\n2. **Retrieve** with TF-IDF\n3. **Synthesize** an extractive answer\n4. **Cite** the matching sources")

engine = build_engine(uploaded_files)
st.markdown(f'<div class="metric">{len(engine.documents)} chunks indexed</div>', unsafe_allow_html=True)
st.markdown("## Ask a question")
query = st.text_input(
    "Question",
    placeholder="What does Atlas do when it cannot find a relevant passage?",
    label_visibility="collapsed",
)

if query:
    answer, retrieved = engine.answer(query)
    st.markdown("### Answer")
    st.markdown(f'<div class="result">{answer}</div>', unsafe_allow_html=True)
    if retrieved:
        st.markdown("### Retrieved passages")
        for index, result in enumerate(retrieved, start=1):
            with st.expander(f"[{index}] {result.chunk.title} · score {result.score:.3f}"):
                st.write(result.chunk.text)
else:
    st.info("Ask a question to run the pipeline. Try the example shown in the input field.")