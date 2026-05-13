import streamlit as st
from rag import load_and_index_pdf, get_qa_chain, ask_question
import uuid
import os
import tempfile

st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Q&A Bot")
st.caption("Upload a PDF and ask questions — powered by RAG + Groq LLaMA 3")

# ✅ Session state — every browser session is completely isolated
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None

# Sidebar: PDF upload + indexing
with st.sidebar:
    st.header("📂 Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        # Show which file is uploaded
        st.caption(f"File: `{uploaded_file.name}`")

        # ✅ Only re-index if a new file is uploaded
        file_changed = st.session_state.indexed_filename != uploaded_file.name

        if file_changed:
            st.info("New file detected. Click below to index it.")

        if st.button("Index PDF", type="primary"):
            # ✅ Use a temp file — gets cleaned up automatically
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                with st.spinner("Reading and indexing PDF..."):
                    vectorstore, n_chunks = load_and_index_pdf(tmp_path)
                    # ✅ Reset chat history when new PDF is indexed
                    st.session_state.qa_chain = get_qa_chain(vectorstore)
                    st.session_state.chat_history = []
                    st.session_state.indexed_filename = uploaded_file.name
                    st.success(f"✅ Indexed **{n_chunks}** chunks from `{uploaded_file.name}`!")
            finally:
                # ✅ Always clean up temp file
                os.unlink(tmp_path)

    st.divider()

    # Show current session status
    if st.session_state.indexed_filename:
        st.success(f"Active PDF: `{st.session_state.indexed_filename}`")
    else:
        st.warning("No PDF indexed yet.")

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# Main chat area
if st.session_state.qa_chain:
    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Question input
    question = st.chat_input(f"Ask anything about {st.session_state.indexed_filename}...")

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                answer, sources = ask_question(
                    st.session_state.qa_chain, question
                )
            st.write(answer)

            # Show source chunks
            with st.expander(f"📚 Sources ({len(sources)} chunks used)"):
                for i, doc in enumerate(sources):
                    page_num = doc.metadata.get('page', '?')
                    page_display = page_num + 1 if isinstance(page_num, int) else page_num
                    st.markdown(f"**Chunk {i+1}** — Page {page_display}")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
else:
    # Landing state
    st.info("👈 Upload a PDF using the sidebar and click **Index PDF** to get started.")
    st.markdown("""
    ### How it works
    1. **Upload** any PDF — resume, research paper, manual, policy doc
    2. **Index** — the document is split into chunks and stored in memory
    3. **Ask** — type any question and get answers from your document
    
    > Each session is completely private — your data is never mixed with other users.
    """)
