import streamlit as st
from rag import load_and_index_pdf, get_qa_chain, ask_question
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="wide"
)

st.title("PDF Q&A Bot")
st.caption("Upload a PDF and ask questions — powered by RAG + GPT")

# Session state to persist the chain across interactions
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: PDF upload + indexing
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Index PDF", type="primary"):
            st.session_state.qa_chain = None
            import gc
            gc.collect()
            with st.spinner("Reading and indexing PDF..."):
                vectorstore, n_chunks = load_and_index_pdf("temp.pdf")
                st.session_state.qa_chain = get_qa_chain(vectorstore)
                st.success(f"Indexed {n_chunks} chunks!")

    st.divider()

    # Load existing index if already built
    if st.button("Load existing index"):
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        st.session_state.qa_chain = get_qa_chain(vectorstore)
        st.success("Index loaded!")

# Main chat area
if st.session_state.qa_chain:
    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Question input
    question = st.chat_input("Ask anything about your PDF...")

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

            # Show source chunks (great for demo!)
            with st.expander(f"Sources ({len(sources)} chunks used)"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}** (Page {doc.metadata.get('page', '?') + 1})")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
else:
    st.info("Upload and index a PDF using the sidebar to get started.")