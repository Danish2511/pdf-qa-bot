import shutil
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def load_and_index_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Still using OpenAI for embeddings — switch to free below if needed
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    # loader = PyPDFLoader(pdf_path)
    # documents = loader.load()

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200
    # )
    # chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore, len(chunks)

def get_qa_chain(vectorstore):
    # Groq LLM — free and very fast
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If you don't know, say "I don't have enough information in the document."

Context: {context}

Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def ask_question(chain_and_retriever, question: str):
    chain, retriever = chain_and_retriever
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources