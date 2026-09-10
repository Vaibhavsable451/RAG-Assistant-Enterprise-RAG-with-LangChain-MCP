"""
Streamlit front end for the RAG app.
Handles document upload/ingestion and the chat-style Q&A interface.
Every query passes through the governance layer before hitting the LLM.
"""
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import settings
from rag.chunking import chunk_documents
from rag.vector_store import build_vector_store, load_vector_store, add_to_vector_store
from rag.retriever import get_relevant_chunks, format_context
from rag.generator import generate_answer
from governance.guardrails import check_input, audit_log

st.set_page_config(page_title="RAG Assistant (Groq)", page_icon="🔎", layout="wide")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔎 RAG Assistant — Groq + LangChain + MCP")

with st.sidebar:
    st.header("📄 Ingest documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Index documents", disabled=not uploaded_files):
        all_docs = []
        with st.spinner("Loading and chunking documents..."):
            for uf in uploaded_files:
                suffix = os.path.splitext(uf.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = uf.name
                all_docs.extend(docs)
                os.unlink(tmp_path)

            chunks = chunk_documents(all_docs)

            if st.session_state.vector_store is None:
                st.session_state.vector_store = build_vector_store(chunks)
            else:
                st.session_state.vector_store = add_to_vector_store(
                    st.session_state.vector_store, chunks
                )
        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    st.caption(f"Model: `{settings.GROQ_MODEL}`")
    st.caption(f"Embeddings: `{settings.EMBEDDING_MODEL}`")
    st.caption(f"PII redaction: {'on' if settings.ENABLE_PII_REDACTION else 'off'}")
    st.caption(f"Content moderation: {'on' if settings.ENABLE_CONTENT_MODERATION else 'off'}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        gov_result = check_input(question)

        if not gov_result.allowed:
            answer = f"⚠️ This query was blocked by governance policy (flags: {gov_result.flags})."
            audit_log("blocked_query", question, flags=gov_result.flags)
        elif st.session_state.vector_store is None:
            answer = "Please upload and index at least one document first."
        else:
            with st.spinner("Retrieving context and generating answer..."):
                chunks = get_relevant_chunks(st.session_state.vector_store, gov_result.redacted_text)
                context = format_context(chunks)
                answer = generate_answer(gov_result.redacted_text, context)
                audit_log("rag_query", gov_result.redacted_text, response=answer)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
