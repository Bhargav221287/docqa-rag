"""Streamlit web interface for the docqa-rag document question-answering system."""

import os
import requests
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="docqa-rag | Grounded Document QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend API Configuration
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("Backend API URL", value=DEFAULT_API_URL)


def check_api_health(url: str) -> dict | None:
    """Check connectivity to the FastAPI backend service."""
    try:
        resp = requests.get(f"{url}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def fetch_documents(url: str) -> list:
    """Fetch the list of currently indexed documents from the backend."""
    try:
        resp = requests.get(f"{url}/documents", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("documents", [])
    except Exception:
        pass
    return []


def delete_document_api(url: str, filename: str) -> bool:
    """Delete a document by filename via backend API."""
    try:
        resp = requests.delete(f"{url}/documents/{filename}", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Navigation & Management
with st.sidebar:
    st.title("⚙️ Control Panel")

    # API Status Check
    health_data = check_api_health(API_URL)
    if health_data:
        st.success(f"🟢 Connected ({health_data.get('provider', 'active')})")
        st.caption(
            f"Model: `{health_data.get('model')}` | Chunks: `{health_data.get('total_chunks')}`"
        )
    else:
        st.error("🔴 Backend Disconnected")
        st.info("Start backend: `uvicorn app.main:app --port 8000`")

    st.divider()

    # Retrieval Tuning
    st.subheader("🎯 Retrieval Parameters")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=4, step=1)
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Chunks with similarity below this cutoff are ignored. If no chunks qualify, system returns fallback.",
    )

    st.divider()

    # Document Upload Section
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload text-based PDF documents to index into ChromaDB.",
    )

    if uploaded_file is not None:
        if st.button("Index Document", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                    if resp.status_code == 201:
                        data = resp.json()
                        st.success(f"✅ {uploaded_file.name} indexed successfully!")
                        st.caption(
                            f"Pages: {data.get('total_pages')} | Chunks: {data.get('chunks_indexed')}"
                        )
                        st.rerun()
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"Upload failed: {detail}")
                except Exception as exc:
                    st.error(f"Connection error: {exc}")

    st.divider()

    # Document Management Section
    st.subheader("📁 Indexed Documents")
    docs = fetch_documents(API_URL) if health_data else []
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{doc.get('filename')}**")
                st.caption(f"{doc.get('total_chunks')} chunks • Pages: {doc.get('pages')}")
            with col2:
                if st.button("🗑️", key=f"del_{doc.get('filename')}", help="Delete document"):
                    if delete_document_api(API_URL, doc.get("filename")):
                        st.toast(f"Deleted {doc.get('filename')}")
                        st.rerun()
                    else:
                        st.error("Failed to delete.")
    else:
        st.write("No documents indexed yet.")


# Main Application Chat Interface
st.title("📄 docqa-rag")
st.markdown(
    """
    **Production-grade Document Question-Answering** powered by FastAPI, ChromaDB, and Grounded RAG.
    Every factual assertion cites its source file and page number. If retrieval confidence is low, the system
    safely returns *"I don't know based on the provided documents"*.
    """
)

# Display Chat History
for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(message["content"])

        # Display citations if available
        sources = message.get("sources", [])
        confidence = message.get("confidence")
        is_grounded = message.get("is_grounded", True)

        if role == "assistant" and confidence is not None:
            # Confidence badge
            if is_grounded:
                conf_color = "green" if confidence >= 0.75 else "orange"
                st.markdown(
                    f":{conf_color}[**Retrieval Confidence:** {confidence * 100:.1f}%]"
                )
            else:
                st.markdown(":red[**Retrieval Confidence:** Below Threshold (Fallback Activated)]")

            if sources:
                with st.expander(f"📚 Sources & Citations ({len(sources)} cited)"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(
                            f"**{i}. {src.get('filename')} — Page {src.get('page')}** "
                            f"(Similarity: `{src.get('similarity', 0.0):.3f}`)"
                        )
                        st.info(f"\"{src.get('snippet')}\"")

# Chat Input & Processing
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Call Backend API
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Retrieving grounded context & synthesizing answer..."):
            try:
                payload = {
                    "question": prompt,
                    "top_k": top_k,
                    "similarity_threshold": threshold,
                }
                resp = requests.post(f"{API_URL}/ask", json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    answer_text = data.get("answer", "")
                    sources = data.get("sources", [])
                    confidence = data.get("confidence", 0.0)
                    is_grounded = data.get("is_grounded", False)

                    st.markdown(answer_text)

                    # Confidence display
                    if is_grounded:
                        conf_color = "green" if confidence >= 0.75 else "orange"
                        st.markdown(
                            f":{conf_color}[**Retrieval Confidence:** {confidence * 100:.1f}%]"
                        )
                    else:
                        st.markdown(":red[**Retrieval Confidence:** Below Threshold (Fallback Activated)]")

                    if sources:
                        with st.expander(f"📚 Sources & Citations ({len(sources)} cited)"):
                            for i, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"**{i}. {src.get('filename')} — Page {src.get('page')}** "
                                    f"(Similarity: `{src.get('similarity', 0.0):.3f}`)"
                                )
                                st.info(f"\"{src.get('snippet')}\"")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources,
                            "confidence": confidence,
                            "is_grounded": is_grounded,
                        }
                    )
                else:
                    detail = resp.json().get("detail", resp.text)
                    err_msg = f"❌ Error ({resp.status_code}): {detail}"
                    st.error(err_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err_msg}
                    )
            except Exception as exc:
                err_msg = f"❌ Connection Error: Could not reach backend API at {API_URL}. Details: {exc}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg}
                )

