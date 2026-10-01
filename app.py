"""
Streamlit front-end for the AI Document Intelligence Assistant.

This is a CLIENT of your FastAPI server — it does not run the RAG pipeline
itself. It sends HTTP requests to api.py (which must be running separately)
and displays whatever comes back. This is the same request/response cycle
you tested manually in the /docs page, just automated behind a UI.

Run with (in a SEPARATE terminal from the one running uvicorn):
    streamlit run app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Document Intelligence Assistant", page_icon="📄")

st.title("📄 AI Document Intelligence Assistant")
st.caption("Ask questions about the documents in sample_docs/ — answered by a local LLM, grounded in retrieved chunks.")

# --- Sidebar: health check + re-ingest ---
with st.sidebar:
    st.header("Server status")

    # Calls GET / — the same health check endpoint from /docs
    try:
        health = requests.get(f"{API_URL}/", timeout=5)
        if health.status_code == 200:
            st.success("API is running")
        else:
            st.error(f"API returned status {health.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Is `uvicorn api:app --reload` running?")

    st.divider()

    st.header("Rebuild index")
    st.caption("Run this after adding new files to sample_docs/")
    if st.button("Re-ingest documents"):
        with st.spinner("Rebuilding index... this can take a minute"):
            try:
                # Calls POST /ingest
                resp = requests.post(f"{API_URL}/ingest", timeout=300)
                if resp.status_code == 200:
                    st.success(resp.json()["message"])
                else:
                    st.error(f"Ingest failed: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API.")

# --- Main area: ask a question ---
question = st.text_input("Ask a question about your documents:", placeholder="What is this document about?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Thinking..."):
        try:
            # Calls POST /ask with a JSON body — exactly what you tested manually in /docs
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=120,
            )
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API. Is `uvicorn api:app --reload` running?")
        else:
            if response.status_code == 200:
                data = response.json()

                st.subheader("Answer")
                st.write(data["answer"])

                st.subheader("Sources")
                for i, source in enumerate(data["sources"], 1):
                    with st.expander(f"[{i}] {source['source'].split(chr(92))[-1].split('/')[-1]}"):
                        st.write(source["snippet"] + "...")
            else:
                st.error(f"Error {response.status_code}: {response.json().get('detail', response.text)}")
