"""
Streamlit front-end for the AI Document Intelligence Assistant.

Run with (api.py must already be running separately):
    streamlit run app.py
"""

import requests
import streamlit as st

try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Document Intelligence Assistant", page_icon="📄")

st.title("📄 AI Document Intelligence Assistant")
st.caption("Ask questions about the documents in sample_docs/ — answered by a local LLM, grounded in retrieved chunks.")

with st.sidebar:
    st.header("Server status")
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
                resp = requests.post(f"{API_URL}/ingest", timeout=300)
                if resp.status_code == 200:
                    st.success(resp.json()["message"])
                else:
                    st.error(f"Ingest failed: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API.")

question = st.text_input("Ask a question about your documents:", placeholder="What is this document about?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Thinking..."):
        try:
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
