"""
AI Document Intelligence Assistant — Core RAG Pipeline
--------------------------------------------------------
Fully free/local stack:
  - Embeddings : sentence-transformers (HuggingFace) — all-MiniLM-L6-v2
  - Vector DB  : Chroma (local, persisted to disk)
  - LLM        : Ollama (runs a local model like llama3 or phi3)
  - Framework  : LangChain (glues everything together)

Run this AFTER completing the local setup in README.md (Ollama installed,
model pulled, Python deps installed).

Usage:
    python rag_assistant.py ingest      # build the vector index from sample_docs/
    python rag_assistant.py ask "your question here"
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

DOCS_DIR = Path(__file__).parent / "sample_docs"
DB_DIR = Path(__file__).parent / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs on CPU
LLM_MODEL = "llama3"                    # change to "phi3" if llama3 is too heavy for your machine

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents():
    """Load every .pdf and .txt file from sample_docs/"""
    docs = []
    for path in DOCS_DIR.glob("*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(path)).load())
    if not docs:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in {DOCS_DIR}. Add some documents first."
        )
    print(f"Loaded {len(docs)} document(s) / page(s).")
    return docs


def chunk_documents(docs):
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def build_index():
    """Ingest pipeline: load -> chunk -> embed -> persist to Chroma."""
    docs = load_documents()
    chunks = chunk_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    # Note: newer Chroma versions auto-persist to persist_directory, no manual .persist() call needed.
    print(f"Vector index built and saved to {DB_DIR}")


PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the
provided context. If the answer isn't in the context, say you don't know —
do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

def ask_question(question: str):
    """Retrieval pipeline, built with LangChain's RetrievalQA chain:
    embed query -> Chroma retriever finds top chunks -> chain fills the prompt
    template -> local Ollama LLM (via LangChain's OllamaLLM wrapper) generates the answer."""
    if not DB_DIR.exists():
        raise RuntimeError("No index found. Run `python rag_assistant.py ingest` first.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 6})  # top 6 relevant chunks (was 3 — widened to catch summary/abstract content buried among reference-list chunks)

    llm = OllamaLLM(model=LLM_MODEL, temperature=0)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",                       # "stuff" = stuff all retrieved chunks into one prompt
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    result = qa_chain.invoke({"query": question})

    print("\n--- ANSWER ---")
    print(result["result"])

    print("\n--- SOURCES ---")
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"[{i}] {source} :: \"{snippet}...\"")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python rag_assistant.py ingest\n  python rag_assistant.py ask \"question\"")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        build_index()
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Please provide a question, e.g.:\n  python rag_assistant.py ask \"What is this document about?\"")
            sys.exit(1)
        ask_question(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
