"""
AI Document Intelligence Assistant — Core RAG Pipeline
--------------------------------------------------------
Fully free/local stack:
  - Embeddings : FastEmbed (ONNX Runtime) — BAAI/bge-small-en-v1.5
                 Chosen over sentence-transformers/torch specifically because
                 it uses a fraction of the memory (no PyTorch, no CUDA libs),
                 which matters when deploying to free-tier hosting (512MB RAM).
  - Vector DB  : Chroma (local, persisted to disk)
  - LLM        : Ollama locally (llama3/phi3), or Groq when deployed (see LLM_BACKEND below)
  - Framework  : LangChain (glues everything together)

Run this AFTER completing the local setup in README.md (Ollama installed,
model pulled, Python deps installed).

Usage:
    python rag_assistant.py ingest      # build the vector index from sample_docs/
    python rag_assistant.py ask "your question here"
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads .env locally if present; harmless no-op when deployed (real env vars are set by the platform instead)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

DOCS_DIR = Path(__file__).parent / "sample_docs"
DB_DIR = Path(__file__).parent / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # same ~384-dim size class as MiniLM, runs via lightweight ONNX (fastembed), not torch
LLM_MODEL = "llama3"                    # used only for local Ollama — change to "phi3" if llama3 is too heavy for your machine

# Which LLM backend to use. Controlled by an environment variable so the SAME
# code works both locally (Ollama, free/private) and when deployed (Groq, since
# free hosting platforms can't run Ollama's background model server).
# Set LLM_BACKEND=groq and GROQ_API_KEY=... in your deployment platform's
# secrets settings. Locally, leave both unset to default to Ollama.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")


def get_llm():
    """Returns the right LangChain LLM object depending on LLM_BACKEND."""
    if LLM_BACKEND == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_BACKEND=groq but GROQ_API_KEY is not set.")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    else:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=LLM_MODEL, temperature=0)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Matches a "References" or "Bibliography" heading on its own line (academic paper convention).
# Once this is found in a page, everything from that point on — and every later page of the
# same file — is the citation list, not real content, so it gets dropped before chunking.
REFERENCES_PATTERN = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)


def strip_references(docs):
    """Remove reference/bibliography sections so they never get embedded or retrieved.
    Limitation: relies on a 'References' heading appearing on its own line — papers that
    format this differently (or have no explicit heading) won't be caught by this heuristic."""
    cleaned = []
    cutoff_sources = set()
    for doc in docs:
        source = doc.metadata.get("source")
        if source in cutoff_sources:
            continue  # already past the references section for this file — skip remaining pages
        match = REFERENCES_PATTERN.search(doc.page_content)
        if match:
            doc.page_content = doc.page_content[: match.start()]
            cutoff_sources.add(source)
            if doc.page_content.strip():
                cleaned.append(doc)
            continue
        cleaned.append(doc)
    return cleaned


def load_documents():
    """Load every .pdf and .txt file from sample_docs/, then strip reference sections."""
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
    before = len(docs)
    docs = strip_references(docs)
    print(f"Loaded {before} document(s) / page(s); {len(docs)} remain after stripping reference sections.")
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

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

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

def ask_question_api(question: str) -> dict:
    """Core retrieval pipeline, returning plain data (no printing) so both the
    terminal command AND the FastAPI endpoint can reuse this same logic.

    Built with LangChain's RetrievalQA chain: embed query -> Chroma retriever
    finds top chunks -> chain fills the prompt template -> local Ollama LLM
    (via LangChain's OllamaLLM wrapper) generates the answer."""
    if not DB_DIR.exists():
        raise RuntimeError("No index found. Run `python rag_assistant.py ingest` first.")

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})  # top 4 relevant chunks — safe since reference-list noise is filtered out before indexing

    llm = get_llm()  # Ollama locally, Groq when deployed — see LLM_BACKEND above

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

    sources = []
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:120].replace("\n", " ")
        sources.append({"source": source, "snippet": snippet})

    return {"answer": result["result"], "sources": sources}


def ask_question(question: str):
    """Terminal command version — calls the same core logic, then prints it nicely."""
    result = ask_question_api(question)

    print("\n--- ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCES ---")
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] {s['source']} :: \"{s['snippet']}...\"")


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
