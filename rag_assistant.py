"""
AI Document Intelligence Assistant — Core RAG Pipeline

Stack: FastEmbed + Chroma + Ollama/Groq + LangChain

Usage:
    python rag_assistant.py ingest
    python rag_assistant.py ask "your question here"
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

DOCS_DIR = Path(__file__).parent / "sample_docs"
DB_DIR = Path(__file__).parent / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3"  # swap to "phi3" on lower-RAM machines

# ollama locally, groq when deployed (set via env var on the hosting platform)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")


def get_llm():
    if LLM_BACKEND == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_BACKEND=groq but GROQ_API_KEY is not set.")
        return ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=api_key)
    else:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=LLM_MODEL, temperature=0)


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

REFERENCES_PATTERN = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)


def strip_references(docs):
    # drop bibliography/citation pages — they're keyword-dense and pollute retrieval
    cleaned = []
    cutoff_sources = set()
    for doc in docs:
        source = doc.metadata.get("source")
        if source in cutoff_sources:
            continue
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


def build_index():
    docs = load_documents()
    chunks = chunk_documents(docs)
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    print(f"Vector index built and saved to {DB_DIR}")


PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the
provided context. If the answer isn't in the context, say you don't know —
do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def ask_question_api(question: str) -> dict:
    if not DB_DIR.exists():
        raise RuntimeError("No index found. Run `python rag_assistant.py ingest` first.")

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = get_llm()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
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
