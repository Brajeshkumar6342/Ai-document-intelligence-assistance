"""
AI Document Intelligence Assistant — Core RAG Pipeline

Stack: FastEmbed + Chroma + rank_bm25 + flashrank + Ollama/Groq + LangChain

Retrieval pipeline: vector search + BM25 keyword search are fused with
Reciprocal Rank Fusion, then the merged candidates are re-ranked by a
cross-encoder (flashrank) before the top few go to the LLM. Both the BM25
and re-ranking steps use CPU-light, non-torch libraries on purpose — the
original torch/sentence-transformers stack caused an out-of-memory crash on
free-tier hosting, so this pipeline deliberately avoids that dependency.

Usage:
    python rag_assistant.py ingest
    python rag_assistant.py ask "your question here"
"""

import os
import pickle
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
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

DOCS_DIR = Path(__file__).parent / "sample_docs"
DB_DIR = Path(__file__).parent / "chroma_db"
CHUNKS_CACHE = Path(__file__).parent / "chunks_cache.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3"  # swap to "phi3" on lower-RAM machines

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

FETCH_K = 10   # candidates pulled from each of vector/BM25 before fusion
FINAL_K = 4    # chunks actually sent to the LLM after re-ranking


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


def invoke_llm(llm, prompt: str) -> str:
    # OllamaLLM returns a plain string; ChatGroq (a chat model) returns a
    # message object with .content — normalize both to plain text here.
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else result


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

REFERENCES_PATTERN = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)


def strip_references(docs):
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

    # BM25 needs the raw chunk text/metadata at query time, so cache it
    # alongside the vector index rather than re-reading sample_docs/ each ask.
    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Vector index built and saved to {DB_DIR}")


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def _load_bm25():
    with open(CHUNKS_CACHE, "rb") as f:
        chunks = pickle.load(f)
    tokenized = [_tokenize(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks


def _bm25_search(query: str, bm25: BM25Okapi, chunks: list, k: int):
    scores = bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in ranked_idx]


def _reciprocal_rank_fusion(rankings: list, k: int = 60):
    # Standard RRF: each doc's fused score is sum(1 / (k + rank)) across every
    # ranking it appears in. Works even though vector similarity and BM25
    # scores aren't on comparable scales, since only rank position is used.
    scores = {}
    doc_lookup = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = (doc.metadata.get("source"), doc.page_content[:80])
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
            doc_lookup[key] = doc
    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_lookup[key] for key in fused_keys]


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        # Small ONNX cross-encoder — no torch, safe for free-tier memory limits.
        _reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _reranker


def _rerank(query: str, docs: list, top_k: int):
    if not docs:
        return docs
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    request = RerankRequest(query=query, passages=passages)
    results = _get_reranker().rerank(request)
    ranked_docs = [docs[r["id"]] for r in results[:top_k]]
    return ranked_docs


PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the
provided context. If the answer isn't in the context, say you don't know —
do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def ask_question_api(question: str) -> dict:
    if not DB_DIR.exists() or not CHUNKS_CACHE.exists():
        raise RuntimeError("No index found. Run `python rag_assistant.py ingest` first.")

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    bm25, chunks = _load_bm25()

    vector_hits = vectordb.similarity_search(question, k=FETCH_K)
    bm25_hits = _bm25_search(question, bm25, chunks, k=FETCH_K)

    fused = _reciprocal_rank_fusion([vector_hits, bm25_hits])
    top_chunks = _rerank(question, fused[: FETCH_K * 2], top_k=FINAL_K)

    context = "\n\n".join(doc.page_content for doc in top_chunks)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    ).format(context=context, question=question)

    llm = get_llm()
    answer = invoke_llm(llm, prompt)

    sources = []
    for doc in top_chunks:
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:120].replace("\n", " ")
        sources.append({"source": source, "snippet": snippet})

    return {"answer": answer, "sources": sources}


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
