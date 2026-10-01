"""
FastAPI wrapper around the RAG pipeline.

This file turns your terminal script into a real web API that any program
(a website, a mobile app, Postman, curl) can call over HTTP.

Run it with:
    uvicorn api:app --reload

Then open http://localhost:8000/docs in a browser — FastAPI auto-generates
an interactive page where you can test every endpoint by clicking buttons,
no extra tools needed.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_assistant import ask_question_api, build_index

# This creates the actual API application object. "app" is the name uvicorn
# looks for by default when you run `uvicorn api:app`.
app = FastAPI(
    title="AI Document Intelligence Assistant API",
    description="A RAG-based API for asking questions about your documents.",
    version="1.0.0",
)


# --- Request/response "shapes" ---
# Pydantic models define exactly what JSON a request must contain, and FastAPI
# automatically validates incoming requests against this shape — reject anything
# that doesn't match, with a clear error, before your code even runs.

class AskRequest(BaseModel):
    question: str   # the client MUST send a "question" field, and it must be a string


class SourceChunk(BaseModel):
    source: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# --- Endpoints ---
# @app.get(...) and @app.post(...) are decorators: they tell FastAPI
# "when a request arrives at this URL with this HTTP method, run this function."

@app.get("/")
def health_check():
    """GET / — a simple endpoint to check the API is alive. No input needed."""
    return {"status": "ok", "message": "RAG Assistant API is running"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """POST /ask — the main endpoint. Takes a question, returns a grounded answer + sources.

    Example request body: {"question": "What is this document about?"}
    """
    try:
        result = ask_question_api(request.question)
    except RuntimeError as e:
        # 400 = client's fault (e.g. no index built yet) — a clear, honest status code
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 500 = something broke on our side
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return AskResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )


@app.post("/ingest")
def ingest():
    """POST /ingest — rebuilds the vector index from whatever is in sample_docs/."""
    try:
        build_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")
    return {"status": "ok", "message": "Index rebuilt successfully"}
