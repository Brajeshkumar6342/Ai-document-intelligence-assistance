"""
FastAPI wrapper around the RAG pipeline.

Run with:
    uvicorn api:app --reload

Docs at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_assistant import ask_question_api, build_index

app = FastAPI(
    title="AI Document Intelligence Assistant API",
    description="A RAG-based API for asking questions about your documents.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    source: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG Assistant API is running"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = ask_question_api(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return AskResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )


@app.post("/ingest")
def ingest():
    try:
        build_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")
    return {"status": "ok", "message": "Index rebuilt successfully"}
