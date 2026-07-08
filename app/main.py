"""
FastAPI backend for the RAG application.

Endpoints:
  POST /ask         - answer a question using only retrieved document context
  POST /contradict   - compare two documents and report whether they contradict
  GET  /documents    - list ingested documents (used by the Streamlit UI)
  GET  /health       - liveness check

Run with: uvicorn app.main:app --reload
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    AskRequest,
    AskResponse,
    ContradictRequest,
    ContradictResponse,
    DocumentsResponse,
    DocumentInfo,
)
from app.rag_pipeline import answer_question
from app.contradiction import compare_documents
from app.vector_store import list_documents, collection_count
from app.llm import LLMError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

app = FastAPI(
    title="RAG Document Q&A API",
    description="Retrieval-Augmented Generation over a local PDF knowledge base.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "ingested_chunks": collection_count()}


@app.get("/documents", response_model=DocumentsResponse)
def get_documents():
    docs = list_documents()
    return DocumentsResponse(
        documents=[
            DocumentInfo(doc_id=doc_id, source=info["source"], pages=info["pages"], chunk_count=info["chunk_count"])
            for doc_id, info in sorted(docs.items())
        ]
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = answer_question(request.question, top_k=request.top_k)
    except LLMError as exc:
        logger.exception("LLM error in /ask")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
    return AskResponse(**result)


@app.post("/contradict", response_model=ContradictResponse)
def contradict(request: ContradictRequest):
    known_docs = set(list_documents().keys())
    for doc_id in (request.doc_id_1, request.doc_id_2):
        if doc_id not in known_docs:
            raise HTTPException(
                status_code=404,
                detail=f"doc_id '{doc_id}' was not found. Known documents: {sorted(known_docs)}",
            )
    try:
        result = compare_documents(request.doc_id_1, request.doc_id_2, topic=request.topic, top_k=request.top_k)
    except LLMError as exc:
        logger.exception("LLM error in /contradict")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /contradict")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
    return ContradictResponse(**result)
