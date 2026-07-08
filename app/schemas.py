"""Pydantic request/response models for the FastAPI endpoints."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question, any supported language")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to use for the answer (default from config)")


class Citation(BaseModel):
    source: str
    page: int
    chunk_id: str
    snippet: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    low_confidence_warning: bool
    detected_language: str
    retrieved_context: list[str]
    response_time_seconds: float


class ContradictRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str
    topic: str | None = None
    top_k: int | None = Field(None, ge=1, le=20)


class ContradictResponse(BaseModel):
    verdict: str  # "contradiction" | "no_contradiction" | "insufficient_evidence"
    reasoning: str
    evidence_doc_1: list[Citation]
    evidence_doc_2: list[Citation]
    response_time_seconds: float


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    pages: list[int]
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
