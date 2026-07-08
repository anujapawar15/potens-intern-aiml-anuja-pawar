"""
Core RAG pipeline for the /ask endpoint:

  1. detect language, translate question to English if needed
  2. embed the (English) question and fetch a wide candidate pool from Chroma
  3. rerank candidates with a cross-encoder, keep the true top_k
  4. build a numbered context block and ask the LLM to answer ONLY from it
  5. compute a confidence score from the reranker's relevance scores
  6. translate the answer back to the question's original language

Hallucination prevention relies on two layers: a strict system prompt that
forbids using outside knowledge and instructs an explicit "not found" reply,
and a confidence score computed independently of the LLM (from retrieval
relevance) so a low score is visible even if the model answers anyway.
"""
import time

from app.config import DEFAULT_TOP_K, RERANK_MULTIPLIER, CONFIDENCE_THRESHOLD
from app.embeddings import embed_query
from app.vector_store import query as vector_query, collection_count
from app.reranker import rerank
from app.llm import chat_completion
from app.translation import detect_language, translate_to_english, translate_from_english

NOT_FOUND_MESSAGE = "The provided documents do not contain enough information to answer this question."

SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer the user's question using ONLY the numbered context excerpts provided below.
Rules:
- Do not use any knowledge beyond the provided context, even if you know the answer.
- Every factual claim in your answer must be supported by the context and cited with the matching bracket number, e.g. [1].
- If the context does not contain enough information to answer the question, respond with exactly this sentence and nothing else: "{not_found}"
- Be concise and direct.
""".format(not_found=NOT_FOUND_MESSAGE)


def _fetch_candidates(query_embedding: list[float], n: int, where: dict | None = None) -> list[dict]:
    result = vector_query(query_embedding, n_results=n, where=where)
    candidates = []
    if not result["ids"] or not result["ids"][0]:
        return candidates
    for doc_text, meta, distance, chunk_id in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0], result["ids"][0]
    ):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "text": doc_text,
                "source": meta["source"],
                "page": meta["page"],
                "doc_id": meta["doc_id"],
                "vector_distance": distance,
            }
        )
    return candidates


def _build_context_block(ranked: list[dict]) -> str:
    lines = []
    for i, c in enumerate(ranked, start=1):
        lines.append(f"[{i}] (source: {c['source']}, page: {c['page']}):\n{c['text']}")
    return "\n\n".join(lines)


def answer_question(question: str, top_k: int | None = None) -> dict:
    start_time = time.perf_counter()
    top_k = top_k or DEFAULT_TOP_K

    detected_lang = detect_language(question)
    english_question = translate_to_english(question, detected_lang)

    if collection_count() == 0:
        return {
            "answer": "No documents have been ingested yet. Run the ingestion script first.",
            "citations": [],
            "confidence": 0.0,
            "low_confidence_warning": True,
            "detected_language": detected_lang,
            "retrieved_context": [],
            "response_time_seconds": time.perf_counter() - start_time,
        }

    query_embedding = embed_query(english_question)
    candidate_pool_size = top_k * RERANK_MULTIPLIER
    candidates = _fetch_candidates(query_embedding, n=candidate_pool_size)

    if not candidates:
        answer_en = NOT_FOUND_MESSAGE
        ranked: list[dict] = []
        confidence = 0.0
    else:
        ranked = rerank(english_question, candidates, top_k=top_k)
        confidence = sum(c["rerank_score"] for c in ranked) / len(ranked)
        context_block = _build_context_block(ranked)
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {english_question}"
        answer_en = chat_completion(SYSTEM_PROMPT, user_prompt, temperature=0.0).strip()

    final_answer = translate_from_english(answer_en, detected_lang)

    citations = [
        {
            "source": c["source"],
            "page": c["page"],
            "chunk_id": c["chunk_id"],
            "snippet": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
            "relevance_score": round(c["rerank_score"], 4),
        }
        for c in ranked
    ]

    return {
        "answer": final_answer,
        "citations": citations,
        "confidence": round(confidence, 4),
        "low_confidence_warning": confidence < CONFIDENCE_THRESHOLD,
        "detected_language": detected_lang,
        "retrieved_context": [c["text"] for c in ranked],
        "response_time_seconds": round(time.perf_counter() - start_time, 3),
    }
