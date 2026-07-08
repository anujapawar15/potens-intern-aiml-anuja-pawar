"""
Core pipeline for the /contradict endpoint.

Retrieves the most relevant chunks from each of the two requested documents
(filtered by topic if one is given, otherwise all chunks up to a cap), then
asks the LLM to judge whether the two sets of excerpts contradict each
other. The prompt forces a JSON response with an explicit
"insufficient_evidence" option so the model states uncertainty instead of
guessing when the retrieved excerpts don't overlap in subject matter.
"""
import time

from app.config import DEFAULT_TOP_K
from app.embeddings import embed_query
from app.vector_store import query as vector_query, get_by_doc_id
from app.reranker import rerank
from app.llm import chat_completion_json, LLMError

MAX_CHUNKS_NO_TOPIC = 10

VALID_VERDICTS = {"contradiction", "no_contradiction", "insufficient_evidence"}

SYSTEM_PROMPT = """You compare excerpts from two documents and decide whether they contradict each other.
Rules:
- Base your judgment ONLY on the provided excerpts. Do not assume facts not present in the text.
- If the two documents don't actually discuss the same topic/rule, or there isn't enough overlapping information to judge, set verdict to "insufficient_evidence" and explain why in reasoning. Do not guess.
- Otherwise set verdict to "contradiction" or "no_contradiction".
- Respond ONLY with a JSON object of the form:
{"verdict": "contradiction" | "no_contradiction" | "insufficient_evidence", "reasoning": "<clear explanation citing bracket labels like [A1], [B2]>"}
"""


def _fetch_doc_chunks(doc_id: str, topic: str | None, top_k: int) -> list[dict]:
    if topic:
        embedding = embed_query(topic)
        result = vector_query(embedding, n_results=top_k * 3, where={"doc_id": doc_id})
        candidates = []
        if result["ids"] and result["ids"][0]:
            for text, meta, chunk_id in zip(result["documents"][0], result["metadatas"][0], result["ids"][0]):
                candidates.append({"chunk_id": chunk_id, "text": text, "source": meta["source"], "page": meta["page"]})
        return rerank(topic, candidates, top_k=top_k) if candidates else []

    raw = get_by_doc_id(doc_id, limit=MAX_CHUNKS_NO_TOPIC)
    chunks = []
    for text, meta, chunk_id in zip(raw["documents"], raw["metadatas"], raw["ids"]):
        chunks.append({"chunk_id": chunk_id, "text": text, "source": meta["source"], "page": meta["page"], "rerank_score": 1.0})
    chunks.sort(key=lambda c: c["chunk_id"])
    return chunks[:MAX_CHUNKS_NO_TOPIC]


def _build_labeled_block(chunks: list[dict], prefix: str) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{prefix}{i}] (source: {c['source']}, page: {c['page']}):\n{c['text']}")
    return "\n\n".join(lines)


def _to_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source": c["source"],
            "page": c["page"],
            "chunk_id": c["chunk_id"],
            "snippet": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
            "relevance_score": round(c.get("rerank_score", 1.0), 4),
        }
        for c in chunks
    ]


def compare_documents(doc_id_1: str, doc_id_2: str, topic: str | None = None, top_k: int | None = None) -> dict:
    start_time = time.perf_counter()
    top_k = top_k or DEFAULT_TOP_K

    chunks_1 = _fetch_doc_chunks(doc_id_1, topic, top_k)
    chunks_2 = _fetch_doc_chunks(doc_id_2, topic, top_k)

    if not chunks_1 or not chunks_2:
        missing = doc_id_1 if not chunks_1 else doc_id_2
        return {
            "verdict": "insufficient_evidence",
            "reasoning": f"No content could be retrieved for document '{missing}'. Check that the doc_id is correct and the document has been ingested.",
            "evidence_doc_1": _to_citations(chunks_1),
            "evidence_doc_2": _to_citations(chunks_2),
            "response_time_seconds": round(time.perf_counter() - start_time, 3),
        }

    block_1 = _build_labeled_block(chunks_1, "A")
    block_2 = _build_labeled_block(chunks_2, "B")
    topic_line = f"Topic to focus on: {topic}\n\n" if topic else ""
    user_prompt = (
        f"{topic_line}Document A excerpts (doc_id={doc_id_1}):\n{block_1}\n\n"
        f"Document B excerpts (doc_id={doc_id_2}):\n{block_2}\n\n"
        "Do these two documents contradict each other on this subject?"
    )

    try:
        result = chat_completion_json(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    except LLMError as exc:
        return {
            "verdict": "insufficient_evidence",
            "reasoning": f"Could not obtain a judgment from the LLM: {exc}",
            "evidence_doc_1": _to_citations(chunks_1),
            "evidence_doc_2": _to_citations(chunks_2),
            "response_time_seconds": round(time.perf_counter() - start_time, 3),
        }

    verdict = result.get("verdict", "insufficient_evidence")
    if verdict not in VALID_VERDICTS:
        verdict = "insufficient_evidence"
    reasoning = result.get("reasoning", "The model did not return a reasoning field.")

    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "evidence_doc_1": _to_citations(chunks_1),
        "evidence_doc_2": _to_citations(chunks_2),
        "response_time_seconds": round(time.perf_counter() - start_time, 3),
    }
