"""Shared citation formatting for the /ask and /contradict responses."""

SNIPPET_LIMIT = 300


def to_citation(chunk: dict) -> dict:
    text = chunk["text"]
    snippet = text[:SNIPPET_LIMIT] + ("..." if len(text) > SNIPPET_LIMIT else "")
    return {
        "source": chunk["source"],
        "page": chunk["page"],
        "chunk_id": chunk["chunk_id"],
        "snippet": snippet,
        "relevance_score": round(chunk.get("rerank_score", 1.0), 4),
    }


def to_citations(chunks: list[dict]) -> list[dict]:
    return [to_citation(chunk) for chunk in chunks]
