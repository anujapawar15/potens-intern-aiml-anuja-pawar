"""
Cross-encoder reranking of vector-search candidates. See README.md section 5
for why vector similarity alone isn't accurate enough on its own.
"""
import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import CROSS_ENCODER_MODEL


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    candidates: list of dicts each containing at least a "text" key.
    Adds a "rerank_score" (0-1, via sigmoid of the cross-encoder logit) to
    each candidate, sorts descending, and returns the top_k.
    """
    if not candidates:
        return []

    encoder = get_cross_encoder()
    pairs = [(query, candidate["text"]) for candidate in candidates]
    raw_scores = encoder.predict(pairs)

    for candidate, score in zip(candidates, raw_scores):
        candidate["rerank_score"] = _sigmoid(float(score))

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]
