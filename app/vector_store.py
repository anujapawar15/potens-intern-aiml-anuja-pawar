"""
ChromaDB persistent vector store wrapper. Embeddings are computed by
app.embeddings rather than via Chroma's built-in embedding function, so the
same model instance and reranking step are shared between ingestion and
query time.
"""
from functools import lru_cache

import chromadb

from app.config import CHROMA_DIR, COLLECTION_NAME


@lru_cache(maxsize=1)
def get_client() -> "chromadb.ClientAPI":
    return chromadb.PersistentClient(
        path=CHROMA_DIR, settings=chromadb.Settings(anonymized_telemetry=False)
    )


def get_collection():
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def reset_collection():
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return get_collection()


def add_chunks(ids, embeddings, documents, metadatas, batch_size: int = 100):
    collection = get_collection()
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def query(embedding: list[float], n_results: int, where: dict | None = None):
    collection = get_collection()
    return collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def get_by_doc_id(doc_id: str, limit: int = 50):
    collection = get_collection()
    return collection.get(where={"doc_id": doc_id}, limit=limit, include=["documents", "metadatas"])


def list_documents() -> dict:
    """Returns {doc_id: {source, pages, chunk_count}} across the whole collection."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    docs: dict = {}
    for meta in result["metadatas"]:
        doc_id = meta["doc_id"]
        entry = docs.setdefault(doc_id, {"source": meta["source"], "pages": set(), "chunk_count": 0})
        entry["pages"].add(meta["page"])
        entry["chunk_count"] += 1
    for entry in docs.values():
        entry["pages"] = sorted(entry["pages"])
    return docs


def collection_count() -> int:
    return get_collection().count()
