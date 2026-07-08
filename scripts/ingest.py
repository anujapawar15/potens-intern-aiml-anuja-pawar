"""
Ingestion pipeline: PDF -> extract -> chunk -> embed -> store in ChromaDB.

Run: python scripts/ingest.py
(Re-run any time to re-ingest; it resets the collection first so re-running
is idempotent and never produces duplicate chunks.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PDF_DIR
from app.chunking import extract_and_chunk_pdf
from app.embeddings import embed_texts
from app.vector_store import reset_collection, add_chunks


def main() -> None:
    pdf_dir = Path(PDF_DIR)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}.")
        print("Run 'python scripts/generate_sample_pdfs.py' first, or add your own PDFs there.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s) in {pdf_dir}:")
    for p in pdf_paths:
        print(f"  - {p.name}")

    all_chunks = []
    for pdf_path in pdf_paths:
        try:
            chunks = extract_and_chunk_pdf(pdf_path)
        except Exception as exc:
            print(f"  [WARN] failed to process {pdf_path.name}: {exc}")
            continue
        if not chunks:
            print(f"  [WARN] no extractable text in {pdf_path.name}, skipping")
            continue
        print(f"  {pdf_path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced from any PDF. Aborting.")
        sys.exit(1)

    print(f"\nEmbedding {len(all_chunks)} chunks ...")
    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts)

    print("Resetting vector store collection ...")
    reset_collection()

    print("Writing chunks + embeddings + metadata to ChromaDB ...")
    add_chunks(
        ids=[c.chunk_id for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"doc_id": c.doc_id, "source": c.source, "page": c.page, "chunk_index": c.chunk_index}
            for c in all_chunks
        ],
    )

    print(f"\nDone. Ingested {len(all_chunks)} chunks from {len(pdf_paths)} document(s).")


if __name__ == "__main__":
    main()
