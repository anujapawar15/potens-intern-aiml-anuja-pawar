"""PDF text extraction + overlapping chunk splitting. See README.md section 4 for the chunking strategy rationale."""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    page: int
    chunk_index: int
    text: str


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-based recursive-ish splitter that snaps to whitespace."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # snap to the last whitespace before `end` so we don't cut a word
            snap = text.rfind(" ", start, end)
            if snap != -1 and snap > start:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        # guarantee forward progress even if whitespace-snapping pulled `end`
        # close to `start` (otherwise a large overlap could stall the loop)
        start = max(start + 1, end - overlap)

    return chunks


def extract_and_chunk_pdf(pdf_path: Path) -> list[Chunk]:
    """Extract text page-by-page from a PDF and split into overlapping chunks."""
    doc_id = pdf_path.stem
    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []
    chunk_counter = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_chunks = _split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk_text in enumerate(page_chunks):
            chunk_counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_p{page_num}_c{idx}",
                    doc_id=doc_id,
                    source=pdf_path.name,
                    page=page_num,
                    chunk_index=chunk_counter,
                    text=chunk_text,
                )
            )
    return chunks
