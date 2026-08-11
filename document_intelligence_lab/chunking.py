from typing import Iterable, List

from .models import Chunk, Document


def chunk_documents(docs: Iterable[Document], chunk_size: int = 55, overlap: int = 12) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        words = doc.text.split()
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            end = min(len(words), start + chunk_size)
            text = " ".join(words[start:end])
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):04d}",
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    title=doc.title,
                    text=text,
                    start=start,
                    end=end,
                    metadata=doc.metadata,
                )
            )
            if end == len(words):
                break
    return chunks

