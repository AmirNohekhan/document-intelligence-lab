from document_intelligence_lab.chunking import chunk_documents
from document_intelligence_lab.models import Document, DocumentType


def _doc(n_words: int, doc_id: str = "D1") -> Document:
    text = " ".join(f"w{i}" for i in range(n_words))
    return Document(doc_id=doc_id, title="t", doc_type=DocumentType.POLICY, text=text)


def test_short_document_yields_single_chunk():
    chunks = chunk_documents([_doc(10)], chunk_size=55, overlap=12)
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0, 10)


def test_empty_document_yields_no_chunks():
    assert chunk_documents([_doc(0)]) == []


def test_chunks_overlap_and_cover_whole_document():
    chunks = chunk_documents([_doc(120)], chunk_size=50, overlap=10)
    assert chunks[0].start == 0
    assert chunks[-1].end == 120
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.end - nxt.start == 10


def test_overlap_not_smaller_than_chunk_size_still_terminates():
    chunks = chunk_documents([_doc(20)], chunk_size=5, overlap=9)
    assert chunks[-1].end == 20


def test_chunk_ids_are_unique_across_documents():
    chunks = chunk_documents([_doc(100, "A"), _doc(100, "B")], chunk_size=30, overlap=5)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.doc_id for c in chunks} == {"A", "B"}
