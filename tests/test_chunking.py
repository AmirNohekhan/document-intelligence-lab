from document_intelligence_lab.chunking import chunk_documents
from document_intelligence_lab.models import Document, DocumentType


def _doc(word_count: int, doc_id: str = "DOC-1") -> Document:
    text = " ".join(f"w{i}" for i in range(word_count))
    return Document(doc_id=doc_id, title="Test", doc_type=DocumentType.POLICY, text=text)


def test_short_document_yields_single_chunk():
    chunks = chunk_documents([_doc(10)])
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == 10


def test_empty_document_yields_no_chunks():
    assert chunk_documents([_doc(0)]) == []


def test_chunks_overlap_and_cover_document():
    chunks = chunk_documents([_doc(100)], chunk_size=55, overlap=12)
    assert [(c.start, c.end) for c in chunks] == [(0, 55), (43, 98), (86, 100)]
    assert chunks[-1].end == 100


def test_chunk_ids_are_sequential_and_reference_document():
    chunks = chunk_documents([_doc(100)])
    assert [c.chunk_id for c in chunks] == ["DOC-1#0000", "DOC-1#0001", "DOC-1#0002"]
    assert all(c.doc_id == "DOC-1" for c in chunks)


def test_overlap_not_smaller_than_chunk_size_still_terminates():
    chunks = chunk_documents([_doc(20)], chunk_size=5, overlap=10)
    assert chunks
    assert chunks[-1].end == 20
