import pytest

from document_intelligence_lab.chunking import chunk_documents
from document_intelligence_lab.models import Document, DocumentType


def make_doc(words: int) -> Document:
    text = " ".join(f"w{i}" for i in range(words))
    return Document(doc_id="D1", title="Doc", doc_type=DocumentType.POLICY, text=text)


def test_chunks_overlap_and_cover_whole_document():
    chunks = chunk_documents([make_doc(20)], chunk_size=10, overlap=4)
    assert [(c.start, c.end) for c in chunks] == [(0, 10), (6, 16), (12, 20)]
    assert chunks[-1].text.split()[-1] == "w19"


def test_short_document_yields_single_chunk():
    chunks = chunk_documents([make_doc(5)], chunk_size=10, overlap=4)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "D1#0000"


def test_empty_document_yields_no_chunks():
    assert chunk_documents([make_doc(0)]) == []


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_parameters_raise(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([make_doc(5)], chunk_size=chunk_size, overlap=overlap)
