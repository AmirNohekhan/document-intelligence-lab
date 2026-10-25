import pytest

from document_intelligence_lab.chunking import chunk_documents
from document_intelligence_lab.models import Document
from document_intelligence_lab.models import DocumentType


def _doc(n_words: int) -> Document:
    return Document(
        doc_id="D1",
        title="Doc",
        doc_type=DocumentType.POLICY,
        text=" ".join(f"w{i}" for i in range(n_words)),
    )


def test_chunks_overlap_and_cover_document():
    chunks = chunk_documents([_doc(100)], chunk_size=40, overlap=10)
    assert chunks[0].start == 0
    assert chunks[1].start == 30
    assert chunks[-1].end == 100


def test_short_document_yields_single_chunk():
    chunks = chunk_documents([_doc(5)], chunk_size=40, overlap=10)
    assert len(chunks) == 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([_doc(20)], chunk_size=size, overlap=overlap)
