from document_intelligence_lab.chunking import chunk_documents
from document_intelligence_lab.models import (
    Chunk,
    Citation,
    Decision,
    DecisionLabel,
    Document,
    DocumentType,
)
from document_intelligence_lab.verification import verify_citations


def _doc(words: int) -> Document:
    return Document(
        doc_id="D1",
        title="Doc",
        doc_type=DocumentType.POLICY,
        text=" ".join(f"w{i}" for i in range(words)),
    )


def test_chunking_overlaps_and_covers_whole_document():
    chunks = chunk_documents([_doc(100)], chunk_size=40, overlap=10)
    assert chunks[0].start == 0 and chunks[0].end == 40
    assert chunks[1].start == 30
    assert chunks[-1].end == 100
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunking_empty_document_yields_no_chunks():
    assert chunk_documents([_doc(0)]) == []


def test_chunking_survives_overlap_larger_than_chunk_size():
    chunks = chunk_documents([_doc(10)], chunk_size=4, overlap=9)
    assert chunks[-1].end == 10


def _decision(quote: str) -> Decision:
    return Decision(
        label=DecisionLabel.COVERED,
        answer="covered",
        confidence=0.9,
        prompt_version="v1",
        model_name="test",
        cost_usd=0.0,
        latency_ms=0.0,
        citations=[Citation(doc_id="D1", chunk_id="D1#0000", quote=quote, supports="x")],
    )


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="D1#0000",
        doc_id="D1",
        doc_type=DocumentType.POLICY,
        title="Doc",
        text="Water damage is covered under Section 4.",
        start=0,
        end=7,
    )


def test_verify_citations_accepts_case_insensitive_quote():
    result = verify_citations(_decision("WATER DAMAGE is covered"), [_chunk()])
    assert result.citations[0].verified is True


def test_verify_citations_rejects_missing_quote_and_empty_quote():
    assert verify_citations(_decision("fire damage"), [_chunk()]).citations[0].verified is False
    assert verify_citations(_decision(""), [_chunk()]).citations[0].verified is False
