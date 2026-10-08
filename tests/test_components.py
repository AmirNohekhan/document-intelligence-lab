import pytest

from document_intelligence_lab.chunking import chunk_documents
from document_intelligence_lab.models import (
    Chunk,
    Citation,
    Decision,
    DecisionLabel,
    Document,
    DocumentType,
)
from document_intelligence_lab.parsing import classify_document, corpus_summary, normalize_text
from document_intelligence_lab.retrieval import RetrievalIndex, tokenize
from document_intelligence_lab.tools import ToolRegistry
from document_intelligence_lab.verification import hallucination_flag, verify_citations


def _doc(doc_id: str, text: str, doc_type: DocumentType = DocumentType.POLICY) -> Document:
    return Document(doc_id=doc_id, title=doc_id, doc_type=doc_type, text=text)


def _chunk(chunk_id: str, text: str, doc_type: DocumentType = DocumentType.POLICY) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="D1", doc_type=doc_type, title="t", text=text, start=0, end=1
    )


def _decision(answer: str, citations) -> Decision:
    return Decision(
        label=DecisionLabel.COVERED,
        answer=answer,
        confidence=0.9,
        citations=citations,
        prompt_version="v1",
        model_name="test",
        cost_usd=0.0,
        latency_ms=0.0,
    )


def test_chunking_overlaps_and_covers_whole_document():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_documents([_doc("D1", " ".join(words))], chunk_size=40, overlap=10)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert chunks[-1].end == 100
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunking_handles_empty_text_and_degenerate_overlap():
    assert chunk_documents([_doc("D1", "   ")]) == []
    chunks = chunk_documents([_doc("D1", "a b c d")], chunk_size=2, overlap=5)
    assert chunks[-1].end == 4


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a \n\t b  ") == "a b"


def test_classify_document_uses_keyword_hints():
    assert classify_document("Invoice from vendor with line items and total") == DocumentType.INVOICE
    assert classify_document("This policy covers fire and excludes flood") == DocumentType.POLICY


def test_corpus_summary_counts_by_type():
    docs = [_doc("A", "x"), _doc("B", "y"), _doc("C", "z", DocumentType.CLAIM)]
    assert corpus_summary(docs) == {"policy": 2, "claim": 1, "documents": 3}


def test_tokenize_lowercases_and_keeps_ids_and_amounts():
    assert tokenize("Claim CLM-1001 for $5,000") == ["claim", "clm-1001", "for", "$5", "000"]


def test_keyword_retrieval_ranks_matching_chunk_first():
    index = RetrievalIndex(
        [_chunk("c1", "fire damage covered by policy"), _chunk("c2", "tenant parking rules")]
    )
    results = index.keyword("fire damage")
    assert [r.chunk.chunk_id for r in results] == ["c1"]
    assert index.keyword("zzz") == []


def test_hybrid_retrieval_merges_sources_and_respects_k():
    index = RetrievalIndex([_chunk(f"c{i}", f"water damage claim {i}") for i in range(5)])
    results = index.hybrid("water damage", k=3)
    assert len(results) == 3
    assert all(r.source == "hybrid" for r in results)


def test_verify_citations_marks_only_quotes_found_in_chunk():
    chunk = _chunk("c1", "The policy covers Sudden Water Damage.")
    decision = _decision(
        "covered",
        [
            Citation(doc_id="D1", chunk_id="c1", quote="sudden water damage", supports="x"),
            Citation(doc_id="D1", chunk_id="c1", quote="flood", supports="x"),
            Citation(doc_id="D1", chunk_id="missing", quote="policy", supports="x"),
            Citation(doc_id="D1", chunk_id="c1", quote="", supports="x"),
        ],
    )
    verified = verify_citations(decision, [chunk])
    assert [c.verified for c in verified.citations] == [True, False, False, False]


def test_hallucination_flag_detects_unverified_citation_and_unsupported_terms():
    chunk = _chunk("c1", "water damage covered")
    good = Citation(doc_id="D1", chunk_id="c1", quote="water", supports="x", verified=True)
    bad = good.copy(update={"verified": False})
    assert not hallucination_flag(_decision("water damage covered", [good]), [chunk])
    assert hallucination_flag(_decision("water damage covered", [bad]), [chunk])
    invented = "earthquake tsunami volcano meteor lightning hurricane"
    assert hallucination_flag(_decision(invented, [good]), [chunk])


def test_tool_registry_dispatches_and_rejects_unknown_tools():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)
    assert registry.call("add", a=1, b=2) == 3
    assert registry.names() == ["add"]
    with pytest.raises(KeyError, match="Unknown tool"):
        registry.call("nope")
