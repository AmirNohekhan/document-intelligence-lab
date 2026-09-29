from document_intelligence_lab.models import (
    Chunk,
    Citation,
    Decision,
    DecisionLabel,
    DocumentType,
)
from document_intelligence_lab.verification import hallucination_flag, verify_citations


def _chunk(text: str = "Water damage is covered under section four of the policy.") -> Chunk:
    return Chunk(
        chunk_id="POL-1#0000",
        doc_id="POL-1",
        doc_type=DocumentType.POLICY,
        title="Policy",
        text=text,
        start=0,
        end=10,
    )


def _decision(quote: str, chunk_id: str = "POL-1#0000") -> Decision:
    return Decision(
        label=DecisionLabel.COVERED,
        answer="covered",
        confidence=0.9,
        citations=[Citation(doc_id="POL-1", chunk_id=chunk_id, quote=quote, supports="coverage")],
        prompt_version="v1",
        model_name="test",
        cost_usd=0.0,
        latency_ms=0.0,
    )


def test_quote_present_in_chunk_is_verified_case_insensitively():
    result = verify_citations(_decision("WATER DAMAGE is covered"), [_chunk()])
    assert result.citations[0].verified is True


def test_quote_missing_from_chunk_is_not_verified():
    result = verify_citations(_decision("flood damage is excluded"), [_chunk()])
    assert result.citations[0].verified is False


def test_unknown_chunk_id_is_not_verified():
    result = verify_citations(_decision("water damage", chunk_id="MISSING#0000"), [_chunk()])
    assert result.citations[0].verified is False


def test_empty_quote_is_not_verified():
    result = verify_citations(_decision(""), [_chunk()])
    assert result.citations[0].verified is False


def test_unverified_citation_triggers_hallucination_flag():
    decision = verify_citations(_decision("flood damage is excluded"), [_chunk()])
    assert hallucination_flag(decision, [_chunk()]) is True
