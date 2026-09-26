from typing import Iterable, List, Set

from .models import Citation, Chunk, Decision
from .retrieval import tokenize


def verify_citations(decision: Decision, chunks: Iterable[Chunk]) -> Decision:
    text_by_chunk = {chunk.chunk_id: chunk.text.lower() for chunk in chunks}
    verified: List[Citation] = []
    for citation in decision.citations:
        source_text = text_by_chunk.get(citation.chunk_id, "")
        ok = citation.quote.lower() in source_text if citation.quote else False
        verified.append(citation.model_copy(update={"verified": ok}))
    return decision.model_copy(update={"citations": verified})


def hallucination_flag(decision: Decision, evidence_chunks: Iterable[Chunk]) -> bool:
    evidence_terms: Set[str] = set()
    for chunk in evidence_chunks:
        evidence_terms.update(tokenize(chunk.text))
    answer_terms = {term for term in tokenize(decision.answer) if len(term) > 4}
    boilerplate = {
        "covered",
        "not_covered",
        "needs_review",
        "review",
        "claim",
        "appears",
        "because",
        "cited",
        "evidence",
        "supports",
        "decision",
        "missing",
        "conservative",
        "should",
    }
    unsupported = answer_terms - evidence_terms - boilerplate
    return len(unsupported) > 4 or any(not citation.verified for citation in decision.citations)
