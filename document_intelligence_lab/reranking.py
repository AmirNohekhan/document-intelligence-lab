from typing import Iterable, List

from .models import RetrievedChunk
from .retrieval import tokenize


IMPORTANT_TYPES = {"policy": 1.25, "inspection": 1.15, "claim": 1.1, "invoice": 0.95, "lease": 0.9}


def rerank(query: str, results: Iterable[RetrievedChunk], k: int = 6) -> List[RetrievedChunk]:
    q_terms = set(tokenize(query))
    reranked = []
    for result in results:
        chunk_terms = set(tokenize(result.chunk.text))
        overlap = len(q_terms & chunk_terms) / max(1, len(q_terms))
        type_boost = IMPORTANT_TYPES.get(result.chunk.doc_type.value, 1.0)
        claim_boost = 1.25 if any(term.startswith("clm-") and term in chunk_terms for term in q_terms) else 1.0
        score = result.score * type_boost * claim_boost + overlap
        reranked.append(result.copy(update={"score": score, "source": f"{result.source}+rerank"}))
    return sorted(reranked, key=lambda item: item.score, reverse=True)[:k]

