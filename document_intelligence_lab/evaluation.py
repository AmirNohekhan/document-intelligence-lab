from statistics import mean
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .chunking import chunk_documents
from .models import Chunk, EvalQuestion, EvalResult, RetrievedChunk, SystemMetrics
from .parsing import parse_documents
from .reasoning import DecisionEngine
from .reranking import rerank
from .retrieval import RetrievalIndex
from .synthetic_data import load_documents, load_eval_questions
from .tools import ToolRegistry, evidence_by_doc_type
from .verification import hallucination_flag, verify_citations


SYSTEMS = ("basic_rag", "hybrid_rag", "rag_reranker", "agentic_workflow")


def build_index(chunk_size: int = 55, overlap: int = 12) -> Tuple[List[Chunk], RetrievalIndex]:
    docs = parse_documents(load_documents())
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    return chunks, RetrievalIndex(chunks)


def infer_claim_id(question: str, index: RetrievalIndex) -> Optional[str]:
    claim_match = re.search(r"\bCLM-\d{4}\b", question, flags=re.IGNORECASE)
    if claim_match:
        return claim_match.group(0).upper()
    lower = question.lower()
    for chunk in index.chunks:
        prop = chunk.metadata.get("property", "").lower()
        if prop and prop in lower:
            return chunk.metadata.get("claim_id")
    return None


def claim_focused_results(claim_id: Optional[str], index: RetrievalIndex, score: float = 10.0) -> List[RetrievedChunk]:
    if not claim_id:
        return []
    focused = []
    for chunk in index.chunks:
        if chunk.metadata.get("claim_id") == claim_id and chunk.doc_type.value in {"policy", "claim", "inspection"}:
            focused.append(RetrievedChunk(chunk=chunk, score=score, source="entity_focus"))
    return focused


def dedupe(results: List[RetrievedChunk], k: int) -> List[RetrievedChunk]:
    seen = set()
    output = []
    for result in sorted(results, key=lambda item: item.score, reverse=True):
        if result.chunk.chunk_id in seen:
            continue
        seen.add(result.chunk.chunk_id)
        output.append(result)
        if len(output) == k:
            break
    return output


def retrieve(system: str, question: EvalQuestion, index: RetrievalIndex, k: int = 6) -> List[RetrievedChunk]:
    claim_id = infer_claim_id(question.question, index)
    if system == "basic_rag":
        return index.keyword(question.question, k=k)
    if system == "hybrid_rag":
        return dedupe(claim_focused_results(claim_id, index, score=4.0) + index.hybrid(question.question, k=k), k)
    if system == "rag_reranker":
        candidates = claim_focused_results(claim_id, index, score=5.0) + index.hybrid(question.question, k=k * 2)
        return rerank(question.question, dedupe(candidates, k=k * 2), k=k)
    if system == "agentic_workflow":
        registry = ToolRegistry()
        registry.register("evidence_by_doc_type", evidence_by_doc_type)
        candidates = claim_focused_results(claim_id, index, score=8.0) + index.hybrid(question.question, k=k * 3)
        broad = rerank(question.question, dedupe(candidates, k=k * 3), k=k * 2)
        focused = []
        for doc_type in ("policy", "claim", "inspection"):
            focused.extend(registry.call("evidence_by_doc_type", results=broad, doc_type=doc_type))
        return dedupe(focused + broad, k=k)
    raise ValueError(f"Unknown system: {system}")


def evaluate_system(system: str, questions: Iterable[EvalQuestion] = None) -> Tuple[SystemMetrics, List[EvalResult]]:
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {', '.join(SYSTEMS)}")
    chunks, index = build_index()
    engine = DecisionEngine()
    eval_questions = list(questions or load_eval_questions())
    results: List[EvalResult] = []
    for question in eval_questions:
        retrieved = retrieve(system, question, index)
        decision = engine.decide(question.question, retrieved, question_id=question.question_id)
        decision = verify_citations(decision, [item.chunk for item in retrieved])
        retrieved_doc_ids = {item.chunk.doc_id for item in retrieved}
        recall = len(set(question.required_doc_ids) & retrieved_doc_ids) / len(question.required_doc_ids)
        citation_doc_ids = {citation.doc_id for citation in decision.citations if citation.verified}
        citation_correct = bool(set(question.required_doc_ids) & citation_doc_ids) and all(c.verified for c in decision.citations)
        hallucinated = hallucination_flag(decision, [item.chunk for item in retrieved])
        results.append(
            EvalResult(
                system=system,
                question_id=question.question_id,
                expected_label=question.expected_label,
                predicted_label=decision.label,
                answer_correct=decision.label == question.expected_label,
                retrieval_recall=recall,
                citation_correct=citation_correct,
                hallucinated=hallucinated,
                abstained=decision.abstained,
                latency_ms=decision.latency_ms,
                cost_usd=decision.cost_usd,
            )
        )
    metrics = summarize(system, results)
    return metrics, results


def summarize(system: str, results: List[EvalResult]) -> SystemMetrics:
    count = len(results) or 1
    return SystemMetrics(
        system=system,
        questions=len(results),
        accuracy=sum(result.answer_correct for result in results) / count,
        retrieval_recall_at_k=mean(result.retrieval_recall for result in results),
        citation_accuracy=sum(result.citation_correct for result in results) / count,
        hallucination_rate=sum(result.hallucinated for result in results) / count,
        abstention_rate=sum(result.abstained for result in results) / count,
        avg_cost_usd=mean(result.cost_usd for result in results),
        avg_latency_ms=mean(result.latency_ms for result in results),
    )


def evaluate_all(systems: Iterable[str] = SYSTEMS) -> Tuple[List[SystemMetrics], Dict[str, List[EvalResult]]]:
    metrics = []
    by_system: Dict[str, List[EvalResult]] = {}
    for system in systems:
        system_metrics, results = evaluate_system(system)
        metrics.append(system_metrics)
        by_system[system] = results
    return metrics, by_system
