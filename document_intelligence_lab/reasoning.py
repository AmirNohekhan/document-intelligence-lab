import time
from typing import List, Optional

from .models import Citation, Decision, DecisionLabel, RetrievedChunk
from .prompts import DEFAULT_PROMPT_VERSION


class DecisionEngine:
    model_name = "local-evidence-reasoner"

    def decide(
        self,
        question: str,
        retrieved: List[RetrievedChunk],
        question_id: Optional[str] = None,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> Decision:
        started = time.perf_counter()
        raw_evidence = " ".join(result.chunk.text.lower() for result in retrieved)
        claim_id = self._extract_claim_id(question, raw_evidence)
        scoped = self._scope_to_claim(retrieved, claim_id)
        evidence = " ".join(result.chunk.text.lower() for result in scoped)
        label = self._label(evidence)
        required_types = {"policy", "claim", "inspection"}
        present_types = {result.chunk.doc_type.value for result in scoped}
        missing = sorted(required_types - present_types)
        if missing or (label == DecisionLabel.NEEDS_REVIEW):
            confidence = 0.58 if label == DecisionLabel.NEEDS_REVIEW else 0.62
            abstained = label == DecisionLabel.NEEDS_REVIEW
        else:
            confidence = 0.84 if label != DecisionLabel.NEEDS_REVIEW else 0.55
            abstained = False

        citations = self._build_citations(scoped)
        answer = self._answer(label, citations, missing)
        latency_ms = (time.perf_counter() - started) * 1000
        token_estimate = sum(len(result.chunk.text.split()) for result in scoped) + len(question.split()) + len(answer.split())
        cost_usd = token_estimate / 1000 * 0.00015
        return Decision(
            question_id=question_id,
            claim_id=claim_id,
            label=label,
            answer=answer,
            confidence=confidence,
            citations=citations,
            missing_evidence=missing,
            abstained=abstained,
            prompt_version=prompt_version,
            model_name=self.model_name,
            cost_usd=round(cost_usd, 6),
            latency_ms=round(latency_ms, 3),
        )

    @staticmethod
    def _extract_claim_id(question: str, evidence: str) -> Optional[str]:
        combined = f"{question} {evidence}".upper().split()
        for token in combined:
            cleaned = token.strip(".,:;()")
            if cleaned.startswith("CLM-"):
                return cleaned
        return None

    @staticmethod
    def _scope_to_claim(retrieved: List[RetrievedChunk], claim_id: Optional[str]) -> List[RetrievedChunk]:
        if not claim_id:
            return retrieved
        scoped = [result for result in retrieved if result.chunk.metadata.get("claim_id") == claim_id]
        return scoped if scoped else retrieved

    @staticmethod
    def _label(evidence: str) -> DecisionLabel:
        if "requires an endorsement" in evidence or "schedule page missing" in evidence or "endorsement status not present" in evidence:
            return DecisionLabel.NEEDS_REVIEW
        not_covered_markers = [
            "policy excludes",
            "excludes loss",
            "wear and tear",
            "flood or surface water",
            "earth movement",
            "mold from unresolved humidity",
        ]
        covered_markers = [
            "policy covers",
            "covers direct physical loss",
            "building was occupied",
            "policy includes sewer backup sublimit",
            "no evidence of intent",
        ]
        if any(marker in evidence for marker in not_covered_markers):
            return DecisionLabel.NOT_COVERED
        if any(marker in evidence for marker in covered_markers):
            return DecisionLabel.COVERED
        return DecisionLabel.NEEDS_REVIEW

    @staticmethod
    def _build_citations(retrieved: List[RetrievedChunk]) -> List[Citation]:
        citations: List[Citation] = []
        for result in retrieved:
            if result.chunk.doc_type.value not in {"policy", "claim", "inspection"}:
                continue
            quote = result.chunk.text.split(".")[0].strip()
            if quote:
                citations.append(
                    Citation(
                        doc_id=result.chunk.doc_id,
                        chunk_id=result.chunk.chunk_id,
                        quote=quote,
                        supports=result.chunk.doc_type.value,
                    )
                )
            if len(citations) == 3:
                break
        return citations

    @staticmethod
    def _answer(label: DecisionLabel, citations: List[Citation], missing: List[str]) -> str:
        if missing:
            return f"{label.value}: missing {', '.join(missing)} evidence, so the decision should be conservative."
        support = "; ".join(citation.quote for citation in citations[:2])
        return f"{label.value}: the cited evidence supports this decision. {support}"
