from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    POLICY = "policy"
    CLAIM = "claim"
    INSPECTION = "inspection"
    LEASE = "lease"
    INVOICE = "invoice"
    EMAIL = "email"


class DecisionLabel(str, Enum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    NEEDS_REVIEW = "needs_review"


class Document(BaseModel):
    doc_id: str
    title: str
    doc_type: DocumentType
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: DocumentType
    title: str
    text: str
    start: int
    end: int
    metadata: Dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    quote: str
    supports: str
    verified: bool = False


class Decision(BaseModel):
    question_id: Optional[str] = None
    claim_id: Optional[str] = None
    label: DecisionLabel
    answer: str
    confidence: float = Field(ge=0, le=1)
    citations: List[Citation]
    missing_evidence: List[str] = Field(default_factory=list)
    abstained: bool = False
    prompt_version: str
    model_name: str
    cost_usd: float
    latency_ms: float


class EvalQuestion(BaseModel):
    question_id: str
    question: str
    claim_id: str
    expected_label: DecisionLabel
    required_doc_ids: List[str]
    required_terms: List[str]
    answer_regex: str


class EvalResult(BaseModel):
    system: str
    question_id: str
    expected_label: DecisionLabel
    predicted_label: DecisionLabel
    answer_correct: bool
    retrieval_recall: float
    citation_correct: bool
    hallucinated: bool
    abstained: bool
    latency_ms: float
    cost_usd: float


class SystemMetrics(BaseModel):
    system: str
    questions: int
    accuracy: float
    retrieval_recall_at_k: float
    citation_accuracy: float
    hallucination_rate: float
    abstention_rate: float
    avg_cost_usd: float
    avg_latency_ms: float

