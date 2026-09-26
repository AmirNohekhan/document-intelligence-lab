import re
from collections import Counter
from typing import Dict, Iterable, List

from .models import Document, DocumentType


TYPE_HINTS: Dict[DocumentType, List[str]] = {
    DocumentType.POLICY: ["policy", "covers", "excludes", "insured", "endorsement"],
    DocumentType.CLAIM: ["claim", "reported", "requested amount", "notice"],
    DocumentType.INSPECTION: ["inspection", "observed", "consistent", "findings"],
    DocumentType.LEASE: ["lease", "tenant", "alterations"],
    DocumentType.INVOICE: ["invoice", "vendor", "line items", "total"],
    DocumentType.EMAIL: ["from:", "subject:", "email"],
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def classify_document(text: str) -> DocumentType:
    lower = text.lower()
    scores = {
        doc_type: sum(1 for hint in hints if hint in lower)
        for doc_type, hints in TYPE_HINTS.items()
    }
    return max(scores.items(), key=lambda item: item[1])[0]


def parse_documents(raw_docs: Iterable[Document]) -> List[Document]:
    parsed: List[Document] = []
    for doc in raw_docs:
        doc_type = doc.doc_type or classify_document(doc.text)
        parsed.append(
            doc.model_copy(update={"doc_type": doc_type, "text": normalize_text(doc.text)})
        )
    return parsed


def corpus_summary(docs: Iterable[Document]) -> Dict[str, int]:
    counts = Counter(doc.doc_type.value for doc in docs)
    counts["documents"] = sum(counts.values())
    return dict(counts)

