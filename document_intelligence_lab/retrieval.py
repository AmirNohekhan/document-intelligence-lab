import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

from .models import Chunk, RetrievedChunk


TOKEN_RE = re.compile(r"[a-zA-Z0-9$-]+")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class RetrievalIndex:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.term_freqs = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.doc_freq: Dict[str, int] = defaultdict(int)
        for tf in self.term_freqs:
            for term in tf:
                self.doc_freq[term] += 1
        self.avg_len = sum(sum(tf.values()) for tf in self.term_freqs) / max(1, len(self.term_freqs))
        self.embeddings = [self._embed(chunk.text) for chunk in self.chunks]

    def keyword(self, query: str, k: int = 6) -> List[RetrievedChunk]:
        q_terms = tokenize(query)
        scored = []
        for chunk, tf in zip(self.chunks, self.term_freqs):
            score = 0.0
            doc_len = sum(tf.values()) or 1
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (len(self.chunks) - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                denom = tf[term] + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(self.avg_len, 1))
                score += idf * (tf[term] * 2.5) / denom
            if score:
                scored.append(RetrievedChunk(chunk=chunk, score=score, source="keyword"))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def semantic(self, query: str, k: int = 6) -> List[RetrievedChunk]:
        q_emb = self._embed(query)
        scored = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            score = self._cosine(q_emb, emb)
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score, source="semantic"))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def hybrid(self, query: str, k: int = 6) -> List[RetrievedChunk]:
        merged: Dict[str, RetrievedChunk] = {}
        for result in self.keyword(query, k=k * 2) + self.semantic(query, k=k * 2):
            current = merged.get(result.chunk.chunk_id)
            if current is None:
                merged[result.chunk.chunk_id] = result
            else:
                merged[result.chunk.chunk_id] = current.model_copy(
                    update={"score": current.score + result.score, "source": "hybrid"}
                )
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:k]

    @staticmethod
    def _embed(text: str, dims: int = 64) -> List[float]:
        vector = [0.0] * dims
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % dims
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

