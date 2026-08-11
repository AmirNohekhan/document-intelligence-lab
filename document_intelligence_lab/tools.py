from typing import Callable, Dict, List

from .models import RetrievedChunk


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[..., object]] = {}

    def register(self, name: str, fn: Callable[..., object]) -> None:
        self._tools[name] = fn

    def call(self, name: str, **kwargs) -> object:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name](**kwargs)

    def names(self) -> List[str]:
        return sorted(self._tools)


def evidence_by_doc_type(results: List[RetrievedChunk], doc_type: str) -> List[RetrievedChunk]:
    return [result for result in results if result.chunk.doc_type.value == doc_type]

