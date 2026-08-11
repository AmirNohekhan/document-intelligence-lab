# Sample Experiment Report

Generated locally with:

```bash
python -m document_intelligence_lab evaluate --out reports
```

| System | Accuracy | Citation accuracy | Retrieval recall@k | Hallucination rate | Cost/query | Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| basic_rag | 0.750 | 1.000 | 0.733 | 0.000 | $0.000016 | 0.07 |
| hybrid_rag | 0.867 | 1.000 | 0.931 | 0.000 | $0.000019 | 0.09 |
| rag_reranker | 0.867 | 1.000 | 0.961 | 0.000 | $0.000019 | 0.11 |
| agentic_workflow | 0.867 | 1.000 | 0.994 | 0.000 | $0.000020 | 0.11 |

The synthetic dataset currently contains 60 documents and 120 labeled questions. The gap between `basic_rag` and the later systems is mainly driven by entity-aware retrieval and claim-scoped reasoning, which prevent unrelated exclusions from contaminating the decision context.

