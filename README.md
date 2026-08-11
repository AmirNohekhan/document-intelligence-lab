# Document Intelligence Lab

An evaluation-first LLM engineering project for messy insurance, financial, and legal documents.

The system ingests synthetic leases, inspection reports, claims notes, invoices, policies, and emails, then answers decision questions such as:

> Does this claim appear covered, and what evidence supports the conclusion?

This is intentionally not another “chat with your PDF” demo. It is a small business-style decision system with retrieval experiments, structured outputs, citations, confidence, abstention, hallucination checks, and repeatable evaluation.

## What It Shows

- Pydantic structured output schemas
- Document parsing and classification
- Chunking experiments
- Keyword, semantic, hybrid, reranked, and agentic retrieval workflows
- Tool/function calling via a local tool registry
- Multi-document reasoning across policies, leases, inspection reports, claims, and invoices
- Citation verification and hallucination detection
- Confidence scoring and abstention
- Prompt versioning
- Cost and latency tracking
- Evaluation over 120 synthetic labeled questions

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m document_intelligence_lab evaluate --out reports
python -m document_intelligence_lab ask "Is claim CLM-1001 covered?"
```

On macOS/Linux, use `source .venv/bin/activate` instead of the Windows activation command.

## Example Commands

Run all experiment variants:

```bash
python -m document_intelligence_lab evaluate --out reports
```

Run one variant:

```bash
python -m document_intelligence_lab evaluate --system hybrid_rag --out reports
```

Ask an ad hoc question:

```bash
python -m document_intelligence_lab ask "Does the water damage claim at Harbor Lofts appear covered?"
```

Inspect generated synthetic data:

```bash
python -m document_intelligence_lab data-summary
```

## Experiment Matrix

The evaluator compares:

| System | Description |
| --- | --- |
| `basic_rag` | Keyword retrieval only |
| `hybrid_rag` | Keyword + semantic retrieval |
| `rag_reranker` | Hybrid retrieval plus evidence-aware reranking |
| `agentic_workflow` | Tool-driven classification, retrieval, reranking, verification, and abstention |

Metrics:

- Answer accuracy
- Retrieval recall@k
- Citation accuracy
- Hallucination rate
- Abstention rate
- Average cost/query
- Average latency

A sample local run is included in [docs/sample_experiment_report.md](docs/sample_experiment_report.md).

## Project Structure

```text
document_intelligence_lab/
  __main__.py            CLI entrypoint
  app.py                 CLI commands
  synthetic_data.py      Public-safe synthetic documents and eval questions
  models.py              Pydantic schemas
  parsing.py             Document parser and classifier
  chunking.py            Chunking strategies
  retrieval.py           Keyword, semantic, and hybrid retrieval
  reranking.py           Evidence-aware reranker
  reasoning.py           Structured decision engine
  tools.py               Function-calling/tool registry
  verification.py        Citation and hallucination checks
  evaluation.py          Metrics and experiment runner
  reporting.py           Markdown/CSV report writer
  prompts.py             Prompt version registry
```

## Notes

The default reasoning backend is deterministic so that evaluation results are reproducible in a portfolio repo. The interfaces are designed so an OpenAI or other LLM provider can be added behind the same `DecisionEngine` contract.
