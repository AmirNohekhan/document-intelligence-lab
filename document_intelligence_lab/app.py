import argparse
import json
from typing import List

from .evaluation import SYSTEMS, build_index, evaluate_all, evaluate_system, retrieve
from .models import EvalQuestion
from .parsing import corpus_summary, parse_documents
from .reasoning import DecisionEngine
from .reporting import write_reports
from .synthetic_data import load_documents, load_eval_questions
from .verification import verify_citations


def cmd_evaluate(args: argparse.Namespace) -> None:
    systems: List[str] = [args.system] if args.system else list(SYSTEMS)
    metrics, results = evaluate_all(systems)
    write_reports(args.out, metrics, results)
    for row in metrics:
        print(
            f"{row.system}: accuracy={row.accuracy:.3f}, citation_accuracy={row.citation_accuracy:.3f}, "
            f"recall@k={row.retrieval_recall_at_k:.3f}, hallucination_rate={row.hallucination_rate:.3f}, "
            f"cost=${row.avg_cost_usd:.6f}, latency={row.avg_latency_ms:.2f}ms"
        )
    print(f"Reports written to {args.out}")


def cmd_ask(args: argparse.Namespace) -> None:
    chunks, index = build_index()
    question = EvalQuestion(
        question_id="ad-hoc",
        question=args.question,
        claim_id="",
        expected_label="needs_review",
        required_doc_ids=[],
        required_terms=[],
        answer_regex="",
    )
    retrieved = retrieve(args.system, question, index)
    decision = DecisionEngine().decide(args.question, retrieved, question_id="ad-hoc")
    decision = verify_citations(decision, [item.chunk for item in retrieved])
    print(json.dumps(decision.dict(), indent=2, default=str))


def cmd_data_summary(_: argparse.Namespace) -> None:
    docs = parse_documents(load_documents())
    questions = load_eval_questions()
    summary = corpus_summary(docs)
    summary["eval_questions"] = len(questions)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document Intelligence Lab")
    sub = parser.add_subparsers(required=True)

    evaluate = sub.add_parser("evaluate", help="Run evaluation experiments")
    evaluate.add_argument("--system", choices=SYSTEMS)
    evaluate.add_argument("--out", default="reports")
    evaluate.set_defaults(func=cmd_evaluate)

    ask = sub.add_parser("ask", help="Ask an ad hoc coverage question")
    ask.add_argument("question")
    ask.add_argument("--system", choices=SYSTEMS, default="agentic_workflow")
    ask.set_defaults(func=cmd_ask)

    data_summary = sub.add_parser("data-summary", help="Show corpus and evaluation dataset size")
    data_summary.set_defaults(func=cmd_data_summary)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

