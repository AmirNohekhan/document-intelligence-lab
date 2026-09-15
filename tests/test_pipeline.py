from document_intelligence_lab.evaluation import evaluate_system
from document_intelligence_lab.models import DecisionLabel, EvalQuestion
from document_intelligence_lab.synthetic_data import load_eval_questions


def test_eval_dataset_has_120_questions():
    assert len(load_eval_questions()) == 120


def test_agentic_workflow_runs_and_beats_minimum_accuracy():
    metrics, results = evaluate_system("agentic_workflow")
    assert len(results) == 120
    assert metrics.accuracy >= 0.70
    assert metrics.citation_accuracy >= 0.70


def test_evaluate_system_handles_question_with_no_required_doc_ids():
    # Ad hoc questions (like the ones built by the `ask` CLI command) have no
    # required_doc_ids. This used to raise ZeroDivisionError when computing
    # retrieval recall instead of treating an empty requirement as satisfied.
    question = EvalQuestion(
        question_id="adhoc-1",
        question="Is claim CLM-1001 covered?",
        claim_id="",
        expected_label=DecisionLabel.NEEDS_REVIEW,
        required_doc_ids=[],
        required_terms=[],
        answer_regex="",
    )
    metrics, results = evaluate_system("hybrid_rag", questions=[question])
    assert len(results) == 1
    assert results[0].retrieval_recall == 1.0
    assert metrics.retrieval_recall_at_k == 1.0

