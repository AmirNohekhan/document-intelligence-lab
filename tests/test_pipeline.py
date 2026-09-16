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
    question = EvalQuestion(
        question_id="ad-hoc",
        question="Is this loss covered?",
        claim_id="",
        expected_label=DecisionLabel.NEEDS_REVIEW,
        required_doc_ids=[],
        required_terms=[],
        answer_regex="",
    )
    metrics, results = evaluate_system("agentic_workflow", [question])
    assert len(results) == 1
    assert results[0].retrieval_recall == 1.0

