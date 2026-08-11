from document_intelligence_lab.evaluation import evaluate_system
from document_intelligence_lab.synthetic_data import load_eval_questions


def test_eval_dataset_has_120_questions():
    assert len(load_eval_questions()) == 120


def test_agentic_workflow_runs_and_beats_minimum_accuracy():
    metrics, results = evaluate_system("agentic_workflow")
    assert len(results) == 120
    assert metrics.accuracy >= 0.70
    assert metrics.citation_accuracy >= 0.70

