import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .models import EvalResult, SystemMetrics


def write_reports(out_dir: str, metrics: Iterable[SystemMetrics], results: Dict[str, List[EvalResult]]) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    metrics = list(metrics)
    write_metrics_csv(path / "metrics.csv", metrics)
    write_results_jsonl(path / "eval_results.jsonl", results)
    write_markdown(path / "experiment_report.md", metrics)


def write_metrics_csv(path: Path, metrics: List[SystemMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SystemMetrics.__fields__.keys()))
        writer.writeheader()
        for row in metrics:
            writer.writerow(row.dict())


def write_results_jsonl(path: Path, results: Dict[str, List[EvalResult]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for system_results in results.values():
            for result in system_results:
                handle.write(json.dumps(result.dict(), default=str) + "\n")


def write_markdown(path: Path, metrics: List[SystemMetrics]) -> None:
    lines = [
        "# Document Intelligence Lab Experiment Report",
        "",
        "| System | Accuracy | Citation accuracy | Retrieval recall@k | Hallucination rate | Cost/query | Latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row.system} | {row.accuracy:.3f} | {row.citation_accuracy:.3f} | "
            f"{row.retrieval_recall_at_k:.3f} | {row.hallucination_rate:.3f} | "
            f"${row.avg_cost_usd:.6f} | {row.avg_latency_ms:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

