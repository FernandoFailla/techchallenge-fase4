from pathlib import Path
from runpy import run_path

_format_metric = run_path(Path(__file__).parents[1] / "scripts" / "export_evidence.py")[
    "_format_metric"
]


def test_optional_fairness_metric_formatting_supports_suppression() -> None:
    assert _format_metric(None) == "indisponível"
    assert _format_metric(0.123456) == "0.1235"
