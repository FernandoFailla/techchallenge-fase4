import pytest

from credit_monitoring.fairness import resolve_fairness_state


@pytest.mark.parametrize(
    ("investigation_required", "evidence_sufficient", "expected"),
    [
        (False, True, "MONITORING_CONTINUES"),
        (True, True, "INVESTIGATION_REQUIRED"),
        (False, False, "EVALUATION_BLOCKED_INSUFFICIENT_DATA"),
        (True, False, "INVESTIGATION_REQUIRED_WITH_INSUFFICIENT_DATA"),
    ],
)
def test_resolve_fairness_state_preserves_investigation_and_evidence_status(
    investigation_required: bool,
    evidence_sufficient: bool,
    expected: str,
) -> None:
    assert (
        resolve_fairness_state(
            investigation_required=investigation_required,
            evidence_sufficient=evidence_sufficient,
        )
        == expected
    )
