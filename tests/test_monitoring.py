import pytest

from credit_monitoring.monitoring import analyze_data_drift
from credit_monitoring.simulation import (
    generate_historical_data,
    scoring_view,
    simulate_data_drift,
)


def test_data_drift_detects_the_two_manipulated_features() -> None:
    reference = scoring_view(generate_historical_data(200, seed=43, id_start=10_000))
    current = simulate_data_drift(
        reference,
        limit_balance_multiplier=0.65,
        age_shift=8,
    )

    analysis = analyze_data_drift(reference, current)

    assert set(analysis.summary.drifted_features) == {"LIMIT_BAL", "AGE"}
    assert analysis.summary.drifted_count == 2
    assert analysis.summary.drifted_share == pytest.approx(0.4)
    assert analysis.summary.alert is True


def test_data_drift_rejects_an_invalid_alert_threshold() -> None:
    reference = scoring_view(generate_historical_data(100, seed=43))

    with pytest.raises(ValueError, match="alert_feature_count must be at least 1"):
        analyze_data_drift(reference, reference, alert_feature_count=0)
