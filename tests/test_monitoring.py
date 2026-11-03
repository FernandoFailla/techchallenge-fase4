import numpy as np
import polars as pl
import pytest

from credit_monitoring.contracts import ID_COLUMN, TARGET_COLUMN
from credit_monitoring.monitoring import (
    PsiLevel,
    analyze_data_drift,
    analyze_population_stability,
    calculate_psi,
    compare_paired_performance,
)
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


def test_psi_uses_reference_bins_and_classifies_shift() -> None:
    reference = pl.Series("feature", np.linspace(0, 1, 500))
    stable = calculate_psi(reference, reference)
    shifted = calculate_psi(reference, pl.Series("feature", np.linspace(0.7, 1.7, 500)))

    assert stable.value == pytest.approx(0.0)
    assert stable.level is PsiLevel.INFORMATIONAL
    assert shifted.level is PsiLevel.RELEVANT
    assert len(shifted.cut_points) == 9


def test_psi_rejects_non_finite_values() -> None:
    reference = pl.Series("feature", np.linspace(0, 1, 500))
    current = reference.clone()
    current[0] = np.nan

    with pytest.raises(ValueError, match="only finite values"):
        calculate_psi(reference, current)

    with pytest.raises(ValueError, match="smoothing must be greater than zero"):
        calculate_psi(reference, reference, smoothing=float("nan"))


def test_psi_detects_shift_from_a_constant_reference() -> None:
    reference = pl.Series("feature", [1.0] * 500)
    current = pl.Series("feature", [2.0] * 500)

    result = calculate_psi(reference, current)

    assert result.level is PsiLevel.RELEVANT


def test_population_alert_requires_two_feature_warnings() -> None:
    reference = pl.DataFrame(
        {
            "first": np.linspace(0, 1, 500),
            "second": np.linspace(1, 2, 500),
            "default_probability": np.linspace(0.05, 0.95, 500),
        }
    )
    current = reference.with_columns(
        (pl.col("first") + 0.7).alias("first"),
        (pl.col("second") + 0.7).alias("second"),
    )

    stability = analyze_population_stability(
        reference,
        current,
        feature_columns=("first", "second"),
    )

    assert stability.feature_warning_count == 2
    assert stability.score_relevant is False
    assert stability.alert is True


def test_paired_bootstrap_flags_adverse_performance() -> None:
    probability = np.tile([0.1, 0.9], 200)
    baseline_target = np.tile([0, 1], 200)
    current_target = baseline_target.copy()
    current_target[0:200:2] = 1
    common = {
        ID_COLUMN: range(1, 401),
        "default_probability": probability,
        "review_required": probability >= 0.5,
    }
    baseline = pl.DataFrame({**common, TARGET_COLUMN: baseline_target})
    current = pl.DataFrame({**common, TARGET_COLUMN: current_target})

    comparison = compare_paired_performance(
        baseline,
        current,
        bootstrap_samples=100,
        seed=42,
    )

    assert comparison.alert is True
    assert {shift.metric for shift in comparison.shifts if shift.alert} >= {
        "roc_auc",
        "brier_score",
    }


def test_paired_bootstrap_preserves_brier_samples_for_imbalanced_targets() -> None:
    probability = np.linspace(0.01, 0.5, 100)
    baseline_target = np.zeros(100, dtype=int)
    baseline_target[-1] = 1
    current_target = np.zeros(100, dtype=int)
    current_target[-2] = 1
    common = {
        ID_COLUMN: range(1, 101),
        "default_probability": probability,
        "review_required": probability >= 0.2,
    }

    comparison = compare_paired_performance(
        pl.DataFrame({**common, TARGET_COLUMN: baseline_target}),
        pl.DataFrame({**common, TARGET_COLUMN: current_target}),
        bootstrap_samples=100,
        seed=42,
    )

    brier = next(shift for shift in comparison.shifts if shift.metric == "brier_score")
    assert brier.confidence_lower <= brier.difference <= brier.confidence_upper
    assert brier.bootstrap_samples == 100
