"""Leakage-safe fitting and evaluation for the real-data baseline."""

from dataclasses import dataclass

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_monitoring.contracts import (
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_model_scoring_batch,
)


@dataclass(frozen=True)
class CreditModel:
    pipeline: Pipeline
    threshold: float
    model_version: str
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS


@dataclass(frozen=True)
class ModelMetrics:
    rows: int
    events: int
    event_rate: float
    roc_auc: float
    average_precision: float
    brier_score: float
    ks_discrimination: float
    f2: float
    precision: float
    recall: float
    review_rate: float


def _validate_features(data: pl.DataFrame) -> None:
    required = {ID_COLUMN, *MODEL_FEATURE_COLUMNS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing modeling columns: {', '.join(missing)}")
    if data.height == 0:
        raise ValueError("modeling data must not be empty")
    if data.select(*MODEL_FEATURE_COLUMNS).null_count().sum_horizontal().item() != 0:
        raise ValueError("modeling data must not contain null values")


def _validate_labeled_data(data: pl.DataFrame) -> None:
    _validate_features(data)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"missing modeling columns: {TARGET_COLUMN}")
    if data.get_column(TARGET_COLUMN).null_count() != 0:
        raise ValueError("modeling data must not contain null values")
    if set(data.get_column(TARGET_COLUMN).unique()) != {0, 1}:
        raise ValueError("modeling target must contain both binary classes")


def fit_credit_model(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    seed: int,
    model_version: str,
) -> CreditModel:
    _validate_labeled_data(train)
    _validate_labeled_data(validation)
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(random_state=seed, max_iter=1_000)),
        ]
    )
    pipeline.fit(
        train.select(MODEL_FEATURE_COLUMNS).to_numpy(),
        train.get_column(TARGET_COLUMN).to_numpy(),
    )

    validation_target = validation.get_column(TARGET_COLUMN).to_numpy()
    validation_probability = pipeline.predict_proba(
        validation.select(MODEL_FEATURE_COLUMNS).to_numpy()
    )[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        validation_target,
        validation_probability,
    )
    if len(thresholds) == 0:
        raise ValueError("validation data did not produce decision thresholds")
    f2_scores = [
        5 * float(p) * float(r) / (4 * float(p) + float(r)) if p + r > 0 else 0.0
        for p, r in zip(precision[:-1], recall[:-1], strict=True)
    ]
    best_index = max(range(len(f2_scores)), key=f2_scores.__getitem__)
    return CreditModel(
        pipeline=pipeline,
        threshold=float(thresholds[best_index]),
        model_version=model_version,
    )


def predict_credit_risk(model: CreditModel, data: pl.DataFrame) -> pl.DataFrame:
    _validate_features(data)
    validated = validate_model_scoring_batch(data.select(ID_COLUMN, *model.feature_columns))
    probability = model.pipeline.predict_proba(validated.select(model.feature_columns).to_numpy())[
        :, 1
    ]
    return pl.DataFrame(
        {
            ID_COLUMN: validated.get_column(ID_COLUMN),
            "default_probability": probability,
        }
    ).with_columns((pl.col("default_probability") >= model.threshold).alias("review_required"))


def _build_model_metrics(target: object, probability: object, prediction: object) -> ModelMetrics:
    false_positive_rate, true_positive_rate, _ = roc_curve(target, probability)
    return ModelMetrics(
        rows=len(target),
        events=int(target.sum()),
        event_rate=float(target.mean()),
        roc_auc=float(roc_auc_score(target, probability)),
        average_precision=float(average_precision_score(target, probability)),
        brier_score=float(brier_score_loss(target, probability)),
        ks_discrimination=float(max(true_positive_rate - false_positive_rate)),
        f2=float(fbeta_score(target, prediction, beta=2, zero_division=0)),
        precision=float(precision_score(target, prediction, zero_division=0)),
        recall=float(recall_score(target, prediction, zero_division=0)),
        review_rate=float(prediction.mean()),
    )


def evaluate_credit_model(model: CreditModel, data: pl.DataFrame) -> ModelMetrics:
    _validate_labeled_data(data)
    target = data.get_column(TARGET_COLUMN).to_numpy()
    probability = model.pipeline.predict_proba(data.select(model.feature_columns).to_numpy())[:, 1]
    prediction = probability >= model.threshold
    return _build_model_metrics(target, probability, prediction)


def evaluate_scored_predictions(data: pl.DataFrame) -> ModelMetrics:
    required = {ID_COLUMN, TARGET_COLUMN, "default_probability", "review_required"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {', '.join(missing)}")
    if data.height == 0 or data.get_column(ID_COLUMN).n_unique() != data.height:
        raise ValueError("scored predictions must contain unique rows")
    if data.select(*required).null_count().sum_horizontal().item() != 0:
        raise ValueError("scored predictions must not contain null values")
    if not data.get_column("default_probability").is_between(0, 1).all():
        raise ValueError("default_probability must be between 0 and 1")
    if set(data.get_column(TARGET_COLUMN).unique()) != {0, 1}:
        raise ValueError("prediction target must contain both binary classes")

    return _build_model_metrics(
        data.get_column(TARGET_COLUMN).to_numpy(),
        data.get_column("default_probability").to_numpy(),
        data.get_column("review_required").to_numpy(),
    )
