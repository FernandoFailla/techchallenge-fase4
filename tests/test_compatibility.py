def test_d4_dependencies_expose_required_apis(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")

    import mlflow
    import pandera.polars as pa
    import polars as pl
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from fairlearn.metrics import MetricFrame, equalized_odds_difference
    from sklearn.linear_model import LogisticRegression

    schema = pa.DataFrameSchema({"ID": pa.Column(int, unique=True)})
    validated = schema.validate(pl.DataFrame({"ID": [1, 2]}), lazy=True)

    assert validated.height == 2
    assert callable(mlflow.start_run)
    assert isinstance(Report([DataDriftPreset()]), Report)
    assert callable(MetricFrame)
    assert callable(equalized_odds_difference)
    assert LogisticRegression().get_params()["random_state"] is None
