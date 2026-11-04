from credit_monitoring.health import collect_process_health


def test_process_health_snapshot_contains_runtime_metrics() -> None:
    health = collect_process_health()

    assert health.cpu_count >= 1
    assert health.max_rss_mb > 0
    assert health.python_version
    assert health.platform
