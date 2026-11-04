"""Small process-health snapshot for local pipeline observability."""

import os
import platform
import resource
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessHealth:
    cpu_count: int
    max_rss_mb: float
    python_version: str
    platform: str


def collect_process_health() -> ProcessHealth:
    return ProcessHealth(
        cpu_count=os.cpu_count() or 1,
        max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        python_version=platform.python_version(),
        platform=sys.platform,
    )
