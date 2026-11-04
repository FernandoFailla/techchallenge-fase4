"""Deterministic hashes for local execution provenance."""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceProvenance:
    revision: str
    worktree_dirty: bool
    tree_sha256: str


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_source_provenance() -> SourceProvenance:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    source_hash = hashlib.sha256()
    for raw_path in sorted(path for path in files if path):
        path = Path(raw_path.decode())
        if path.parts[0] not in {"src", "scripts", "configs"} and path.name not in {
            "Makefile",
            "pyproject.toml",
            "uv.lock",
        }:
            continue
        source_hash.update(raw_path)
        source_hash.update(b"\0")
        source_hash.update(path.read_bytes())
        source_hash.update(b"\0")
    return SourceProvenance(
        revision=revision,
        worktree_dirty=dirty,
        tree_sha256=source_hash.hexdigest(),
    )
