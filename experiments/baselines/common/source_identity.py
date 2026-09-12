"""Baseline-local source identity and sanitized error helpers.

This module deliberately has no dependency on ``atomic_skillgraph`` so the
isolated Python 3.9 SkillGen runtime can import the baseline controller.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from .manifest import sha256_json


_CODE_SUFFIXES = frozenset(
    {".py", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".txt", ".lock"}
)
_CODE_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".external",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "build",
        "data",
        "data_v3",
        "dist",
        "env",
        "outputs",
        "reports",
        "results",
        "runs",
        "runs_v3",
        "frozen",
        "snapshots",
        "traces",
        "venv",
    }
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*)['\"]?[^\s,'\";]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def sanitize_error_text(value: Any) -> str:
    """Keep failure diagnostics useful without persisting provider secrets."""

    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            (lambda match: match.group(1) + "[REDACTED]")
            if pattern.groups
            else "[REDACTED]",
            text,
        )
    return text[:4000]


def _excluded(relative: Path) -> bool:
    return any(
        part in _CODE_EXCLUDED_DIRS
        or part.endswith(".egg-info")
        or part.startswith(".venv")
        for part in relative.parts
    )


def hash_code(root: str | Path) -> str:
    """Hash experiment source/config files without runtime environments."""

    source_root = Path(root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if source_root.is_file():
        files = [source_root]
    else:
        files: list[Path] = []
        for directory, dirnames, filenames in os.walk(source_root):
            directory_path = Path(directory)
            relative_directory = directory_path.relative_to(source_root)
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if not _excluded(relative_directory / name)
            ]
            for name in filenames:
                path = directory_path / name
                if (
                    path.is_file()
                    and path.suffix.casefold() in _CODE_SUFFIXES
                    and not _excluded(path.relative_to(source_root))
                ):
                    files.append(path)
    records = []
    for path in sorted(files, key=lambda item: item.as_posix()):
        relative = (
            path.name
            if source_root.is_file()
            else path.relative_to(source_root).as_posix()
        )
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sha256_json(records)
