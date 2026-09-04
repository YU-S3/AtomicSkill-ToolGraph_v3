"""Fail-closed integrity checks shared by all baseline runs.

- no API key value may ever land on disk inside a run directory;
- completed episodes must carry provider usage evidence (missing usage is an
  infrastructure failure, never silently treated as zero cost);
- infrastructure failures must never be reported as task failures.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .schema import CommonEpisodeRecord

# Long token-looking strings that must never appear in any run artifact.
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
]


def assert_no_secrets_on_disk(root: str | Path, *, api_key_env: str) -> None:
    """Scan a run directory for the live API key value and key-like patterns."""

    root = Path(root)
    if not root.is_dir():
        return
    live_key = os.environ.get(api_key_env, "").strip()
    violations: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if live_key and live_key in content:
            violations.append(f"{path.relative_to(root)}: contains {api_key_env} value")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(content):
                violations.append(f"{path.relative_to(root)}: contains a key-like string")
                break
    if violations:
        raise RuntimeError(
            "API credentials found on disk inside the run directory: "
            + "; ".join(sorted(set(violations))[:5])
        )


def validate_episode_usage(episodes: list[CommonEpisodeRecord]) -> None:
    """Fail closed unless every non-infrastructure episode has usage evidence."""

    for episode in episodes:
        if episode.infrastructure_failure:
            continue
        if episode.target_llm_calls <= 0:
            raise RuntimeError(
                f"episode {episode.task_id} ({episode.phase}) completed without any "
                "provider usage evidence; missing usage must not be treated as zero"
            )


def split_infrastructure_failures(
    episodes: list[CommonEpisodeRecord],
) -> tuple[list[CommonEpisodeRecord], list[CommonEpisodeRecord]]:
    """Return (task_outcome_episodes, infrastructure_failed_episodes)."""

    normal = [episode for episode in episodes if not episode.infrastructure_failure]
    failed = [episode for episode in episodes if episode.infrastructure_failure]
    return normal, failed


def run_manifest_payload(
    *,
    method: str,
    external_repo: str,
    external_commit: str,
    ours_controller_commit: str,
    train_manifest_hash: str | None,
    validation_manifest_hash: str | None,
    test_manifest_hash: str | None,
    alfworld_version: str,
    alfworld_data_signature: str,
    model: str,
    provider: str,
    method_specific_decoding: dict[str, Any],
    max_environment_actions: int,
    run_seed: int,
    phase: str,
    output_dir: str,
) -> dict[str, Any]:
    """The provenance block every formal run must persist (§45)."""

    return {
        "schema_version": 1,
        "method": method,
        "phase": phase,
        "external_repo": external_repo,
        "external_commit": external_commit,
        "ours_controller_commit": ours_controller_commit,
        "train_manifest_hash": train_manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
        "test_manifest_hash": test_manifest_hash,
        "alfworld_version": alfworld_version,
        "alfworld_data_signature": alfworld_data_signature,
        "model": model,
        "provider": provider,
        "method_specific_decoding": dict(method_specific_decoding),
        "max_environment_actions": int(max_environment_actions),
        "run_seed": int(run_seed),
        "output_dir": str(output_dir),
    }
