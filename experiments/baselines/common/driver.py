"""Common driver protocol for baseline methods (§42 of the design document).

``BaselineMethodDriver`` is a Protocol: a method implementation must expose a
preflight, a train step, a freeze step, and a frozen evaluation step.
The controller only sequences these steps; method logic stays upstream or in
the per-method adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .manifest import TaskManifestSet
from .model_config import ModelConfig
from .schema import CommonEpisodeRecord
from .usage import UsageSnapshot


@dataclass(frozen=True)
class RunContext:
    campaign_id: str
    method_id: str
    run_seed: int
    output_dir: Path
    repo_root: Path
    external_repo: Path | None
    external_commit: str | None
    model_config: ModelConfig
    max_environment_actions: int
    alfworld_data: Path
    config_hash: str
    code_hash: str
    # One immutable controller identity shared by every worker phase of this
    # run.  The resolved config path is the exact file whose digest appears in
    # ``identity``; method drivers must not substitute a hard-coded config.
    run_id: str
    resolved_config_path: Path
    identity: dict[str, str]
    train_manifest_path: Path | None = None
    validation_manifest_path: Path | None = None
    test_manifest_path: Path | None = None
    campaign: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None


@dataclass
class TrainResult:
    """Everything a train phase must leave behind for freeze + reporting."""

    episodes: list[CommonEpisodeRecord] = field(default_factory=list)
    usage: UsageSnapshot = field(default_factory=UsageSnapshot)
    persistent_artifact_files: dict[str, Path] = field(default_factory=dict)
    method_metrics: dict[str, Any] = field(default_factory=dict)
    validation_episodes: list[CommonEpisodeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": len(self.episodes),
            "validation_episode_count": len(self.validation_episodes),
            "usage": self.usage.to_dict(),
            "persistent_artifact_files": sorted(self.persistent_artifact_files),
            "method_metrics": self.method_metrics,
        }


@dataclass
class SmokeResult:
    """Evidence emitted by the isolated real-API/ALFWorld smoke phase."""

    episodes: list[CommonEpisodeRecord] = field(default_factory=list)
    usage: UsageSnapshot = field(default_factory=UsageSnapshot)


class BaselineMethodDriver(Protocol):
    method_id: str

    def preflight(self, ctx: RunContext) -> None:
        """Check source SHA, dependencies, model, ALFWorld, manifests, output dir."""

    def smoke(
        self,
        ctx: RunContext,
        train_manifest: TaskManifestSet,
    ) -> SmokeResult:
        """Run the isolated real-provider/ALFWorld smoke without training."""

    def train(
        self,
        ctx: RunContext,
        train_manifest: TaskManifestSet,
        validation_manifest: TaskManifestSet | None,
    ) -> TrainResult:
        """Learn with the upstream algorithm on the exact common manifest."""

    def freeze(self, ctx: RunContext, train_result: TrainResult) -> Any:
        """Produce the immutable persistent-knowledge snapshot + digest."""

    def evaluate_train(
        self,
        ctx: RunContext,
        frozen: Any,
        train_manifest: TaskManifestSet,
    ) -> list[CommonEpisodeRecord]:
        """Read-only frozen-skill re-evaluation on the exact Train manifest.

        This is explicitly an in-sample ``train_eval`` diagnostic, not a
        held-out Test/generalization result.
        """


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
