"""Strict readers and ALFWorld selectors for frozen reference manifests.

The reference manifests are experiment inputs, not mutable run state.  This
module therefore fails closed on their complete schema, canonical digest,
physical game-file identity, and train/test disjointness.  It deliberately
does not create or rewrite manifests.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from experiments.protocol import (
    ALFWORLD_FORMAL_TASK_TYPES,
    ProtocolError,
    sha256_json,
    task_signature,
)


REFERENCE_MANIFEST_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "benchmark",
        "source_split",
        "seed",
        "digest",
        "tasks",
    }
)
_TASK_FIELDS = frozenset(
    {
        "index",
        "task_id",
        "task_type",
        "source_split",
        "env_index",
        "gamefile_rel",
        "gamefile_sha256",
        "task_signature",
    }
)
_HARNESS_SPLITS = MappingProxyType(
    {
        "train": "train",
        "valid_seen": "eval_in_distribution",
        "valid_unseen": "eval_out_of_distribution",
    }
)


@dataclass(frozen=True)
class ReferenceManifestSpec:
    manifest_id: str
    source_split: str
    seed: int
    digest: str
    task_count: int
    family_counts: Mapping[str, int]


FORMAL_REFERENCE_MANIFEST_SPECS: Mapping[str, ReferenceManifestSpec] = (
    MappingProxyType(
        {
            "train_120": ReferenceManifestSpec(
                manifest_id="train_120",
                source_split="train",
                seed=42,
                digest="b4c1b07193ead0999f689365ffb698a4626ad21fbad1141ea6ad44729d883183",
                task_count=120,
                family_counts=MappingProxyType(
                    {name: 20 for name in ALFWORLD_FORMAL_TASK_TYPES}
                ),
            ),
            "test_ood_full_134": ReferenceManifestSpec(
                manifest_id="test_ood_full_134",
                source_split="valid_unseen",
                seed=42,
                digest="b897da6fc4d2b7c209a9e229f776e37ccde2ea90042737d005c07ff0fa549fee",
                task_count=134,
                family_counts=MappingProxyType(
                    {
                        "pick_and_place_simple": 24,
                        "look_at_obj_in_light": 18,
                        "pick_clean_then_place_in_recep": 31,
                        "pick_heat_then_place_in_recep": 23,
                        "pick_cool_then_place_in_recep": 21,
                        "pick_two_obj_and_place": 17,
                    }
                ),
            ),
        }
    )
)


def _require_exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], *, subject: str
) -> None:
    actual = frozenset(payload)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ProtocolError(
            f"{subject} schema mismatch: missing={missing}, unexpected={unexpected}"
        )


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{field} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    text = _require_string(value, field=field)
    if _SHA256.fullmatch(text) is None:
        raise ProtocolError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _validate_gamefile_rel(value: Any, *, source_split: str, field: str) -> str:
    text = _require_string(value, field=field)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\\" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or str(posix) != text
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ProtocolError(f"{field} must be a normalized relative POSIX path")
    expected_prefix = ("json_2.1.1", source_split)
    if posix.parts[:2] != expected_prefix or posix.name != "game.tw-pddl":
        raise ProtocolError(
            f"{field} must identify json_2.1.1/{source_split}/.../game.tw-pddl"
        )
    return text


@dataclass(frozen=True)
class ReferenceTask:
    index: int
    task_id: str
    task_type: str
    source_split: str
    env_index: int
    gamefile_rel: str
    gamefile_sha256: str
    task_signature: str

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, manifest_split: str, position: int
    ) -> "ReferenceTask":
        if not isinstance(payload, Mapping):
            raise ProtocolError(f"reference manifest task[{position}] must be a mapping")
        _require_exact_fields(payload, _TASK_FIELDS, subject=f"task[{position}]")
        source_split = _require_string(
            payload["source_split"], field=f"task[{position}].source_split"
        )
        if source_split != manifest_split:
            raise ProtocolError(
                f"task[{position}] source_split {source_split!r} differs from "
                f"manifest source_split {manifest_split!r}"
            )
        task_type = _require_string(
            payload["task_type"], field=f"task[{position}].task_type"
        )
        if task_type not in ALFWORLD_FORMAL_TASK_TYPES:
            raise ProtocolError(
                f"task[{position}].task_type is not a formal ALFWorld family: "
                f"{task_type!r}"
            )
        return cls(
            index=_require_int(payload["index"], field=f"task[{position}].index"),
            task_id=_require_string(
                payload["task_id"], field=f"task[{position}].task_id"
            ),
            task_type=task_type,
            source_split=source_split,
            env_index=_require_int(
                payload["env_index"], field=f"task[{position}].env_index"
            ),
            gamefile_rel=_validate_gamefile_rel(
                payload["gamefile_rel"],
                source_split=source_split,
                field=f"task[{position}].gamefile_rel",
            ),
            gamefile_sha256=_require_sha256(
                payload["gamefile_sha256"],
                field=f"task[{position}].gamefile_sha256",
            ),
            task_signature=_require_sha256(
                payload["task_signature"],
                field=f"task[{position}].task_signature",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceManifest:
    manifest_id: str
    benchmark: str
    source_split: str
    seed: int
    tasks: tuple[ReferenceTask, ...]
    digest: str

    def digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REFERENCE_MANIFEST_SCHEMA_VERSION,
            "manifest_id": self.manifest_id,
            "benchmark": self.benchmark,
            "source_split": self.source_split,
            "seed": self.seed,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def computed_digest(self) -> str:
        return sha256_json(self.digest_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.digest_payload()
        payload["digest"] = self.digest
        return payload


def _validate_manifest_identity(manifest: ReferenceManifest) -> None:
    indexes = [task.index for task in manifest.tasks]
    if indexes != list(range(len(manifest.tasks))):
        raise ProtocolError("reference manifest task indexes must be contiguous from zero")

    unique_fields: tuple[tuple[str, Sequence[Any]], ...] = (
        ("task_id", [task.task_id for task in manifest.tasks]),
        ("env_index", [task.env_index for task in manifest.tasks]),
        ("task_signature", [task.task_signature for task in manifest.tasks]),
        ("gamefile_rel", [task.gamefile_rel for task in manifest.tasks]),
    )
    for field, values in unique_fields:
        if len(set(values)) != len(values):
            raise ProtocolError(f"reference manifest {field} values must be unique")


def load_reference_manifest(path: str | Path) -> ReferenceManifest:
    """Load one reference manifest with exact schema and canonical digest checks."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"reference manifest is unreadable: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"reference manifest root must be a mapping: {source}")
    _require_exact_fields(payload, _ROOT_FIELDS, subject="reference manifest")
    schema_version = _require_int(
        payload["schema_version"], field="schema_version", minimum=1
    )
    if schema_version != REFERENCE_MANIFEST_SCHEMA_VERSION:
        raise ProtocolError(
            f"reference manifest schema_version must be "
            f"{REFERENCE_MANIFEST_SCHEMA_VERSION}, got {schema_version}"
        )
    manifest_id = _require_string(payload["manifest_id"], field="manifest_id")
    benchmark = _require_string(payload["benchmark"], field="benchmark")
    if benchmark != "alfworld":
        raise ProtocolError(f"reference manifest benchmark must be 'alfworld', got {benchmark!r}")
    source_split = _require_string(payload["source_split"], field="source_split")
    if source_split not in _HARNESS_SPLITS:
        raise ProtocolError(f"unsupported ALFWorld source_split: {source_split!r}")
    seed = _require_int(payload["seed"], field="seed")
    declared_digest = _require_sha256(payload["digest"], field="digest")
    raw_tasks = payload["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ProtocolError("reference manifest tasks must be a non-empty list")
    tasks = tuple(
        ReferenceTask.from_payload(
            item, manifest_split=source_split, position=position
        )
        for position, item in enumerate(raw_tasks)
    )
    manifest = ReferenceManifest(
        manifest_id=manifest_id,
        benchmark=benchmark,
        source_split=source_split,
        seed=seed,
        tasks=tasks,
        digest=declared_digest,
    )
    _validate_manifest_identity(manifest)
    computed = manifest.computed_digest()
    if computed != declared_digest:
        raise ProtocolError(
            f"reference manifest digest mismatch: declared {declared_digest}, "
            f"computed {computed}"
        )
    return manifest


def load_formal_reference_manifest(
    path: str | Path, *, manifest_id: str | None = None
) -> ReferenceManifest:
    """Load one of the two frozen 120-train / 134-test reference manifests."""

    manifest = load_reference_manifest(path)
    expected_id = manifest_id or Path(path).stem
    spec = FORMAL_REFERENCE_MANIFEST_SPECS.get(expected_id)
    if spec is None:
        raise ProtocolError(f"unknown formal reference manifest: {expected_id!r}")
    if manifest.manifest_id != expected_id:
        raise ProtocolError(
            f"formal reference manifest id mismatch: expected {expected_id!r}, "
            f"got {manifest.manifest_id!r}"
        )
    if manifest.source_split != spec.source_split:
        raise ProtocolError(
            f"{expected_id} source_split must be {spec.source_split!r}, "
            f"got {manifest.source_split!r}"
        )
    if manifest.seed != spec.seed:
        raise ProtocolError(
            f"{expected_id} seed must be {spec.seed}, got {manifest.seed}"
        )
    if manifest.digest != spec.digest:
        raise ProtocolError(
            f"{expected_id} frozen digest mismatch: expected {spec.digest}, "
            f"got {manifest.digest}"
        )
    if len(manifest.tasks) != spec.task_count:
        raise ProtocolError(
            f"{expected_id} must contain {spec.task_count} tasks, "
            f"got {len(manifest.tasks)}"
        )
    observed_counts = Counter(task.task_type for task in manifest.tasks)
    if dict(observed_counts) != dict(spec.family_counts):
        raise ProtocolError(
            f"{expected_id} family counts differ from frozen reference: "
            f"{dict(observed_counts)}"
        )
    return manifest


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_observed_gamefile(
    value: Any, *, data_root: Path, manifest_task: ReferenceTask
) -> tuple[Path, str]:
    text = _require_string(value, field=f"task[{manifest_task.index}].context.game_file")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = data_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(
            f"task[{manifest_task.index}] game_file is missing: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise ProtocolError(
            f"task[{manifest_task.index}] game_file is not a regular file: {resolved}"
        )
    try:
        relative = resolved.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise ProtocolError(
            f"task[{manifest_task.index}] game_file escapes ALFWorld data root: {resolved}"
        ) from exc
    _validate_gamefile_rel(
        relative,
        source_split=manifest_task.source_split,
        field=f"task[{manifest_task.index}].observed_gamefile_rel",
    )
    return resolved, relative


def select_reference_tasks(
    harness: Any,
    manifest: ReferenceManifest,
    *,
    alfworld_data: str | Path | None = None,
) -> list[Any]:
    """Select by ``env_index`` and verify every manifest identity field in order.

    ``harness.load_tasks(limit=0)`` is intentionally used so selection cannot be
    influenced by task-family scanning or an incidental prefix limit.
    """

    _validate_manifest_identity(manifest)
    if manifest.computed_digest() != manifest.digest:
        raise ProtocolError("reference manifest changed after loading")
    expected_harness_split = _HARNESS_SPLITS[manifest.source_split]
    observed_harness_split = str(_field(harness, "split", ""))
    if observed_harness_split != expected_harness_split:
        raise ProtocolError(
            f"harness split mismatch for {manifest.manifest_id}: expected "
            f"{expected_harness_split!r}, got {observed_harness_split!r}"
        )
    root_value = alfworld_data
    if root_value is None:
        root_value = _field(harness, "alfworld_data", None)
    if not root_value:
        raise ProtocolError("ALFWorld data root is required for reference selection")
    try:
        data_root = Path(root_value).resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"ALFWorld data root is missing: {root_value}") from exc
    if not data_root.is_dir():
        raise ProtocolError(f"ALFWorld data root is not a directory: {data_root}")

    loaded = harness.load_tasks(limit=0)
    if not isinstance(loaded, Sequence):
        raise ProtocolError("harness.load_tasks(limit=0) must return a sequence")
    by_env_index: dict[int, Any] = {}
    for position, task in enumerate(loaded):
        context = _field(task, "context", {}) or {}
        env_index = _field(context, "env_index", None)
        if isinstance(env_index, bool) or not isinstance(env_index, int) or env_index < 0:
            raise ProtocolError(
                f"loaded harness task[{position}] has invalid context.env_index"
            )
        if env_index in by_env_index:
            raise ProtocolError(f"harness returned duplicate env_index {env_index}")
        by_env_index[env_index] = task

    selected: list[Any] = []
    for expected in manifest.tasks:
        actual = by_env_index.get(expected.env_index)
        if actual is None:
            raise ProtocolError(
                f"manifest task[{expected.index}] env_index {expected.env_index} "
                "was not returned by the harness"
            )
        observed_id = str(_field(actual, "task_id", ""))
        if observed_id != expected.task_id:
            raise ProtocolError(
                f"task[{expected.index}] task_id mismatch: expected "
                f"{expected.task_id!r}, got {observed_id!r}"
            )
        observed_type = str(_field(actual, "task_type", ""))
        if observed_type != expected.task_type:
            raise ProtocolError(
                f"task[{expected.index}] task_type mismatch: expected "
                f"{expected.task_type!r}, got {observed_type!r}"
            )
        observed_signature = task_signature(actual)
        if observed_signature != expected.task_signature:
            raise ProtocolError(
                f"task[{expected.index}] task_signature mismatch: expected "
                f"{expected.task_signature}, got {observed_signature}"
            )
        context = _field(actual, "context", {}) or {}
        resolved_gamefile, observed_rel = _resolve_observed_gamefile(
            _field(context, "game_file", None),
            data_root=data_root,
            manifest_task=expected,
        )
        if observed_rel != expected.gamefile_rel:
            raise ProtocolError(
                f"task[{expected.index}] gamefile_rel mismatch: expected "
                f"{expected.gamefile_rel!r}, got {observed_rel!r}"
            )
        observed_sha256 = _sha256_file(resolved_gamefile)
        if observed_sha256 != expected.gamefile_sha256:
            raise ProtocolError(
                f"task[{expected.index}] gamefile SHA-256 mismatch: expected "
                f"{expected.gamefile_sha256}, got {observed_sha256}"
            )
        selected.append(actual)
    return selected


def validate_reference_disjoint(
    train: ReferenceManifest, test: ReferenceManifest
) -> dict[str, Any]:
    """Fail closed unless the frozen train and test identities are disjoint."""

    if train.source_split != "train":
        raise ProtocolError(
            f"train manifest must use source_split 'train', got {train.source_split!r}"
        )
    if test.source_split == "train":
        raise ProtocolError("test manifest must not use source_split 'train'")
    checks: tuple[tuple[str, set[Any], set[Any]], ...] = (
        (
            "task_id",
            {task.task_id for task in train.tasks},
            {task.task_id for task in test.tasks},
        ),
        (
            "task_signature",
            {task.task_signature for task in train.tasks},
            {task.task_signature for task in test.tasks},
        ),
        (
            "gamefile_rel",
            {task.gamefile_rel for task in train.tasks},
            {task.gamefile_rel for task in test.tasks},
        ),
    )
    for name, left, right in checks:
        overlap = sorted(left & right)
        if overlap:
            raise ProtocolError(
                f"reference train/test manifests share {name}: {overlap[:5]}"
            )
    return {
        "passed": True,
        "train_manifest_id": train.manifest_id,
        "train_digest": train.digest,
        "train_tasks": len(train.tasks),
        "test_manifest_id": test.manifest_id,
        "test_digest": test.digest,
        "test_tasks": len(test.tasks),
    }


__all__ = [
    "FORMAL_REFERENCE_MANIFEST_SPECS",
    "REFERENCE_MANIFEST_SCHEMA_VERSION",
    "ReferenceManifest",
    "ReferenceManifestSpec",
    "ReferenceTask",
    "load_formal_reference_manifest",
    "load_reference_manifest",
    "select_reference_tasks",
    "validate_reference_disjoint",
]
