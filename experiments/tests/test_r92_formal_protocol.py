from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap

import pytest

from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.system import load_config
from experiments import run_v3_frozen_eval as frozen_runner
from experiments import run_v3_smoke as smoke_runner
from experiments import run_v3_train as train_runner
from experiments.protocol import (
    ManifestStore,
    ProtocolError,
    RunManifest,
    RunState,
    TaskManifest,
    hash_knowledge,
)
from experiments.run_v3_frozen_eval import (
    _frozen_protocol,
    _validate_formal_config as validate_frozen_config,
    _verify_source_train,
)
from experiments.run_v3_train import (
    _train_protocol,
    _validate_formal_config as validate_train_config,
)


ROOT = Path(__file__).resolve().parents[2]


def _config(name: str) -> dict[str, object]:
    return load_config(ROOT / "configs" / name)


def _output(config: dict[str, object]) -> Path:
    experiment = dict(config["experiment"])  # type: ignore[arg-type]
    return (ROOT / str(experiment["output_dir"])).resolve()


def _without_release_identity(config: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(config)
    normalized.pop("_config_path", None)
    normalized.pop("repair_revision", None)
    normalized.pop("data_dir", None)
    normalized.pop("trace_data_dir", None)
    experiment = dict(normalized["experiment"])  # type: ignore[arg-type]
    for field in (
        "name",
        "output_dir",
        "task_manifest_path",
        "frozen_snapshot_dir",
        "source_train_run_dir",
        "source_frozen_snapshot_dir",
    ):
        experiment.pop(field, None)
    normalized["experiment"] = experiment
    return normalized


def test_r92_train_is_new_identity_with_exact_r91_formal_settings() -> None:
    config = _config("alfworld_train_full_120_r92_seed42.yaml")
    baseline = _config("alfworld_train_full_120_r9_seed42.yaml")

    assert _train_protocol(config) == ("r92_full120", 42, 20, 120)
    assert config["repair_revision"] == "R9.2"
    assert _without_release_identity(config) == _without_release_identity(baseline)
    validate_train_config(config, _output(config))


def test_r92_frozen_is_new_identity_with_exact_r91_formal_settings() -> None:
    config = _config("alfworld_frozen_eval_134_r92_seed42.yaml")
    baseline = _config("alfworld_frozen_eval_134_r9_seed42.yaml")

    assert _frozen_protocol(config) == ("r92_frozen134", 42, 0, 134)
    assert config["repair_revision"] == "R9.2"
    assert _without_release_identity(config) == _without_release_identity(baseline)
    validate_frozen_config(config, _output(config))


def test_r92_real_smoke_retargets_replay_manifest_before_system_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class SystemConstructionReached(RuntimeError):
        pass

    class CapturingSystem:
        def __init__(self, config: dict[str, object], *, readonly: bool) -> None:
            captured["config"] = copy.deepcopy(config)
            captured["readonly"] = readonly
            raise SystemConstructionReached

    base_output = tmp_path / "real_smoke"
    monkeypatch.setattr(
        smoke_runner,
        "load_config",
        lambda _path: {"experiment": {"output_dir": str(base_output)}},
    )
    monkeypatch.setattr(
        smoke_runner, "validate_deepseek_formal_llm", lambda _config: None,
    )
    monkeypatch.setattr(
        smoke_runner,
        "ensure_provider_capability",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(smoke_runner, "hash_config", lambda _path: "config")
    monkeypatch.setattr(smoke_runner, "hash_code", lambda _path: "code")
    monkeypatch.setattr(smoke_runner, "AtomicSkillGraphSystem", CapturingSystem)

    with pytest.raises(SystemConstructionReached):
        smoke_runner.run_real_alfworld(tmp_path / "config.yaml")

    configured = captured["config"]
    assert isinstance(configured, dict)
    experiment = dict(configured["experiment"])
    output_dir = Path(str(experiment["output_dir"]))
    assert Path(str(experiment["task_manifest_path"])) == (
        output_dir / "task_manifest.json"
    )
    assert captured["readonly"] is False


@pytest.mark.parametrize("kind", ("train", "frozen"))
def test_r92_formal_protocol_rejects_missing_repair_identity(kind: str) -> None:
    if kind == "train":
        config = _config("alfworld_train_full_120_r92_seed42.yaml")
        validator = validate_train_config
    else:
        config = _config("alfworld_frozen_eval_134_r92_seed42.yaml")
        validator = validate_frozen_config
    config.pop("repair_revision")

    with pytest.raises(ProtocolError, match="repair_revision"):
        validator(config, _output(config))


def _assigned_expression(function: object, target_name: str) -> ast.expr:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    matches = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        )
    ]
    assert len(matches) == 1
    return matches[0]


def _eval_expression(expression: ast.expr, values: dict[str, object]) -> object:
    compiled = compile(
        ast.fix_missing_locations(ast.Expression(body=expression)),
        "<runner-expression>",
        "eval",
    )
    return eval(compiled, {"str": str}, values)  # noqa: S307 - trusted local AST


def test_r92_train_freeze_provenance_records_repair_revision() -> None:
    manifest = SimpleNamespace(
        run_id="alfworld_train_full_120_r92_seed42",
        manifest_hash="manifest-hash",
        config_hash="config-hash",
        code_commit="code-hash",
        task_manifest_hash="task-manifest-hash",
        knowledge_digest="initial-digest",
        metadata={
            "llm_config_hash": "llm-hash",
            "repair_revision": "R9.2",
        },
    )
    system = SimpleNamespace(knowledge_digest=lambda: "final-digest")
    expression = _assigned_expression(train_runner.run, "freeze_provenance")

    provenance = _eval_expression(
        expression, {"manifest": manifest, "system": system}
    )

    assert isinstance(provenance, dict)
    assert provenance["repair_revision"] == "R9.2"
    assert provenance["source_code_commit"] == "code-hash"


def _source_train_authority(
    tmp_path: Path,
) -> tuple[Path, RunManifest, dict[str, object], str]:
    run_id = "alfworld_train_full_30_r92_authority_fixture"
    train_run_dir = tmp_path / run_id
    source_data_dir = train_run_dir / "data_v3"
    trace_dir = train_run_dir / "traces"
    source_data_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)
    tasks = tuple(
        TaskManifest(
            ordinal=index,
            task_id=f"task-{index}",
            task_signature=f"signature-{index}",
            knowledge_milestone=(
                "initial:empty" if index == 0 else f"after:task-{index - 1}"
            ),
            benchmark="alfworld",
            split="train",
            metadata_json=json.dumps({"task_type": "fixture"}),
        )
        for index in range(30)
    )
    with StateDatabase(source_data_dir / "state.sqlite3") as database:
        frozen_digest = hash_knowledge(source_data_dir, database=database)
        manifest = RunManifest.create(
            run_id=run_id,
            phase="train",
            config_hash="config-hash",
            code_commit="code-hash",
            knowledge_digest=frozen_digest,
            tasks=tasks,
            metadata={
                "condition": "full",
                "tasks_per_type": 5,
                "total_tasks": 30,
                "llm_config_hash": "llm-hash",
                "repair_revision": "R9.2",
            },
        )
        store = ManifestStore(tmp_path, database)
        store.persist_before_run(manifest)
        store.mark_run_state(run_id, RunState.RUNNING)
        for item in tasks:
            trace_id = f"trace-{item.ordinal}"
            store.mark_task_running(run_id, item.task_id)
            (trace_dir / f"{trace_id}.json").write_text(
                json.dumps({
                    "trace_id": trace_id,
                    "task": {
                        "task_id": item.task_id,
                        "task_signature": item.task_signature,
                    },
                }),
                encoding="utf-8",
            )
            store.mark_task_completed(
                run_id,
                item.task_id,
                trace_id=trace_id,
                result={
                    "knowledge_digest_before": frozen_digest,
                    "knowledge_digest_after": frozen_digest,
                },
            )
        store.mark_run_state(run_id, RunState.COMPLETED)
        assert hash_knowledge(source_data_dir, database=database) == frozen_digest

    provenance: dict[str, object] = {
        "source_run_id": manifest.run_id,
        "source_run_manifest_hash": manifest.manifest_hash,
        "source_config_hash": manifest.config_hash,
        "source_code_commit": manifest.code_commit,
        "source_task_manifest_hash": manifest.task_manifest_hash,
        "source_initial_knowledge_digest": manifest.knowledge_digest,
        "source_final_knowledge_digest": frozen_digest,
        "source_llm_config_hash": "llm-hash",
        "repair_revision": "R9.2",
    }
    return train_run_dir, manifest, {"provenance": provenance}, frozen_digest


def test_r92_frozen_source_verifier_accepts_matching_revision(
    tmp_path: Path,
) -> None:
    train_dir, manifest, freeze_manifest, frozen_digest = (
        _source_train_authority(tmp_path)
    )

    _verify_source_train(
        train_run_dir=train_dir,
        train_manifest=manifest,
        freeze_manifest=freeze_manifest,
        frozen_digest=frozen_digest,
        current_code_digest="code-hash",
        current_llm_hash="llm-hash",
    )


@pytest.mark.parametrize("revision", (None, "R9.1", "R9.3"))
def test_r92_frozen_source_verifier_rejects_freeze_revision_drift(
    tmp_path: Path,
    revision: str | None,
) -> None:
    train_dir, manifest, freeze_manifest, frozen_digest = (
        _source_train_authority(tmp_path)
    )
    provenance = dict(freeze_manifest["provenance"])  # type: ignore[arg-type]
    if revision is None:
        provenance.pop("repair_revision")
    else:
        provenance["repair_revision"] = revision
    freeze_manifest["provenance"] = provenance

    with pytest.raises(ProtocolError, match="provenance"):
        _verify_source_train(
            train_run_dir=train_dir,
            train_manifest=manifest,
            freeze_manifest=freeze_manifest,
            frozen_digest=frozen_digest,
            current_code_digest="code-hash",
            current_llm_hash="llm-hash",
        )


@pytest.mark.parametrize(
    "revision,guard_rejects",
    (("R9.2", False), (None, True), ("R9.1", True), ("R9.3", True)),
)
def test_r92_frozen_runner_source_manifest_revision_guard(
    revision: str | None,
    guard_rejects: bool,
) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(frozen_runner.run)))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "repair_revision" in ast.unparse(node.test)
        and "r92_frozen134" in ast.unparse(node.test)
        and any(isinstance(item, ast.Raise) for item in node.body)
    ]
    assert len(guards) == 1
    manifest = SimpleNamespace(metadata={})
    if revision is not None:
        manifest.metadata["repair_revision"] = revision

    observed = _eval_expression(
        guards[0].test,
        {"protocol": "r92_frozen134", "train_manifest": manifest},
    )

    assert observed is guard_rejects
