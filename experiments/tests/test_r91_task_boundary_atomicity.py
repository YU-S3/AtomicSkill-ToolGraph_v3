from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CompositeOccurrence,
    CompositeSkill,
    TaskContract,
)
from atomic_skillgraph.core.edges import GlobalRelationType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.governance.ledger import EvidenceEvent, EvidenceEventType
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments.fakes import FakeHarness, fake_task
from experiments.protocol import (
    AttemptTraceLedger,
    ManifestStore,
    RunManifest,
    RunState,
    TaskCheckpointStore,
    TaskManifest,
    artifact_audit_snapshot,
    audit_failed_attempt,
    require_r9_formal_freeze_audit,
)
from experiments.run_v3_train import _run_final_batch_maintenance


COMPOSITE_REF = "skill://r91-boundary-composite@1.0.0"
RUN_ID = "r91-boundary-run"
CONFIG_HASH = "r91-config-hash"
CODE_COMMIT = "r91-code-commit"


def _config(
    data_dir: Path,
    trace_dir: Path,
    *,
    frozen: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "data_dir": str(data_dir),
        "trace_data_dir": str(trace_dir),
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "model": "fixture-model",
            "api_key_env": "MODEL_API_KEY",
        },
        "cold_start": {"enabled": False},
        "experiment": {
            "condition": "full",
            "runtime_mode": "frozen" if frozen else "online",
            "freeze_skills": frozen,
            "allow_long_term_knowledge_writes": not frozen,
            "output_dir": str(trace_dir),
        },
    }


def _register_candidate(
    system: AtomicSkillGraphSystem,
    harness: FakeHarness,
) -> None:
    system.skills.register_composite(CompositeSkill(
        ref=SkillRef.parse(COMPOSITE_REF),
        summary="task-boundary fixture",
        occurrences=[],
        control_sequence=[],
        data_edges=[],
        dependency_edges=[],
        goal_contract=harness.task_contract(fake_task("contract", "apple_1")),
        guideline={},
        insight={},
        validator_spec={},
        metadata={"harness_profiles": [harness.profile_name]},
        status=SkillStatus.CANDIDATE,
    ))


def _deployment_failure(task_id: str) -> EvidenceEvent:
    return EvidenceEvent.create(
        task_id=task_id,
        trace_id=f"trace-{task_id}",
        occurrence_id="deployment",
        attempt_id=f"attempt-{task_id}",
        sequence_no=0,
        artifact_ref=COMPOSITE_REF,
        artifact_kind="composite",
        event=EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
    )


def _seed_two_trials(system: AtomicSkillGraphSystem) -> None:
    for index in range(2):
        event = _deployment_failure(f"prior-task-{index}")
        system._commit_evidence([event])
        system._review_task_deployments([event])
    assert system.projection is not None
    assert system.projection.stats(
        COMPOSITE_REF, "composite"
    ).independent_deployment_trial_count == 2
    assert system.skills.get_composite(COMPOSITE_REF).status is SkillStatus.CANDIDATE


def _install_failed_deployment_runtime(system: AtomicSkillGraphSystem) -> None:
    def failed_runtime(task, *, mode, trace_builder, attempt_id=""):
        del mode, attempt_id
        trace = trace_builder.trace
        trace.task_contract = to_primitive(system.harness.task_contract(task))
        trace.runtime_plan = {
            "source": "stored_composite",
            "source_composite_ref": COMPOSITE_REF,
            "failure_stage": "runtime",
        }
        return trace_builder.finish()

    system.orchestrator.run_task = failed_runtime
    system.extraction_policy.decide = lambda _trace: SimpleNamespace(
        should_extract=False,
        reasons=["r91_boundary_fixture"],
    )
    assert system.evolution_maintenance is not None
    system.evolution_maintenance.prepare_failure_repairs = (
        lambda *_args, **_kwargs: []
    )


def _manifest(
    system: AtomicSkillGraphSystem,
    output_dir: Path,
    task,
) -> tuple[ManifestStore, RunManifest, AttemptTraceLedger, TaskCheckpointStore]:
    item = TaskManifest.from_task(
        task,
        ordinal=0,
        knowledge_milestone="initial:r91-boundary",
    )
    manifest = RunManifest.create(
        run_id=RUN_ID,
        phase="train",
        config_hash=CONFIG_HASH,
        code_commit=CODE_COMMIT,
        knowledge_digest=system.knowledge_digest(),
        tasks=(item,),
    )
    store = ManifestStore(output_dir / "run_manifest", system.database)
    store.persist_before_run(manifest)
    store.mark_run_state(RUN_ID, RunState.RUNNING)
    attempts = AttemptTraceLedger(
        output_dir / "attempt_history",
        system.traces.root,
    )
    checkpoint = TaskCheckpointStore(
        output_dir / ".task_checkpoint",
        system.data_dir,
    )
    return store, manifest, attempts, checkpoint


def _start_attempt(
    system: AtomicSkillGraphSystem,
    store: ManifestStore,
    manifest: RunManifest,
    attempts: AttemptTraceLedger,
    checkpoint: TaskCheckpointStore,
):
    item = manifest.tasks[0]
    sequence = store.mark_task_running(RUN_ID, item.task_id, max_attempts=3)
    attempt = attempts.begin(
        run_id=RUN_ID,
        task_id=item.task_id,
        task_signature=item.task_signature,
        attempt_kind="task",
        sequence=sequence,
    )
    checkpoint.create(
        system.database,
        run_id=RUN_ID,
        task_id=item.task_id,
        before_digest=system.knowledge_digest(),
        config_hash=CONFIG_HASH,
        code_commit=CODE_COMMIT,
    )
    return attempt


def _task_state(system: AtomicSkillGraphSystem, task_id: str) -> str:
    row = system.database.execute(
        "SELECT state FROM run_tasks WHERE run_id=? AND task_id=?",
        (RUN_ID, task_id),
    ).fetchone()
    assert row is not None
    return str(row["state"])


def _trace_path(system: AtomicSkillGraphSystem) -> Path:
    paths = sorted(system.traces.root.glob("trace_*.json"))
    assert paths
    return paths[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r91_review_failure_is_not_completed_and_resume_applies_governance(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    data_dir = output_dir / "data_v3"
    trace_dir = output_dir / "trace_output"
    config = _config(data_dir, trace_dir)
    harness = FakeHarness()
    task = fake_task("boundary-task", "apple_1")

    with AtomicSkillGraphSystem(
        config, harness=harness, provider=object(),
    ) as system:
        _register_candidate(system, harness)
        _seed_two_trials(system)
        _install_failed_deployment_runtime(system)
        store, manifest, attempts, checkpoint = _manifest(
            system, output_dir, task,
        )
        attempt = _start_attempt(system, store, manifest, attempts, checkpoint)
        assert system.lifecycle is not None

        def fail_before_review(*, artifact_refs=None):
            assert artifact_refs == [COMPOSITE_REF]
            assert system.projection is not None
            assert system.projection.stats(
                COMPOSITE_REF, "composite"
            ).independent_deployment_trial_count == 3
            assert _trace_path(system).is_file()
            raise RuntimeError("injected failure before task deployment review")

        system.lifecycle.review = fail_before_review
        with pytest.raises(RuntimeError, match="injected failure") as raised:
            system.run_task(task, attempt_id=attempt.attempt_id)

        trace_path = _trace_path(system)
        trace_hash = _sha256(trace_path)

        def mark_failed() -> None:
            store.mark_task_failed(
                RUN_ID,
                task.task_id,
                infrastructure=True,
                result={"error_type": type(raised.value).__name__},
            )
            store.mark_run_state(RUN_ID, RunState.INFRASTRUCTURE_FAILED)

        assert audit_failed_attempt(
            primary=raised.value,
            attempt=attempt,
            attempt_ledger=attempts,
            receipt_root=output_dir / "failure_receipts",
            update_state=mark_failed,
            capture_reason="task_exception",
        ) == []
        assert _task_state(system, task.task_id) == "infrastructure_failed"
        assert checkpoint.root.is_dir()
        assert _sha256(trace_path) == trace_hash

    assert checkpoint.recover_if_present(
        run_id=RUN_ID,
        config_hash=CONFIG_HASH,
        code_commit=CODE_COMMIT,
        resume=True,
    ) == task.task_id

    with AtomicSkillGraphSystem(
        config, harness=FakeHarness(), provider=object(),
    ) as resumed:
        resumed_store = ManifestStore(output_dir / "run_manifest", resumed.database)
        resumed_store.mark_run_state(RUN_ID, RunState.RUNNING)
        _install_failed_deployment_runtime(resumed)
        assert resumed.projection is not None
        assert resumed.projection.stats(
            COMPOSITE_REF, "composite"
        ).independent_deployment_trial_count == 2
        retry = _start_attempt(
            resumed, resumed_store, manifest, attempts, checkpoint,
        )
        trace = resumed.run_task(task, attempt_id=retry.attempt_id)
        attempts.capture(retry, reason="run_task_returned")
        resumed_store.mark_task_completed(
            RUN_ID, task.task_id, trace_id=trace.trace_id, result={},
        )
        checkpoint.clear()

        assert _task_state(resumed, task.task_id) == "completed"
        assert resumed.skills.get_composite(
            COMPOSITE_REF
        ).status is SkillStatus.SUPPRESSED
        assert resumed.projection.stats(
            COMPOSITE_REF, "composite"
        ).independent_deployment_trial_count == 3


def test_r91_reviewed_but_uncompleted_task_rolls_back_without_duplicate_trial(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    data_dir = output_dir / "data_v3"
    trace_dir = output_dir / "trace_output"
    config = _config(data_dir, trace_dir)
    harness = FakeHarness()
    task = fake_task("post-review-task", "apple_1")

    with AtomicSkillGraphSystem(
        config, harness=harness, provider=object(),
    ) as system:
        _register_candidate(system, harness)
        _seed_two_trials(system)
        _install_failed_deployment_runtime(system)
        store, manifest, attempts, checkpoint = _manifest(
            system, output_dir, task,
        )
        attempt = _start_attempt(system, store, manifest, attempts, checkpoint)
        assert system.lifecycle is not None
        real_review = system.lifecycle.review
        immutable_hashes: list[tuple[str, str]] = []

        def hash_around_real_review(*, artifact_refs=None):
            path = _trace_path(system)
            before = _sha256(path)
            result = real_review(artifact_refs=artifact_refs)
            after = _sha256(path)
            immutable_hashes.append((before, after))
            return result

        system.lifecycle.review = hash_around_real_review
        trace = system.run_task(task, attempt_id=attempt.attempt_id)
        capture = attempts.capture(attempt, reason="run_task_returned")
        assert capture["trace_ids"] == [trace.trace_id]
        assert immutable_hashes and immutable_hashes[0][0] == immutable_hashes[0][1]
        assert system.skills.get_composite(
            COMPOSITE_REF
        ).status is SkillStatus.SUPPRESSED
        assert _task_state(system, task.task_id) == "running"
        first_trace_path = system.traces.root / f"{trace.trace_id}.json"
        first_trace_hash = _sha256(first_trace_path)

    assert checkpoint.recover_if_present(
        run_id=RUN_ID,
        config_hash=CONFIG_HASH,
        code_commit=CODE_COMMIT,
        resume=True,
    ) == task.task_id

    with AtomicSkillGraphSystem(
        config, harness=FakeHarness(), provider=object(),
    ) as resumed:
        resumed_store = ManifestStore(output_dir / "run_manifest", resumed.database)
        resumed_store.mark_run_state(RUN_ID, RunState.RUNNING)
        _install_failed_deployment_runtime(resumed)
        assert resumed.skills.get_composite(
            COMPOSITE_REF
        ).status is SkillStatus.CANDIDATE
        assert resumed.projection is not None
        assert resumed.projection.stats(
            COMPOSITE_REF, "composite"
        ).independent_deployment_trial_count == 2

        retry = _start_attempt(
            resumed, resumed_store, manifest, attempts, checkpoint,
        )
        retry_trace = resumed.run_task(task, attempt_id=retry.attempt_id)
        attempts.capture(retry, reason="run_task_returned")
        resumed_store.mark_task_completed(
            RUN_ID, task.task_id, trace_id=retry_trace.trace_id, result={},
        )
        checkpoint.clear()

        assert resumed.projection.stats(
            COMPOSITE_REF, "composite"
        ).independent_deployment_trial_count == 3
        assert resumed.skills.get_composite(
            COMPOSITE_REF
        ).status is SkillStatus.SUPPRESSED
        assert _sha256(first_trace_path) == first_trace_hash


def test_r91_frozen_run_preserves_snapshot_digest(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "data_v3"
    snapshot_dir = tmp_path / "snapshot" / "data_v3"
    with AtomicSkillGraphSystem(
        _config(source_dir, tmp_path / "source-traces"),
        harness=FakeHarness(),
        provider=object(),
    ) as source:
        source_digest = source.knowledge_digest()
        source.freeze(snapshot_dir)

    frozen_config = _config(
        snapshot_dir,
        tmp_path / "frozen-traces",
        frozen=True,
    )
    with AtomicSkillGraphSystem(
        frozen_config,
        harness=FakeHarness(),
        provider=object(),
    ) as frozen:
        digest_before = frozen.knowledge_digest()

        def frozen_runtime(task, *, mode, trace_builder, attempt_id=""):
            del mode, attempt_id
            trace_builder.trace.task_contract = to_primitive(
                frozen.harness.task_contract(task)
            )
            trace_builder.trace.runtime_plan = {
                "source": "full_dynamic",
                "source_composite_ref": None,
                "failure_stage": "runtime",
            }
            return trace_builder.finish()

        frozen.orchestrator.run_task = frozen_runtime
        trace = frozen.run_task(fake_task("frozen-boundary", "apple_1"))
        assert frozen.traces.exists(trace.trace_id)
        assert frozen.knowledge_digest() == digest_before == source_digest


def test_r91_final_maintenance_invalidates_parent_before_frozen_snapshot(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "formal-run"
    data_dir = output_dir / "data_v3"
    trace_dir = output_dir / "trace_output"
    child_ref = SkillRef("r91-final-child", "1.0.0")
    parent_ref = SkillRef("r91-final-parent", "1.0.0")
    harness = FakeHarness()

    with AtomicSkillGraphSystem(
        _config(data_dir, trace_dir),
        harness=harness,
        provider=object(),
    ) as system:
        initial_snapshot = artifact_audit_snapshot(system.database)
        initial_digest = system.knowledge_digest()
        system.skills.register_atomic(AbstractAtomicSkill(
            ref=child_ref,
            summary="final maintenance child",
            inputs=[],
            outputs=[],
            preconditions=[],
            effects=[],
            validator_spec={},
            failure_modes=[],
            guideline={},
            metadata={"harness_profiles": [harness.profile_name]},
            status=SkillStatus.ACTIVE,
        ))
        occurrence = CompositeOccurrence(
            step_id="step-final",
            occurrence_id="occ-final",
            node_ref=child_ref,
            binding_specs={},
        )
        system.skills.register_composite(CompositeSkill(
            ref=parent_ref,
            summary="final maintenance parent",
            occurrences=[occurrence],
            control_sequence=[occurrence.step_id],
            data_edges=[],
            dependency_edges=[],
            goal_contract=TaskContract(),
            guideline={},
            insight={},
            validator_spec={},
            metadata={"harness_profiles": [harness.profile_name]},
            status=SkillStatus.ACTIVE,
        ))
        system._add_structural_edge(
            str(parent_ref),
            str(child_ref),
            GlobalRelationType.CONTAINS,
            "trace-final-dependency",
            occurrence_id=occurrence.occurrence_id,
        )
        system.skills.set_recommended(child_ref)
        system.skills.set_recommended(parent_ref)
        failures = [
            EvidenceEvent.create(
                task_id=f"prior-child-failure-{index}",
                trace_id=f"trace-prior-child-failure-{index}",
                occurrence_id="occ-final",
                attempt_id=f"attempt-prior-child-failure-{index}",
                sequence_no=index,
                artifact_ref=str(child_ref),
                artifact_kind="atomic",
                event=EvidenceEventType.DIRECT_FAILURE,
                failure_layer="atomic",
                metadata={"intrinsic_failure": True},
            )
            for index in range(3)
        ]
        system._commit_evidence(failures)
        assert system.skills.get_atomic(child_ref).status is SkillStatus.ACTIVE
        assert system.skills.get_composite(parent_ref).status is SkillStatus.ACTIVE

        task = fake_task("final-maintenance-task", "apple_1")
        task_item = TaskManifest.from_task(
            task,
            ordinal=0,
            knowledge_milestone="initial:r91-d06",
        )
        manifest = RunManifest.create(
            run_id="r91-d06-run",
            phase="train",
            config_hash=CONFIG_HASH,
            code_commit=CODE_COMMIT,
            knowledge_digest=initial_digest,
            tasks=(task_item,),
            metadata={
                "initial_artifact_snapshot": initial_snapshot,
                "initial_artifact_snapshot_digest": initial_snapshot[
                    "snapshot_digest"
                ],
            },
        )
        store = ManifestStore(output_dir / "run_manifest", system.database)
        store.persist_before_run(manifest)
        store.mark_run_state(manifest.run_id, RunState.RUNNING)
        store.mark_task_running(manifest.run_id, task.task_id)
        before_maintenance = system.knowledge_digest()
        store.mark_task_completed(
            manifest.run_id,
            task.task_id,
            trace_id="trace-final-maintenance-task",
            result={"knowledge_digest_after": before_maintenance},
        )
        checkpoint = TaskCheckpointStore(
            output_dir / ".task_checkpoint", system.data_dir,
        )
        assert system.evolution_maintenance is not None
        system.evolution_maintenance.build_batch_reviews = (
            lambda *_args, **_kwargs: []
        )
        system.evolution_maintenance.build_typed_reviews = (
            lambda *_args, **_kwargs: []
        )
        system.evolution_maintenance.build_composite_sequence_reviews = (
            lambda *_args, **_kwargs: []
        )

        maintenance_audit = _run_final_batch_maintenance(
            system,
            store,
            checkpoint,
            manifest,
            config_digest=manifest.config_hash,
            code_digest=manifest.code_commit,
        )

        assert maintenance_audit["pending_count"] == 0
        assert system.skills.get_atomic(
            child_ref
        ).status is SkillStatus.SUPPRESSED
        assert system.skills.get_composite(
            parent_ref
        ).status is SkillStatus.SUPPRESSED
        assert not checkpoint.root.exists()
        formal_audit = require_r9_formal_freeze_audit(
            system.database, system.skills,
        )
        assert formal_audit["r9_formal_freeze_audit_passed"] is True
        assert formal_audit["active_composite_frozen_closure"][
            "active_composite_count"
        ] == 0

        snapshot_dir = output_dir / "frozen" / "data_v3"
        system.freeze(snapshot_dir, provenance={
            "r9_formal_freeze_audit_passed": True,
            "r9_formal_freeze_audit": formal_audit,
        })
        frozen_digest = system.knowledge_digest()

    with AtomicSkillGraphSystem(
        _config(
            snapshot_dir,
            output_dir / "frozen_trace_output",
            frozen=True,
        ),
        harness=FakeHarness(),
        provider=object(),
    ) as frozen:
        assert frozen.readonly is True
        assert frozen.knowledge_digest() == frozen_digest
        assert frozen.skills.get_atomic(
            child_ref
        ).status is SkillStatus.SUPPRESSED
        assert frozen.skills.get_composite(
            parent_ref
        ).status is SkillStatus.SUPPRESSED
