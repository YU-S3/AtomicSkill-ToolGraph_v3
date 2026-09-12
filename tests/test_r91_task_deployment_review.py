from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.contracts import CompositeSkill
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.evolution.atomicizer import Atomicizer
from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
from atomic_skillgraph.governance.ledger import EvidenceEvent, EvidenceEventType
from atomic_skillgraph.system import AtomicSkillGraphSystem, _PreparedEvolution
from atomic_skillgraph.traces.schema import RuntimeSpan
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from experiments.fakes import FakeHarness, fake_task
from tests.test_v32_r4_learning_retention import _normalized_take, _take_proposal


COMPOSITE_REF = "skill://r91-task-review@1.0.0"


def _config(data_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 3,
        "data_dir": str(data_dir),
        "trace_data_dir": str(data_dir.parent / "traces"),
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "model": "fixture-model",
            "api_key_env": "MODEL_API_KEY",
        },
        "experiment": {
            "condition": "full",
            "runtime_mode": "online",
            "freeze_skills": False,
            "output_dir": str(data_dir.parent),
        },
    }


def _register_composite(
    system: AtomicSkillGraphSystem,
    harness: FakeHarness,
    *,
    status: SkillStatus = SkillStatus.CANDIDATE,
) -> None:
    system.skills.register_composite(CompositeSkill(
        ref=SkillRef.parse(COMPOSITE_REF),
        summary="hold a task-selected target",
        occurrences=[],
        control_sequence=[],
        data_edges=[],
        dependency_edges=[],
        goal_contract=harness.task_contract(fake_task("contract", "apple_1")),
        guideline={},
        insight={},
        validator_spec={},
        metadata={"harness_profiles": [harness.profile_name]},
        status=status,
    ))


def _deployment_event(
    task_id: str,
    event: EvidenceEventType,
    *,
    sequence_no: int = 0,
) -> EvidenceEvent:
    return EvidenceEvent.create(
        task_id=task_id,
        trace_id=f"trace-{task_id}",
        occurrence_id="deployment",
        attempt_id=f"attempt-{task_id}",
        sequence_no=sequence_no,
        artifact_ref=COMPOSITE_REF,
        artifact_kind="composite",
        event=event,
    )


def _status(system: AtomicSkillGraphSystem) -> SkillStatus:
    return system.skills.get_composite(COMPOSITE_REF).status


def _install_runtime_outcome(
    system: AtomicSkillGraphSystem,
    *,
    successful: bool = False,
    infrastructure_failure: bool = False,
    source: str = "stored_composite",
    task_rescue_required: bool = False,
) -> None:
    def runtime(task, *, mode, trace_builder, attempt_id=""):
        del mode, attempt_id
        trace = trace_builder.trace
        trace.task_contract = to_primitive(system.harness.task_contract(task))
        trace.runtime_plan = {
            "source": source,
            "source_composite_ref": (
                COMPOSITE_REF if source == "stored_composite" else None
            ),
            "failure_stage": "runtime",
        }
        trace.benchmark_success = successful
        trace.task_contract_success = successful
        trace.strict_task_success = successful
        trace.graph_self_sufficient_success = successful
        trace.graph_full_completion = successful
        trace.task_rescue_required = task_rescue_required
        trace.infrastructure_failure = infrastructure_failure
        if successful:
            trace.runtime_spans.append(RuntimeSpan(
                span_id=f"span-{task.task_id}",
                kind=source,
                occurrence_id="occ-1",
                action_start=0,
                action_end=1,
                parent_span_id=None,
                learnable=True,
            ))
        return trace_builder.finish()

    system.orchestrator.run_task = runtime
    system.extraction_policy.decide = lambda _trace: SimpleNamespace(
        should_extract=False,
        reasons=["r91_fixture_no_extraction"],
    )
    assert system.evolution_maintenance is not None
    system.evolution_maintenance.prepare_failure_repairs = (
        lambda *_args, **_kwargs: []
    )


def _apply_event(
    system: AtomicSkillGraphSystem,
    event: EvidenceEvent,
) -> None:
    system._commit_evidence([event])
    system._review_task_deployments([event])


def test_r91_normal_failures_review_before_return_and_next_retrieval(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"),
        harness=harness,
        provider=object(),
    ) as system:
        _register_composite(system, harness)
        _install_runtime_outcome(system)
        assert system.lifecycle is not None
        real_review = system.lifecycle.review
        review_observations: list[tuple[str, int]] = []

        def observed_review(artifact_refs=None):
            payloads = list(system.traces.iter_payloads())
            current_task = system._current_task_id
            assert any(
                payload["task"]["task_id"] == current_task
                for payload in payloads
            )
            assert system.projection is not None
            trial_count = system.projection.stats(
                COMPOSITE_REF, "composite"
            ).independent_deployment_trial_count
            review_observations.append((current_task, trial_count))
            return real_review(artifact_refs=artifact_refs)

        system.lifecycle.review = observed_review
        system.run_maintenance = lambda **_kwargs: pytest.fail(
            "normal failures must not require periodic maintenance"
        )

        probe_task = fake_task("probe-before", "apple_1")
        before = system.planner.composite_retriever.retrieve_complete(
            probe_task,
            harness.task_contract(probe_task),
            mode=RuntimeMode.ONLINE,
            harness_profile=harness.profile_name,
        )
        assert [str(item.ref) for item in before.candidates] == [COMPOSITE_REF]

        for index in range(3):
            trace = system.run_task(
                fake_task(f"failure-{index}", "apple_1"),
                attempt_id=f"attempt-failure-{index}",
            )
            expected = (
                SkillStatus.SUPPRESSED
                if index == 2
                else SkillStatus.CANDIDATE
            )
            assert trace.benchmark_success is False
            assert _status(system) is expected

        probe_after = fake_task("probe-after", "apple_1")
        after = system.planner.composite_retriever.retrieve_complete(
            probe_after,
            harness.task_contract(probe_after),
            mode=RuntimeMode.ONLINE,
            harness_profile=harness.profile_name,
        )
        assert COMPOSITE_REF not in {
            str(item.ref) for item in after.candidates
        }
        assert review_observations == [
            ("failure-0", 1),
            ("failure-1", 2),
            ("failure-2", 3),
        ]


def test_r91_candidate_one_success_in_five_trials_is_suppressed(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        _register_composite(system, harness)
        outcomes = [
            EvidenceEventType.DEPLOYMENT_SUCCESS,
            EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
            EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
            EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
            EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
        ]
        for index, outcome in enumerate(outcomes):
            _apply_event(system, _deployment_event(f"trial-{index}", outcome))
        assert _status(system) is SkillStatus.SUPPRESSED


def test_r91_candidate_two_independent_successes_still_promote(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        _register_composite(system, harness)
        for index in range(2):
            _apply_event(system, _deployment_event(
                f"success-{index}", EvidenceEventType.DEPLOYMENT_SUCCESS,
            ))
        assert _status(system) is SkillStatus.ACTIVE


def test_r91_active_three_unsuccessful_deployments_suppress_and_clear_pointer(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        _register_composite(system, harness, status=SkillStatus.ACTIVE)
        system.skills.set_recommended(COMPOSITE_REF)
        for index in range(3):
            _apply_event(system, _deployment_event(
                f"active-failure-{index}",
                EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
            ))
        assert _status(system) is SkillStatus.SUPPRESSED
        pointer = system.database.execute(
            "SELECT artifact_ref FROM recommended_pointers WHERE logical_id=?",
            (SkillRef.parse(COMPOSITE_REF).logical_id,),
        ).fetchone()
        assert pointer is None


def test_r91_no_deployment_refs_never_expand_to_full_review(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        _register_composite(system, harness)
        pending = [
            _deployment_event(
                f"pending-{index}",
                EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
            )
            for index in range(3)
        ]
        system._commit_evidence(pending)
        selected = EvidenceEvent.create(
            task_id="selected-only",
            trace_id="trace-selected-only",
            occurrence_id="occ-1",
            attempt_id="attempt-selected-only",
            sequence_no=0,
            artifact_ref=COMPOSITE_REF,
            artifact_kind="composite",
            event=EvidenceEventType.SELECTED,
        )
        assert system._review_task_deployments([selected]) is None
        assert system._review_task_deployments([]) is None
        assert _status(system) is SkillStatus.CANDIDATE


def test_r91_replayed_deployment_is_idempotent_and_does_not_revive(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        _register_composite(system, harness)
        events = [
            _deployment_event(
                f"replay-{index}",
                EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
            )
            for index in range(3)
        ]
        for event in events:
            _apply_event(system, event)
        assert _status(system) is SkillStatus.SUPPRESSED

        system._commit_evidence(events)
        result = system._review_task_deployments(events)
        assert result is not None
        assert result.changed_count == 0
        assert _status(system) is SkillStatus.SUPPRESSED
        assert system.projection is not None
        stats = system.projection.stats(COMPOSITE_REF, "composite")
        assert stats.independent_deployment_trial_count == 3


def test_r91_multiple_selected_occurrences_are_one_deployment_trial(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        _register_composite(system, harness)
        selected = [
            EvidenceEvent.create(
                task_id="multi-occurrence-task",
                trace_id="trace-multi-occurrence-task",
                occurrence_id=f"occ-{index}",
                attempt_id=f"attempt-occ-{index}",
                sequence_no=index,
                artifact_ref=COMPOSITE_REF,
                artifact_kind="composite",
                event=EvidenceEventType.SELECTED,
            )
            for index in range(2)
        ]
        outcome = EvidenceEvent.create(
            task_id="multi-occurrence-task",
            trace_id="trace-multi-occurrence-task",
            occurrence_id="deployment",
            attempt_id="attempt-deployment",
            sequence_no=2,
            artifact_ref=COMPOSITE_REF,
            artifact_kind="composite",
            event=EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
        )

        system._commit_evidence([*selected, outcome])
        result = system._review_task_deployments([*selected, outcome])

        assert result is not None and result.reviewed_count == 1
        assert system.projection is not None
        stats = system.projection.stats(COMPOSITE_REF, "composite")
        assert stats.selected_count == 2
        assert stats.independent_deployment_trial_count == 1


def test_r91_light_review_precedes_existing_fifth_success_maintenance(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        _register_composite(system, harness)
        _install_runtime_outcome(system, successful=True)
        system._online_successes = 4
        system._last_maintenance_success_count = 0
        order: list[str] = []
        assert system.lifecycle is not None
        real_review = system.lifecycle.review

        def observed_review(artifact_refs=None):
            order.append("task_review")
            return real_review(artifact_refs=artifact_refs)

        def observed_maintenance(*, triggering_task_id, milestone):
            assert triggering_task_id == "fifth-success"
            assert milestone == "online_success_5"
            order.append("periodic_maintenance")
            system._last_maintenance_success_count = system._online_successes
            return SimpleNamespace()

        system.lifecycle.review = observed_review
        system.run_maintenance = observed_maintenance
        trace = system.run_task(
            fake_task("fifth-success", "apple_1"),
            attempt_id="attempt-fifth-success",
        )
        assert trace.learning_eligible is True
        assert order == ["task_review", "periodic_maintenance"]


def test_r91_infrastructure_failure_skips_task_deployment_review(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        _register_composite(system, harness)
        for index in range(2):
            _apply_event(system, _deployment_event(
                f"prior-failure-{index}",
                EvidenceEventType.DEPLOYMENT_UNSUCCESSFUL,
            ))
        _install_runtime_outcome(system, infrastructure_failure=True)
        assert system.lifecycle is not None
        review_calls: list[object] = []

        def forbidden_review(*args, **kwargs):
            review_calls.append((args, kwargs))
            raise AssertionError("infrastructure failure must not run deployment review")

        system.lifecycle.review = forbidden_review
        trace = system.run_task(
            fake_task("infrastructure-failure", "apple_1"),
            attempt_id="attempt-infrastructure-failure",
        )

        assert trace.infrastructure_failure is True
        assert review_calls == []
        assert _status(system) is SkillStatus.CANDIDATE
        assert system.projection is not None
        assert system.projection.stats(
            COMPOSITE_REF, "composite"
        ).independent_deployment_trial_count == 2


def test_r91_newly_admitted_composite_is_not_reviewed_as_deployed(
    tmp_path: Path,
) -> None:
    harness = FakeHarness()
    task = fake_task("dynamic-admission", "apple_1")
    with AtomicSkillGraphSystem(
        _config(tmp_path / "data_v3"), harness=harness, provider=object(),
    ) as system:
        canonical = Atomicizer().validate_and_canonicalize(
            [_take_proposal()], _normalized_take(),
        )
        atomic = system._canonical_atomic_for_occurrence(canonical[0])
        staged = system._stage_atomic_only_occurrence(canonical[0], atomic)
        composite = system.composite_builder.validate_and_build(
            CompositeExtractionProposal(
                [staged.occurrence.occurrence_id],
                [],
                [],
                "hold the task-selected item",
                {},
                {},
            ),
            [staged.occurrence],
            harness.task_contract(task),
            contract_matcher=ExactContractMatcher({"item": "apple_1"}),
            task_bindings=dict(task.context["semantic_bindings"]),
        )
        prepared = _PreparedEvolution([staged], composite, {}, "")
        _install_runtime_outcome(system, successful=True, source="full_dynamic")
        system.extraction_policy.decide = lambda _trace: SimpleNamespace(
            should_extract=True,
            reasons=["r91_new_admission_fixture"],
        )
        system._prepare_evolution = lambda _trace, _task: prepared
        assert system.lifecycle is not None
        real_review = system.lifecycle.review
        review_calls: list[tuple[str, ...]] = []

        def observed_review(*, artifact_refs=None):
            review_calls.append(tuple(artifact_refs or ()))
            return real_review(artifact_refs=artifact_refs)

        system.lifecycle.review = observed_review
        trace = system.run_task(task, attempt_id="attempt-dynamic-admission")

        admitted_ref = str(trace.metadata["evolution_applied"]["composite_ref"])
        assert admitted_ref
        assert system.skills.get_composite(
            admitted_ref
        ).status is SkillStatus.CANDIDATE
        assert review_calls == []
        assert system.projection is not None
        stats = system.projection.stats(admitted_ref, "composite")
        assert stats.independent_deployment_trial_count == 0
        assert stats.independent_deployment_success_count == 0


def test_r91_readonly_helper_fails_closed() -> None:
    system = object.__new__(AtomicSkillGraphSystem)
    system.readonly = True
    with pytest.raises(RuntimeError, match="read-only"):
        system._review_task_deployments([])
