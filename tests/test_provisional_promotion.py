from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.contracts import ContractSource, SemanticPredicate, TaskContract
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import RuntimeMode
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.atomicizer import AtomicOccurrenceProposal, Atomicizer
from atomic_skillgraph.evolution.contract_canonicalizer import (
    AtomicContractCanonicalizer,
    atomic_contract_signature,
)
from atomic_skillgraph.evolution.provisional_promotion import (
    ProvisionalPromotionCompiler,
    commit_prepared_promotion,
)
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.knowledge.failure_knowledge_store import (
    ProvisionalAtomicRecord,
    ProvisionalStatus,
    provisional_ref_for,
)
from atomic_skillgraph.runtime.cold_start_executor import ProvisionalTrialResult
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.traces.schema import (
    EnvironmentActionRecord,
    RuntimeSpan,
    TaskRecord,
    TraceRecord,
)
from atomic_skillgraph.validation.tool_validator import ToolValidator


class _Harness:
    profile_name = "fake_v3"

    def __init__(self, replay_passed: bool = True) -> None:
        self.replay_passed = replay_passed
        self.replayed_trace_ids: list[str] = []

    def replay_tool(self, _task, _tool, case) -> bool:
        self.replayed_trace_ids.append(str(case.get("trace_id", "")))
        return self.replay_passed

    def supports_constraint(self, _kind: str, _verifier_id: str = "") -> bool:
        return True


def _trace(*, strict_success: bool = True) -> TraceRecord:
    contract = TaskContract(
        target_effects=[SemanticPredicate(
            "agent.holds", {"object": "apple"},
        )],
        source=ContractSource.ADAPTER_DERIVED,
        confidence=1.0,
        validator_id="fake",
    )
    trace = TraceRecord.create(
        TaskRecord(
            "success-task", "fake", "hold an apple", "hold", "signature",
        ),
        to_primitive(contract),
        {},
        {"source": "cold_start"},
    )
    trace.environment_actions = [EnvironmentActionRecord(
        "action-1", 0, "TAKE",
        {"object": "apple_1", "source": "table_1"},
        True, "taken", True, True, 1, "trial-span",
    )]
    trace.runtime_spans = [RuntimeSpan(
        "trial-span", "runtime_provisional_seeded", "cold::acquire",
        0, 1, None, True,
    )]
    trace.benchmark_success = strict_success
    trace.task_contract_success = strict_success
    trace.strict_task_success = strict_success
    return trace


def _provisional_and_trial(trace: TraceRecord):
    normalized = TraceNormalizer().build(trace)
    occurrence = Atomicizer().validate_and_canonicalize([
        AtomicOccurrenceProposal(
            "source", "acquire_target_object", 0, 0,
            {"object": "apple_1", "source": "table_1"},
            {"held_object": "apple_1"},
            [],
            [SemanticPredicate("agent.holds", {"object": "apple_1"})],
            "accepted TAKE",
        ),
    ], normalized)[0]
    compiled = ToolCompiler().compile([occurrence])[0]
    canonicalizer = AtomicContractCanonicalizer()
    bundle = canonicalizer.canonicalize(
        compiled.atomic, compiled.tool, compiled.implementation,
    )
    assert bundle.tool is not None and bundle.implementation is not None
    signature = atomic_contract_signature(bundle.atomic)
    contract = {
        "summary": bundle.atomic.summary,
        "inputs": to_primitive(bundle.atomic.inputs),
        "outputs": to_primitive(bundle.atomic.outputs),
        "preconditions": to_primitive(bundle.atomic.preconditions),
        "effects": to_primitive(bundle.atomic.effects),
        "validator_spec": to_primitive(bundle.atomic.validator_spec),
    }
    provisional = ProvisionalAtomicRecord(
        provisional_ref=provisional_ref_for(signature),
        contract_signature=signature,
        canonical_intent="acquire_target_object",
        atomic_contract=contract,
        seeded_guideline={"intent": "establish the declared local Effect"},
        harness_profile="fake_v3",
        source_trace_id="failed-source-trace",
        source_task_id="failed-source-task",
        source_span={"event_start": 0, "event_end": 1},
        source_replay={"passed": True},
        aligned_plan_step_ids=("acquire",),
        progress_relation="consumed_prerequisite",
        status=ProvisionalStatus.TRIAL_READY,
    )
    bindings = {
        bundle.input_role_map["object"]: "apple_1",
        bundle.input_role_map["source"]: "table_1",
        bundle.output_role_map["held_object"]: "apple_1",
    }
    trial = ProvisionalTrialResult(
        provisional_ref=provisional.provisional_ref,
        step_id="acquire",
        local_effect_passed=True,
        progress_before_digest="before",
        progress_after_digest="after",
        action_span=(0, 1),
        witness_refs=["action:action-1:revision:1"],
        failure_code="",
        resolved_bindings=bindings,
    )
    return provisional, trial


def _compiler(harness: _Harness) -> ProvisionalPromotionCompiler:
    return ProvisionalPromotionCompiler(
        normalizer=TraceNormalizer(),
        atomicizer=Atomicizer(),
        tool_compiler=ToolCompiler(),
        admission=Admission(ToolValidator()),
        harness=harness,
    )


def test_task_failure_keeps_local_success_isolated_from_verified_bank() -> None:
    trace = _trace(strict_success=False)
    provisional, trial = _provisional_and_trial(trace)
    looked_up: list[str] = []
    prepared = _compiler(_Harness()).prepare(
        trace,
        [trial],
        provisional_lookup=lambda ref: looked_up.append(ref) or provisional,
        task=SimpleNamespace(task_id=trace.task.task_id),
    )
    assert prepared == []
    assert looked_up == []


def test_strict_success_recompiles_current_span_and_never_creates_composite() -> None:
    trace = _trace(strict_success=True)
    provisional, trial = _provisional_and_trial(trace)
    harness = _Harness()
    compiler = _compiler(harness)
    prepared = compiler.prepare(
        trace,
        [trial],
        provisional_lookup=lambda _ref: provisional,
        task=SimpleNamespace(task_id=trace.task.task_id),
    )
    assert len(prepared) == 1
    promotion = prepared[0]
    assert promotion.source_replay["source_trace_id"] == trace.trace_id
    assert harness.replayed_trace_ids == [trace.trace_id]
    assert all(
        artifact.metadata["origin"] == "promoted_from_provisional"
        and artifact.metadata["provisional_ref"] == provisional.provisional_ref
        for artifact in (
            promotion.compiled.atomic,
            promotion.compiled.implementation,
            promotion.compiled.tool,
        )
    )
    assert len(promotion.candidate_refs) == 3
    assert not any("composite" in ref.casefold() for ref in promotion.candidate_refs)


def test_contract_mismatch_and_fresh_replay_failure_are_rejected() -> None:
    trace = _trace(strict_success=True)
    provisional, trial = _provisional_and_trial(trace)

    mismatch = replace(
        provisional,
        contract_signature="different_contract",
        provisional_ref=provisional_ref_for("different_contract"),
    )
    mismatch_trial = replace(trial, provisional_ref=mismatch.provisional_ref)
    compiler = _compiler(_Harness())
    assert compiler.prepare(
        trace, [mismatch_trial],
        provisional_lookup=lambda _ref: mismatch,
        task=SimpleNamespace(task_id=trace.task.task_id),
    ) == []
    assert compiler.last_rejections[0].code == "provisional_promotion_contract_mismatch"

    compiler = _compiler(_Harness(replay_passed=False))
    assert compiler.prepare(
        trace, [trial],
        provisional_lookup=lambda _ref: provisional,
        task=SimpleNamespace(task_id=trace.task.task_id),
    ) == []
    assert compiler.last_rejections[0].code == "provisional_promotion_replay_failed"


def test_commit_records_only_verified_non_composite_refs_and_propagates_store() -> None:
    trace = _trace(strict_success=True)
    provisional, trial = _provisional_and_trial(trace)
    promotion = _compiler(_Harness()).prepare(
        trace, [trial],
        provisional_lookup=lambda _ref: provisional,
        task=SimpleNamespace(task_id=trace.task.task_id),
    )[0]

    class Store:
        def promote_provisional(self, ref, refs, **kwargs):
            return {"ref": ref, "refs": tuple(refs), **kwargs}

    committed = commit_prepared_promotion(
        promotion,
        store=Store(),
        verified_refs=promotion.candidate_refs,
    )
    assert committed["ref"] == provisional.provisional_ref
    assert committed["task_id"] == trace.task.task_id
    with pytest.raises(ValueError, match="only Atomic/Implementation/Tool"):
        commit_prepared_promotion(
            promotion,
            store=Store(),
            verified_refs=(*promotion.candidate_refs, "skill://composite_x@1.0.0"),
        )


def test_promotion_evidence_is_trace_first_and_appended_with_runtime_events() -> None:
    trace = _trace(strict_success=True)
    provisional, trial = _provisional_and_trial(trace)
    prepared = _compiler(_Harness()).prepare(
        trace, [trial],
        provisional_lookup=lambda _ref: provisional,
        task=SimpleNamespace(task_id=trace.task.task_id),
    )[0]
    promotion_event = SimpleNamespace(event_id="promotion-event")

    promotion_system = SimpleNamespace(
        aligner=SimpleNamespace(
            align_atomic=lambda atomic: atomic.ref,
            align_tool_with_replays=lambda *_args, **_kwargs: SimpleNamespace(
                admitted=True,
                ref=prepared.compiled.tool.ref,
                operation="reuse",
                source_ref=None,
            ),
            align_implementation=lambda implementation, *_args: implementation.ref,
        ),
        admission=Admission(ToolValidator()),
        harness=_Harness(),
        credit=SimpleNamespace(assign=lambda _trace: [promotion_event]),
        _add_structural_edge=lambda *_args, **_kwargs: None,
        _commit_evidence=lambda _events: pytest.fail(
            "promotion evidence must not reach the ledger during registry apply"
        ),
    )
    _refs, staged_events = AtomicSkillGraphSystem._apply_prepared_promotion(
        promotion_system,
        prepared,
        trace,
        SimpleNamespace(task_id=trace.task.task_id),
    )
    assert staged_events == [promotion_event]

    runtime_event = SimpleNamespace(event_id="runtime-event")
    order: list[str] = []

    def failure_side(trace_arg, _task, **_kwargs):
        assert trace_arg.evidence_event_refs == ["runtime-event"]
        trace_arg.evidence_event_refs.append(promotion_event.event_id)
        return {}, list(staged_events)

    def save_atomic(trace_arg):
        assert trace_arg.evidence_event_refs == [
            "runtime-event", "promotion-event",
        ]
        order.append("save_atomic")

    def append_evidence(events):
        assert order == ["save_atomic"]
        assert [item.event_id for item in events] == [
            "runtime-event", "promotion-event",
        ]
        order.append("ledger_append")

    pipeline = SimpleNamespace(
        orchestrator=SimpleNamespace(run_task=lambda *_args, **_kwargs: trace),
        _attach_provider_requests=lambda *_args, **_kwargs: None,
        _require_resource_usage_complete=lambda *_args, **_kwargs: None,
        failure_processor=SimpleNamespace(localize=lambda *_args: None),
        _provisional_trials=lambda *_args: [],
        _prepare_failure_extraction=lambda *_args, **_kwargs: None,
        _prepare_provisional_promotions=lambda *_args, **_kwargs: [prepared],
        extraction_policy=SimpleNamespace(decide=lambda *_args: SimpleNamespace(
            should_extract=False, reasons=["test_disabled"],
        )),
        evolution_maintenance=None,
        _observed_sessions=[],
        _attach_external_sessions=lambda *_args, **_kwargs: None,
        usage=SimpleNamespace(events=[]),
        _provider_override=object(),
        credit=SimpleNamespace(assign=lambda *_args: [runtime_event]),
        _commit_failure_side_task_evidence=failure_side,
        _finalize_v31_metrics=lambda *_args, **_kwargs: None,
        traces=SimpleNamespace(save_atomic=save_atomic),
        _commit_evidence=append_evidence,
        _online_successes=0,
        _maybe_run_maintenance=lambda: None,
        _persist_maintenance_state=lambda: None,
    )
    result = AtomicSkillGraphSystem._run_task_pipeline(
        pipeline,
        SimpleNamespace(task_id=trace.task.task_id),
        run_mode=RuntimeMode.ONLINE,
        trace_builder=SimpleNamespace(trace=trace),
        usage_start=0,
        sessions_start=0,
        provider_offsets={},
        failure_side_read_start=0,
    )
    assert result is trace
    assert order == ["save_atomic", "ledger_append"]
