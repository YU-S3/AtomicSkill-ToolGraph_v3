from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import atomic_skillgraph.system as system_module
from atomic_skillgraph.core.contracts import SemanticPredicate, TaskContract
from atomic_skillgraph.core.errors import AtomicSkillGraphError, FailureLayer
from atomic_skillgraph.core.results import RuntimeLinearPlan
from atomic_skillgraph.evolution.atomicizer import (
    AtomicOccurrenceProposal,
    Atomicizer,
)
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.harness.alfworld import AlfWorldValidatorChannel
from atomic_skillgraph.harness.protocol import (
    HarnessActionResult,
    HarnessActionSpec,
    HarnessTask,
)
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.traces.schema import (
    EnvironmentActionRecord,
    RuntimeSpan,
    TaskRecord,
    TraceBuilder,
    TraceRecord,
)
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher


def _spec(
    action_id: str,
    revision: int,
    action_type: str,
    arguments: dict[str, str],
) -> HarnessActionSpec:
    return HarnessActionSpec(
        action_id,
        revision,
        action_type,
        arguments,
        action_type,
        action_type,
        {},
    )


def _fact(
    revision: int,
    predicate: str,
    args: dict[str, str],
    *,
    domain: str = "world",
) -> dict:
    return {
        "predicate": predicate,
        "args": dict(args),
        "effect_domain": domain,
        "witness_ref": AlfWorldValidatorChannel._fact_ref(
            revision, predicate, args,
        ),
    }


def _trace(
    actions: list[EnvironmentActionRecord],
    snapshots: list[dict] | None,
    *,
    method_patch: str = "3.2",
) -> TraceRecord:
    task = TaskRecord(
        "semantic-authority",
        "alfworld",
        "exercise semantic authority",
        "pick_and_place_simple",
        "semantic-authority-signature",
    )
    trace = TraceRecord.create(task, {}, {}, {"source": "full_dynamic"})
    trace.metadata["method_patch"] = method_patch
    if snapshots is not None:
        trace.metadata["semantic_state_snapshots"] = copy.deepcopy(snapshots)
    trace.environment_actions = list(actions)
    trace.runtime_spans = [RuntimeSpan(
        "span", "full_dynamic", "", 0, len(actions), None, True,
    )]
    return trace


def _snapshot(
    sequence_index: int,
    revision: int,
    facts: list[dict],
    *,
    origin: str,
    action_id: str = "",
    accepted: bool = True,
    done: bool = False,
    won: bool = False,
) -> dict:
    return {
        "sequence_index": sequence_index,
        "revision": revision,
        "origin": origin,
        "action_id": action_id,
        "occurrence_id": "",
        "accepted": accepted,
        "done": done,
        "won": won,
        "facts": copy.deepcopy(facts),
    }


def test_gate37_validator_snapshot_uses_schema_domain_and_exact_fact_ref() -> None:
    channel = AlfWorldValidatorChannel()
    channel.record(
        _spec("r000_a001", 0, "GO_TO", {"destination": "desk_1"}),
        accepted=True,
        revision=1,
        done=False,
        won=False,
    )
    channel.record(
        _spec("r001_a001", 1, "TOGGLE_ON", {"object": "lamp_1"}),
        accepted=True,
        revision=2,
        done=False,
        won=False,
    )

    snapshot = channel.snapshot()
    assert snapshot["revision"] == 2
    assert snapshot["facts"]
    for fact in snapshot["facts"]:
        assert fact["effect_domain"] in {"world", "evidence"}
        assert fact["witness_ref"] == channel._fact_ref(
            2, fact["predicate"], fact["args"],
        )


class _TimelineHarness:
    profile_name = "semantic_timeline_test"

    def __init__(self) -> None:
        self.revision = 0
        self.channel = AlfWorldValidatorChannel()

    def _catalog(self) -> list[HarnessActionSpec]:
        return [
            _spec(
                f"r{self.revision:03d}_a001",
                self.revision,
                "GO_TO",
                {"destination": "desk_1"},
            )
        ]

    def reset(self, _task: HarnessTask) -> HarnessActionResult:
        self.revision = 0
        self.channel.reset()
        return HarnessActionResult(
            True, "reset", False, False, 0, self._catalog(), {"reset": True},
        )

    def validator_channel(self) -> AlfWorldValidatorChannel:
        return self.channel

    def step(self, *, accepted: bool) -> tuple[HarnessActionResult, HarnessActionSpec]:
        spec = self._catalog()[0]
        if accepted:
            self.revision += 1
        self.channel.record(
            spec,
            accepted=accepted,
            revision=self.revision,
            done=False,
            won=False,
        )
        return HarnessActionResult(
            accepted,
            "accepted" if accepted else "rejected",
            False,
            False,
            self.revision,
            self._catalog(),
            {},
        ), spec


def test_gate37_task_context_records_reset_and_every_step_snapshot() -> None:
    task = HarnessTask(
        "timeline-task",
        "visit a desk",
        "alfworld",
        metadata={"task_signature": "timeline-task"},
    )
    plan = RuntimeLinearPlan.full_dynamic(
        task.task_id, TaskContract(), reason="timeline fixture",
    )
    trace = _trace([], [], method_patch="3.2")
    builder = TraceBuilder(trace)
    harness = _TimelineHarness()
    ctx = TaskRuntimeContext.create(
        task,
        plan,
        harness,
        builder,
        RuntimeBudget(global_action_budget=4, node_action_budget=4),
    )
    ctx.begin_occurrence(SimpleNamespace(occurrence_id="occ"))

    for accepted in (True, False):
        result, spec = harness.step(accepted=accepted)
        record = EnvironmentActionRecord(
            spec.action_id,
            spec.revision,
            spec.action_type,
            dict(spec.arguments),
            result.accepted,
            result.observation,
            result.done,
            result.won,
            result.new_revision,
            "span",
        )
        trace.environment_actions.append(record)
        ctx.update_after_action(
            result,
            {
                **record.__dict__,
                "occurrence_id": "occ",
                "origin": "environment_action",
            },
        )

    timeline = trace.metadata["semantic_state_snapshots"]
    assert [item["sequence_index"] for item in timeline] == [0, 1, 2]
    assert [item["origin"] for item in timeline] == [
        "reset", "environment_action", "environment_action",
    ]
    assert [item["revision"] for item in timeline] == [0, 1, 1]
    assert [item["accepted"] for item in timeline] == [True, True, False]
    assert len(trace.metadata["atomic_evidence_snapshots"]) == 1

    normalized = TraceNormalizer().build(trace)
    assert normalized["semantic_authority_source"] == (
        "validator_snapshot_v3_2"
    )
    assert normalized["actions"][1]["authoritative_positive_effects"] == []


def test_gate38_same_revision_identical_snapshots_fold() -> None:
    action = EnvironmentActionRecord(
        "r000_a001", 0, "OPEN", {"object": "cabinet_1"},
        False, "rejected", False, False, 0, "span",
    )
    reset = _snapshot(0, 0, [], origin="reset")
    repeated = _snapshot(
        1, 0, [], origin="environment_action",
        action_id=action.action_id, accepted=False,
    )
    normalized = TraceNormalizer().build(_trace([action], [reset, repeated]))
    assert normalized["actions"][0]["authoritative_positive_effects"] == []
    assert normalized["actions"][0]["authoritative_negative_effects"] == []


@pytest.mark.parametrize(
    "snapshots, message",
    [
        (None, "missing semantic_state_snapshots"),
        (
            [_snapshot(0, 0, [], origin="reset")],
            "after revision 1 has no semantic snapshot",
        ),
        (
            [
                _snapshot(0, 0, [], origin="reset"),
                _snapshot(1, 1, [], origin="environment_action"),
                _snapshot(
                    2, 1, [_fact(1, "agent.holds", {"object": "apple_1"})],
                    origin="environment_action",
                ),
            ],
            "conflicts at revision 1",
        ),
    ],
)
def test_gate38_current_snapshot_integrity_fails_closed(
    snapshots: list[dict] | None,
    message: str,
) -> None:
    action = EnvironmentActionRecord(
        "r000_a001", 0, "TAKE", {"object": "apple_1"},
        True, "taken", False, False, 1, "span",
    )
    with pytest.raises(AtomicSkillGraphError, match=message) as caught:
        TraceNormalizer().build(_trace([action], snapshots))
    assert caught.value.code == "semantic_snapshot_integrity_error"
    assert caught.value.layer is FailureLayer.INFRASTRUCTURE


@pytest.mark.parametrize("mutation", ["domain", "witness", "duplicate"])
def test_gate38_invalid_snapshot_fact_fails_closed(mutation: str) -> None:
    fact = _fact(1, "agent.holds", {"object": "apple_1"})
    if mutation == "domain":
        fact["effect_domain"] = "benchmark"
    elif mutation == "witness":
        fact["witness_ref"] = ""
    facts = [fact, copy.deepcopy(fact)] if mutation == "duplicate" else [fact]
    snapshots = [
        _snapshot(0, 0, [], origin="reset"),
        _snapshot(1, 1, facts, origin="environment_action"),
    ]
    action = EnvironmentActionRecord(
        "r000_a001", 0, "TAKE", {"object": "apple_1"},
        True, "taken", False, False, 1, "span",
    )
    with pytest.raises(AtomicSkillGraphError) as caught:
        TraceNormalizer().build(_trace([action], snapshots))
    assert caught.value.code == "semantic_snapshot_integrity_error"
    assert caught.value.layer is FailureLayer.INFRASTRUCTURE


@pytest.mark.parametrize("revision", [True, "0"])
def test_gate38_snapshot_revision_must_be_non_boolean_integer(
    revision: object,
) -> None:
    reset = _snapshot(0, 0, [], origin="reset")
    reset["revision"] = revision
    with pytest.raises(AtomicSkillGraphError) as caught:
        TraceNormalizer().build(_trace([], [reset]))
    assert caught.value.code == "semantic_snapshot_integrity_error"
    assert caught.value.layer is FailureLayer.INFRASTRUCTURE


@pytest.mark.parametrize("mutation", ["predicate", "args", "domain_conflict"])
def test_gate38_snapshot_fact_shape_and_domain_are_strict(
    mutation: str,
) -> None:
    first = _fact(0, "agent.holds", {"object": "apple_1"})
    second = _fact(1, "agent.holds", {"object": "apple_1"})
    if mutation == "predicate":
        second.pop("predicate")
    elif mutation == "args":
        second["args"] = ["apple_1"]
    elif mutation == "domain_conflict":
        second["effect_domain"] = "evidence"
    snapshots = [
        _snapshot(0, 0, [first], origin="reset"),
        _snapshot(1, 1, [second], origin="environment_action"),
    ]
    action = EnvironmentActionRecord(
        "r000_a001", 0, "LOOK", {}, True, "looked", False, False, 1,
        "span",
    )
    with pytest.raises(AtomicSkillGraphError) as caught:
        TraceNormalizer().build(_trace([action], snapshots))
    assert caught.value.code == "semantic_snapshot_integrity_error"
    assert caught.value.layer is FailureLayer.INFRASTRUCTURE


def test_gate38_missing_intermediate_before_snapshot_fails_closed() -> None:
    actions = [
        EnvironmentActionRecord(
            "a0", 0, "LOOK", {}, True, "looked", False, False, 1,
            "span",
        ),
        EnvironmentActionRecord(
            "a1", 2, "LOOK", {}, True, "looked", False, False, 3,
            "span",
        ),
    ]
    snapshots = [
        _snapshot(0, 0, [], origin="reset"),
        _snapshot(1, 1, [], origin="environment_action"),
        _snapshot(2, 3, [], origin="environment_action"),
    ]
    with pytest.raises(
        AtomicSkillGraphError, match="before revision 2 has no semantic snapshot",
    ) as caught:
        TraceNormalizer().build(_trace(actions, snapshots))
    assert caught.value.code == "semantic_snapshot_integrity_error"
    assert caught.value.layer is FailureLayer.INFRASTRUCTURE


def test_gate39_ambiguous_use_does_not_fabricate_semantic_effect() -> None:
    channel = AlfWorldValidatorChannel()
    snapshots = [_snapshot(0, 0, [], origin="reset")]
    actions: list[EnvironmentActionRecord] = []
    steps = [
        ("GO_TO", {"destination": "desk_1"}, "arrived"),
        ("TAKE", {"object": "book_1", "source": "desk_1"}, "taken"),
        ("USE", {"object": "lamp_1"}, "Nothing unambiguously changes."),
    ]
    for index, (action_type, arguments, observation) in enumerate(steps):
        spec = _spec(f"r{index:03d}_a001", index, action_type, arguments)
        channel.record(
            spec,
            accepted=True,
            revision=index + 1,
            done=False,
            won=False,
            observation=observation,
        )
        actions.append(EnvironmentActionRecord(
            spec.action_id, index, action_type, arguments, True,
            observation, False, False, index + 1, "span",
        ))
        snapshot = channel.snapshot()
        snapshots.append(_snapshot(
            index + 1,
            index + 1,
            snapshot["facts"],
            origin="environment_action",
            action_id=spec.action_id,
        ))

    normalized = TraceNormalizer().build(_trace(actions, snapshots))
    effects = normalized["actions"][-1]["authoritative_positive_effects"]
    assert not any(
        fact["predicate"] in {
            "light.on", "light.off", "object.observed_with",
        }
        for fact in effects
    )


def test_gate38_legacy_trace_retains_action_reducer() -> None:
    action = EnvironmentActionRecord(
        "r000_a001", 0, "TAKE", {"object": "apple_1"},
        True, "taken", False, False, 1, "span",
    )
    normalized = TraceNormalizer().build(
        _trace([action], None, method_patch="3.1")
    )
    assert normalized["semantic_authority_source"] == "legacy_action_reducer"
    assert normalized["actions"][0]["authoritative_positive_effects"][0][
        "predicate"
    ] == "agent.holds"


def test_gate40_current_e1_reads_only_final_boundary_effect_authority() -> None:
    witness_ref = (
        "alfworld_action_fact:r1:agent.holds:object=apple_1"
    )
    fact = {
        **_fact(1, "agent.holds", {"object": "apple_1"}),
        "event_index": 0,
        "revision": 1,
        "source_kind": "semantic_snapshot_delta",
        "action_id": "r000_a001",
    }
    normalized = {
        "trace_id": "trace_gate40",
        "semantic_authority_source": "validator_snapshot_v3_2",
        "actions": [{
            "event_index": 0,
            "action_id": "r000_a001",
            "action_type": "TAKE",
            "arguments": {"object": "apple_1"},
            "accepted": True,
            "before_revision": 0,
            "after_revision": 1,
            "span_id": "span",
            "authoritative_before_state_facts": [],
            "authoritative_positive_effects": [fact],
        }],
        "runtime_spans": [{
            "span_id": "span",
            "occurrence_id": "",
            "action_start": 0,
            "action_end": 1,
        }],
        "boundary_authorities": {
            "inputs": [{
                "authority_ref": "action_arg:r000_a001:object",
                "event_id": "r000_a001",
                "argument_role": "object",
                "kind": "action_argument",
                "source_kind": "action_argument",
                "role": "object",
                "value": "apple_1",
            }],
            "effects": [],
        },
        # This historical bridge must not be a current E1 fallback.
        "after_state_facts": [fact],
    }
    proposal = AtomicOccurrenceProposal(
        "take",
        "take the object",
        0,
        0,
        {"object": "apple_1"},
        {"held_object": "apple_1"},
        [],
        [SemanticPredicate(
            "agent.holds", {"object": "apple_1"},
            effect_domain="world",
        )],
        "accepted TAKE established possession",
        support_event_ids=["r000_a001"],
        precondition_witness_refs=[],
        effect_witness_refs=[witness_ref],
        input_provenance_refs={
            "object": "action_arg:r000_a001:object",
        },
        output_derivations={
            "held_object": {
                "kind": "input_identity", "input_role": "object",
            },
        },
        input_provenance_contract="code_authority_v3_2",
    )

    with pytest.raises(ValueError, match="lacks accepted state/validator"):
        Atomicizer().validate_and_canonicalize([proposal], normalized)

    normalized["boundary_authorities"]["effects"] = [fact]
    canonical = Atomicizer().validate_and_canonicalize(
        [proposal], normalized,
    )
    assert canonical[0].effects[0].predicate == "agent.holds"


def test_gate40_current_e1_precondition_has_no_reducer_or_top_level_fallback() -> None:
    held_ref = "action:a0:revision:1"
    clean_ref = (
        "alfworld_action_fact:r2:object.cleaned:object=apple_1"
    )
    held_fact = {
        "predicate": "agent.holds",
        "args": {"object": "apple_1"},
        "effect_domain": "world",
        "witness_ref": held_ref,
        "revision": 1,
        "event_index": 0,
    }
    clean_fact = {
        **_fact(2, "object.cleaned", {"object": "apple_1"}),
        "revision": 2,
        "event_index": 1,
        "source_kind": "semantic_snapshot_delta",
        "action_id": "a1",
    }
    normalized = {
        "trace_id": "trace_precondition_boundary",
        "semantic_authority_source": "validator_snapshot_v3_2",
        "actions": [
            {
                "event_index": 0,
                "action_id": "a0",
                "action_type": "TAKE",
                "arguments": {"object": "apple_1"},
                "accepted": True,
                "before_revision": 0,
                "after_revision": 1,
                "span_id": "span",
                "authoritative_before_state_facts": [],
                "authoritative_positive_effects": [held_fact],
            },
            {
                "event_index": 1,
                "action_id": "a1",
                "action_type": "CLEAN",
                "arguments": {"object": "apple_1"},
                "accepted": True,
                "before_revision": 1,
                "after_revision": 2,
                "span_id": "span",
                # Deliberately empty: current E1 may not recover the held
                # precondition from either source below.
                "authoritative_before_state_facts": [],
                "authoritative_positive_effects": [clean_fact],
            },
        ],
        "runtime_spans": [{
            "span_id": "span",
            "occurrence_id": "",
            "action_start": 0,
            "action_end": 2,
        }],
        "before_state_facts": [held_fact],
        "boundary_authorities": {
            "inputs": [{
                "authority_ref": "action_arg:a1:object",
                "event_id": "a1",
                "argument_role": "object",
                "kind": "action_argument",
                "source_kind": "action_argument",
                "role": "object",
                "value": "apple_1",
            }],
            "effects": [clean_fact],
        },
    }
    proposal = AtomicOccurrenceProposal(
        "clean",
        "clean the held object",
        1,
        1,
        {"object": "apple_1"},
        {"cleaned_object": "apple_1"},
        [SemanticPredicate(
            "agent.holds", {"object": "apple_1"},
            effect_domain="world",
        )],
        [SemanticPredicate(
            "object.cleaned", {"object": "apple_1"},
            effect_domain="world",
        )],
        "accepted CLEAN establishes cleaned state",
        support_event_ids=["a1"],
        precondition_witness_refs=[held_ref],
        effect_witness_refs=[clean_ref],
        input_provenance_refs={"object": "action_arg:a1:object"},
        output_derivations={
            "cleaned_object": {
                "kind": "input_identity", "input_role": "object",
            },
        },
        input_provenance_contract="code_authority_v3_2",
    )

    with pytest.raises(ValueError, match="precondition lacks before-state"):
        Atomicizer().validate_and_canonicalize([proposal], normalized)


def test_gate40_failed_runtime_trial_range_cannot_use_ordinary_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class _CaptureExtractor:
        def __init__(self, _session):
            pass

        def propose_atomics(self, normalized, *_args, **_kwargs):
            captured.update(normalized)
            raise RuntimeError("captured")

    monkeypatch.setattr(system_module, "ExtractorSession", _CaptureExtractor)
    monkeypatch.setattr(
        system_module,
        "relevant_known_atomic_contracts",
        lambda *_args, **_kwargs: [],
    )
    ordinary = {
        **_fact(1, "agent.at_location", {"location": "desk_1"}),
        "event_index": 0,
        "revision": 1,
        "source_kind": "semantic_snapshot_delta",
        "action_id": "a0",
    }
    trial_delta = {
        **_fact(2, "agent.holds", {"object": "apple_1"}),
        "event_index": 1,
        "revision": 2,
        "source_kind": "semantic_snapshot_delta",
        "action_id": "a1",
    }
    system = system_module.AtomicSkillGraphSystem.__new__(
        system_module.AtomicSkillGraphSystem
    )
    system.normalizer = SimpleNamespace(build=lambda _trace: {
        "trace_id": "trace",
        "source_task": {},
        "semantic_authority_source": "validator_snapshot_v3_2",
        "actions": [
            {
                "event_index": 0,
                "action_id": "a0",
                "arguments": {},
                "accepted": True,
                "after_revision": 1,
                "authoritative_positive_effects": [ordinary],
            },
            {
                "event_index": 1,
                "action_id": "a1",
                "arguments": {},
                "accepted": True,
                "after_revision": 2,
                "authoritative_positive_effects": [trial_delta],
            },
        ],
        "runtime_spans": [],
        "validations": [],
        "boundary_authorities": {"inputs": [], "effects": []},
    })
    system._extractor_session = lambda _task_id: object()
    system.skills = object()
    system.harness = SimpleNamespace(
        task_contract=lambda _task: TaskContract(),
        contract_matcher=lambda: ExactContractMatcher(),
        semantic_predicate_schema=lambda: [],
    )
    trace = SimpleNamespace(
        trace_id="trace",
        metadata={
            "runtime_tool_trials": {
                "failed": {
                    "trial_event_start": 1,
                    "trial_event_end": 1,
                    "r1": {"admission_eligible": False},
                }
            }
        },
        runtime_plan={},
    )

    with pytest.raises(RuntimeError, match="captured"):
        system._prepare_evolution(trace, SimpleNamespace(task_id="task"))

    assert [
        effect["predicate"]
        for effect in captured["boundary_authorities"]["effects"]
    ] == ["agent.at_location"]


@pytest.mark.parametrize(("start", "end"), [("0", 0), (0, 2), (1, -1)])
def test_gate40_malformed_runtime_trial_range_fails_closed(
    start: object,
    end: object,
) -> None:
    system = system_module.AtomicSkillGraphSystem.__new__(
        system_module.AtomicSkillGraphSystem
    )
    system.normalizer = SimpleNamespace(build=lambda _trace: {
        "trace_id": "trace",
        "source_task": {},
        "semantic_authority_source": "validator_snapshot_v3_2",
        "actions": [{
            "event_index": 0,
            "action_id": "a0",
            "arguments": {},
            "accepted": True,
            "after_revision": 1,
            "authoritative_positive_effects": [],
        }],
        "runtime_spans": [],
        "validations": [],
        "boundary_authorities": {"inputs": [], "effects": []},
    })
    trace = SimpleNamespace(
        metadata={
            "runtime_tool_trials": {
                "malformed": {
                    "trial_event_start": start,
                    "trial_event_end": end,
                    "r1": {"started": True, "admission_eligible": False},
                }
            }
        },
        runtime_plan={},
    )

    with pytest.raises(AtomicSkillGraphError) as caught:
        system._prepare_evolution(trace, SimpleNamespace(task_id="task"))
    assert caught.value.code == "semantic_snapshot_integrity_error"
    assert caught.value.layer is FailureLayer.INFRASTRUCTURE


@pytest.mark.parametrize("mutation", ["domain", "revision"])
def test_gate40_runtime_r1_requires_exact_snapshot_domain_and_event_revision(
    mutation: str,
) -> None:
    raw_authority = {
        "predicate": "agent.holds",
        "args": {"object": "apple_1"},
        "effect_domain": "world",
        "event_index": 0,
        "revision": 1,
        "source_kind": "occurrence_action_delta",
        "source_occurrence_id": "occ",
    }
    if mutation == "domain":
        raw_authority["effect_domain"] = "evidence"
    else:
        raw_authority["revision"] = 2
    trial = {
        "draft_id": "take_tool",
        "trial_bindings": {"object": "apple_1"},
        "r1_outputs": {},
        "r1_witness_refs": [
            "alfworld_action_fact:r1:agent.holds:object=apple_1"
        ],
        "declared_effects": [{
            "predicate": "agent.holds",
            "args": {"object": "$object"},
            "effect_domain": "world",
        }],
        "after_revision": 1,
        "trial_event_start": 0,
        "trial_event_end": 0,
        "r1_effect_event_authorities": [raw_authority],
        "r1": {"admission_eligible": True},
    }
    actions = [{
        "event_index": 0,
        "accepted": True,
        "after_revision": 1,
        "authoritative_positive_effects": [{
            "predicate": "agent.holds",
            "args": {"object": "apple_1"},
            "effect_domain": "world",
        }],
    }]

    assert system_module.AtomicSkillGraphSystem._runtime_trial_effect_authorities(
        trial, actions,
    ) == []
