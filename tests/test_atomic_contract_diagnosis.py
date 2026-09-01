from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from atomic_skillgraph.agents.protocol import AgentTurn, NativeToolCall
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CapabilityRequirement,
    ParameterSpec,
    PlannerRequirementBundle,
    RequirementSearchResult,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.knowledge.query import (
    atomic_contract_compatible,
    diagnose_atomic_contract_compatibility,
)
from atomic_skillgraph.planner.atomic_retriever import AtomicRetriever
from atomic_skillgraph.planner.requirement_agent import RequirementAgent


def _requirement(
    *,
    predicate: str = "object.at_location",
    args: tuple[str, ...] = ("object", "location"),
    cardinality: int = 1,
    input_types: tuple[str, ...] = (),
) -> CapabilityRequirement:
    return CapabilityRequirement(
        requirement_id="req_move",
        intent="move object to location",
        desired_effects=[SemanticPredicate(
            predicate,
            {role: f"${role}" for role in args},
            cardinality,
        )],
        expected_inputs=[
            ParameterSpec(f"required_{index}", semantic_type)
            for index, semantic_type in enumerate(input_types)
        ],
        expected_outputs=[],
        precondition_hints=[],
        semantic_variants=["move object"],
        required=True,
        rationale="required transition",
    )


def _atomic(
    logical_id: str,
    *,
    predicate: str = "object.at_location",
    args: tuple[str, ...] = ("object", "location"),
    cardinality: int = 1,
    input_types: tuple[str, ...] = (),
    summary: str = "unrelated",
    metadata: dict[str, Any] | None = None,
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary=summary,
        inputs=[
            ParameterSpec(f"input_{index:03d}", semantic_type, True, True, "concrete")
            for index, semantic_type in enumerate(input_types)
        ],
        outputs=[ParameterSpec("output_000", "entity")],
        preconditions=[],
        effects=[SemanticPredicate(
            predicate,
            {role: f"old_concrete_{role}" for role in args},
            cardinality,
        )],
        validator_spec={},
        failure_modes=[],
        guideline={"private_tool_body": "forbidden_primitive_sequence"},
        metadata=dict(metadata or {}),
        status=SkillStatus.ACTIVE,
    )


def test_atomic_bool_matcher_is_exactly_the_diagnosis_decision() -> None:
    requirement = _requirement(input_types=("object",))
    matching = _atomic("matching", input_types=("entity",))
    incompatible = _atomic("wrong", predicate="object.cleaned", args=("object",))
    for atomic in (matching, incompatible):
        report = diagnose_atomic_contract_compatibility(requirement, atomic)
        assert atomic_contract_compatible(requirement, atomic) is report.passed


def test_atomic_diagnosis_reports_typed_effect_failures() -> None:
    required = _requirement(cardinality=2)

    missing_predicate = diagnose_atomic_contract_compatibility(
        required,
        _atomic("predicate", predicate="object.cleaned", args=("object",)),
    )
    assert missing_predicate.failure_codes == (
        "atomic_effect_predicate_missing",
    )

    missing_role = diagnose_atomic_contract_compatibility(
        required,
        _atomic("role", args=("object",), cardinality=2),
    )
    assert "atomic_effect_argument_role_missing" in missing_role.failure_codes
    assert missing_role.effect_details[0].missing_argument_roles == ("location",)

    insufficient = diagnose_atomic_contract_compatibility(
        required,
        _atomic("cardinality", cardinality=1),
    )
    assert insufficient.failure_codes == (
        "atomic_effect_cardinality_insufficient",
    )
    assert insufficient.effect_details[0].best_offered_cardinality == 1


def test_generic_entity_satisfies_specific_required_input_but_primitive_does_not() -> None:
    required = _requirement(input_types=("object",))
    entity = diagnose_atomic_contract_compatibility(
        required,
        _atomic("entity_input", input_types=("entity",)),
    )
    primitive = diagnose_atomic_contract_compatibility(
        required,
        _atomic("string_input", input_types=("string",)),
    )
    assert entity.passed
    assert entity.input_details[0].compatible_offered_roles == ("input_000",)
    assert not primitive.passed
    assert primitive.missing_required_input_types == ("object",)
    assert primitive.failure_codes == (
        "atomic_required_input_type_unavailable",
    )


class _SkillBank:
    def __init__(self, atomics: list[AbstractAtomicSkill]) -> None:
        self._atomics = list(atomics)

    def atomics(self, *, mode: RuntimeMode | str) -> list[AbstractAtomicSkill]:
        return list(self._atomics)


def test_repair_hints_are_sanitized_bounded_and_deterministically_ranked() -> None:
    requirement = _requirement(input_types=("location", "device"))
    atomics = [
        _atomic(
            "z_high_lexical",
            input_types=("location",),
            summary="move object to location",
            metadata={"source_trace": "forbidden_trace_payload"},
        ),
        _atomic("a_low_lexical", input_types=("location",)),
        _atomic("b_more_missing", input_types=()),
        _atomic(
            "c_wrong_effect",
            predicate="object.cleaned",
            args=("object",),
            input_types=("entity",),
        ),
    ]
    result = AtomicRetriever(
        _SkillBank(atomics),
        top_k=3,
        max_top_k=5,
    ).retrieve(
        [requirement],
        mode=RuntimeMode.ONLINE,
        harness_profile="alfworld",
    ).results[0]

    assert not result.covered
    assert len(result.repair_hints) == 3
    assert [hint["atomic_ref"] for hint in result.repair_hints] == [
        "skill://z_high_lexical@1.0.0",
        "skill://a_low_lexical@1.0.0",
        "skill://b_more_missing@1.0.0",
    ]
    first = result.repair_hints[0]
    assert set(first) == {"atomic_ref", "compatibility", "contract_view"}
    assert set(first["contract_view"]) == {"inputs", "outputs", "effects"}
    assert first["contract_view"]["effects"][0]["args"] == {
        "location": "<role>",
        "object": "<role>",
    }
    serialized = json.dumps(first, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "forbidden_primitive_sequence",
        "forbidden_trace_payload",
        "old_concrete_object",
        "old_concrete_location",
        "parameter_mapping",
    ):
        assert forbidden not in serialized
    assert all(
        "compatibility" in rejection
        for rejection in result.rejection_reasons
        if "effect_or_io_contract_mismatch" in rejection["reasons"]
    )


def test_incompatible_effect_hint_does_not_relax_code_matcher() -> None:
    requirement = _requirement()
    atomic = _atomic(
        "different_effect",
        predicate="object.cleaned",
        args=("object",),
    )
    result = AtomicRetriever(
        _SkillBank([atomic]),
        top_k=3,
        max_top_k=5,
    ).retrieve(
        [requirement],
        mode=RuntimeMode.ONLINE,
        harness_profile="alfworld",
    ).results[0]
    assert result.repair_hints
    assert not result.candidates
    assert not atomic_contract_compatible(requirement, atomic)


def _submission_payload() -> dict[str, Any]:
    return {
        "requirements": [{
            "requirement_id": "req_move",
            "intent": "move object",
            "desired_effects": [{
                "predicate": "object.at_location",
                "args": {"object": "$object", "location": "$location"},
                "cardinality": 1,
                "distinct_by": "",
            }],
            "expected_inputs": [],
            "expected_outputs": [],
            "precondition_hints": [],
            "semantic_variants": ["move"],
            "required": True,
            "rationale": "required transition",
        }],
        "repeat_blocks": [],
    }


class _CaptureSession:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.usage_buckets: list[str] = []
        self.acknowledged: list[str] = []

    def set_usage_bucket(self, value: str) -> None:
        self.usage_buckets.append(value)

    def next_turn(self, prompt: str, *, tools: list[Any]) -> AgentTurn:
        self.prompts.append(prompt)
        tool = tools[0]
        return AgentTurn(
            content="",
            tool_calls=[NativeToolCall(
                f"call_{len(self.prompts)}",
                tool.name,
                _submission_payload(),
            )],
            finish_reason="tool_calls",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            reasoning_tokens=0,
            latency_ms=0.0,
        )

    def acknowledge_tool_result(self, call_id: str, _payload: Any) -> None:
        self.acknowledged.append(call_id)


def test_p1r_receives_sanitized_compatibility_details_but_p1_does_not_read_bank() -> None:
    session = _CaptureSession()
    agent = RequirementAgent(session)
    task = SimpleNamespace(goal="move an object")
    contract = TaskContract(target_effects=[SemanticPredicate(
        "object.at_location",
        {"object": "$object", "location": "$location"},
    )])
    requirement = _requirement(input_types=("location",))
    hint = {
        "atomic_ref": "skill://known_atomic@1.0.0",
        "compatibility": {
            "effects_passed": True,
            "inputs_passed": False,
            "failure_codes": ["atomic_required_input_type_unavailable"],
            "missing_required_input_types": ["location"],
        },
        "contract_view": {"inputs": [], "outputs": [], "effects": []},
    }
    search = [RequirementSearchResult(
        requirement=requirement,
        candidates=[],
        covered=False,
        rejection_reasons=[{
            "atomic_ref": "skill://full_bank_row_must_stay_audit_only@1.0.0",
            "reasons": ["effect_or_io_contract_mismatch"],
            "validator_hidden_state": "forbidden_hidden_payload",
        }],
        repair_hints=[hint],
    )]

    agent.propose(task, contract, "observation", "alfworld")
    assert "skill://known_atomic@1.0.0" not in session.prompts[0]

    agent.repair(
        task,
        contract,
        PlannerRequirementBundle([requirement]),
        search,
        [],
    )
    repair_prompt = session.prompts[1]
    assert "skill://known_atomic@1.0.0" in repair_prompt
    assert "atomic_required_input_type_unavailable" in repair_prompt
    assert "sanitized interface hints" in repair_prompt
    assert "Do not change the Requirement merely to force a match" in repair_prompt
    assert "full_bank_row_must_stay_audit_only" not in repair_prompt
    assert "forbidden_hidden_payload" not in repair_prompt
