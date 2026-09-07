from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import pytest

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
    ToolAsset,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.harness.protocol import HarnessActionSpec, PredicateSpec
from atomic_skillgraph.tooling.proposal import tool_proposal_from_dict
from atomic_skillgraph.tooling.validator import (
    ToolStaticValidator,
    _concrete_ids_from_nodes,
)


class _ToolHarness:
    profile_name = "r5_fixture"

    def __init__(self, catalog_values: tuple[str, ...] = ()) -> None:
        self._catalog_values = catalog_values

    def primitive_action_schema(self) -> list[dict[str, Any]]:
        return [
            {"action_type": "MOVE", "argument_roles": ["object", "destination"]},
            {"action_type": "TAKE", "argument_roles": ["item"]},
        ]

    def semantic_predicate_schema(self) -> list[PredicateSpec]:
        return [
            PredicateSpec(
                "object.at_location", "world", ("location", "object"),
                {"location": "entity", "object": "entity"}, "fixture",
            ),
            PredicateSpec(
                "agent.holds", "world", ("object",),
                {"object": "entity"}, "fixture",
            ),
        ]

    def action_catalog(self) -> list[HarnessActionSpec]:
        return [
            HarnessActionSpec(
                action_id=f"r000_a{index:03d}",
                revision=0,
                action_type="TAKE",
                arguments={"item": value},
                display_text=f"take {value}",
                raw_action=f"take {value}",
                metadata={},
            )
            for index, value in enumerate(self._catalog_values, 1)
        ]


def _task4_payload() -> dict[str, Any]:
    # Exact original task-4 place proposal, including its untouched
    # path_expectations and source-instance rationale.
    return {
        "proposal_version": "1",
        "decision": "create",
        "summary": "place_object_at_location",
        "atomic_ref": "skill://atomic_place_object_at_location_238d043f8003@1.0.0",
        "inputs": [
            {
                "name": "destination",
                "semantic_type": "entity",
                "required": True,
                "runtime_resolvable": True,
                "required_resolution": "concrete",
                "description": "",
            },
            {
                "name": "object",
                "semantic_type": "entity",
                "required": True,
                "runtime_resolvable": True,
                "required_resolution": "concrete",
                "description": "",
            },
        ],
        "outputs": [
            {
                "name": "place_location",
                "semantic_type": "entity",
                "required": True,
                "runtime_resolvable": False,
                "required_resolution": "semantic",
                "description": "",
            },
            {
                "name": "placed_object",
                "semantic_type": "entity",
                "required": True,
                "runtime_resolvable": False,
                "required_resolution": "semantic",
                "description": "",
            },
        ],
        "program": [
            {
                "node_id": "act_place",
                "op": "ACTION",
                "action_type": "MOVE",
                "argument_mapping": {
                    "object": {"kind": "skill_input", "source_role": "object"},
                    "destination": {
                        "kind": "skill_input",
                        "source_role": "destination",
                    },
                },
                "expected_effects": [
                    {
                        "predicate": "object.at_location",
                        "args": {
                            "location": {
                                "kind": "skill_input",
                                "source_role": "destination",
                            },
                            "object": {
                                "kind": "skill_input",
                                "source_role": "object",
                            },
                        },
                        "cardinality": 1,
                        "distinct_by": "",
                        "effect_domain": "world",
                    }
                ],
            },
            {
                "node_id": "ret_outputs",
                "op": "RETURN",
                "output_sources": {
                    "place_location": {
                        "source": "tool_input",
                        "field": "destination",
                    },
                    "placed_object": {
                        "source": "tool_input",
                        "field": "object",
                    },
                },
            },
        ],
        "max_actions": 1,
        "final_effects": [
            {
                "predicate": "object.at_location",
                "args": {
                    "location": {
                        "kind": "skill_input",
                        "source_role": "place_location",
                    },
                    "object": {
                        "kind": "skill_input",
                        "source_role": "placed_object",
                    },
                },
                "cardinality": 1,
                "distinct_by": "",
                "effect_domain": "world",
            }
        ],
        "evidence_outputs": [],
        "path_expectations": [
            {
                "path": "act_place",
                "must_verify": (
                    "MOVE(object=<object input>, destination=<destination input>) "
                    "is accepted and produces world evidence "
                    "object.at_location(location=<destination input>, "
                    "object=<object input>) with cardinality 1"
                ),
            },
            {
                "path": "ret_outputs",
                "must_verify": (
                    "returned place_location equals the destination input and "
                    "placed_object equals the object input of the witnessed "
                    "object.at_location fact"
                ),
            },
        ],
        "rationale": (
            "The supplied success evidence (r027_a043) shows a single accepted "
            "MOVE(object, destination) primitive producing the world fact "
            "object.at_location(location=countertop_1, object=ladle_1), where "
            "destination=countertop_1 and object=ladle_1 are exactly the supplied "
            "inputs. The Atomic is therefore implemented by one MOVE action mapping "
            "object->object and destination->destination; its immediate step effect "
            "is object.at_location(location=<destination>, object=<object>). Since "
            "the witnessed object.at_location location argument is precisely the "
            "destination input and its object argument is precisely the object input "
            "(the evidence witness refs confirm the same identities), returning "
            "place_location from the destination input and placed_object from the "
            "object input is a direct return of the same witnessed identity permitted "
            "by the effect_witness derivation, not an inference from prose. "
            "final_effects is an exact copy of canonical_atomic.effects, keeping "
            "formal roles place_location and placed_object. Only one ACTION occurs, "
            "so max_actions=1 is a true bound; no loop is justified because the "
            "evidence contains a single MOVE repetition. No episode entity or task "
            "identifier is embedded as a constant."
        ),
    }


def _task4_atomic(payload: dict[str, Any]) -> AbstractAtomicSkill:
    proposal = tool_proposal_from_dict(payload)
    return AbstractAtomicSkill(
        SkillRef.parse(payload["atomic_ref"]),
        "place object at location",
        list(proposal.inputs),
        list(proposal.outputs),
        [],
        list(proposal.final_effects),
        {},
        [],
        {},
        {},
    )


def _validate_task4(
    payload: dict[str, Any],
    *,
    harness: _ToolHarness | None = None,
    history: Any = None,
):
    proposal = tool_proposal_from_dict(payload)
    return ToolStaticValidator().validate_proposal(
        proposal,
        _task4_atomic(_task4_payload()),
        harness or _ToolHarness(),
        historical_evidence_support=history,
    )


def test_original_task4_proposal_is_not_rejected_for_cardinality_prose() -> None:
    payload = _task4_payload()

    report = _validate_task4(payload)

    assert report.passed, report.messages
    assert "tool_ir_episode_concrete_id" not in report.failure_codes
    # Rationale remains untouched and intentionally outside the new scan surface.
    assert "countertop_1" in payload["rationale"]
    assert payload["path_expectations"][0]["must_verify"].endswith(
        "cardinality 1"
    )


@pytest.mark.parametrize(
    "description",
    [
        "with cardinality 1",
        "with cardinality 2.",
        "repeat 3 times; step 2",
        "step 3",
        "step 3.",
    ],
)
def test_quantity_annotations_are_not_episode_instances(description: str) -> None:
    payload = _task4_payload()
    payload["path_expectations"][0]["must_verify"] = description

    report = _validate_task4(payload)

    assert "tool_ir_episode_concrete_id" not in report.failure_codes


def test_native_quantities_are_never_stringified_into_episode_ids() -> None:
    program = [
        {
            "node_id": "act",
            "op": "ACTION",
            "argument_mapping": {
                "item": {"kind": "constant", "constant": 1},
            },
            "expected_effects": [
                {"predicate": "p", "args": {"x": True, "y": None}}
            ],
        },
        {
            "node_id": "branch",
            "op": "IF",
            "condition": {
                "source": "tool_input",
                "field": "x",
                "op": "equals",
                "value": 1.5,
            },
            "then_branch": [],
            "else_branch": [],
        },
        {
            "node_id": "ret",
            "op": "RETURN",
            "output_sources": {
                "result": {"source": "constant", "value": "1"}
            },
        },
    ]

    assert _concrete_ids_from_nodes(program) == []


def test_nested_executable_constants_report_exact_path_value_and_category() -> None:
    payload = _task4_payload()
    payload["program"][0]["argument_mapping"]["object"] = {
        "kind": "constant",
        "constant": {"nested": ["cup_3"]},
    }

    report = _validate_task4(payload)

    assert "tool_ir_episode_concrete_id" in report.failure_codes
    message = next(
        item for item in report.messages
        if "episode concrete literal:" in item
    )
    assert "path=program[0].argument_mapping.object.constant.nested[0]" in message
    assert 'value="cup_3"' in message
    assert 'matched_text="cup_3"' in message
    assert "category=executable_literal" in message


def test_all_executable_literal_surfaces_remain_strict() -> None:
    program = [
        {
            "node_id": "act",
            "op": "ACTION",
            "argument_mapping": {
                "item": {"kind": "constant", "constant": "object_1"}
            },
            "expected_effects": [
                {
                    "predicate": "p",
                    "args": {
                        "x": {
                            "kind": "constant",
                            "constant": "effect_object 2",
                        }
                    },
                }
            ],
        },
        {
            "node_id": "condition",
            "op": "STOP_WHEN",
            "condition": {
                "source": "tool_input",
                "field": "x",
                "op": "equals",
                "value": {"nested": ["cabinet 999"]},
            },
        },
        {
            "node_id": "loop",
            "op": "FOR_EACH",
            "collection_source": {
                "source": "local_deterministic",
                "values": [{"kind": "skill_input", "constant": "cup_3"}],
                "where": {"object": "drawer_4"},
            },
            "iteration_variable": "candidate",
            "max_iterations": 1,
            "body": [],
        },
        {
            "node_id": "ret",
            "op": "RETURN",
            "output_sources": {
                "result": {
                    "kind": "constant",
                    "constant": {"nested": ["plate_5"]},
                }
            },
        },
    ]

    hits = _concrete_ids_from_nodes(program)

    assert hits == [
        "object_1", "effect_object 2", "cabinet 999",
        "cup_3", "drawer_4", "plate_5",
    ]


def _numbered_role_payload() -> tuple[dict[str, Any], AbstractAtomicSkill]:
    atomic = AbstractAtomicSkill(
        SkillRef("atomic_numbered_role", "1.0.0"),
        "take item",
        [ParameterSpec("item_1", "entity")],
        [ParameterSpec("held_1", "entity")],
        [],
        [SemanticPredicate("agent.holds", {"object": "$held_1"})],
        {},
        [],
        {},
        {},
    )
    payload = {
        "proposal_version": "1",
        "decision": "create",
        "summary": "take item",
        "atomic_ref": str(atomic.ref),
        "inputs": [vars(item) for item in atomic.inputs],
        "outputs": [vars(item) for item in atomic.outputs],
        "program": [
            {
                "node_id": "take_1",
                "op": "ACTION",
                "action_type": "TAKE",
                "argument_mapping": {
                    "item": {"kind": "skill_input", "source_role": "item_1"}
                },
                "expected_effects": [
                    {"predicate": "agent.holds", "args": {"object": "$item_1"}}
                ],
            },
            {
                "node_id": "return_1",
                "op": "RETURN",
                "output_sources": {
                    "held_1": {"source": "tool_input", "field": "item_1"}
                },
            },
        ],
        "max_actions": 1,
        "final_effects": [
            {"predicate": "agent.holds", "args": {"object": "$held_1"}}
        ],
        "evidence_outputs": [],
        "path_expectations": [
            {
                "path": "take_1",
                "must_verify": "$item_1 and <item_1 input> are formal references",
            }
        ],
        "rationale": "portable",
    }
    return payload, atomic


def test_numbered_formal_roles_are_not_literals_but_same_constant_is() -> None:
    payload, atomic = _numbered_role_payload()
    validator = ToolStaticValidator()

    formal = validator.validate_proposal(
        tool_proposal_from_dict(payload), atomic, _ToolHarness(),
    )
    literal_payload = copy.deepcopy(payload)
    literal_payload["program"][0]["argument_mapping"]["item"] = {
        "kind": "constant",
        "constant": "item_1",
    }
    literal = validator.validate_proposal(
        tool_proposal_from_dict(literal_payload), atomic, _ToolHarness(),
    )

    assert formal.passed, formal.messages
    assert "tool_ir_episode_concrete_id" in literal.failure_codes


def test_invalid_numbered_reference_keeps_scope_failure_not_literal_failure() -> None:
    payload, atomic = _numbered_role_payload()
    payload["program"][0]["argument_mapping"]["item"]["source_role"] = "ghost_9"

    report = ToolStaticValidator().validate_proposal(
        tool_proposal_from_dict(payload), atomic, _ToolHarness(),
    )

    assert "tool_ir_input_closure_invalid" in report.failure_codes
    assert "tool_ir_episode_concrete_id" not in report.failure_codes


def test_annotation_uses_canonical_token_and_public_spaced_alias_only() -> None:
    canonical = _task4_payload()
    canonical["path_expectations"][0]["must_verify"] = "verify cup_3."
    unknown_spaced = _task4_payload()
    unknown_spaced["path_expectations"][0]["must_verify"] = "verify cabinet 7."

    canonical_report = _validate_task4(canonical)
    unknown_report = _validate_task4(unknown_spaced)
    known_report = _validate_task4(
        unknown_spaced, harness=_ToolHarness(("cabinet_7",)),
    )

    assert "tool_ir_episode_concrete_id" in canonical_report.failure_codes
    assert "tool_ir_episode_concrete_id" not in unknown_report.failure_codes
    assert "tool_ir_episode_concrete_id" in known_report.failure_codes
    assert "path=path_expectations[0].must_verify" in known_report.messages[-1]
    assert 'matched_text="cabinet 7"' in known_report.messages[-1]
    assert "category=annotation" in known_report.messages[-1]


@pytest.mark.parametrize(
    ("unrelated_kind", "unrelated_constant"),
    [("skill_input", None), ("local_variable", None), ("constant", "safe")],
)
def test_path_expectation_kind_cannot_suppress_sibling_annotation_scan(
    unrelated_kind: str,
    unrelated_constant: str | None,
) -> None:
    payload = _task4_payload()
    expectation = payload["path_expectations"][0]
    expectation["kind"] = unrelated_kind
    if unrelated_constant is not None:
        expectation["constant"] = unrelated_constant
    expectation["must_verify"] = "verify cup_3"

    report = _validate_task4(payload)

    assert "tool_ir_episode_concrete_id" in report.failure_codes
    assert "path=path_expectations[0].must_verify" in report.messages[-1]


@pytest.mark.parametrize("extra_field", ["value", "constant"])
def test_evidence_output_extra_data_cannot_bypass_annotation_scan(
    extra_field: str,
) -> None:
    payload = _task4_payload()
    payload["evidence_outputs"] = [{
        "source": "tool_input",
        "field": "destination",
        "role": "place_location",
        extra_field: "cup_3",
    }]

    report = _validate_task4(payload)

    assert "tool_ir_episode_concrete_id" in report.failure_codes
    assert f"path=evidence_outputs[0].{extra_field}" in report.messages[-1]
    assert "category=annotation" in report.messages[-1]


@pytest.mark.parametrize(
    "argument_annotation",
    [
        {"kind": "skill_input", "source_role": "object", "note": "cup_3"},
        {"kind": "constant", "constant": "safe", "note": "cup_3"},
        {"kind": "bogus", "note": "cup_3"},
    ],
)
def test_path_expectation_nested_kind_cannot_hide_annotation(
    argument_annotation: dict[str, str],
) -> None:
    payload = _task4_payload()
    payload["path_expectations"][0]["arguments"] = {
        "object": argument_annotation,
    }

    report = _validate_task4(payload)

    assert "tool_ir_episode_concrete_id" in report.failure_codes
    assert (
        "path=path_expectations[0].arguments.object.note"
        in report.messages[-1]
    )
    assert "category=annotation" in report.messages[-1]


@pytest.mark.parametrize(
    "annotation",
    [
        {"field": "cup_3"},
        {"note": {"field": "cup_3"}},
    ],
)
def test_annotation_subtree_cannot_reinterpret_nested_keys_as_syntax(
    annotation: dict[str, Any],
) -> None:
    payload = _task4_payload()
    payload["path_expectations"][0]["must_verify"] = annotation

    report = _validate_task4(payload)

    assert "tool_ir_episode_concrete_id" in report.failure_codes
    assert "path=path_expectations[0].must_verify" in report.messages[-1]
    assert ".field" in report.messages[-1]
    assert 'matched_text="cup_3"' in report.messages[-1]


@pytest.mark.parametrize(
    "field_name", ["step", "cardinality", "max_iterations", "max_actions"]
)
def test_path_expectation_string_in_quantity_or_unknown_step_is_scanned(
    field_name: str,
) -> None:
    payload = _task4_payload()
    payload["path_expectations"][0][field_name] = "cup_3"

    report = _validate_task4(payload)

    assert "tool_ir_episode_concrete_id" in report.failure_codes
    assert f"path=path_expectations[0].{field_name}" in report.messages[-1]


def test_path_expectation_native_quantities_and_declared_numbered_path_are_safe() -> None:
    payload, atomic = _numbered_role_payload()
    payload["path_expectations"][0].update({
        "step": "take_1",
        "cardinality": 1,
        "max_iterations": 2.0,
        "max_actions": 3,
    })

    report = ToolStaticValidator().validate_proposal(
        tool_proposal_from_dict(payload), atomic, _ToolHarness(),
    )

    assert "tool_ir_episode_concrete_id" not in report.failure_codes


def test_history_iterator_is_materialized_once_and_shared_with_loop_check() -> None:
    payload, atomic = _numbered_role_payload()
    payload["program"] = [
        {
            "node_id": "loop",
            "op": "FOR_EACH",
            "collection_source": {
                "source": "local_deterministic",
                "values": ["portable_a", "portable_b"],
            },
            "iteration_variable": "candidate",
            "max_iterations": 2,
            "body": [
                {
                    "node_id": "take_each",
                    "op": "ACTION",
                    "action_type": "TAKE",
                    "argument_mapping": {
                        "item": {
                            "kind": "local_variable",
                            "source_role": "candidate",
                        }
                    },
                    "expected_effects": [
                        {
                            "predicate": "agent.holds",
                            "args": {
                                "object": {
                                    "kind": "local_variable",
                                    "source_role": "candidate",
                                }
                            },
                        }
                    ],
                }
            ],
        },
        payload["program"][1],
    ]
    payload["max_actions"] = 2
    payload["path_expectations"] = [
        {"path": "take_each", "must_verify": "verify cabinet 7"}
    ]
    yielded: list[str] = []

    def history() -> Iterator[dict[str, Any]]:
        for value in ("cabinet_7", "cabinet_8"):
            yielded.append(value)
            yield {
                "accepted": True,
                "action_type": "TAKE",
                "arguments": {"item": value},
            }

    report = ToolStaticValidator().validate_proposal(
        tool_proposal_from_dict(payload), atomic, _ToolHarness(),
        historical_evidence_support=history(),
    )

    assert yielded == ["cabinet_7", "cabinet_8"]
    assert "tool_ir_historical_loop_evidence_insufficient" not in report.failure_codes
    assert "tool_ir_episode_concrete_id" in report.failure_codes
    assert 'matched_text="cabinet 7"' in report.messages[-1]


@pytest.mark.parametrize("local_role", ["candidate_7", "item_1"])
def test_numbered_local_reference_uses_node_id_path_scope(
    local_role: str,
) -> None:
    payload, atomic = _numbered_role_payload()
    payload["program"] = [
        {
            "node_id": "loop_1",
            "op": "FOR_EACH",
            "collection_source": {
                "source": "local_deterministic",
                "values": ["portable_a", "portable_b"],
            },
            "iteration_variable": local_role,
            "max_iterations": 2,
            "body": [
                {
                    "node_id": "take_each",
                    "op": "ACTION",
                    "action_type": "TAKE",
                    "argument_mapping": {
                        "item": {
                            "kind": "local_variable",
                            "source_role": local_role,
                        }
                    },
                    "expected_effects": [
                        {
                            "predicate": "agent.holds",
                            "args": {
                                "object": {
                                    "kind": "local_variable",
                                    "source_role": local_role,
                                }
                            },
                        }
                    ],
                }
            ],
        },
        payload["program"][1],
    ]
    payload["max_actions"] = 2
    payload["path_expectations"] = [{
        "node_id": "take_each",
        "arguments": {
            "item": {
                "kind": "local_variable",
                "source_role": local_role,
            }
        },
        "must_verify": f"uses <{local_role} local>",
    }]
    history = [
        {
            "accepted": True,
            "action_type": "TAKE",
            "arguments": {"item": "portable_a"},
        },
        {
            "accepted": True,
            "action_type": "TAKE",
            "arguments": {"item": "portable_b"},
        },
    ]

    report = ToolStaticValidator().validate_proposal(
        tool_proposal_from_dict(payload),
        atomic,
        _ToolHarness(),
        historical_evidence_support=history,
    )

    assert "tool_ir_episode_concrete_id" not in report.failure_codes


def test_persisted_tool_reuses_the_same_episode_literal_scanner() -> None:
    payload = _task4_payload()
    payload["path_expectations"][0]["must_verify"] = "verify cup_3"
    atomic = _task4_atomic(payload)
    tool = ToolAsset(
        ToolRef("r5_place_tool", "1.0.0"),
        payload["summary"],
        {
            "type": "object",
            "properties": {
                item["name"]: {"type": "string"}
                for item in payload["inputs"]
            },
            "required": [item["name"] for item in payload["inputs"]],
        },
        {
            "output_schema": {
                "type": "object",
                "properties": {
                    item["name"]: {"type": "string"}
                    for item in payload["outputs"]
                },
                "required": [item["name"] for item in payload["outputs"]],
            }
        },
        "tool_ir_v1",
        {
            "program": copy.deepcopy(payload["program"]),
            "max_actions": payload["max_actions"],
            "final_effects": copy.deepcopy(payload["final_effects"]),
            "evidence_outputs": copy.deepcopy(payload["evidence_outputs"]),
            "path_expectations": copy.deepcopy(payload["path_expectations"]),
        },
        [],
        {},
        {},
        {"tool_builder_rationale": payload["rationale"]},
    )

    report = ToolStaticValidator().validate_tool_asset(
        tool, atomic, _ToolHarness(),
    )

    assert "tool_ir_episode_concrete_id" in report.failure_codes
    assert "path=path_expectations[0].must_verify" in report.messages[-1]
