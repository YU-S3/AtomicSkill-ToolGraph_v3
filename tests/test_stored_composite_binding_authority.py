from __future__ import annotations

import json
from typing import Any, Mapping

from experiments.fakes import FakeAgentFactory, FakeReply
from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    GroundingConstraint,
    GroundingConstraintKind,
    ToolBinding,
)
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ContractSource,
    ImplementationAtom,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
    ToolAsset,
)
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.results import (
    NodeExecutionStatus,
    PrimitiveToolStep,
    RuntimeLinearPlan,
    RuntimeOccurrence,
)
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.extractor_session import CompositeExtractionProposal
from atomic_skillgraph.harness.action_catalog import HarnessActionCatalog
from atomic_skillgraph.harness.alfworld import (
    AlfWorldContractMatcher,
    AlfWorldValidatorChannel,
    semantic_value_compatible,
)
from atomic_skillgraph.harness.protocol import (
    HarnessActionResult,
    HarnessActionSpec,
    HarnessTask,
)
from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler
from atomic_skillgraph.runtime.orchestrator import RuntimeOrchestrator
from atomic_skillgraph.validation.engine import ValidationEngine


def _predicate(name: str, **roles: str) -> SemanticPredicate:
    return SemanticPredicate(
        name,
        {
            argument: BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role=input_role,
            )
            for argument, input_role in roles.items()
        },
    )


def _occurrence(
    occurrence_id: str,
    intent: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    effect: SemanticPredicate,
    index: int,
) -> CanonicalAtomicOccurrence:
    return CanonicalAtomicOccurrence(
        occurrence_id,
        occurrence_id,
        intent,
        index,
        index,
        inputs,
        outputs,
        [
            ParameterSpec(role, "entity", True, True, "concrete")
            for role in inputs
        ],
        [ParameterSpec(role, "entity") for role in outputs],
        [],
        [effect],
        [],
        [],
        {"task_id": "train_pick_place"},
        "trace_train_pick_place",
        SkillRef(f"atomic_{intent.replace(' ', '_')}", "1.0.0"),
        [f"witness:{occurrence_id}"],
    )


def _four_node_composite():
    canonical = [
        _occurrence(
            "source_nav", "navigate",
            {"destination": "cabinet_1"},
            {"reached_location": "cabinet_1"},
            _predicate("agent.at_location", location="destination"),
            0,
        ),
        _occurrence(
            "take", "take",
            {"object": "apple_1", "source": "cabinet_1"},
            {"object": "apple_1"},
            _predicate("agent.holds", object="object"),
            1,
        ),
        _occurrence(
            "target_nav", "navigate",
            {"destination": "sidetable_1"},
            {"reached_location": "sidetable_1"},
            _predicate("agent.at_location", location="destination"),
            2,
        ),
        _occurrence(
            "put", "put",
            {"object": "apple_1", "destination": "sidetable_1"},
            {"placed_object": "apple_1"},
            _predicate(
                "object.at_location", object="object", location="destination",
            ),
            3,
        ),
    ]
    edges = [
        {
            "edge_id": "source_to_take",
            "edge_type": "data_flow",
            "source_step": "source_nav",
            "target_step": "take",
            "source_role": "reached_location",
            "target_role": "source",
        },
        {
            "edge_id": "take_to_put",
            "edge_type": "data_flow",
            "source_step": "take",
            "target_step": "put",
            "source_role": "object",
            "target_role": "object",
        },
        {
            "edge_id": "target_to_put",
            "edge_type": "data_flow",
            "source_step": "target_nav",
            "target_step": "put",
            "source_role": "reached_location",
            "target_role": "destination",
        },
    ]
    contract = TaskContract([
        SemanticPredicate(
            "object.at_location", {"object": "apple", "location": "sidetable"},
        ),
    ])
    composite = CompositeBuilder().validate_and_build(
        CompositeExtractionProposal(
            [item.occurrence_id for item in canonical],
            [],
            edges,
            "pick and place through source and target navigation",
            {},
            {},
        ),
        canonical,
        contract,
        contract_matcher=AlfWorldContractMatcher(),
        task_bindings={"object": "apple", "destination": "sidetable"},
    )
    return composite


def test_runtime_resolvable_source_is_not_bound_by_same_task_role() -> None:
    composite = _four_node_composite()
    by_id = {item.occurrence_id: item for item in composite.occurrences}

    assert "destination" not in by_id["source_nav"].binding_specs
    target = by_id["target_nav"].binding_specs["destination"]
    assert target == BindingExpression(
        BindingExprKind.SKILL_INPUT, source_role="destination",
    )
    assert composite.metadata["binding_origins"]["source_nav"]["destination"] == {
        "kind": "runtime",
    }
    assert composite.metadata["binding_origins"]["target_nav"]["destination"] == {
        "kind": "task", "source_role": "destination",
    }


def test_task_object_anchor_propagates_backward_through_identity_dataflow() -> None:
    composite = _four_node_composite()
    by_id = {item.occurrence_id: item for item in composite.occurrences}

    assert by_id["take"].binding_specs["object"] == BindingExpression(
        BindingExprKind.SKILL_INPUT, source_role="object",
    )
    assert by_id["put"].binding_specs["object"].kind is BindingExprKind.DATA_FLOW
    assert by_id["put"].binding_specs["destination"].kind is BindingExprKind.DATA_FLOW


class _PickPlaceHarness:
    """Small public-affordance world for the four-node reuse regression."""

    profile_name = "fake_v3"
    semantic_value_compatible = staticmethod(semantic_value_compatible)

    def __init__(self) -> None:
        self._catalog = HarnessActionCatalog(self._parse_action)
        self._validator = AlfWorldValidatorChannel()
        self._task: HarnessTask | None = None
        self._revision = 0
        self._location = ""
        self._held = False
        self._won = False

    @staticmethod
    def _parse_action(
        raw: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
        return (
            str(raw["action_type"]),
            dict(raw.get("arguments") or {}),
            str(raw.get("display_text") or raw["action_type"]),
            {},
        )

    def reset(self, task: HarnessTask) -> HarnessActionResult:
        self._task = task
        self._revision = 0
        self._location = "start_1"
        self._held = self._won = False
        self._validator.reset()
        return HarnessActionResult(
            True,
            "The apple is somewhere in the room.",
            False,
            False,
            0,
            self._replace_catalog(),
            {"reset": True},
        )

    def action_catalog(self) -> list[HarnessActionSpec]:
        return self._catalog.items()

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        if self._task is None:
            raise RuntimeError("harness not reset")
        spec = self._catalog.get(action_id, revision)
        object_instance = str(self._task.context["object_instance"])
        source_instance = str(self._task.context["source_instance"])
        destination_instance = str(self._task.context["destination_instance"])
        accepted = False
        if spec.action_type == "GO_TO":
            self._location = str(spec.arguments["destination"])
            accepted = True
        elif spec.action_type == "LOOK":
            accepted = True
        elif (
            spec.action_type == "TAKE"
            and spec.arguments == {
                "object": object_instance,
                "source": source_instance,
            }
            and self._location == source_instance
            and not self._held
        ):
            self._held = True
            accepted = True
        elif (
            spec.action_type == "PUT"
            and spec.arguments == {
                "object": object_instance,
                "destination": destination_instance,
            }
            and self._location == destination_instance
            and self._held
        ):
            self._held = False
            self._won = True
            accepted = True
        self._revision += 1
        self._validator.record(
            spec,
            accepted=accepted,
            revision=self._revision,
            done=self._won,
            won=self._won,
        )
        return HarnessActionResult(
            accepted,
            f"{spec.display_text}: {'accepted' if accepted else 'rejected'}",
            self._won,
            self._won,
            self._revision,
            self._replace_catalog(),
            {},
        )

    def validator_channel(self) -> AlfWorldValidatorChannel:
        return self._validator

    def compile_primitive(
        self, primitive: PrimitiveToolStep, bindings: dict[str, Any],
    ) -> HarnessActionSpec:
        expected: dict[str, Any] = {}
        for role, raw in primitive.argument_mapping.items():
            expression = (
                BindingExpression.from_dict(raw)
                if isinstance(raw, dict) and "kind" in raw
                else raw
            )
            if isinstance(expression, BindingExpression):
                expected[role] = (
                    expression.constant
                    if expression.kind is BindingExprKind.CONSTANT
                    else bindings.get(expression.source_role)
                )
            else:
                expected[role] = expression
        for spec in self.action_catalog():
            if spec.action_type == primitive.action_type and spec.arguments == expected:
                return spec
        raise KeyError(f"no current affordance matches {primitive.action_type} {expected}")

    def supports_constraint(self, kind: str, verifier_id: str = "") -> bool:
        return kind in {
            "argument_exists",
            "argument_concrete",
            "harness_affordance",
            "current_context",
        } or bool(verifier_id)

    def _replace_catalog(self) -> list[HarnessActionSpec]:
        if self._task is None:
            return self._catalog.replace([], self._revision)
        object_instance = str(self._task.context["object_instance"])
        source_instance = str(self._task.context["source_instance"])
        destination_instance = str(self._task.context["destination_instance"])
        actions: list[dict[str, Any]] = [
            {
                "action_type": "GO_TO",
                "arguments": {"destination": source_instance},
                "display_text": "go to drawer 2",
            },
            {
                "action_type": "GO_TO",
                "arguments": {"destination": destination_instance},
                "display_text": "go to desk 1",
            },
            {
                "action_type": "LOOK",
                "arguments": {},
                "display_text": "look",
            },
        ]
        if self._location == source_instance and not self._held:
            actions.append({
                "action_type": "TAKE",
                "arguments": {
                    "object": object_instance,
                    "source": source_instance,
                },
                "display_text": "take apple 2 from drawer 2",
            })
        if self._location == destination_instance and self._held:
            actions.append({
                "action_type": "PUT",
                "arguments": {
                    "object": object_instance,
                    "destination": destination_instance,
                },
                "display_text": "put apple 2 in/on desk 1",
            })
        return self._catalog.replace(actions, self._revision)


def _runtime_assets():
    def build(
        name: str,
        action_type: str,
        inputs: list[str],
        output_role: str,
        output_input_role: str,
        effect: SemanticPredicate,
    ):
        atomic_ref = SkillRef(f"atomic_{name}", "1.0.0")
        tool_ref = ToolRef(f"tool_{name}", "1.0.0")
        implementation_ref = SkillRef(f"impl_{name}", "1.0.0")
        atomic = AbstractAtomicSkill(
            atomic_ref,
            name,
            [ParameterSpec(role, "entity", True, True, "concrete") for role in inputs],
            [ParameterSpec(output_role, "entity")],
            [],
            [effect],
            {"validator_id": "harness_atomic_effect"},
            [],
            {"steps": [action_type]},
            {},
            SkillStatus.CANDIDATE,
        )
        argument_mapping = {
            role: BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)
            for role in inputs
        }
        tool = ToolAsset(
            tool_ref,
            name,
            {
                "type": "object",
                "properties": {role: {"type": "string"} for role in inputs},
                "required": inputs,
            },
            {
                "output_schema": {
                    "type": "object",
                    "properties": {output_role: {"type": "string"}},
                    "required": [output_role],
                    "additionalProperties": False,
                },
            },
            "primitive_ir",
            {
                "steps": [{
                    "action_type": action_type,
                    "argument_mapping": argument_mapping,
                }],
                "output_mapping": {
                    output_role: to_primitive(BindingExpression(
                        BindingExprKind.SKILL_INPUT,
                        source_role=output_input_role,
                    )),
                },
            },
            [],
            {"reviewed": True},
            {},
            {},
            ToolStatus.CANDIDATE,
        )
        implementation = ImplementationAtom(
            implementation_ref,
            atomic_ref,
            [ToolBinding(tool_ref, "primary", argument_mapping)],
            [GroundingConstraint(
                f"{name}_affordance",
                GroundingConstraintKind.HARNESS_AFFORDANCE,
                action_type=action_type,
                argument_mapping=argument_mapping,
                required_resolution="relation_verified" if len(inputs) > 1 else "concrete",
            )],
            {
                "mode": "serial",
                "output_mapping": {
                    output_role: BindingExpression(
                        BindingExprKind.TOOL_OUTPUT,
                        source_role=output_role,
                        source_step="primary",
                    ),
                },
            },
            {"harness_profiles": ["fake_v3"]},
            {"preferred": True},
            SkillStatus.CANDIDATE,
        )
        return atomic, implementation, tool

    nav = build(
        "navigate", "GO_TO", ["destination"], "reached_location", "destination",
        _predicate("agent.at_location", location="destination"),
    )
    take = build(
        "take", "TAKE", ["object", "source"], "held_object", "object",
        _predicate("agent.holds", object="object"),
    )
    put = build(
        "put", "PUT", ["object", "destination"], "placed_object", "object",
        _predicate("object.at_location", object="object", location="destination"),
    )
    return nav, take, put


def test_four_node_new_scene_reuse_closes_binding_and_dataflow() -> None:
    nav, take, put = _runtime_assets()
    atomics = {str(item[0].ref): item[0] for item in (nav, take, put)}
    implementations = {str(item[1].ref): item[1] for item in (nav, take, put)}
    tools = {str(item[2].ref): item[2] for item in (nav, take, put)}

    class Skills:
        def get_atomic(self, ref):
            return atomics[str(ref)]

        def get_implementation(self, ref):
            return implementations[str(ref)]

    class Tools:
        def get(self, ref):
            return tools[str(ref)]

        def tools(self):
            return list(tools.values())

    source_nav = RuntimeOccurrence(
        "source_nav", "source_nav", nav[0].ref, [], {}, [nav[1].ref], nav[0].effects,
    )
    take_occurrence = RuntimeOccurrence(
        "take", "take", take[0].ref, [],
        {
            "object": BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role="object",
            ),
            "source": BindingExpression(
                BindingExprKind.DATA_FLOW,
                source_role="reached_location",
                source_step="source_nav",
            ),
        },
        [take[1].ref],
        take[0].effects,
    )
    target_nav = RuntimeOccurrence(
        "target_nav", "target_nav", nav[0].ref, [],
        {
            "destination": BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role="destination",
            ),
        },
        [nav[1].ref],
        nav[0].effects,
    )
    put_occurrence = RuntimeOccurrence(
        "put", "put", put[0].ref, [],
        {
            "object": BindingExpression(
                BindingExprKind.DATA_FLOW,
                source_role="held_object",
                source_step="take",
            ),
            "destination": BindingExpression(
                BindingExprKind.DATA_FLOW,
                source_role="reached_location",
                source_step="target_nav",
            ),
        },
        [put[1].ref],
        put[0].effects,
    )
    edges = [
        GraphEdge(
            "source_to_take", GraphEdgeType.DATA_FLOW,
            "source_nav", "take", "reached_location", "source",
        ),
        GraphEdge(
            "take_to_put", GraphEdgeType.DATA_FLOW,
            "take", "put", "held_object", "object",
        ),
        GraphEdge(
            "target_to_put", GraphEdgeType.DATA_FLOW,
            "target_nav", "put", "reached_location", "destination",
        ),
    ]
    contract = TaskContract(
        [SemanticPredicate(
            "object.at_location", {"object": "apple", "location": "desk"},
        )],
        source=ContractSource.ADAPTER_DERIVED,
        confidence=1.0,
        validator_id="fake_pick_place",
    )
    plan = RuntimeLinearPlan(
        "reuse_task",
        "stored_composite",
        "skill://pick_place@1.0.0",
        [source_nav, take_occurrence, target_nav, put_occurrence],
        ["source_nav", "take", "target_nav", "put"],
        edges,
        [],
        contract,
        {"final_outcome": "stored_composite"},
    )

    class Planner:
        def build_plan(self, *_args, **_kwargs):
            return plan

    harness = _PickPlaceHarness()
    factory = FakeAgentFactory()
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("environment_action", {"action_id": "r000_a003"}),
        FakeReply.tool("$learned", {"destination": "drawer_2"}),
    ])
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("$learned", {"object": "apple_2"}),
    ])
    factory.enqueue("runtime_preparation", [
        FakeReply.tool("$learned", {"destination": "desk_1"}),
    ])
    runtime = RuntimeOrchestrator(
        Planner(),
        harness,
        InvocationCompiler(
            Skills(), Tools(), harness, mode=RuntimeMode.ONLINE,
        ),
        ValidationEngine(),
        factory,
        runtime_config={
            "global_action_budget": 10,
            "node_action_budget": 4,
            "learned_toolcall_repair_limit": 2,
        },
    )
    task = HarnessTask(
        "reuse_task",
        "put an apple on a desk",
        "fake",
        "pick_and_place_simple",
        {
            "semantic_bindings": {"object": "apple", "destination": "desk"},
            "binding_types": {"object": "entity", "destination": "entity"},
            "object_instance": "apple_2",
            "source_instance": "drawer_2",
            "destination_instance": "desk_1",
            "initial_observation": "The apple is somewhere in the room.",
        },
        {"task_signature": "reuse_task:apple:desk"},
    )

    trace = runtime.run_task(task)

    assert trace.benchmark_success is True
    assert trace.graph_self_sufficient_success is True
    assert trace.task_rescue_required is False
    assert len(trace.node_records) == 4
    assert all(
        item.status in {
            NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS,
            NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS,
        }
        for item in trace.node_records
    )
    assert len(trace.implementation_invocations) == 4
    assert all(
        item.result["started"] and item.result["completed"]
        for item in trace.implementation_invocations
    ), json.dumps([item.result for item in trace.implementation_invocations], indent=2)
    assert len(trace.tool_executions) == 4
    assert all(item.result["started"] and item.result["completed"] for item in trace.tool_executions)
    assert any(
        item["occurrence_id"] == "take"
        and item["role"] == "source"
        and item["reason"] == "data_flow"
        for item in trace.binding_changes
    )
    first_session = next(
        item for item in trace.agent_sessions
        if item.occurrence_id == "source_nav"
    )
    first_prompt = next(
        item["content"] for item in first_session.snapshot["messages"]
        if item["role"] == "user"
    )
    payload = json.loads(first_prompt.split("\n\nPOLICY_CONTEXT_JSON\n", 1)[1])
    assert payload["task_semantic_context"]["destination"] == "desk"
    assert "destination" not in payload["current_occurrence_semantic_anchors"]
    assert any(
        item["arguments"].get("destination") == "drawer_2"
        for item in payload["current_action_catalog"]
    )
    first_tool_result = next(
        json.loads(item["content"])
        for item in first_session.snapshot["messages"]
        if item["role"] == "tool"
        and json.loads(item["content"]).get("new_revision") == 1
    )
    assert any(
        item["arguments"].get("destination") == "drawer_2"
        and item["revision"] == 1
        for item in first_tool_result["action_catalog"]
    )
    factory.assert_exhausted()
