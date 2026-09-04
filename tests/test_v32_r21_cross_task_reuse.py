"""Deterministic Gate29: cross-task Runtime self-tool reuse (v3.2-R2.1 freeze).

Task A must create persistent locate knowledge through the real production
chain: Runtime Automation draft -> R0 -> ToolBuilder -> task-local trial ->
R1 -> Task success -> Success Extractor -> Atomicizer -> Tool Admission ->
Registry.  Task B, on the same persisted bank with a different concrete
entity, must resolve its missing binding through SupportRetriever and execute
the learned zero-LLM Tool.  The fake Harness and deterministic Agent replies
are allowed; Extractor, Admission, Registry, and Lifecycle are never bypassed.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from atomic_skillgraph.core.bindings import BindingExprKind, BindingExpression
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ContractSource,
    EffectDomain,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.results import PrimitiveToolStep, RuntimeLinearPlan
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import Atomicizer
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.extractor_session import ExtractorSession
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceLedger
from atomic_skillgraph.governance.projections import LifecycleProjection
from atomic_skillgraph.harness.action_catalog import HarnessActionCatalog
from atomic_skillgraph.harness.alfworld import AlfWorldValidatorChannel
from atomic_skillgraph.harness.protocol import (
    HarnessActionResult,
    HarnessActionSpec,
    HarnessTask,
    PredicateSpec,
)
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.graph_store import GraphStore
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.planner.pipeline import PlannerPipeline
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler
from atomic_skillgraph.runtime.orchestrator import RuntimeOrchestrator
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.tooling.builder_session import ToolBuilderSession
from atomic_skillgraph.tooling.proposal import ToolProvenance
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.traces.schema import TaskRecord, TraceBuilder, TraceRecord
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import FakeAgentFactory, FakeReply

import re


def _room_for(target: Any) -> str:
    family = re.sub(r"(?:_|\s)\d+$", "", str(target)).casefold() or "unknown"
    # Deterministic, digit-suffixed concrete location per entity family so the
    # production validator's concrete-entity gate applies like real ALFWorld ids.
    return f"room_{sum(ord(char) for char in family) % 900 + 100}"


def _locating_task(
    task_id: str,
    target_item: str,
    *,
    expose_object: bool = True,
) -> HarnessTask:
    context: dict[str, Any] = {
        "target_item": target_item,
        "initial_observation": f"Target {target_item} is available.",
    }
    if expose_object:
        context["semantic_bindings"] = {"object": target_item}
    return HarnessTask(
        task_id=task_id,
        goal=f"Hold the target item ({target_item}).",
        benchmark="fake",
        task_type="locating_v1",
        context=context,
        metadata={"task_signature": f"fake:{task_id}:{target_item}"},
    )


class LocatingValidatorChannel(AlfWorldValidatorChannel):
    """Production-shaped channel; SEARCH records the discovery fact."""

    def record(
        self,
        spec: HarnessActionSpec,
        *,
        accepted: bool,
        revision: int,
        done: bool,
        won: bool,
        observation: str = "",
        metadata: dict[str, Any] | None = None,
        catalog: list[HarnessActionSpec] | None = None,
    ) -> None:
        super().record(
            spec, accepted=accepted, revision=revision, done=done, won=won,
            observation=observation, metadata=metadata, catalog=catalog,
        )
        if spec.action_type == "SEARCH" and accepted:
            target = str(spec.arguments.get("target", ""))
            room = _room_for(target)
            self._discovered[(target, room)] = (target, room)
            self._rebuild_facts()


class LocatingHarness:
    """Revisioned locate/pick world: SEARCH discovers, TAKE wins."""

    profile_name = "fake_v3"

    def __init__(self) -> None:
        self._catalog = HarnessActionCatalog(self._parse_action)
        self._validator = LocatingValidatorChannel()
        self._task: HarnessTask | None = None
        self._revision = 0
        self._held = False
        self._done = False
        self._won = False

    @staticmethod
    def _parse_action(raw: Mapping[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
        return (
            str(raw["action_type"]),
            copy.deepcopy(dict(raw.get("arguments", {}))),
            str(raw.get("display_text", raw["action_type"])),
            copy.deepcopy(dict(raw.get("metadata", {}))),
        )

    @property
    def current_task(self) -> HarnessTask | None:
        return self._task

    def reset(self, task: HarnessTask) -> HarnessActionResult:
        if "target_item" not in task.context:
            raise ValueError("LocatingHarness task requires context.target_item")
        self._task = task
        self._revision = 0
        self._held = self._done = self._won = False
        self._validator.reset()
        catalog = self._replace_catalog()
        return HarnessActionResult(
            True,
            f"Target {task.context['target_item']} is available.",
            False,
            False,
            self._revision,
            catalog,
            {"reset": True},
        )

    def action_catalog(self) -> list[HarnessActionSpec]:
        return self._catalog.items()

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        if self._task is None:
            raise RuntimeError("LocatingHarness must be reset before action execution")
        if self._done or self._won:
            from atomic_skillgraph.core.errors import AtomicSkillGraphError, FailureLayer

            raise AtomicSkillGraphError(
                "harness_terminal_latched",
                "benchmark terminal is latched; no further env.step is allowed",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        spec = self._catalog.get(action_id, revision)
        target = str(self._task.context["target_item"])
        room = _room_for(target)
        accepted = False
        observation = "Nothing happens."
        if spec.action_type == "SEARCH" and spec.arguments == {"target": target}:
            accepted = True
            observation = f"You search and find {target} at {room}."
        elif (
            spec.action_type == "TAKE"
            and spec.arguments == {"object": target, "location": room}
            and not self._held
        ):
            accepted = True
            self._held = True
            observation = f"You take {target}."
        self._revision += 1
        self._won = accepted and spec.action_type == "TAKE"
        self._done = self._won
        catalog = self._replace_catalog()
        self._validator.record(
            spec,
            accepted=accepted,
            revision=self._revision,
            done=self._done,
            won=self._won,
            observation=observation,
            catalog=catalog,
        )
        return HarnessActionResult(
            accepted,
            observation,
            self._done,
            self._won,
            self._revision,
            catalog,
            {"action_type": spec.action_type},
        )

    def task_contract(self, task: HarnessTask) -> TaskContract:
        return TaskContract(
            target_effects=[
                SemanticPredicate(
                    "agent.holds",
                    {
                        "object": BindingExpression(
                            BindingExprKind.SKILL_INPUT,
                            source_role="object",
                        )
                    },
                )
            ],
            source=ContractSource.ADAPTER_DERIVED,
            confidence=1.0,
            validator_id="fake_v3_goal",
        )

    def contract_matcher(self):
        from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher

        bindings = {}
        if self._task is not None:
            bindings["object"] = self._task.context.get("target_item")
        return ExactContractMatcher(bindings)

    def validator_channel(self) -> LocatingValidatorChannel:
        return self._validator

    def semantic_predicate_schema(self) -> list[PredicateSpec]:
        return [
            PredicateSpec(
                "agent.holds", "world", ("object",),
                {"object": "entity"}, "fake_v3_action_facts",
            ),
            PredicateSpec(
                "entity.discovered_at", "evidence", ("entity", "location"),
                {"entity": "entity", "location": "location"},
                "fake_v3_action_facts",
            ),
        ]

    def primitive_action_schema(self) -> list[dict[str, Any]]:
        return [
            {"action_type": "SEARCH", "argument_roles": ["target"]},
            {"action_type": "TAKE", "argument_roles": ["object", "location"]},
        ]

    def compile_primitive(
        self,
        primitive: PrimitiveToolStep,
        bindings: dict[str, Any],
    ) -> HarnessActionSpec:
        expected: dict[str, Any] = {}
        for role, raw_expression in primitive.argument_mapping.items():
            expression = raw_expression
            if isinstance(expression, Mapping) and "kind" in expression:
                expression = BindingExpression.from_dict(dict(expression))
            if isinstance(expression, BindingExpression):
                value = (
                    expression.constant
                    if expression.kind is BindingExprKind.CONSTANT
                    else bindings.get(expression.source_role)
                )
            else:
                value = expression
            expected[role] = value
        for spec in self.action_catalog():
            if spec.action_type == primitive.action_type and all(
                spec.arguments.get(role) == value
                for role, value in expected.items()
            ):
                return spec
        raise KeyError(
            f"no locating {primitive.action_type} affordance matches {expected!r}"
        )

    def execute_primitive(
        self,
        primitive: PrimitiveToolStep,
        bindings: dict[str, Any],
    ) -> HarnessActionResult:
        spec = self.compile_primitive(primitive, bindings)
        return self.execute_action(spec.action_id, spec.revision)

    def supports_constraint(self, kind: str, verifier_id: str = "") -> bool:
        return kind in {
            "argument_exists",
            "argument_concrete",
            "harness_affordance",
            "current_context",
        } or bool(verifier_id)

    def replay_tool(
        self,
        task: HarnessTask,
        tool: Any,
        case: Mapping[str, Any],
    ) -> bool:
        """Replay a compiled Tool at its recorded boundary state."""

        if str(getattr(tool, "artifact_kind", "")) == "tool_ir_v1":
            target = str(dict(case.get("bindings") or {}).get("target", ""))
            replay_task = _locating_task(
                str((case.get("source_task") or {}).get("task_id", "replay")),
                target or "apple_1",
            )
            self.reset(replay_task)
            plan = RuntimeLinearPlan.full_dynamic(
                replay_task.task_id, self.task_contract(replay_task),
                reason="tool_ir_replay",
            )
            trace = TraceRecord.create(
                TaskRecord(
                    replay_task.task_id, replay_task.benchmark,
                    replay_task.goal, replay_task.task_type,
                    str(replay_task.metadata.get("task_signature", "")),
                    dict(replay_task.metadata),
                ),
                {},
                {},
                {"source": "full_dynamic", "failure_stage": "tool_ir_replay"},
            )
            ctx = TaskRuntimeContext.create(
                replay_task, plan, self, TraceBuilder(trace),
                RuntimeBudget(global_action_budget=10, node_action_budget=5),
            )
            bindings = copy.deepcopy(dict(case.get("bindings") or {}))
            result = ToolRunner(ValidationEngine().tool).run(
                tool, bindings, ctx, occurrence_id="tool_ir_replay",
            )
            return bool(
                result.executed_action_count > 0
                and result.atomic_effect_passed
                and result.completed
            )
        self.reset(task)
        bindings = copy.deepcopy(dict(case.get("bindings", {})))
        try:
            for raw in list(tool.artifact.get("steps", [])):
                primitive = PrimitiveToolStep(
                    action_type=str(raw["action_type"]),
                    argument_mapping=copy.deepcopy(
                        dict(raw.get("argument_mapping", {}))
                    ),
                )
                result = self.execute_primitive(primitive, bindings)
                if not result.accepted:
                    return False
        except Exception:
            return False
        return bool(list(tool.artifact.get("steps", [])))

    def _replace_catalog(self) -> list[HarnessActionSpec]:
        if self._task is None:
            return self._catalog.replace([], self._revision)
        target = str(self._task.context["target_item"])
        room = _room_for(target)
        actions: list[dict[str, Any]] = []
        if not self._held:
            actions.extend([
                {
                    "action_type": "SEARCH",
                    "arguments": {"target": target},
                    "display_text": f"search {target}",
                },
                {
                    "action_type": "TAKE",
                    "arguments": {"object": target, "location": room},
                    "display_text": f"take {target}",
                },
                {
                    "action_type": "TAKE",
                    "arguments": {"object": target, "location": "room_wrong"},
                    "display_text": f"take {target} from wrong room",
                },
                {
                    "action_type": "TAKE",
                    "arguments": {"object": "distractor_1", "location": "room_distractor"},
                    "display_text": "take distractor_1",
                },
            ])
        return self._catalog.replace(actions, self._revision)


def _param_spec(value: Any) -> dict[str, Any]:
    return {
        "name": str(value.name),
        "semantic_type": str(value.semantic_type),
        "required": bool(value.required),
        "runtime_resolvable": bool(value.runtime_resolvable),
        "required_resolution": str(value.required_resolution),
        "description": str(value.description or ""),
    }


_LOCATE_PROGRAM = [
    {
        "node_id": "search",
        "op": "ACTION",
        "action_type": "SEARCH",
        "argument_mapping": {
            "target": {"kind": "skill_input", "source_role": "target"}
        },
        "expected_effects": [{
            "predicate": "entity.discovered_at",
            "args": {"entity": "$entity", "location": "$location"},
            "effect_domain": "evidence",
        }],
    },
    {
        "node_id": "ret",
        "op": "RETURN",
        "output_sources": {
            "entity": {
                "source": "semantic_evidence",
                "where": {"predicate": "entity.discovered_at"},
                "project": {"kind": "argument", "role": "entity"},
            },
            "location": {
                "source": "semantic_evidence",
                "where": {"predicate": "entity.discovered_at"},
                "project": {"kind": "argument", "role": "location"},
            },
        },
    },
]


def _locate_proposal(atomic: AbstractAtomicSkill) -> dict[str, Any]:
    return {
        "proposal_version": "1",
        "decision": "create",
        "summary": "locate target entity",
        "atomic_ref": str(atomic.ref),
        "inputs": [_param_spec(item) for item in atomic.inputs],
        "outputs": [_param_spec(item) for item in atomic.outputs],
        "program": copy.deepcopy(_LOCATE_PROGRAM),
        "max_actions": 1,
        "final_effects": [
            {
                "predicate": "entity.discovered_at",
                "args": {"entity": "$entity", "location": "$location"},
                "effect_domain": "evidence",
            }
        ],
        "evidence_outputs": [],
        "path_expectations": [],
        "rationale": "Bounded deterministic target search.",
    }


def _locate_draft_call() -> dict[str, Any]:
    return {
        "draft_id": "locate_1",
        "intent": "locate target",
        "inputs": [{"name": "target", "semantic_type": "entity"}],
        "outputs": [
            {"name": "entity", "semantic_type": "entity"},
            {"name": "location", "semantic_type": "location"},
        ],
        "preconditions": [],
        "effects": [
            {
                "predicate": "entity.discovered_at",
                "args": {"entity": "$entity", "location": "$location"},
                "effect_domain": "evidence",
            }
        ],
        "rationale": "Automate the mechanical target search.",
        "source_occurrence_id": "take_occurrence",
        "input_binding_specs": {
            "target": {
                "kind": "current_occurrence_anchor",
                "source_role": "object",
            }
        },
    }


def _locate_e1(
    target: str,
    location: str,
    event_id: str,
    witness_ref: str = "effect:w_locate",
) -> dict[str, Any]:
    return {
        "phase_id": "locate_p1",
        "intent": "locate target entity",
        "event_start": 0,
        "event_end": 1,
        "support_event_ids": [event_id],
        "input_roles": {"target": target},
        "output_roles": {"entity": target, "location": location},
        "preconditions": [],
        "precondition_witness_refs": [],
        "effects": [
            {
                "predicate": "entity.discovered_at",
                "args": {"entity": target, "location": location},
                "effect_domain": "evidence",
            }
        ],
        "rationale": "The accepted SEARCH transition discovered the target.",
        "input_provenance_refs": {"target": "runtime_input:locate_1:target"},
        "output_derivations": {
            "entity": {
                "kind": "effect_witness",
                "predicate": "entity.discovered_at",
                "argument_role": "entity",
            },
            "location": {
                "kind": "effect_witness",
                "predicate": "entity.discovered_at",
                "argument_role": "location",
            },
        },
        "effect_witness_refs": [witness_ref],
    }


def _atomic_view(occurrence: Any) -> AbstractAtomicSkill:
    derivations = dict(getattr(occurrence, "output_derivations", None) or {})
    output_identity = [
        {
            "output_role": str(role),
            "input_role": str(derivation.get("input_role", "")),
        }
        for role, derivation in derivations.items()
        if derivation.get("kind") == "input_identity"
    ]
    return AbstractAtomicSkill(
        occurrence.proposed_ref,
        occurrence.intent,
        occurrence.input_specs,
        occurrence.output_specs,
        occurrence.preconditions,
        occurrence.effects,
        {
            "validator_id": "harness_atomic_effect",
            "identity_strict": True,
            "output_identity": output_identity,
            "output_derivations": {
                str(role): dict(derivation)
                for role, derivation in derivations.items()
            },
        },
        [],
        {"steps": [], "runtime_automation": True},
        {"source_trace_ids": [occurrence.source_trace_id]},
        SkillStatus.DRAFT,
    )


_TASK_LOCAL_ATOMIC_REF = "skill://atomic_locate_target_task_local@1.0.0"


def _task_local_locate_proposal() -> dict[str, Any]:
    return {
        "proposal_version": "1",
        "decision": "create",
        "summary": "locate target entity",
        "atomic_ref": _TASK_LOCAL_ATOMIC_REF,
        "inputs": [
            {"name": "target", "semantic_type": "entity", "required": True,
             "runtime_resolvable": False, "required_resolution": "semantic",
             "description": ""},
        ],
        "outputs": [
            {"name": "entity", "semantic_type": "entity", "required": True,
             "runtime_resolvable": False, "required_resolution": "semantic",
             "description": ""},
            {"name": "location", "semantic_type": "location", "required": True,
             "runtime_resolvable": False, "required_resolution": "semantic",
             "description": ""},
        ],
        "program": copy.deepcopy(_LOCATE_PROGRAM),
        "max_actions": 1,
        "final_effects": [
            {
                "predicate": "entity.discovered_at",
                "args": {"entity": "$entity", "location": "$location"},
                "effect_domain": "evidence",
            }
        ],
        "evidence_outputs": [],
        "path_expectations": [],
        "rationale": "Bounded deterministic target search.",
    }


def _take_e1(target: str, location: str, event_id: str) -> dict[str, Any]:
    return {
        "phase_id": "take_p1",
        "intent": "take target object",
        "event_start": 0,
        "event_end": 1,
        "support_event_ids": [event_id],
        "input_roles": {"object": target, "location": location},
        "input_provenance_refs": {
            "object": f"action_arg:{event_id}:object",
            "location": f"action_arg:{event_id}:location",
        },
        "output_roles": {"held_object": target},
        "output_derivations": {
            "held_object": {
                "kind": "input_identity", "input_role": "object",
            },
        },
        "preconditions": [],
        "precondition_witness_refs": [],
        "effects": [{
            "predicate": "agent.holds",
            "args": {"object": target},
            "effect_domain": "world",
        }],
        "effect_witness_refs": [f"action:{event_id}:revision:1"],
        "rationale": "The accepted TAKE transition establishes possession.",
    }


def _register_take_graph(
    trace,
    task,
    factory: FakeAgentFactory,
    skills: SkillRegistry,
    tools: ToolRegistry,
    validation: ValidationEngine,
    harness: LocatingHarness,
    ledger: EvidenceLedger,
    projection: LifecycleProjection,
) -> dict[str, Any]:
    normalized = TraceNormalizer().build(trace)
    session = factory.new_session(
        "extractor",
        [FakeReply.structured({"occurrences": [
            _take_e1(
                str(task.context["target_item"]),
                _room_for(task.context["target_item"]),
                str(trace.environment_actions[0].action_id),
            ),
        ]})],
    )
    extractor = ExtractorSession(session)
    proposals = extractor.propose_atomics(normalized)
    canonical = Atomicizer().validate_and_canonicalize(proposals, normalized)

    aligner = Aligner(skills, tools)
    compiled = list(ToolCompiler().compile(canonical))
    assert len(compiled) == 1
    item = compiled[0]
    assert item.tool is not None and item.implementation is not None
    bundle = aligner.stage_atomic(item.atomic, item.tool, item.implementation)
    staged_occurrence = aligner.atomic_canonicalizer.rewrite_canonical_occurrence(
        item.occurrence, bundle, atomic_ref=bundle.atomic.ref,
    )
    item = type(item)(staged_occurrence, bundle.atomic, bundle.tool, bundle.implementation)
    canonical = [staged_occurrence]

    def e2_reply(request):
        return {
            "selected_existing_edge_ids": [],
            "selected_new_edge_candidate_ids": [],
            "summary": "take target object",
            "guideline": {"canonical": True},
            "insight": {"source": "gate29_bootstrap"},
        }

    session.enqueue(FakeReply.structured(e2_reply))
    e2 = extractor.propose_composite(canonical, [])
    assert session.remaining_replies == 0

    admission = Admission(validation.tool)
    atomic_ref = aligner.align_atomic(item.atomic)
    admitted_tool = admission.admit_tool(
        item.tool,
        replay=lambda tool, case: harness.replay_tool(task, tool, case),
        atomic=item.atomic,
        harness=harness,
    )
    assert admitted_tool.status is ToolStatus.CANDIDATE
    tool_ref = aligner.align_tool(admitted_tool)
    admitted_implementation = admission.admit_implementation(
        item.implementation,
        admitted_tool,
        atomic=item.atomic,
        harness=harness,
    )
    assert admitted_implementation.status is SkillStatus.CANDIDATE
    implementation_ref = aligner.align_implementation(
        admitted_implementation, atomic_ref, tool_ref,
    )
    occurrence_to_atomic = {staged_occurrence.occurrence_id: atomic_ref}
    candidate_composite = CompositeBuilder().validate_and_build(
        e2,
        canonical,
        harness.task_contract(task),
        contract_matcher=harness.contract_matcher(),
        task_bindings=task.context["semantic_bindings"],
    )
    composite_ref = aligner.align_composite(candidate_composite, occurrence_to_atomic)
    events = CreditAssigner().assign_evolution(
        trace,
        [atomic_ref],
        [implementation_ref],
        [tool_ref],
        composite_ref,
    )
    assert ledger.append_transaction(events).inserted_count == len(events)
    projection.consume_new_events()
    return {
        "atomic_ref": atomic_ref,
        "tool_ref": tool_ref,
        "implementation_ref": implementation_ref,
        "composite_ref": composite_ref,
    }


def _learn_locate(
    trace,
    task,
    factory: FakeAgentFactory,
    skills: SkillRegistry,
    tools: ToolRegistry,
    validation: ValidationEngine,
    harness: LocatingHarness,
    ledger: EvidenceLedger,
    projection: LifecycleProjection,
    graph: GraphStore,
) -> dict[str, Any]:
    normalized = TraceNormalizer().build(trace)
    trials = dict(trace.metadata.get("runtime_tool_trials") or {})
    trial = dict(trials["locate_1"])
    boundary_inputs = [
        {
            **dict(authority),
            "draft_id": str(trial.get("draft_id", "")),
            "trial_event_start": int(trial["trial_event_start"]),
            "trial_event_end": int(trial["trial_event_end"]),
            "source_kind": str(authority.get("kind", "")),
            "role": role,
        }
        for role, authority in dict(trial.get("input_authorities") or {}).items()
    ]
    from atomic_skillgraph.system import AtomicSkillGraphSystem

    effect_facts = (
        AtomicSkillGraphSystem._runtime_trial_effect_authorities(
            trial, list(normalized.get("actions") or []),
        )
    )
    assert effect_facts
    normalized["boundary_authorities"] = {
        "inputs": boundary_inputs,
        "effects": [{
            "witness_ref": str(fact["witness_ref"]),
            "predicate": str(fact["predicate"]),
            "args": dict(fact["args"]),
            "effect_domain": str(fact["effect_domain"]),
        } for fact in effect_facts],
    }
    search = next(
        action for action in normalized["actions"]
        if action["action_type"] == "SEARCH"
    )
    normalized["after_state_facts"] = effect_facts
    session = factory.new_session(
        "extractor",
        [FakeReply.structured({"occurrences": [
            _locate_e1(
                str(trial["r1_outputs"]["entity"]),
                str(trial["r1_outputs"]["location"]),
                str(search["action_id"]),
                str(effect_facts[0]["witness_ref"]),
            ),
        ]})],
    )
    extractor = ExtractorSession(session)
    proposals = extractor.propose_atomics(normalized)
    canonical = Atomicizer().validate_and_canonicalize(proposals, normalized)
    occurrence = canonical[0]
    atomic_view = _atomic_view(occurrence)

    provenance = ToolProvenance(
        source="success_evolution",
        atomic_ref=str(atomic_view.ref),
        source_trace_id=str(trace.trace_id),
        occurrence_id=occurrence.occurrence_id,
        task_id=task.task_id,
    )
    builder_session = factory.new_session(
        "tool_builder",
        [FakeReply.tool("create_tool", _locate_proposal(atomic_view))],
    )
    proposal = ToolBuilderSession(builder_session).build(
        atomic=atomic_view,
        provenance=provenance,
        evidence_support=[],
        semantic_delta={"before_facts": [], "after_facts": []},
        harness_interface={
            "profile": harness.profile_name,
            "predicate_vocabulary": to_primitive(
                harness.semantic_predicate_schema()
            ),
            "primitive_actions": [
                dict(item) for item in harness.primitive_action_schema()
            ],
        },
        bucket="tool_builder_evolution",
    )
    static = ToolStaticValidator().validate_proposal(proposal, atomic_view, harness)
    assert static.passed, static.failure_codes
    item = ToolCompiler().compile_proposal(
        occurrence, atomic_view, proposal, provenance,
    )
    assert item.tool is not None and item.implementation is not None

    aligner = Aligner(skills, tools)
    atomic_ref = aligner.align_atomic(item.atomic)
    admission = Admission(validation.tool)
    admitted_tool = admission.admit_tool(
        item.tool,
        replay=lambda tool, case: harness.replay_tool(task, tool, case),
        atomic=item.atomic,
        harness=harness,
    )
    assert admitted_tool.status is ToolStatus.CANDIDATE
    tool_alignment = aligner.align_tool_with_replays(
        admitted_tool,
        admission=admission,
        replay=lambda tool, case: harness.replay_tool(task, tool, case),
    )
    assert tool_alignment.admitted
    tool_ref = tool_alignment.ref
    admitted_implementation = admission.admit_implementation(
        item.implementation,
        admitted_tool,
        atomic=item.atomic,
        harness=harness,
    )
    assert admitted_implementation.status is SkillStatus.CANDIDATE
    implementation_ref = aligner.align_implementation(
        admitted_implementation, atomic_ref, tool_ref,
    )
    from atomic_skillgraph.core.edges import GlobalGraphEdge, GlobalRelationType
    from atomic_skillgraph.core.refs import content_hash

    def edge(source_ref: str, target_ref: str, relation: GlobalRelationType) -> None:
        payload = {
            "source_ref": source_ref, "target_ref": target_ref,
            "relation": relation.value, "trace_id": str(trace.trace_id),
        }
        graph.add(GlobalGraphEdge(
            content_hash(payload)[:24], source_ref, target_ref, relation,
            {"support_trace_ids": [str(trace.trace_id)]},
        ))

    edge(str(implementation_ref), str(atomic_ref), GlobalRelationType.IMPLEMENTS)
    edge(str(implementation_ref), str(tool_ref), GlobalRelationType.CONTAINS)
    events = CreditAssigner().assign_evolution(
        trace, [atomic_ref], [implementation_ref], [tool_ref], None,
    )
    assert ledger.append_transaction(events).inserted_count == len(events)
    projection.consume_new_events()
    return {
        "atomic_ref": atomic_ref,
        "tool_ref": tool_ref,
        "implementation_ref": implementation_ref,
        "occurrence": occurrence,
    }


def test_gate29_cross_task_runtime_tool_reuse(tmp_path: Path) -> None:
    data_dir = tmp_path / "bank"
    database = StateDatabase(data_dir / "state.sqlite3")
    artifacts = ArtifactStore(data_dir, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    graph = GraphStore(database, skills)
    ledger = EvidenceLedger(database)
    projection = LifecycleProjection(database, ledger)
    validation = ValidationEngine()
    harness = LocatingHarness()
    factory = FakeAgentFactory()
    planner = PlannerPipeline(skills, graph, factory)
    invocation_compiler = InvocationCompiler(
        skills, tools, harness, mode=RuntimeMode.ONLINE,
    )
    runtime = RuntimeOrchestrator(
        planner,
        harness,
        invocation_compiler,
        validation,
        factory,
        runtime_config={
            "global_action_budget": 20,
            "node_action_budget": 10,
            "learned_toolcall_repair_limit": 2,
        },
    )
    runtime.attach_runtime_automation(
        tool_builder_factory=lambda kind, occurrence_id: factory.new_session(
            "tool_builder",
            [FakeReply.tool("create_tool", _task_local_locate_proposal())],
        ),
        tool_compiler=ToolCompiler(),
    )

    # Bootstrap: a distinct take graph so the planner can produce a planned
    # node for Task A and Task B without any planner LLM turn.
    factory.enqueue(
        "runtime_dynamic",
        [FakeReply.tool("environment_action", {"action_id": "r000_a002"})],
    )
    bootstrap_task = _locating_task("task_bootstrap_take", "orange_1")
    bootstrap_trace = runtime.run_task(bootstrap_task)
    assert bootstrap_trace.runtime_plan["source"] == "full_dynamic"
    assert bootstrap_trace.benchmark_success is True
    take_refs = _register_take_graph(
        bootstrap_trace, bootstrap_task, factory, skills, tools, validation,
        harness, ledger, projection,
    )

    # Gate29A: Task A proposes the automation draft, the task-local trial
    # passes R1 through validate_execution_result, and the Success Extractor
    # chain persists Atomic/Tool/Implementation candidates.
    task_a = _locating_task("task_a_locate", "apple_1")
    factory.enqueue(
        "runtime_preparation",
        [
            FakeReply.tool("propose_runtime_automation_atomic", _locate_draft_call()),
            FakeReply.tool("environment_action", {
                "action_id": "r001_a002", "intent": "attempt_current_atomic",
            }),
        ],
    )
    trace_a = runtime.run_task(task_a)
    assert trace_a.runtime_plan["source"] == "stored_composite"
    assert trace_a.benchmark_success is True
    drafts = dict(trace_a.metadata.get("runtime_automation_drafts") or {})
    outcome = dict(drafts["locate_1"])
    assert outcome.get("r0_passed") is True
    assert outcome.get("r1_passed") is True
    assert outcome.get("trial", {}).get("r1_outputs") == {
        "entity": "apple_1",
        "location": _room_for("apple_1"),
    }

    locate_refs = _learn_locate(
        trace_a, task_a, factory, skills, tools, validation, harness,
        ledger, projection, graph,
    )
    learned_atomic = skills.get_atomic(locate_refs["atomic_ref"])
    assert [item.name for item in learned_atomic.inputs] == ["target"]
    assert [item.name for item in learned_atomic.outputs] == ["entity", "location"]
    assert {
        expression.source_role
        for expression in learned_atomic.effects[0].args.values()
    } == {"entity", "location"}
    assert learned_atomic.status is SkillStatus.CANDIDATE
    assert tools.get(locate_refs["tool_ref"]).status is ToolStatus.CANDIDATE
    assert (
        skills.get_implementation(locate_refs["implementation_ref"]).status
        is SkillStatus.CANDIDATE
    )

    # Gate29B: a different task and a different concrete entity reuse the
    # persisted locate Tool through SupportRetriever with zero LLM ToolBuilder
    # calls and at least one LLM-bypassed environment action.
    task_b = _locating_task("task_b_reuse", "mug_1", expose_object=False)
    assert task_b.task_id != task_a.task_id
    assert task_b.context["target_item"] != task_a.context["target_item"]
    tool_builder_sessions_before = len(factory.sessions_of("tool_builder"))
    factory.enqueue(
        "runtime_preparation",
        [
            FakeReply.tool("invoke_support_atomic", {
                "support_atomic_ref": str(locate_refs["atomic_ref"]),
                "arguments": {"target": "mug_1"},
                "output_mapping": {"location": "location"},
            }),
            FakeReply.tool("environment_action", {
                "action_id": "r001_a002", "intent": "attempt_current_atomic",
            }),
        ],
    )
    trace_b = runtime.run_task(task_b)
    assert trace_b.benchmark_success is True
    assert len(factory.sessions_of("tool_builder")) == tool_builder_sessions_before
    augmentations = list(
        trace_b.metadata.get("runtime_graph_augmentation") or []
    )
    assert augmentations
    assert augmentations[0]["support_atomic_ref"] == str(locate_refs["atomic_ref"])
    tool_executions = list(trace_b.tool_executions)
    assert len(tool_executions) >= 1
    assert any(
        str(item.tool_ref) == str(locate_refs["tool_ref"])
        for item in tool_executions
    )
    search_actions = [
        action for action in trace_b.environment_actions
        if action.action_type == "SEARCH"
    ]
    assert len(search_actions) >= 1
    published = [
        change for change in trace_b.binding_changes
        if change.get("reason") == "validated_output_published"
        and change.get("role") == "location"
        and dict(change.get("current") or {}).get("value") == _room_for("mug_1")
    ]
    assert published

    factory.assert_exhausted()
    artifacts.verify_all()
    database.close()
