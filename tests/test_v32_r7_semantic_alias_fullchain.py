"""R7 synthetic full-chain gate for semantic aliases and frozen reuse."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from atomic_skillgraph.core.bindings import BindingExprKind, BindingExpression
from atomic_skillgraph.core.contracts import (
    ContractSource,
    EffectDomain,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.results import NodeExecutionStatus
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import Atomicizer
from atomic_skillgraph.evolution.composite_builder import CompositeBuilder
from atomic_skillgraph.evolution.extractor_session import ExtractorSession
from atomic_skillgraph.evolution.tool_compiler import CompiledKnowledge, ToolCompiler
from atomic_skillgraph.evolution.trace_normalizer import TraceNormalizer
from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceLedger
from atomic_skillgraph.governance.lifecycle import LifecycleController
from atomic_skillgraph.governance.projections import LifecycleProjection
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
from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler
from atomic_skillgraph.runtime.orchestrator import RuntimeOrchestrator
from atomic_skillgraph.validation.contract_matcher import ExactContractMatcher
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import (
    FakeAgentFactory,
    FakeHarness,
    FakeReply,
    FakeValidatorChannel,
)


def _look_task(
    task_id: str,
    target: str,
    light: str,
) -> HarnessTask:
    return HarnessTask(
        task_id=task_id,
        goal=f"Observe {target} with {light}.",
        benchmark="fake",
        task_type="semantic_alias_look",
        context={
            "target_item": target,
            "light": light,
            "binding_types": {"object": "string", "light": "string"},
            "semantic_bindings": {"object": target, "light": light},
            "initial_observation": f"{target} and {light} are available.",
        },
        metadata={"task_signature": f"fake-look:{task_id}:{target}:{light}"},
    )


class _SemanticAliasValidator(FakeValidatorChannel):
    def snapshot(self) -> dict[str, Any]:
        value = super().snapshot()
        value["done"] = bool(value["won"])
        value["facts"] = [
            {
                **fact,
                "effect_domain": (
                    "evidence"
                    if fact["predicate"] == "object.observed_with"
                    else "world"
                ),
            }
            for fact in value["facts"]
        ]
        return value


class _SemanticAliasLookHarness(FakeHarness):
    """Two-step world whose primitive role collides across two identities."""

    def __init__(self) -> None:
        super().__init__()
        self._validator = _SemanticAliasValidator()

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        if self._task is None:
            raise RuntimeError("harness must be reset before execution")
        spec = self._catalog.get(action_id, revision)
        target = str(self._task.context["target_item"])
        light = str(self._task.context["light"])
        accepted = False
        observation = "Nothing happens."
        if (
            spec.action_type == "TAKE"
            and spec.arguments == {"object": target}
            and not self._held
        ):
            accepted = True
            self._held = True
            observation = f"You take {target}."
        elif (
            spec.action_type == "USE"
            and spec.arguments == {"object": light}
            and self._held
            and not self._observed
        ):
            accepted = True
            self._observed = True
            observation = f"You observe {target} with {light}."
        self._revision += 1
        if accepted and spec.action_type == "TAKE":
            self._validator.record_fact(
                "agent.holds", {"object": target}, self._revision,
            )
        if accepted and spec.action_type == "USE":
            self._validator.record_fact(
                "object.observed_with",
                {"object": target, "light": light},
                self._revision,
            )
        self._won = self._observed
        self._done = self._won
        self._validator.won = self._won
        self._validator.revision = self._revision
        catalog = self._replace_catalog()
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
            target_effects=[SemanticPredicate(
                "object.observed_with",
                {
                    "object": BindingExpression(
                        BindingExprKind.SKILL_INPUT,
                        source_role="object",
                    ),
                    "light": BindingExpression(
                        BindingExprKind.SKILL_INPUT,
                        source_role="light",
                    ),
                },
                effect_domain=EffectDomain.EVIDENCE,
            )],
            source=ContractSource.ADAPTER_DERIVED,
            confidence=1.0,
            validator_id="fake_semantic_alias_look",
        )

    def contract_matcher(self) -> ExactContractMatcher:
        return ExactContractMatcher(
            dict(self._task.context["semantic_bindings"])
            if self._task is not None
            else {}
        )

    def semantic_predicate_schema(self) -> list[PredicateSpec]:
        return [
            PredicateSpec(
                "agent.holds", "world", ("object",),
                {"object": "entity"}, "fake_action_facts",
            ),
            PredicateSpec(
                "object.observed_with", "evidence", ("object", "light"),
                {"object": "entity", "light": "entity"},
                "fake_action_facts",
            ),
        ]

    def primitive_action_schema(self) -> list[dict[str, Any]]:
        return [
            {"action_type": "TAKE", "argument_roles": ["object"]},
            {"action_type": "USE", "argument_roles": ["object"]},
            {"action_type": "INSPECT_LIGHT", "argument_roles": ["light"]},
        ]

    def _replace_catalog(self) -> list[HarnessActionSpec]:
        if self._task is None:
            return self._catalog.replace([], self._revision)
        target = str(self._task.context["target_item"])
        light = str(self._task.context["light"])
        actions: list[dict[str, Any]] = []
        if not self._held:
            actions.append({
                "action_type": "TAKE",
                "arguments": {"object": target},
                "display_text": f"take {target}",
            })
        if not self._observed:
            actions.append({
                "action_type": "USE",
                "arguments": {"object": light},
                "display_text": f"use {light}",
            })
            # A separate current affordance exposes the reusable semantic
            # role at Runtime.  R7 alias authority is still derived solely
            # from the executed USE(object=...) transition above.
            actions.append({
                "action_type": "INSPECT_LIGHT",
                "arguments": {"light": light},
                "display_text": f"inspect {light}",
            })
        return self._catalog.replace(actions, self._revision)


def _proposal(
    target: str,
    light: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    take_id = str(actions[0]["action_id"])
    use_id = str(actions[1]["action_id"])
    observed = next(
        fact
        for fact in actions[1]["authoritative_positive_effects"]
        if fact["predicate"] == "object.observed_with"
    )
    return {
        "phase_id": "observe_with_light",
        "intent": "observe target with illuminating device",
        "event_start": 0,
        "event_end": 2,
        "support_event_ids": [take_id, use_id],
        "input_roles": {"object": target, "light": light},
        "input_provenance_refs": {
            "object": f"action_arg:{take_id}:object",
            "light": f"semantic_alias:{use_id}:object:light",
        },
        "output_roles": {"observed_object": target},
        "output_derivations": {
            "observed_object": {
                "kind": "input_identity",
                "input_role": "object",
            },
        },
        "preconditions": [],
        "precondition_witness_refs": [],
        "effects": [{
            "predicate": "object.observed_with",
            "args": {"object": target, "light": light},
            "effect_domain": "evidence",
        }],
        "effect_witness_refs": [str(observed["witness_ref"])],
        "rationale": "The accepted transitions establish illuminated observation.",
    }


def test_semantic_alias_full_chain_reaches_frozen_stored_composite(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "bank"
    database = StateDatabase(data_dir / "state.sqlite3")
    artifacts = ArtifactStore(data_dir, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    graph = GraphStore(database, skills)
    ledger = EvidenceLedger(database)
    projection = LifecycleProjection(database, ledger)
    harness = _SemanticAliasLookHarness()
    factory = FakeAgentFactory()
    validation = ValidationEngine()
    runtime = RuntimeOrchestrator(
        PlannerPipeline(skills, graph, factory),
        harness,
        InvocationCompiler(skills, tools, harness, mode=RuntimeMode.ONLINE),
        validation,
        factory,
        runtime_config={
            "global_action_budget": 10,
            "node_action_budget": 5,
            "learned_toolcall_repair_limit": 2,
        },
    )

    # Empty-bank success supplies the real accepted actions and Validator
    # snapshots; no test-only boundary input is injected.
    factory.enqueue("runtime_dynamic", [
        FakeReply.tool("environment_action", {"action_id": "r000_a001"}),
        FakeReply.tool("environment_action", {"action_id": "r001_a001"}),
    ])
    source_task = _look_task("look_train_0", "key_1", "device_1")
    source_trace = runtime.run_task(source_task)
    assert source_trace.benchmark_success is True
    source_trace.metadata["method_patch"] = "3.2"
    normalized = TraceNormalizer().build(source_trace)
    aliases = [
        item
        for item in normalized["boundary_authorities"]["inputs"]
        if item.get("kind") == "semantic_alias"
    ]
    assert any(
        item["role"] == "light"
        and item["value"] == "device_1"
        and item["source_authority_ref"].endswith(":object")
        for item in aliases
    )
    normalized["boundary_authorities"]["effects"] = [
        copy.deepcopy(fact)
        for action in normalized["actions"]
        for fact in action["authoritative_positive_effects"]
    ]

    session = factory.new_session("extractor", [
        FakeReply.structured({
            "occurrences": [_proposal(
                "key_1", "device_1", normalized["actions"],
            )],
        }),
    ])
    extractor = ExtractorSession(session)
    canonical = Atomicizer().validate_and_canonicalize(
        extractor.propose_atomics(normalized), normalized,
    )
    assert len(canonical) == 1
    assert canonical[0].effects[0].predicate == "object.observed_with"

    aligner = Aligner(skills, tools)
    compiled = list(ToolCompiler().compile(canonical))
    assert len(compiled) == 1
    item = compiled[0]
    assert item.tool is not None and item.implementation is not None
    bundle = aligner.stage_atomic(item.atomic, item.tool, item.implementation)
    staged = aligner.atomic_canonicalizer.rewrite_canonical_occurrence(
        item.occurrence, bundle, atomic_ref=bundle.atomic.ref,
    )
    canonical = [staged]
    item = CompiledKnowledge(
        staged, bundle.atomic, bundle.tool, bundle.implementation,
    )

    session.enqueue(FakeReply.structured({
        "selected_existing_edge_ids": [],
        "selected_new_edge_candidate_ids": [],
        "summary": "observe target with illuminating device",
        "guideline": {"canonical": True},
        "insight": {"source": "r7_semantic_alias_fullchain"},
    }))
    e2 = extractor.propose_composite(canonical, [])
    admission = Admission(validation.tool)
    atomic_ref = aligner.align_atomic(item.atomic)
    admitted_tool = admission.admit_tool(
        item.tool,
        replay=lambda candidate, case: harness.replay_tool(
            source_task, candidate, case,
        ),
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
    composite = CompositeBuilder().validate_and_build(
        e2,
        canonical,
        harness.task_contract(source_task),
        contract_matcher=harness.contract_matcher(),
        task_bindings=source_task.context["semantic_bindings"],
    )
    assert composite.validator_spec["task_contract_covered"] is True
    composite_ref = aligner.align_composite(
        composite, {staged.occurrence_id: atomic_ref},
    )
    evolution_events = CreditAssigner().assign_evolution(
        source_trace,
        [atomic_ref],
        [implementation_ref],
        [tool_ref],
        composite_ref,
    )
    assert ledger.append_transaction(evolution_events).inserted_count == len(
        evolution_events
    )
    projection.consume_new_events()

    refs = [
        str(atomic_ref), str(implementation_ref), str(tool_ref),
        str(composite_ref),
    ]
    assigner = CreditAssigner()
    # Two independent graph-self-sufficient successes exercise the unmodified
    # formal Atomic/Composite promotion thresholds.
    for index in (1, 2):
        factory.enqueue("runtime_preparation", [
            FakeReply.tool("$learned", {
                "object": f"key_{index + 1}",
                "light": f"device_{index + 1}",
            }),
        ])
        trace = runtime.run_task(_look_task(
            f"look_train_{index}", f"key_{index + 1}",
            f"device_{index + 1}",
        ))
        assert trace.runtime_plan["source"] == "stored_composite"
        assert trace.benchmark_success is True
        assert trace.graph_self_sufficient_success is True
        assert trace.node_records[0].status in {
            NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS,
            NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS,
        }
        events = assigner.assign(trace)
        assert ledger.append_transaction(events).inserted_count == len(events)
        projection.consume_new_events()
        LifecycleController(database, projection).review(refs)

    assert skills.get_atomic(atomic_ref).status is SkillStatus.ACTIVE
    assert skills.get_composite(composite_ref).status is SkillStatus.ACTIVE

    frozen_runtime = RuntimeOrchestrator(
        PlannerPipeline(skills, graph, factory),
        harness,
        InvocationCompiler(skills, tools, harness, mode=RuntimeMode.FROZEN),
        validation,
        factory,
        runtime_config={
            "global_action_budget": 10,
            "node_action_budget": 5,
            "learned_toolcall_repair_limit": 2,
        },
    )
    # Frozen mode cannot execute the still-candidate terminal Implementation;
    # the active stored graph therefore exercises its normal Seeded fallback.
    factory.enqueue("runtime_seeded", [
        FakeReply.tool("environment_action", {
            "action_id": "r000_a001",
            "intent": "attempt_current_atomic",
        }),
        FakeReply.tool("environment_action", {
            "action_id": "r001_a001",
            "intent": "attempt_current_atomic",
        }),
    ])
    held_out = frozen_runtime.run_task(
        _look_task("look_held_out", "key_6", "device_6"),
        mode=RuntimeMode.FROZEN,
    )
    assert held_out.runtime_plan["source"] == "stored_composite"
    assert held_out.runtime_plan["source_composite_ref"] == str(composite_ref)
    assert held_out.graph_self_sufficient_success is True
    assert held_out.benchmark_success is True
    factory.assert_exhausted()
    artifacts.verify_all()
    database.close()
