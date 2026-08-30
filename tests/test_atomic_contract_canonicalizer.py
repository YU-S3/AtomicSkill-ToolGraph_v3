from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    GroundingConstraint,
    GroundingConstraintKind,
    ToolBinding,
)
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CompositeOccurrence,
    CompositeSkill,
    ImplementationAtom,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
    ToolAsset,
)
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
from atomic_skillgraph.evolution.contract_canonicalizer import (
    AtomicContractCanonicalizer,
    atomic_contract_signature,
)
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.graph_store import GraphStore
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.planner.compiler import PlanCompiler
from atomic_skillgraph.planner.validator import PlannerValidator
from atomic_skillgraph.runtime.invocation_compiler import InvocationCompiler


class _Harness:
    def supports_constraint(self, _kind: str, _verifier_id: str) -> bool:
        return True


def _runtime(tmp_path: Path):
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    return database, skills, tools, GraphStore(database, skills)


def _take_candidate(input_role: str, output_role: str, suffix: str):
    input_expression = BindingExpression(
        BindingExprKind.SKILL_INPUT,
        source_role=input_role,
    )
    effect = SemanticPredicate(
        "agent.holds",
        {"object": input_expression},
    )
    atomic = AbstractAtomicSkill(
        SkillRef(f"draft_take_{suffix}", "0.1.0"),
        "take an object",
        [ParameterSpec(input_role, "entity")],
        [ParameterSpec(output_role, "entity")],
        [],
        [effect],
        {
            "input_role": input_role,
            "output_role": f"${output_role}",
        },
        [],
        {},
        {"trace_alias": suffix},
        SkillStatus.CANDIDATE,
    )
    tool_ref = ToolRef(f"take_tool_{suffix}", "1.0.0")
    tool = ToolAsset(
        tool_ref,
        "take primitive",
        {
            "type": "object",
            "properties": {input_role: {"type": "string"}},
            "required": [input_role],
            "additionalProperties": False,
        },
        {
            "output_schema": {
                "type": "object",
                "properties": {output_role: {"type": "string"}},
                "required": [output_role],
            }
        },
        "primitive_ir",
        {
            "steps": [{
                "action_type": "TAKE",
                "argument_mapping": {"object": input_expression},
            }],
            "output_mapping": {output_role: input_expression},
        },
        [{"bindings": {input_role: "apple_1"}, "effects": [effect]}],
        {"reviewed": True, "allowed_action_types": ["TAKE"]},
        {},
        {},
        ToolStatus.CANDIDATE,
    )
    implementation = ImplementationAtom(
        SkillRef(f"take_impl_{suffix}", "1.0.0"),
        atomic.ref,
        [ToolBinding(
            tool_ref,
            "take",
            {input_role: input_expression},
            0,
        )],
        [GroundingConstraint(
            f"take_affordance_{suffix}",
            GroundingConstraintKind.HARNESS_AFFORDANCE,
            "TAKE",
            {"object": input_expression},
            "relation_verified",
            "alfworld_action_catalog",
        )],
        {
            "mode": "serial",
            "output_mapping": {
                output_role: BindingExpression(
                    BindingExprKind.TOOL_OUTPUT,
                    source_role=output_role,
                    source_step="take",
                )
            },
        },
        {},
        {"reliability": 1.0},
        SkillStatus.CANDIDATE,
    )
    return atomic, tool, implementation


def _place_atomic(
    item_role: str,
    destination_role: str,
    output_role: str,
    suffix: str,
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        SkillRef(f"draft_place_{suffix}", "0.1.0"),
        "place an object",
        [
            ParameterSpec(item_role, "entity"),
            ParameterSpec(destination_role, "entity"),
        ],
        [ParameterSpec(output_role, "entity")],
        [],
        [SemanticPredicate(
            "object.at_location",
            {
                "object": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role=item_role,
                ),
                "receptacle": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role=destination_role,
                ),
            },
        )],
        {},
        [],
        {},
        {"trace_alias": suffix},
        SkillStatus.CANDIDATE,
    )


def test_atomic_role_normalization_rewrites_implementation_and_composite(
    tmp_path: Path,
) -> None:
    canonicalizer = AtomicContractCanonicalizer()
    trace_one = _take_candidate("object", "held_object", "trace_one")
    trace_two = _take_candidate("item", "acquired_item", "trace_two")
    first = canonicalizer.canonicalize(*trace_one)
    second = canonicalizer.canonicalize(*trace_two)

    assert atomic_contract_signature(first.atomic) == atomic_contract_signature(second.atomic)
    assert first.atomic.ref == second.atomic.ref
    assert [item.name for item in first.atomic.inputs] == ["input_000"]
    assert [item.name for item in first.atomic.outputs] == ["output_000"]
    assert first.atomic.effects[0].args["object"].source_role == "input_000"
    assert first.atomic.validator_spec == {
        "input_role": "input_000",
        "output_role": "$output_000",
    }

    trace_occurrence = CanonicalAtomicOccurrence(
        "occ_trace_one",
        "take",
        trace_one[0].summary,
        0,
        1,
        {"object": "apple_1"},
        {"held_object": "apple_1"},
        list(trace_one[0].inputs),
        list(trace_one[0].outputs),
        list(trace_one[0].preconditions),
        list(trace_one[0].effects),
        [],
        [],
        {},
        "trace_one",
        trace_one[0].ref,
    )
    staged_occurrence = canonicalizer.rewrite_canonical_occurrence(
        trace_occurrence,
        first,
    )
    assert staged_occurrence.proposed_ref == first.atomic.ref
    assert staged_occurrence.input_bindings == {"input_000": "apple_1"}
    assert staged_occurrence.output_bindings == {"output_000": "apple_1"}

    for bundle in (first, second):
        assert bundle.tool is not None
        assert bundle.implementation is not None
        assert set(bundle.tool.signature["properties"]) == {"input_000"}
        assert bundle.tool.signature["required"] == ["input_000"]
        assert set(bundle.tool.interface["output_schema"]["properties"]) == {
            "output_000"
        }
        step_expression = bundle.tool.artifact["steps"][0]["argument_mapping"][
            "object"
        ]
        assert step_expression.source_role == "input_000"
        assert set(bundle.tool.artifact["output_mapping"]) == {"output_000"}
        tool_binding = bundle.implementation.tool_bindings[0]
        assert set(tool_binding.parameter_mapping) == {"input_000"}
        assert tool_binding.parameter_mapping["input_000"].source_role == "input_000"
        assert (
            bundle.implementation.grounding_constraints[0]
            .argument_mapping["object"]
            .source_role
            == "input_000"
        )
        output_expression = bundle.implementation.execution_policy[
            "output_mapping"
        ]["output_000"]
        assert output_expression.source_role == "output_000"
        assert bundle.implementation.abstract_ref == bundle.atomic.ref

    database, skills, tools, graph = _runtime(tmp_path)
    try:
        compiler = InvocationCompiler(skills, tools, _Harness())
        assert compiler.compile(
            first.atomic,
            first.implementation,
            [first.tool],
            {},
        ).atomic_ref == first.atomic.ref
        assert compiler.compile(
            second.atomic,
            second.implementation,
            [second.tool],
            {},
        ).atomic_ref == second.atomic.ref

        place = canonicalizer.canonicalize(
            _place_atomic("item", "destination", "placed_item", "trace_one")
        )
        skills.register_atomic(replace(first.atomic, status=SkillStatus.CANDIDATE))
        skills.register_atomic(replace(place.atomic, status=SkillStatus.CANDIDATE))

        source_role = trace_one[0].outputs[0].name
        target_item_role = "item"
        raw_composite = CompositeSkill(
            SkillRef("take_then_place", "1.0.0"),
            "take then place",
            [
                CompositeOccurrence(
                    "take", "occ_take", trace_one[0].ref,
                    {
                        "object": BindingExpression(
                            BindingExprKind.CONSTANT,
                            constant="apple_1",
                        )
                    },
                ),
                CompositeOccurrence(
                    "place", "occ_place", _place_atomic(
                        "item", "destination", "placed_item", "unused"
                    ).ref,
                    {
                        target_item_role: BindingExpression(
                            BindingExprKind.DATA_FLOW,
                            source_role=source_role,
                            source_step="take",
                        ),
                        "destination": BindingExpression(
                            BindingExprKind.CONSTANT,
                            constant="table_1",
                        ),
                    },
                ),
            ],
            ["take", "place"],
            [GraphEdge(
                "take_to_place",
                GraphEdgeType.DATA_FLOW,
                "take",
                "place",
                source_role,
                target_item_role,
                "extractor_validated",
                evidence_refs=("trace_one",),
            )],
            [],
            TaskContract(),
            {},
            {},
            {},
            {},
            SkillStatus.CANDIDATE,
        )
        rewritten = canonicalizer.rewrite_composite(
            raw_composite,
            {"take": first, "place": place},
        )
        assert rewritten.occurrences[0].node_ref == first.atomic.ref
        assert rewritten.occurrences[1].node_ref == place.atomic.ref
        assert set(rewritten.occurrences[0].binding_specs) == {"input_000"}
        target_item = place.input_role_map[target_item_role]
        assert rewritten.data_edges[0].source_role == "output_000"
        assert rewritten.data_edges[0].target_role == target_item
        assert (
            rewritten.occurrences[1]
            .binding_specs[target_item]
            .source_role
            == "output_000"
        )

        plan = PlanCompiler(skills).from_composite(
            SimpleNamespace(task_id="canonical-composite"),
            TaskContract(),
            rewritten,
            mode=RuntimeMode.ONLINE,
            audit={},
        )
        validation = PlannerValidator(skills, graph).validate(
            plan,
            mode=RuntimeMode.ONLINE,
            harness_profile="alfworld_v3",
        )
        assert validation.passed, (validation.checks, validation.failure_codes)
    finally:
        database.close()


def test_existing_edge_lookup_uses_resolved_persistent_atomic_refs(
    tmp_path: Path,
) -> None:
    database, skills, tools, graph = _runtime(tmp_path)
    try:
        aligner = Aligner(skills, tools)
        source_one = _take_candidate("object", "held_object", "source_one")
        target_one = _place_atomic(
            "item", "destination", "placed_item", "target_one"
        )
        source_ref = aligner.align_atomic(source_one[0])
        target_ref = aligner.align_atomic(target_one)
        source_persistent = skills.get_atomic(source_ref)
        target_persistent = skills.get_atomic(target_ref)
        assert [item.name for item in source_persistent.inputs] == ["input_000"]
        assert [item.name for item in source_persistent.outputs] == ["output_000"]

        graph_composite = CompositeSkill(
            SkillRef("persistent_workflow", "1.0.0"),
            "persistent workflow",
            [
                CompositeOccurrence("s1", "occ1", source_ref, {}),
                CompositeOccurrence("s2", "occ2", target_ref, {}),
            ],
            ["s1", "s2"],
            [GraphEdge(
                "persistent_edge",
                GraphEdgeType.DATA_FLOW,
                "s1",
                "s2",
                source_persistent.outputs[0].name,
                next(
                    item.name
                    for item in target_persistent.inputs
                    if any(
                        expression.source_role == item.name
                        for expression in target_persistent.effects[0].args.values()
                    )
                ),
                "extractor_validated",
                evidence_refs=("trace_one",),
            )],
            [],
            TaskContract(),
            {},
            {},
            {},
            {},
            SkillStatus.ACTIVE,
        )
        skills.register_composite(graph_composite)

        source_two = _take_candidate("item", "acquired_item", "source_two")
        target_two = _place_atomic(
            "thing", "receptacle", "stored_thing", "target_two"
        )
        proposed_refs = {str(source_two[0].ref), str(target_two.ref)}
        before = database.execute(
            "SELECT COUNT(*) AS count FROM artifact_index"
        ).fetchone()["count"]
        staged_source = aligner.stage_atomic(*source_two)
        staged_target = aligner.stage_atomic(target_two)
        after = database.execute(
            "SELECT COUNT(*) AS count FROM artifact_index"
        ).fetchone()["count"]

        assert after == before
        assert staged_source.atomic.ref == source_ref
        assert staged_source.implementation is not None
        assert staged_source.implementation.abstract_ref == source_ref
        assert staged_target.atomic.ref == target_ref
        assert skills.find_equivalent_atomic(staged_source.atomic) == source_ref
        assert skills.find_equivalent_atomic(staged_target.atomic) == target_ref
        assert graph.existing_edges(
            [str(staged_source.atomic.ref), str(staged_target.atomic.ref)],
            mode=RuntimeMode.ONLINE,
        )[0].edge_id == "persistent_edge"
        assert graph.existing_edges(
            proposed_refs,
            mode=RuntimeMode.ONLINE,
        ) == []
    finally:
        database.close()


def test_staged_aliases_persist_one_role_schema_across_all_layers(
    tmp_path: Path,
) -> None:
    """Exercise the same staged-to-persistent boundary used by System.

    The second trace deliberately uses different local aliases.  No layer may
    reintroduce those aliases after staging, and the resulting persistent
    Composite must remain planner-valid and its Implementation compilable.
    """

    database, skills, tools, graph = _runtime(tmp_path)
    try:
        aligner = Aligner(skills, tools)
        first = aligner.stage_atomic(
            *_take_candidate("object", "held_object", "persistent_one")
        )
        second = aligner.stage_atomic(
            *_take_candidate("item", "acquired_item", "persistent_two")
        )
        assert first.atomic.ref == second.atomic.ref

        before = database.execute(
            "SELECT COUNT(*) AS count FROM artifact_index"
        ).fetchone()["count"]
        # Re-stage the alias after the initial pure resolution check: E2
        # staging itself must remain read-only.
        second = aligner.stage_atomic(
            *_take_candidate("item", "acquired_item", "persistent_two")
        )
        after = database.execute(
            "SELECT COUNT(*) AS count FROM artifact_index"
        ).fetchone()["count"]
        assert after == before

        assert first.tool is not None and first.implementation is not None
        atomic_ref = aligner.align_atomic(first.atomic)
        tool_ref = aligner.align_tool(first.tool)
        implementation_ref = aligner.align_implementation(
            first.implementation,
            atomic_ref,
            tool_ref,
        )

        first_occurrence = CompositeOccurrence(
            "first",
            "occ_first",
            first.atomic.ref,
            {
                "input_000": BindingExpression(
                    BindingExprKind.CONSTANT,
                    constant="apple_1",
                )
            },
        )
        second_occurrence = CompositeOccurrence(
            "second",
            "occ_second",
            second.atomic.ref,
            {
                "input_000": BindingExpression(
                    BindingExprKind.DATA_FLOW,
                    source_role="output_000",
                    source_step="first",
                )
            },
        )
        candidate_composite = CompositeSkill(
            SkillRef("persistent_alias_workflow", "1.0.0"),
            "canonical staged aliases",
            [first_occurrence, second_occurrence],
            ["first", "second"],
            [GraphEdge(
                "canonical_dataflow",
                GraphEdgeType.DATA_FLOW,
                "first",
                "second",
                "output_000",
                "input_000",
                "extractor_validated",
                evidence_refs=("persistent_one", "persistent_two"),
            )],
            [],
            TaskContract(),
            {},
            {},
            {},
            {},
            SkillStatus.CANDIDATE,
        )
        composite_ref = aligner.align_composite(
            candidate_composite,
            {"occ_first": atomic_ref, "occ_second": atomic_ref},
        )

        persistent_atomic = skills.get_atomic(atomic_ref)
        persistent_tool = tools.get(tool_ref)
        persistent_implementation = skills.get_implementation(
            implementation_ref
        )
        persistent_composite = skills.get_composite(composite_ref)

        assert [item.name for item in persistent_atomic.inputs] == ["input_000"]
        assert [item.name for item in persistent_atomic.outputs] == ["output_000"]
        assert set(persistent_tool.signature["properties"]) == {"input_000"}
        assert set(
            persistent_tool.interface["output_schema"]["properties"]
        ) == {"output_000"}
        assert persistent_implementation.abstract_ref == persistent_atomic.ref
        assert set(
            persistent_implementation.tool_bindings[0].parameter_mapping
        ) == {"input_000"}
        assert set(
            persistent_implementation.execution_policy["output_mapping"]
        ) == {"output_000"}
        assert {
            (edge.source_role, edge.target_role)
            for edge in persistent_composite.data_edges
        } == {("output_000", "input_000")}
        assert all(
            set(occurrence.binding_specs) == {"input_000"}
            for occurrence in persistent_composite.occurrences
        )

        invocation = InvocationCompiler(skills, tools, _Harness()).compile(
            persistent_atomic,
            persistent_implementation,
            [persistent_tool],
            {},
        )
        assert invocation.atomic_ref == persistent_atomic.ref

        plan = PlanCompiler(skills).from_composite(
            SimpleNamespace(task_id="persistent-alias-workflow"),
            TaskContract(),
            persistent_composite,
            mode=RuntimeMode.ONLINE,
            audit={},
        )
        validation = PlannerValidator(skills, graph).validate(
            plan,
            mode=RuntimeMode.ONLINE,
            harness_profile="alfworld_v3",
        )
        assert validation.passed, (validation.checks, validation.failure_codes)
    finally:
        database.close()
