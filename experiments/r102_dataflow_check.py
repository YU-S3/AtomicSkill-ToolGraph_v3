"""Real API/ALFWorld T1 route fixture: authored Tools, explicit edges, actual entry checks.

The two generic contracts and Active fixture status are declared test inputs,
not learned lifecycle evidence. The model authors both programs and chooses
all concrete values from public state. No task win/benchmark score is claimed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import RuntimeOccurrence, RuntimeLinearPlan
from atomic_skillgraph.core.serialization import atomic_create_json, atomic_write_json, to_primitive
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.composite_executor import VerifiedCompositeExecutor
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config, _reconcile_events, _require_formal_usage
from atomic_skillgraph.tooling.builder_session import ToolBuilderSession
from atomic_skillgraph.tooling.proposal import ToolProvenance
from atomic_skillgraph.tooling.runtime_interface import public_primitive_action_schema
from atomic_skillgraph.traces.schema import NodeTraceRecord
from experiments.protocol import hash_code, hash_config, capture_execution_manifest
from experiments.run_v3_self_tooling_targeted import isolated_config
from experiments.run_v3_r102_targeted import action_source_audit

REPO = Path(__file__).resolve().parents[1]


def contracts():
    param = lambda name: ParameterSpec(name, 'entity', runtime_resolvable=True, required_resolution='concrete')
    producer = AbstractAtomicSkill(SkillRef('r102_t1_reach', '1.0.0'), 'Reach the supplied destination',
        [param('destination')], [param('object'), param('location')], [],
        [SemanticPredicate('agent.at_location', {'location': '$destination'})],
        {'output_derivations': {role: {'kind': 'input_identity', 'input_role': 'destination'} for role in ('object', 'location')}},
        [], {'steps': ['Move to the supplied destination and return its verified identity.'], 'notes': []},
        {'acceptance_fixture': True}, SkillStatus.ACTIVE)
    consumer = AbstractAtomicSkill(SkillRef('r102_t1_open', '1.0.0'), 'Open the supplied container at the supplied current location',
        [param('object'), param('location')], [param('opened')],
        [SemanticPredicate('agent.at_location', {'location': '$location'})],
        [SemanticPredicate('container.open', {'container': '$object'})],
        {'output_derivations': {'opened': {'kind': 'input_identity', 'input_role': 'object'}}},
        [], {'steps': ['At the supplied location, open the supplied container.'], 'notes': []},
        {'acceptance_fixture': True}, SkillStatus.ACTIVE)
    return producer, consumer


def run(config_path, output):
    started = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = isolated_config(load_config(config_path), output)
    config['experiment']['task_manifest_path'] = None
    if config.get('repair_revision') != 'R10.2':
        raise ValueError('R10.2 configuration required')
    manifest = {'formal_experiment': False, 'fixture_active_status_not_lifecycle_evidence': True,
        'code_hash': hash_code(REPO), 'config_hash': hash_config(config), 'env_index': 0,
        'contracts': to_primitive(contracts())}
    atomic_create_json(output / 'declared_manifest.json', manifest)
    from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
    capability = ensure_provider_capability(config, output_dir=output,
        config_hash=manifest['config_hash'], code_hash=manifest['code_hash'], run_if_missing=True)
    capture_execution_manifest(REPO, output, manifest['config_hash'], capability)
    with AtomicSkillGraphSystem(config) as system:
        task = system.harness.load_tasks(limit=1)[0]
        system.harness.reset(task)
        system._current_task_id, system._current_task_usage_start = task.task_id, 0
        compiled = []
        for atomic in contracts():
            system.skills.register_atomic(atomic)
            provenance = ToolProvenance(source='r102_dataflow_fixture', atomic_ref=str(atomic.ref),
                source_trace_id='fixture', occurrence_id=atomic.ref.logical_id)
            proposal = ToolBuilderSession(system._tool_builder_session('tool_builder_evolution', atomic.ref.logical_id)).build(
                atomic=atomic, provenance=provenance, harness_interface={
                    'profile': system.harness.profile_name,
                    'predicate_vocabulary': to_primitive(system.harness.semantic_predicate_schema()),
                    'primitive_actions': public_primitive_action_schema(system.harness)})
            report = system.tool_static_validator.validate_proposal(proposal, atomic, system.harness)
            atomic_write_json(output / (atomic.ref.logical_id + '_builder.json'), {'proposal': to_primitive(proposal), 'static': to_primitive(report)})
            if not report.passed or proposal.decision != 'create':
                raise RuntimeError('Real Builder failed T1 static admission; raw submission retained')
            canonical = CanonicalAtomicOccurrence(atomic.ref.logical_id, 'fixture', atomic.summary, 0, 0,
                {}, {}, atomic.inputs, atomic.outputs, atomic.preconditions, atomic.effects, [], [],
                to_primitive(task), 'fixture', atomic.ref)
            built = system.tool_compiler.compile_proposal(canonical, atomic, proposal, provenance)
            built.tool.status, built.implementation.status = ToolStatus.ACTIVE, SkillStatus.ACTIVE
            system.tools.register(built.tool)
            system.skills.register_implementation(built.implementation)
            compiled.append(built)
        first, second = compiled
        occurrences = [RuntimeOccurrence('upstream', 'upstream', first.atomic.ref, [], {}, [first.implementation.ref], first.atomic.effects),
            RuntimeOccurrence('downstream', 'downstream', second.atomic.ref, [], {
                role: BindingExpression(BindingExprKind.DATA_FLOW, source_role=role, source_step='upstream')
                for role in ('object', 'location')}, [second.implementation.ref], second.atomic.effects)]
        edges = [GraphEdge(f'edge_{role}', GraphEdgeType.DATA_FLOW, 'upstream', 'downstream', role, role)
                 for role in ('object', 'location')]
        plan = RuntimeLinearPlan(task.task_id, 'atomic_composition', None, occurrences,
            ['upstream', 'downstream'], [], edges, system.harness.task_contract(task), {'acceptance_fixture': True})
        ctx = TaskRuntimeContext.create(task, plan, system.harness, system.orchestrator.create_trace_builder(task),
            RuntimeBudget(global_action_budget=100, node_action_budget=35))
        ctx.runtime_config = config['runtime']
        ctx.task_goal = ('TARGETED DATAFLOW ACCEPTANCE, not the episode goal: choose a currently closed container '
            'destination from public observation/catalog and invoke the offered reach implementation. '
            'Do not open it yourself: the next declared graph node opens the returned container. '
            'Choose the concrete destination yourself; do not change either contract.')
        trace = ctx.trace_builder.trace
        trace.runtime_plan = to_primitive(plan)
        trace.metadata['acceptance_fixture'] = manifest
        results, provider_counts = [], []
        for occurrence in occurrences:
            ctx.budget.begin_node(occurrence.occurrence_id)
            ctx.binding_store.apply_data_flow(plan, occurrence.step_id, ctx.validated_outputs, revision=ctx.world_revision)
            ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
            ctx.begin_occurrence(occurrence)
            result = VerifiedCompositeExecutor(system.orchestrator.node_executor).run_occurrence(occurrence, ctx)
            results.append(result)
            provider_counts.append(len(system.usage.events))
            trace.node_records.append(NodeTraceRecord(occurrence.occurrence_id, occurrence.step_id, str(occurrence.node_ref),
                result.node_status, to_primitive(result), {}, dict(result.validated_outputs)))
            if not result.atomic_effect_passed:
                break
            refs = system.orchestrator._latest_atomic_witnesses(ctx, occurrence.occurrence_id)
            ctx.binding_store.publish_validated_outputs(occurrence, result.validated_outputs, refs, ctx.world_revision)
            ctx.validated_outputs[occurrence.occurrence_id] = dict(result.validated_outputs)
            for role, value in result.validated_outputs.items():
                ctx.evidence_store.add_validated_tool_output(role, value, refs)
        system._attach_external_sessions(trace, system._observed_sessions)
        trace.llm_usage = [event.to_dict() for event in system.usage.events]
        trace.metadata['usage_reconciliation'] = _reconcile_events(system.usage.events)
        _require_formal_usage(system.usage.events, trace.agent_turns)
        trace.finish()
        system.traces.save_atomic(trace)
        result = {'passed': len(results) == 2 and all(r.atomic_effect_passed for r in results)
            and provider_counts[0] == provider_counts[1] and results[1].node_status.value == 'direct_autonomous_success',
            'trace_id': trace.trace_id, 'results': to_primitive(results), 'provider_counts': provider_counts,
            'action_source_audit': action_source_audit(trace), 'wall_seconds': time.monotonic() - started,
            'code_unchanged': hash_code(REPO) == manifest['code_hash'], **manifest}
        atomic_write_json(output / 'summary.json', result)
        print(result, flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/alfworld_train_full_120_r102_seed42.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.config, args.output)['passed'] else 1)
