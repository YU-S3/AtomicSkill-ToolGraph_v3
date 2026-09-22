"""Public-catalog-only, no-model acceptance for the actual old-first bank.

This is a bounded test driver, not a Runtime policy or an evaluation score.
"""
import copy
import json
from pathlib import Path
from atomic_skillgraph.core.bindings import BindingExpression
from atomic_skillgraph.core.edges import GraphEdge
from atomic_skillgraph.core.results import RuntimeOccurrence, RuntimeLinearPlan
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
from atomic_skillgraph.traces.compiler_observer import initialize, finalize, node_window
from atomic_skillgraph.system import AtomicSkillGraphSystem
from .released_bank_checks import NoModel
from .run_v3_r103_validation import resolve_tasks


def run(config, entry, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config); config['trace_data_dir'] = str(output)
    edit = json.loads((Path(config['data_dir'])/'edit_plan.lock.json').read_text())
    with AtomicSkillGraphSystem(config, readonly=True, provider=NoModel()) as system:
        before = system.knowledge_digest()
        task = resolve_tasks(system, [entry])[0]
        roles = edit['existing_roles']
        search_job = next(j for j in edit['program_realization_jobs'] if j.get('query_input_role'))
        serial = next(j for j in edit['program_realization_jobs'] if len(j.get('tool_bindings', [])) > 1)
        def occurrence(step, ref):
            atomic = system.skills.get_atomic(ref)
            return RuntimeOccurrence(step, step, atomic.ref, [], {},
                [i.ref for i in system.skills.implementations_for(atomic.ref, mode='frozen')], atomic.effects)
        nav = occurrence('navigate', roles['navigate']); opening = occurrence('open', roles['open'])
        find = occurrence('discover', search_job['atomic_ref']); take = occurrence('take', roles['take'])
        deliver = occurrence('deliver', serial['atomic_ref'])
        taken_role = next(p.name for p in system.skills.get_atomic(take.node_ref).outputs
                          if p.name in {'object', 'held_object'})
        occurrences = [nav, opening, find, take, deliver]
        edges = [GraphEdge('found_entity','data_flow','discover','take','entity','object'),
                 GraphEdge('found_location','data_flow','discover','take','location','source'),
                 GraphEdge('held_object','data_flow','take','deliver',taken_role,'object'),
                 GraphEdge('return_location','data_flow','discover','deliver','location','destination')]
        for edge in edges:
            target = next(o for o in occurrences if o.step_id == edge.target_step)
            target.binding_specs[edge.target_role] = BindingExpression('data_flow',
                source_step=edge.source_step, source_role=edge.source_role)
        plan = RuntimeLinearPlan(task.task_id, 'atomic_composition', None, occurrences,
            [o.step_id for o in occurrences], edges, [], system.harness.task_contract(task), {})
        builder = system.orchestrator.create_trace_builder(task)
        builder.trace.metadata['experiment_kind'] = 'controlled_oldfirst_automation_acceptance'
        initialize(builder, persistent_refs=[str(t.ref) for t in system.tools.tools()], request_snapshot=lambda: [])
        ctx = TaskRuntimeContext.create(task, plan, system.harness, builder, RuntimeBudget(global_action_budget=100))
        ctx.runtime_config = config['runtime']; executor = system.orchestrator.node_executor

        def begin(occ):
            ctx.begin_occurrence(occ); ctx.budget.begin_node(occ.occurrence_id)
            ctx.binding_store.apply_data_flow(plan, occ.step_id, revision=ctx.world_revision)
            ctx.binding_store.resolve_occurrence_specs(occ, ctx.world_revision,
                input_specs=system.skills.get_atomic(occ.node_ref).inputs)
            return system.invocation_compiler.compile_candidates(occ, ctx.binding_store,
                task_id=ctx.task_id, evidence_store=ctx.evidence_store, revision=ctx.world_revision,
                task_contract=ctx.task_contract)

        def publish(occ, result):
            if not (result.completed and result.atomic_effect_passed and not result.failure_code):
                builder.trace.metadata['acceptance_failure'] = to_primitive(result)
                system.traces.save_atomic(builder.finish())
                raise AssertionError(to_primitive(result))
            ctx.binding_store.publish_validated_outputs(occ, result.validated_outputs, result.atomic_witness_refs,
                ctx.world_revision, certified_bindings=result.validated_output_bindings)
            ctx.validated_outputs[occ.occurrence_id] = result.validated_outputs

        def explicit(occ, args):
            routes = begin(occ)
            preferred = [r for r in routes if r.implementation.quality.get('preferred')]
            selected = preferred if len(preferred) == 1 else routes
            if len(selected) != 1: raise AssertionError('test route is not unique')
            compiled = selected[0]
            preflight = system.invocation_compiler.preflight(compiled, call_name=compiled.spec.name,
                call_id='test_'+occ.step_id, arguments=args, occurrence=occ, binding_store=ctx.binding_store,
                evidence_store=ctx.evidence_store, revision=ctx.world_revision)
            if not preflight.passed: raise AssertionError(to_primitive(preflight))
            result = execute_invocation(executor.implementation_runner, compiled, preflight, occ, ctx,
                agent_prepared=True, authorizing_native_call_id='test_'+occ.step_id)
            publish(occ, result); return result

        scope = list(dict.fromkeys(a.arguments['destination'] for a in ctx.action_catalog if a.action_type == 'GO_TO'))[:8]
        found = None
        for destination in scope:
            explicit(nav, {'destination': destination})
            offered = next((a for a in ctx.action_catalog if a.action_type == 'OPEN'), None)
            if offered:
                explicit(opening, {'container': offered.arguments['object']})
            found = next((a for a in ctx.action_catalog if a.action_type == 'TAKE'), None)
            if found: break
        if not found: raise AssertionError('fixed bounded public scope contains no takeable object')
        source = found.arguments['source']
        public_other = next(a.arguments['destination'] for a in ctx.action_catalog
                            if a.action_type == 'GO_TO' and a.arguments['destination'] != source)
        explicit(find, {search_job['query_input_role']: found.arguments['object'],
                        'locations': [public_other, source], 'allow_open': True})
        explicit(take, {})
        explicit(nav, {'destination': public_other})
        routes = begin(deliver); ctx.graph_bootstrap_completed = True
        with node_window(deliver, ctx):
            result = executor.try_autonomous(deliver, routes, ctx)
        if result is None: raise AssertionError('declared serial dataflow successor not ready')
        publish(deliver, result)
        trace = builder.finish(); finalize(trace); system.traces.save_atomic(trace)
        metrics = trace.metadata['compiler_observability']
        multi = [p for p in metrics['program_invocation_links'] if p['complete_success']
                 and len(p['canonical_action_indices'] or []) >= 2 and p['provider_request_refs'] == []]
        auto = [p for p in metrics['program_invocation_links'] if p['complete_success']
                and p['origin'] == 'graph_entry_auto' and p['provider_request_refs'] == []]
        flows = [p for p in metrics['dataflow_consumptions'] if p['consumer_occurrence_id'] == deliver.occurrence_id
                 and p['producer_invocation_id']]
        report = {'passed': bool(multi and auto and flows and before == system.knowledge_digest()),
            'entry_selection': 'fixed public-catalog test, not natural planner', 'trace_id': trace.trace_id,
            'multi_action_intervals': multi, 'automatic_intervals': auto, 'dataflow_consumptions': flows,
            'bank_digest_before': before, 'bank_digest_after': system.knowledge_digest()}
        atomic_write_json(output/'acceptance.json', report)
        from .protocol import hash_code
        from atomic_skillgraph.deployment.release_protocol import sha
        report.update(code_hash=hash_code(Path(__file__).resolve().parents[1]),
            trace_sha256=sha(output/'traces'/f'{trace.trace_id}.json'), task_identity=entry)
        atomic_write_json(output/'acceptance.json', report)
        if not report['passed']: raise AssertionError(report)
        return report
