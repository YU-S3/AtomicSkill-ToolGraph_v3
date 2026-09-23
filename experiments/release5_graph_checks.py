"""Train-side controlled acceptance of the published graph; never Runtime policy."""
import copy
from pathlib import Path
from types import SimpleNamespace
from atomic_skillgraph.core.results import RuntimeOccurrence
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.agents.protocol import NativeToolCall
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
from atomic_skillgraph.runtime.composite_executor import VerifiedCompositeExecutor
from atomic_skillgraph.runtime.loop_guard import ActionLoopGuard
from atomic_skillgraph.traces.compiler_observer import initialize, finalize, node_window
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.planner.compiler import PlanCompiler
from atomic_skillgraph.harness.alfworld import entity_matches
from .released_bank_checks import NoModel
from .run_v3_r103_validation import resolve_tasks


def run(config, entry, graph_ref, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = []
    for initially_open in (False, True):
        target = output/('open' if initially_open else 'closed')
        local = copy.deepcopy(config); local['trace_data_dir'] = str(target)
        with AtomicSkillGraphSystem(local, readonly=True, provider=NoModel()) as system:
            digest = system.knowledge_digest()
            task = resolve_tasks(system, [entry])[0]
            graph = system.skills.get_composite(graph_ref)
            plan = PlanCompiler(system.skills).from_composite(task, system.harness.task_contract(task), graph,
                mode='frozen', audit={})
            builder = system.orchestrator.create_trace_builder(task)
            initialize(builder, persistent_refs=[str(t.ref) for t in system.tools.tools()], request_snapshot=lambda: [])
            ctx = TaskRuntimeContext.create(task, plan, system.harness, builder, RuntimeBudget(global_action_budget=200))
            ctx.runtime_config = config['runtime']
            ex = system.orchestrator.node_executor
            nodes = {o.step_id: o for o in plan.occurrences}
            def helper(step, source):
                a = system.skills.get_atomic(nodes[source].node_ref)
                return RuntimeOccurrence(step, step, a.ref, [], {},
                    [i.ref for i in system.skills.implementations_for(a.ref, mode='frozen')], a.effects)
            nav, opening = helper('fixture_nav', 'station_nav'), helper('fixture_open', 'station_open')
            fixture_sequence = 0

            def begin(occ):
                ctx.begin_occurrence(occ); ctx.budget.begin_node(occ.occurrence_id)
                if occ.step_id in nodes:
                    ctx.binding_store.apply_data_flow(plan, occ.step_id, revision=ctx.world_revision)
                atomic = system.skills.get_atomic(occ.node_ref)
                ctx.binding_store.resolve_occurrence_specs(occ, ctx.world_revision, input_specs=atomic.inputs)
                return system.invocation_compiler.compile_candidates(occ, ctx.binding_store, task_id=ctx.task_id,
                    evidence_store=ctx.evidence_store, revision=ctx.world_revision, task_contract=ctx.task_contract)

            def publish(occ, result):
                if result is None or not result.atomic_effect_passed:
                    raise AssertionError(to_primitive(result))
                ctx.binding_store.publish_validated_outputs(occ, result.validated_outputs, result.atomic_witness_refs,
                    ctx.world_revision, certified_bindings=result.validated_output_bindings)
                ctx.validated_outputs[occ.occurrence_id] = result.validated_outputs
                return result

            def explicit(occ, args):
                nonlocal fixture_sequence
                if occ.step_id not in nodes:
                    # Repeated preparation calls are distinct occurrences, not
                    # a rewrite of the identity/output of an earlier call.
                    fixture_sequence += 1
                    occ = copy.deepcopy(occ)
                    occ.occurrence_id = occ.step_id = f'{occ.step_id}_{fixture_sequence}'
                routes = begin(occ)
                satisfied = ex._complete_from_current_effect(occ, ctx, mode='entry', preferred_values=[],
                    preferred_bindings=args)
                if satisfied is not None:
                    return publish(occ, satisfied)
                for compiled in routes:
                    checked = system.invocation_compiler.preflight(compiled, call_name=compiled.spec.name,
                        call_id='fixture_'+occ.step_id, arguments=args, occurrence=occ, binding_store=ctx.binding_store,
                        evidence_store=ctx.evidence_store, revision=ctx.world_revision)
                    if checked.passed:
                        return publish(occ, execute_invocation(ex.implementation_runner, compiled, checked, occ, ctx,
                            agent_prepared=True, authorizing_native_call_id='fixture_'+occ.step_id))
                raise AssertionError({'occurrence': occ.step_id, 'args': args, 'preflight': to_primitive(checked) if routes else 'no route'})

            semantic = task.context['semantic_bindings']
            destinations = [a.arguments['destination'] for a in ctx.action_catalog if a.action_type == 'GO_TO']
            final_destination = next(d for d in destinations if entity_matches(d, semantic['destination']))
            found = None
            for location in destinations:
                if any(a.action_type == 'GO_TO' and a.arguments['destination'] == location for a in ctx.action_catalog):
                    explicit(nav, {'destination': location})
                else:
                    continue
                if any(a.action_type == 'OPEN' and a.arguments['object'] == location for a in ctx.action_catalog):
                    explicit(opening, {'container': location})
                found = next((a for a in ctx.action_catalog if a.action_type == 'TAKE'
                    and entity_matches(a.arguments['object'], semantic['object'])), None)
                if found:
                    break
            if not found:
                raise AssertionError('fixed train fixture target not found in public catalog')
            explicit(nodes['take'], dict(found.arguments))
            station = None
            for location in destinations:
                if any(a.action_type == 'GO_TO' and a.arguments['destination'] == location for a in ctx.action_catalog):
                    explicit(nav, {'destination': location})
                cool = next((a for a in ctx.action_catalog if a.action_type == 'COOL'), None)
                if cool:
                    station = cool.arguments['station']; break
            if station is None or station == final_destination:
                raise AssertionError('fixture must expose distinct processing and destination roles')
            if any(a.action_type == 'GO_TO' and a.arguments['destination'] == station for a in ctx.action_catalog):
                explicit(nav, {'destination': station})
            if initially_open:
                if any(a.action_type == 'OPEN' and a.arguments['object'] == station for a in ctx.action_catalog):
                    explicit(opening, {'container': station})
            else:
                close = next((a for a in ctx.action_catalog if a.action_type == 'CLOSE' and a.arguments['object'] == station), None)
                if close:
                    ex._execute_environment_call(NativeToolCall('fixture_close', 'environment_action',
                        {'action_id': close.action_id, 'intent': 'explore'}), SimpleNamespace(session_id='fixture'), None, ctx,
                        span_id='', origin='controlled_fixture', loop_guard=ActionLoopGuard())
            explicit(nav, {'destination': final_destination})
            routes = begin(nodes['station_nav'])
            state = ex._activate_occurrence_state(nodes['station_nav'], system.skills.get_atomic(nodes['station_nav'].node_ref), routes, ctx)
            explicit(nodes['station_nav'], {'destination': station})
            ctx.graph_bootstrap_completed = True
            begin(nodes['station_open'])
            opened = publish(nodes['station_open'], VerifiedCompositeExecutor(ex).run_occurrence(nodes['station_open'], ctx))
            routes = begin(nodes['cool'])
            with node_window(nodes['cool'], ctx):
                cooled = publish(nodes['cool'], ex.try_autonomous(nodes['cool'], routes, ctx))
            # The final task anchor is still semantic: the test explicitly
            # chooses its public concrete instance, just as an Agent must.
            # Object identity is consumed only from the revised COOL edge.
            delivered = explicit(nodes['deliver'], {'destination': final_destination})
            trace = builder.finish(); finalize(trace); system.traces.save_atomic(trace)
            flows = trace.metadata['compiler_observability']['dataflow_consumptions']
            for producer, output_role, consumer, input_role in [('station_nav', 'location', 'cool', 'station'),
                                                               ('cool', 'object', 'deliver', 'object')]:
                if not any(f['producer_occurrence_id'] == producer and f['producer_output_role'] == output_role
                    and f['consumer_occurrence_id'] == consumer and f['consumer_input_role'] == input_role
                    and f['consumed'] and f['producer_invocation_id'] for f in flows):
                    raise AssertionError('revised edge was not consumed from a certified program RETURN')
            obligations = state['downstream_obligations']['output_obligations']
            if {o['consumer_step'] for o in obligations} != {'station_open', 'cool'}:
                raise AssertionError('direct downstream station use missing from policy')
            expected_open = 'already_satisfied' if initially_open else 'direct_autonomous_success'
            if opened.node_status != expected_open or cooled.node_status != 'direct_autonomous_success':
                raise AssertionError('opened/unopened automatic boundary differs')
            row = {'initially_open': initially_open, 'station': station, 'destination': final_destination,
                'downstream_obligations': state['downstream_obligations'], 'open_result': to_primitive(opened),
                'cool_result': to_primitive(cooled), 'deliver_result': to_primitive(delivered),
                'trace_id': trace.trace_id, 'dataflow_consumptions': flows, 'provider_calls': len(system.usage.events),
                'digest_unchanged': digest == system.knowledge_digest()}
            if not (flows and row['digest_unchanged'] and row['provider_calls'] == 0):
                raise AssertionError(row)
            rows.append(row)
    from .protocol import hash_code
    from atomic_skillgraph.deployment.release_protocol import sha
    report = {'passed': True, 'scope': 'fixed train controlled test only, not natural score', 'cases': rows,
        'code_hash': hash_code(Path(__file__).resolve().parents[1]), 'bank_digest': digest,
        'trace_hashes': {str(p.relative_to(output)): sha(p) for p in output.glob('*/traces/*.json')}}
    atomic_write_json(output/'acceptance.json', report)
    return report


if __name__ == '__main__':
    import argparse
    import json
    from .release4_coverage import entries_for, episode_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entry = next(e for e in entries_for(args.release_root) if e['task_type'] == 'pick_cool_then_place_in_recep')
    graph_ref = json.loads((args.release_root/'seed42_graph_revision_audit.json').read_text())['target_ref']
    run(episode_config(args.release_root, 42, entry), entry, graph_ref, args.output)
