"""Real-JVM public search and identity-bundle/dataflow diagnostics (Train only)."""
import argparse
from pathlib import Path
from atomic_skillgraph.core.results import RuntimeLinearPlan,RuntimeOccurrence
from atomic_skillgraph.core.edges import GraphEdge
from atomic_skillgraph.core.serialization import atomic_write_json,to_primitive
from atomic_skillgraph.deployment.scienceworld_reference import workflow_entry,room_search,container_search,bounded_wait
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.checkpoint import capture, restore
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.traces.schema import TraceRecord,TaskRecord,TraceBuilder
from atomic_skillgraph.validation.tool_validator import ToolValidator
from atomic_skillgraph.validation.atomic_validator import AtomicValidator
from .scienceworld_manifest import load,task_from_entry


def run(output):
    output = Path(output)
    manifest = load(Path(__file__).resolve().parents[1]/'data/scienceworld_manifests/train_120.json')
    entry = next(e for e in manifest['tasks'] if e['task_type']=='2-1')
    harness = ScienceWorldAdapter()
    task = task_from_entry(entry,'')
    plan = RuntimeLinearPlan.full_dynamic(task.task_id,harness.task_contract(task),reason='public_search_acceptance')
    trace = TraceRecord.create(TaskRecord(task.task_id,'scienceworld',task.goal,task.task_type,entry['task_signature']),{},{},to_primitive(plan))
    ctx = TaskRuntimeContext.create(task,plan,harness,TraceBuilder(trace),RuntimeBudget())
    runner = ToolRunner(ToolValidator())
    records = []
    try:
        for count in (1,2,3):
            a,i,t = workflow_entry(count)
            arguments = {r.name:'thermometer' for r in a.inputs if r.name.startswith('query')}
            arguments.update(locations=['hallway','bathroom','kitchen'],wait_steps=1)
            before = len(trace.environment_actions)
            result = runner.run(t,arguments,ctx,occurrence_id=f'entry{count}')
            assert result.completed and len(trace.environment_actions)==before, to_primitive(result)
            occurrence = RuntimeOccurrence('entry',f'entry{count}',a.ref,[],{},[i.ref],[])
            r1 = AtomicValidator().validate_execution_result(a,occurrence,arguments,result.output_candidates,
                harness.validator_channel(),current_revision=ctx.world_revision)
            assert r1.passed, to_primitive(r1)
            assert all(b.resolution.value=='semantic' for k,b in r1.validated_output_bindings.items() if k.startswith('query'))
            ctx.binding_store.publish_validated_outputs(occurrence,result.output_candidates,r1.witness_refs,
                ctx.world_revision,certified_bindings=r1.validated_output_bindings)
            wa,wi,wt = bounded_wait()
            downstream = RuntimeOccurrence('wait',f'wait{count}',wa.ref,[],{},[wi.ref],wa.effects)
            pair = RuntimeLinearPlan(task.task_id,'acceptance',None,[occurrence,downstream],['entry','wait'],
                [GraphEdge(f'data{count}','data_flow','entry','wait','wait_steps','wait_steps','registered')],[],
                plan.task_contract,{})
            ctx.binding_store.apply_data_flow(pair,'wait',revision=ctx.world_revision)
            values = ctx.binding_store.snapshot_for_node(downstream)
            assert values['wait_steps'].value == 1
            waited = runner.run(wt,{'wait_steps':values['wait_steps'].value},ctx,occurrence_id=downstream.occurrence_id)
            assert waited.completed and waited.executed_action_count==1,to_primitive(waited)
            records.append({'gate':'SW-WF00','arity':count,'entry':to_primitive(result),'r1':to_primitive(r1),'dataflow_wait':to_primitive(waited)})
            atomic_write_json(output/'progress.json',records)
        # The fixture declares its scopes explicitly; no runtime task/answer table.
        a,i,t = room_search()
        searched = runner.run(t,{'query':'thermometer','locations':['hallway','bathroom','kitchen']},ctx,occurrence_id='room_search')
        assert searched.completed and searched.executed_action_count==3,to_primitive(searched)
        records.append({'gate':'room_found','result':to_primitive(searched)})
        absent = runner.run(t,{'query':'nonexistent diagnostic object','locations':['hallway','bathroom','kitchen']},ctx,occurrence_id='absent')
        assert not absent.completed and absent.executed_action_count==3,to_primitive(absent)
        records.append({'gate':'room_absent_no_fabricated_output','result':to_primitive(absent)})
        ca,ci,ct = container_search()
        for name,query,container,expected in (
            ('container_found','sodium chloride','glass jar','found'),
            ('container_empty','nonexistent diagnostic object','freezer','scope_exhausted'),
            ('container_partial','nonexistent diagnostic object','cupboard','incomplete')):
            arguments = {'query':query,'location':'kitchen','containers':[container]}
            checkpoint = capture(ctx,name)
            result = runner.run(ct,arguments,ctx,occurrence_id=name)
            row = {'gate':name,'result':to_primitive(result),
                   'frame':harness.public_container_inspection_frame()}
            records.append(row)
            atomic_write_json(output/'progress.json',records)
            if expected == 'found':
                assert result.completed and result.executed_action_count==3,to_primitive(result)
                occurrence = RuntimeOccurrence(name,name,ca.ref,[],{},[ci.ref],ca.effects)
                r1 = AtomicValidator().validate_execution_result(ca,occurrence,arguments,result.output_candidates,
                    harness.validator_channel(),current_revision=ctx.world_revision,
                    semantic_compatible=harness.semantic_value_compatible)
                row['r1'] = to_primitive(r1)
                assert r1.passed,to_primitive(r1)
            else:
                assert not result.completed and not result.output_candidates,to_primitive(result)
                diagnostic = result.tool_path_evidence.get('final_effect_result',{}).get('scope_diagnostic',{})
                assert (diagnostic.get('outcome') == 'scope_exhausted') == (expected=='scope_exhausted'),to_primitive(result)
                restore(ctx,checkpoint,'acceptance_failed_search')
                observation = list(ctx.search_history.attempts.values())[-1]
                assert observation.world_disposition == 'rolled_back'
                assert not observation.current_truth_authorized and not observation.output_authorized
                assert observation.checks[0].outcome == ('no_matching_candidate' if expected=='scope_exhausted' else 'incomplete_inspection')
                row['rolled_back_history'] = to_primitive(observation)
        # Public catalog decides which containers can be inspected in this fixed
        # fixture; preserve every result, including partial/ambiguous listings.
        containers = sorted({a.arguments['container'] for a in harness.action_catalog() if a.action_type=='OPEN'})[:8]
        container_results = []
        for container in containers:
            for kind in ('OPEN','LOOK_IN'):
                matches = [a for a in harness.action_catalog() if a.action_type==kind and a.arguments=={'container':container}]
                if len(matches)==1:
                    r = harness.execute_action(matches[0].action_id,matches[0].revision)
                    container_results.append({'action':kind,'container':container,'result':to_primitive(r),
                        'frame':harness.public_container_inspection_frame()})
        checkpoint = harness.capture_runtime_checkpoint()
        for _ in range(2): harness.restore_runtime_checkpoint(checkpoint)
        atomic_write_json(output/'trace.json',to_primitive(trace))
        atomic_write_json(output/'result.json',{'passed':True,'source':entry,'checks':records,
            'container_observations':container_results,'prefix_replays':2,
            'not_a_p0_workflow_gate':True,'not_a_publication_proof':True})
    finally:
        atomic_write_json(output/'trace.json',to_primitive(trace))
        atomic_write_json(output/'progress.json',records)
        harness._close_backend()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
