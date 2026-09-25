"""Real JVM/public ToolRunner gate for the two generic IR extensions, no LLM."""
import argparse
from pathlib import Path
from atomic_skillgraph.core.results import RuntimeLinearPlan
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.deployment.scienceworld_reference import author, predicate, argument
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.traces.schema import TaskRecord, TraceRecord, TraceBuilder
from atomic_skillgraph.validation.tool_validator import ToolValidator
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from .scienceworld_manifest import load, task_from_entry


def run(output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    root = Path(__file__).resolve().parents[1]
    entry = next(e for e in load(root/'data/scienceworld_manifests/train_120.json')['tasks'] if e['task_type']=='2-1')
    harness = ScienceWorldAdapter()
    task = task_from_entry(entry,'')
    plan = RuntimeLinearPlan.full_dynamic(task.task_id,harness.task_contract(task),reason='generic_ir_probe')
    trace = TraceRecord.create(TaskRecord(task.task_id,'scienceworld','IR fixture',task.task_type,entry['task_signature']),{},{},to_primitive(plan))
    ctx = TaskRuntimeContext.create(task,plan,harness,TraceBuilder(trace),RuntimeBudget())
    runner = ToolRunner(ToolValidator())
    records = []
    try:
        wait_effects = [predicate('time.progressed',evidence=None)]
        wait = author('caller_bounded_wait',['steps'], {'evidence_ref':('time.progressed','evidence',{})},wait_effects,
            [{'op':'FOR_EACH','node_id':'wait_steps','collection_source':{'source':'bounded_count','count':{'source':'tool_input','field':'steps'}},
              'iteration_variable':'step_index','max_iterations':8,'body':[
                  {'op':'ACTION','node_id':'wait_once','action_type':'WAIT1','argument_mapping':{},'expected_effects':to_primitive(wait_effects)}]}],
            max_actions=8,input_types={'steps':'integer'})
        wait[2].signature['properties']['steps'].update(minimum=0,maximum=8)
        projection = {**argument('job',True),'field_path':['destination']}
        nav = author('caller_authorized_sequence',['jobs','destination'], {'location':'destination'},
            [predicate('agent.at_location',location='destination')],
            [{'op':'FOR_EACH','node_id':'visit_jobs','collection_source':{'source':'tool_input','field':'jobs'},
              'iteration_variable':'job','max_iterations':8,'body':[
                  {'op':'ACTION','node_id':'visit','action_type':'TELEPORT',
                   'argument_mapping':{'destination':projection}, 'expected_effects':[
                       {'predicate':'agent.at_location','args':{'location':projection},'effect_domain':'world'}]}]}],
            max_actions=8,input_types={'jobs':'array'})
        nav[2].signature['properties']['jobs'].update(maxItems=8,items={'type':'object','required':['destination'],
            'additionalProperties':False,'properties':{'destination':{'type':'string'}}})
        for atomic, _, tool in (wait,nav):
            tool.artifact['value_contract_version'] = 2
            report = ToolStaticValidator().validate_tool_asset(tool,atomic,harness)
            assert report.passed, report
        def execute(tool, bindings, count, *, rejected=False):
            before = len(trace.environment_actions)
            result = runner.run(tool,bindings,ctx,occurrence_id='probe',execution_scope='runtime_trial')
            assert len(trace.environment_actions)-before == count, result
            if rejected: assert result.failure_code == 'tool_input_schema_invalid' and not result.started
            else: assert result.completed and result.atomic_effect_passed, result
            records.append({'tool':str(tool.ref),'bindings':bindings,'result':to_primitive(result)})
            atomic_write_json(output/'progress.json',records)
        execute(wait[2],{'steps':8},8)
        execute(wait[2],{'steps':9},0,rejected=True)
        places = sorted({a.arguments['destination'] for a in harness.action_catalog() if a.action_type=='TELEPORT'})[:2]
        assert len(places)==2
        execute(nav[2],{'jobs':[{'destination':p} for p in places],'destination':places[-1]},2)
        execute(nav[2],{'jobs':[{'destination':places[0]}]*9,'destination':places[0]},0,rejected=True)
        checkpoint = harness.capture_runtime_checkpoint()
        for _ in range(2): harness.restore_runtime_checkpoint(checkpoint)
        atomic_write_json(output/'trace.json',to_primitive(trace))
        atomic_write_json(output/'result.json',{'passed':True,'source_identity':entry,'checks':records,'prefix_replays':2})
    finally:
        harness._close_backend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
