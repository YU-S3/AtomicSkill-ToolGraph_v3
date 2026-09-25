"""Production ToolRunner acceptance on a fixed Train fixture; no model or gold."""
import argparse
from pathlib import Path
from atomic_skillgraph.core.results import RuntimeLinearPlan
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.deployment.scienceworld_reference import basic_assets, preparation_assets, authority
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.traces.schema import TaskRecord, TraceRecord, TraceBuilder
from atomic_skillgraph.validation.tool_validator import ToolValidator
from .scienceworld_manifest import load, task_from_entry


def run(output):
    root=Path(__file__).resolve().parents[1]
    output=Path(output)
    entry=next(e for e in load(root/'data/scienceworld_manifests/train_120.json')['tasks'] if e['task_type']=='2-1')
    harness=ScienceWorldAdapter()
    task=task_from_entry(entry,'')
    plan=RuntimeLinearPlan.full_dynamic(task.task_id,harness.task_contract(task),reason='program_fixture_not_policy')
    trace=TraceRecord.create(TaskRecord(task.task_id,'scienceworld','program fixture',task.task_type,entry['task_signature']),{}, {},to_primitive(plan))
    ctx=TaskRuntimeContext.create(task,plan,harness,TraceBuilder(trace),RuntimeBudget())
    assets={a.metadata['canonical_intent']:(a,i,t) for a,i,t in basic_assets()+preparation_assets()}
    runner=ToolRunner(ToolValidator())
    results=[]
    try:
        def invoke(intent, bindings, expected=True):
            result=runner.run(assets[intent][2],bindings,ctx,occurrence_id=intent,execution_scope='runtime_trial')
            results.append({'intent':intent,'bindings':bindings,'result':to_primitive(result)})
            atomic_write_json(output/'progress.json',results)
            if bool(result.completed and result.atomic_effect_passed) != expected:
                raise AssertionError((intent,to_primitive(result)))
            return result
        invoke('relocate_to_location',{'destination':'kitchen'})
        invoke('inspect_current_scope',{'location':'kitchen'})
        first=invoke('wait_one_step',{})
        second=invoke('wait_one_step',{})
        assert first.output_candidates != second.output_candidates
        measure=next(a for a in harness.action_catalog() if a.action_type=='USE' and
            a.arguments['instrument']=='thermometer' and a.arguments['target']=='air')
        for _ in range(2):
            invoke('measure_temperature',{'thermometer':measure.arguments['instrument'],'target':measure.arguments['target']})
        invoke('discover_entity_in_authorized_rooms',{'query':'air','locations':['kitchen']})
        before=len(harness._prefix)
        invoke('relocate_to_location',{'destination':'unlisted location'},expected=False)
        assert len(harness._prefix)==before
        checkpoint=harness.capture_runtime_checkpoint()
        for _ in range(2):
            harness.restore_runtime_checkpoint(checkpoint)
        atomic_write_json(output/'scienceworld_bank_authority.json',authority(harness))
        atomic_write_json(output/'trace.json',to_primitive(trace))
        atomic_write_json(output/'result.json',{'passed':True,'source_identity':entry,'checks':results,'prefix_replays':2})
    finally:
        harness._close_backend()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    run(p.parse_args().output)
