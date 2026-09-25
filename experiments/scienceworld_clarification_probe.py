"""Train-side public parser-state regression, with two exact prefix replays."""
import argparse
from pathlib import Path
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.harness.scienceworld_actions import resolve
from .scienceworld_manifest import load, task_from_entry


def run(output):
    root = Path(__file__).resolve().parents[1]
    entry = next(x for x in load(root/'data/scienceworld_manifests/train_120.json')['tasks'] if x['task_type']=='5-1')
    harness = ScienceWorldAdapter()
    events = []
    try:
        harness.reset(task_from_entry(entry, ''))
        def step(kind, args):
            spec = resolve(harness.action_catalog(),kind,args,harness.validator_channel().revision)
            result = harness.execute_action(spec.action_id,spec.revision)
            events.append({'action':to_primitive(spec),'result':to_primitive(result),
                'catalog':to_primitive(harness.action_catalog())})
            return result
        step('TELEPORT',{'destination':'kitchen'})
        seeds = [x for x in harness.action_catalog() if x.action_type=='PICK_UP' and 'seed' in x.arguments['entity']]
        if not seeds:
            raise AssertionError('Fixed Train fixture no longer offers seed pickup')
        step('PICK_UP',seeds[0].arguments)
        choices = [x for x in harness.action_catalog() if x.action_type=='CLARIFY']
        if not choices:
            raise AssertionError('Fixed Train fixture no longer exercises public clarification')
        step('CLARIFY',choices[0].arguments)
        checkpoint = harness.capture_runtime_checkpoint()
        for _ in range(2):
            harness.restore_runtime_checkpoint(checkpoint)
        atomic_write_json(output,{'passed':True,'source_identity':entry,'events':events,'replays':2})
    finally:
        harness._close_backend()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    run(p.parse_args().output)
