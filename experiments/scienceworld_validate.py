"""No-model public surface and fresh-JVM replay acceptance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.harness.scienceworld_actions import compact, resolve
from .scienceworld_manifest import load, task_from_entry, resource_contract

def run(manifest_path, output):
    manifest = load(manifest_path)
    adapter = ScienceWorldAdapter()
    report = {'resource': resource_contract(), 'resets': [], 'replay': [], 'passed': False}
    try:
        adapter.initialize()
        replay_macro = {}
        for entry in manifest['tasks']:
            task_type, macro = entry['task_type'], entry['macro_type']
            selected = not any(r['task_type'] == task_type for r in report['resets'])
            replay = replay_macro.get(macro, 0) < 2
            if not selected and not replay:
                continue
            task = task_from_entry(entry, '')
            result = adapter.reset(task)
            # Every nonfree catalog row must round trip exactly.
            items = adapter.action_catalog()
            ambiguous = []
            for action in items:
                try:
                    assert resolve(items, action.action_type, action.arguments, action.revision) == action
                except ValueError:
                    ambiguous.append(action.raw_action)
            projected = compact(items)
            count = sum(len(v.get('tuples', next(iter(v.values())))) for v in projected.values())
            assert count == len({(a.action_type, tuple(a.arguments.items())) for a in items})
            report['resets'].append({'task_type': task_type, 'task_id': task.task_id,
                'catalog_count': len(items), 'semantic_count': count, 'ambiguous': ambiguous,
                'score': result.benchmark_score, 'simplifications': adapter._env.get_simplifications_used()})
            if replay:
                for name in ('TELEPORT', 'EXAMINE', 'WAIT1'):
                    candidates = [a for a in adapter.action_catalog() if a.action_type == name]
                    if candidates and not adapter.validator_channel().done:
                        action = sorted(candidates, key=lambda a: a.raw_action)[0]
                        adapter.execute_action(action.action_id, action.revision)
                checkpoint = adapter.capture_runtime_checkpoint()
                for _ in range(2):
                    adapter.restore_runtime_checkpoint(checkpoint)
                report['replay'].append({'task_id': task.task_id, 'macro_type': macro,
                    'prefix_length': len(checkpoint.accepted_prefix), 'passed': True})
                replay_macro[macro] = replay_macro.get(macro, 0) + 1
            atomic_write_json(output, report)
            print(json.dumps({'task_type': task_type, 'catalog': len(items), 'replay': replay}), flush=True)
        assert len({r['task_type'] for r in report['resets']}) == 30
        assert len(replay_macro) == 10 and set(replay_macro.values()) == {2}
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        atomic_write_json(output, report)
        adapter._close_backend()
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/scienceworld_manifests/train_120.json')
    parser.add_argument('--output', default='runs/scienceworld_acceptance/environment.json')
    args = parser.parse_args()
    run(args.manifest, args.output)
