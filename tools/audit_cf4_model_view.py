"""Project all recorded fixed-six HTTP materials offline; bytes are not tokens."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.model_view import project
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.prompts import PLANNER_PROMPT, RUNTIME_PROMPT
from atomic_skillgraph.harness.alfworld_simple import SimpleAlfWorld
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.experiments.run_empirical import write_json, code_identity


def size(value): return len(json.dumps(value, ensure_ascii=False).encode())


def references(value):
    if isinstance(value, dict):
        if set(value) == {'result_id', 'path'}: yield value
        else:
            for child in value.values(): yield from references(child)
    elif isinstance(value, list):
        for child in value: yield from references(child)


def audit(root, bank_path, output):
    root, output = Path(root), Path(output)
    before = tree_identity(root)
    frozen_before = tree_identity(bank_path)
    traces = {t['task']['task_id']: t for p in (root / 'val/traces').glob('*.json') for t in [json.loads(p.read_text())]}
    calls = [json.loads(line) for line in (root / 'llm_calls.jsonl').read_text().splitlines()]
    if len(calls) != 82: raise ValueError('Expected the complete original 82 HTTP requests')
    bank, rows, projected = Bank(bank_path, readonly=True), [], []
    try:
        for call in calls:
            trace = traces[call['task_id']]
            task = PublicTask(**trace['task'])
            context = TaskContext()
            material = json.loads(call['request_messages'][1]['content'])
            if 'node_interface' in material:
                material.setdefault('output_aliases', {})
            existing = list(references(material))
            if existing: context.scope = existing[0]['result_id'].split(':')[0]
            for event in trace['tools']:
                if event.get('result_id'): context.results[event['result_id']] = event['result']
            context.results.update(trace.get('result_store', {}).get('program_results', {}))
            context_task = task.inputs.get('environment_task', {}).get('context', {})
            proxy = SimpleNamespace(task=task, initial_observation=context_task.get('initial_observation', ''),
                capabilities=SimpleNamespace(interaction='interactive'))
            proxy.model_task = lambda t=None, proxy=proxy: SimpleAlfWorld.model_task(proxy, t)
            if call['stage'] == 'planner':
                material['related_interfaces'] = bank.planning_cards(task.goal)
                material['program_options'] = bank.program_options(task.goal)
            after = project(call['stage'], material, task=task, adapter=proxy, context=context)
            for ref in references(after): context.resolve(ref)
            messages = deepcopy(call['request_messages'])
            messages[0]['content'] = PLANNER_PROMPT if call['stage'] == 'planner' else RUNTIME_PROMPT
            messages[1]['content'] = json.dumps(after, ensure_ascii=False)
            before_tools = material.get('tools', material.get('current_tools', []))
            after_tools = after.get('calls', {}).get('tools', after.get('current_tools', []))
            same_tools = before_tools == after_tools
            if not same_tools or after.get('task', after.get('original_task', {}))['goal'] != task.goal:
                raise AssertionError('Projection changed the task or legal call set')
            rows.append({'call_id': call['call_id'], 'task_id': task.task_id, 'stage': call['stage'],
                'before_messages_bytes': size(call['request_messages']), 'after_messages_bytes': size(messages),
                'before_tool_schema_bytes': size(call['request_tools']), 'after_tool_schema_bytes': size(call['request_tools']),
                'goal_retained': True, 'legal_calls_retained': same_tools, 'result_refs_resolved': len(list(references(after))),
                'backend_identity_occurrences_before': sum(call['request_messages'][1]['content'].count(k) for k in ('env_index', 'native_task_id', 'game_file', 'task_signature')),
                'backend_identity_occurrences_after': sum(messages[1]['content'].count(k) for k in ('env_index', 'native_task_id', 'game_file', 'task_signature')),
                'backend_identity_repeated_after': sum(max(0, messages[1]['content'].count(k)-1) for k in ('env_index', 'native_task_id', 'game_file', 'task_signature')),
                'input_ref_store_sha256': digest(context.results), 'input_ref_store_saved_before_send': 'offline reconstruction; production checkpoint tested separately'})
            used_ids = {ref['result_id'] for ref in references(after)}
            write_json(output / 'reference_stores' / (call['call_id'] + '.json'), {rid: context.results[rid] for rid in sorted(used_ids)})
            projected.append({'call_id': call['call_id'], 'messages': messages, 'tools': call['request_tools']})
    finally: bank.close()
    if tree_identity(root) != before or tree_identity(bank_path) != frozen_before:
        raise AssertionError('Read-only original evidence changed')
    report = {'kind': 'offline counterfactual material projection, no provider/environment calls',
        'source_identity': code_identity(), 'original_run': str(root), 'original_bank': str(bank_path),
        'requests': len(rows), 'live_model_requests': 0, 'original_run_hash_before_after': before,
        'frozen_hash_before_after': frozen_before, 'rows': rows,
        'totals': {key: sum(r[key] for r in rows) for key in ('before_messages_bytes', 'after_messages_bytes',
            'before_tool_schema_bytes', 'after_tool_schema_bytes', 'backend_identity_occurrences_before', 'backend_identity_occurrences_after', 'backend_identity_repeated_after')},
        'claim_limit': 'Serialized bytes only; no measured token saving, cost saving, choice preference or new task success.'}
    write_json(output / 'model_view_projection.json', report)
    write_json(output / 'projected_requests.json', projected)
    return report['totals']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.original_run, args.bank, args.output)))
