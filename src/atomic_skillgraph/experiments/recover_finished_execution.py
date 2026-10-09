"""Explicit zero-LLM scoring child for an already sealed executor completion."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

from ..empirical import POLICY_DEFAULTS
from ..empirical.bank import Bank
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import PublicTask, digest
from ..empirical.system import EmpiricalSystem, validate_config
from ..empirical.trial_snapshot import validate_finished_execution, restore_trial_workspace, seal_trial_workspace
from ..harness.registry import create_simple_harness
from .run_empirical import code_identity, write_json
from .formal_log import tree_identity, utc
from .episode_projection import episode_projection


def read(path): return json.loads(Path(path).read_text())


def inspect(source, task_id):
    source = Path(source).resolve()
    if (source/'test').exists(): raise ValueError('Scoring recovery requires a Train source before Test')
    execution = read(source/'train/execution_manifest.json')
    if execution['config'] != read(source/'train/config.json'):
        raise ValueError('Parent execution/config identity differs')
    db = sqlite3.connect((source/'train/run.sqlite3').as_uri()+'?mode=ro', uri=True)
    try: rows = db.execute('SELECT id,attempts,status,result FROM tasks ORDER BY rowid').fetchall()
    finally: db.close()
    pending = [r for r in rows if r[2] != 'completed']
    if len(pending) != 1 or pending[0][0] != task_id: raise ValueError('Expected one pending Train boundary')
    if [r[0] for r in rows] != [t['task_id'] for t in execution['tasks'][:len(rows)]]:
        raise ValueError('Recovery prefix differs from task identity')
    task = next(t for t in execution['tasks'] if t['task_id'] == task_id)
    attempt = pending[0][1]
    cp = TaskCheckpoint(source/'train/checkpoints'/task_id/str(attempt))
    if cp.state['stage'] != 'task_started' or cp.state.get('trace', {}).get('score') is not None:
        raise ValueError('Expected executor completion before scoring')
    finished, receipt, events = validate_finished_execution(cp)
    saved_task = cp.state.get('trace', {}).get('task')
    if saved_task is not None and saved_task != task: raise ValueError('Checkpoint task identity differs')
    audit = read(source/'train/requests'/f'{task_id}_attempt{attempt}.json')
    if saved_task is None:
        public = {'goal': task['goal'], 'inputs': task['inputs']}
        materials = [json.loads(m['content']) for q in audit['requests'] for m in q.get('messages', [])
                     if m['role'] == 'user' and isinstance(m['content'], str) and m['content'].startswith('{')]
        if not any(m.get('task', m.get('original_task')) == public for m in materials):
            raise ValueError('Paid Runtime material does not confirm the executor task')
    requests = [q['id'] for q in audit['requests']]
    usage = [u['event_id'] for u in audit['usage']]
    if len(set(requests)) != len(requests) or len(set(usage)) != len(usage): raise ValueError('Duplicate inherited billing')
    bank = Bank(source/'train/bank', readonly=True, seed=execution['config']['experiment']['seed'])
    try:
        if any(c['task']['task_id'] == task_id for c in bank.train_cases()):
            raise ValueError('Executor-only recovery cannot inherit a started learning case')
        bank_digest = bank.digest()
    finally: bank.close()
    report = {'source': str(source), 'parent_run_id': read(source/'run_manifest.json')['run_id'],
        'parent_code': execution['code'], 'child_code': code_identity(), 'task': task, 'attempt': attempt,
        'checkpoint_sha256': digest(cp.state), 'workspace_sha256': receipt['sha256'],
        'execution_sha256': digest(finished), 'bank_digest': bank_digest,
        'prefix': [r[0] for r in rows[:-1]], 'source_tree': tree_identity(source),
        'request_ids': requests, 'usage_ids': usage, 'inherited_tokens': sum(u['total_tokens'] for u in audit['usage']),
        'native_events_sha256': digest(events), 'new_model_calls': 0, 'automatic_resume': False}
    report['recovery_id'] = digest(report)
    return report


def recover(source, output, task_id, *, dry_run=False, interrupt_after=None):
    report = inspect(source, task_id)
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source): raise ValueError('Recovery must have a separate child directory')
    if dry_run:
        write_json(output, report)
        return report
    if output.exists():
        receipt = read(output/'recovery_receipt.json')
        if receipt['recovery_id'] != report['recovery_id']: raise ValueError('Recovery identity differs')
        return receipt
    staging = output.with_name(output.name+'.staging-'+report['recovery_id'][:12])
    if not staging.exists():
        shutil.copytree(source, staging)
        write_json(staging/'recovery_inspection.json', report)
    elif read(staging/'recovery_inspection.json')['recovery_id'] != report['recovery_id']:
        raise ValueError('Staging identity differs')
    if (staging/'recovery_receipt.json').exists():
        ready = read(staging/'recovery_receipt.json')
        if ready['recovery_id'] != report['recovery_id']: raise ValueError('Ready child identity differs')
        if tree_identity(source) != report['source_tree']: raise RuntimeError('Recovery changed its source')
        staging.rename(output)
        return ready
    if interrupt_after == 'copy': raise InterruptedError('after copy')
    task = PublicTask(**report['task'])
    cp = TaskCheckpoint(staging/'train/checkpoints'/task_id/str(report['attempt']))
    original_receipt = cp.state['executor_finished_workspace']
    receipt = deepcopy(original_receipt)
    receipt['root'] = str(cp.root/'executor_finished_workspace')
    config = deepcopy(read(source/'train/config.json'))
    for section, defaults in POLICY_DEFAULTS.items(): config[section] = {**config.get(section, {}), **defaults}
    config['data_dir'] = str(staging/'train/bank')
    config['experiment']['output_dir'] = str(staging/'train')
    config = validate_config(config)
    adapter = create_simple_harness(config)
    def forbid(*args, **kwargs): raise RuntimeError('Scoring recovery cannot invoke a model')
    system = EmpiricalSystem(config, harness=adapter, provider=SimpleNamespace(complete=forbid))
    try:
        system.checkpoint = cp
        system.audit_path = staging/'train/requests'/f'{task_id}_attempt{report["attempt"]}.json'
        adapter.reset(task)
        restore_trial_workspace(adapter, receipt)
        adapter.evaluation_receipts = cp.root/'evaluation_receipts'
        evaluation_path = cp.root/'recovery_evaluation.json'
        if evaluation_path.exists():
            evaluation = read(evaluation_path)
            if evaluation['recovery_id'] != report['recovery_id']: raise ValueError('Saved evaluation identity differs')
            sealed, score = evaluation['sealed'], evaluation['score']
            adapter.score_audit = evaluation['scoring_audit']
        else:
            sealed = adapter.submit(cp.state['executor_finished'].get('prediction'))
            if sealed.get('bundle'):
                stable_bundle = cp.root/'sealed_submission'
                if not stable_bundle.exists(): shutil.copytree(sealed['bundle'], stable_bundle)
                sealed['bundle'] = str(stable_bundle)
            score = adapter.evaluate(sealed)
            write_json(evaluation_path, {'recovery_id': report['recovery_id'], 'sealed': sealed,
                'score': score, 'scoring_audit': getattr(adapter, 'score_audit', {})})
        if interrupt_after == 'score': raise InterruptedError('after score, before completion')
        trace = deepcopy(cp.state.get('trace') or {'schema':'empirical.trace.v1', 'task':report['task'],
            'attempt_id':task_id+':'+str(report['attempt']), 'knowledge_before':report['bank_digest'],
            'initial_plan':cp.state.get('initial_plan'), 'learning_status':'not_started',
            'trace_origin':'explicit_recovery_from_confirmed_executor_boundary'})
        trace.update(read(system.audit_path))
        native_path = cp.root/'native_events.json'
        native_events = read(native_path) if native_path.exists() else []
        context = cp.state.get('executor_state', {}).get('context', {})
        native_ids = {e.get('result_id') for e in native_events}
        trace['result_store'] = {'native_index':[{'result_id':e.get('result_id'), 'event_id':e.get('event_id'), 'index':e['index']}
            for e in native_events if e.get('result_id')], 'program_results':{rid:v for rid,v in context.get('results', {}).items() if rid not in native_ids},
            'local_reads':context.get('local_reads', [])}
        system.complete_execution(task, trace,
            SimpleNamespace(events=native_events,
                environment_steps=sum(e.get('environment_step', 0) for e in native_events)),
            cp.state['executor_finished'], learn=False, sealed=sealed, score=score)
        # Learning remains a real pending boundary, with a verified public workspace.
        learning_workspace = seal_trial_workspace(adapter, cp.root/'learning_workspace') if not cp.state.get('learning_workspace') else cp.state['learning_workspace']
        trace.update(learning_status='deferred', learning_error=None, knowledge_after=system.bank.digest(),
            is_recovery=True, new_model_calls=0, new_train_examples_consumed=0)
        trace['learning'] = {'decision_origin': 'host', 'reason': 'explicit_scoring_recovery_pending_learning'}
        cp.advance('task_execution_finished', trace=trace, learning_workspace=learning_workspace,
                   executor_finished_workspace=receipt, recovery_id=report['recovery_id'])
        if system.bank.digest() != report['bank_digest']: raise RuntimeError('Scoring recovery changed Bank')
    finally: system.close()
    # Remap only runtime artifacts/configs; inherited request bodies retain original paid identity.
    def remap(v):
        if isinstance(v, dict): return {k:remap(x) for k,x in v.items()}
        if isinstance(v, list): return [remap(x) for x in v]
        return str(output)+v[len(str(staging)):] if isinstance(v, str) and (v == str(staging) or v.startswith(str(staging)+'/')) else v
    config = remap(config)
    write_json(staging/'train/config.json', config)
    execution = read(staging/'train/execution_manifest.json')
    execution.update(config=config, code=code_identity())
    write_json(staging/'train/execution_manifest.json', execution)
    cp.advance('task_execution_finished', **remap({k:v for k,v in cp.state.items() if k not in {'stage','schema'}}))
    snapshot = cp.root/'executor_finished_workspace/snapshot.json'
    write_json(snapshot, cp.state['executor_finished_workspace'])
    if (cp.root/'learning_workspace/snapshot.json').exists(): write_json(cp.root/'learning_workspace/snapshot.json', cp.state['learning_workspace'])
    write_json(staging/'train/traces'/(task_id+'.json'), cp.state['trace'])
    events_path = staging/'episodes.jsonl'
    if events_path.exists():
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
        event_id = 'recovery:'+report['recovery_id']
        if not any(e['event_id'] == event_id for e in events):
            original = next(e for e in reversed(events) if e['task_id'] == task_id)
            events.append({**original, 'event_id': event_id, 'supersedes_event_id': original['event_id'],
                'episode_event_type': 'scoring_recovery', 'solve_status': 'completed', 'learning_status': 'deferred',
                'success': score['hard'], 'official_score': score['raw_score'], 'is_recovery': True,
                'new_model_calls': 0, 'task_end_time': utc()})
            events_path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        write_json(staging/'episode_projection.json', episode_projection(events))
    receipt = {**report, 'scored_task_id': task_id, 'score': score, 'scoring_audit': adapter.score_audit,
        'completed_tasks': len(report['prefix']), 'learning_status': 'deferred', 'task_committed': False,
        'status': 'scored_pending_learning', 'new_model_calls': 0, 'new_train_examples_consumed': 0,
        'path_mapping': {str(source): str(output)}, 'time': utc()}
    write_json(staging/'continuation_manifest.json', receipt)
    write_json(staging/'completion.json', {'status': 'scored_pending_learning', 'is_recovery': True})
    manifest = read(staging/'run_manifest.json')
    manifest['parent_identity'] = deepcopy(manifest.get('identity', {}))
    manifest['identity'] = {**manifest.get('identity', {}),
        'source_commit': report['child_code']['git_sha'], 'source_sha256': report['child_code']['source_sha256'],
        'implementation_revision': config['experiment']['implementation_revision'],
        'scorer_version': adapter.score_audit['scorer_version'], 'recovery_id': report['recovery_id'],
        'initial_bank': 'inherited_scoring_recovery'}
    write_json(staging/'resolved_config.json', {'train': config})
    manifest.update(run_id=report['recovery_id'], run_status='scored_pending_learning', is_recovery=True,
        parent_run_id=report['parent_run_id'], child_source=report['child_code'], automatic_formal_resume=False)
    manifest['config_hash'] = digest({'train': config})
    write_json(staging/'run_manifest.json', manifest)
    if tree_identity(source) != report['source_tree']: raise RuntimeError('Recovery changed its source')
    write_json(staging/'recovery_receipt.json', receipt)
    staging.rename(output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(recover(args.source_run, args.output, args.task_id, dry_run=args.dry_run), ensure_ascii=False))


if __name__ == '__main__': main()
