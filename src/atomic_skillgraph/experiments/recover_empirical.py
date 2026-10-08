"""Explicit, zero-provider recovery child for saved solve/failed learning boundaries."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3

from ..empirical import IMPLEMENTATION_REVISION
from ..empirical.bank import Bank
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import digest
from ..empirical.program_submission import ProgramContractError, validate_program_declaration
from ..empirical.system import validate_config
from .episode_projection import episode_projection, terminal_episode
from .formal_log import tree_identity, utc
from .run_empirical import code_identity, write_json, commit_trace


def read(path): return json.loads(Path(path).read_text())


def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def inspect_source(source, task_id):
    source = Path(source).resolve()
    if (source/'test').exists(): raise ValueError('Recovery source must not have started Test')
    manifest, execution = read(source/'run_manifest.json'), read(source/'train/execution_manifest.json')
    database = sqlite3.connect((source/'train/run.sqlite3').as_uri() + '?mode=ro', uri=True)
    try: stored = database.execute('SELECT id,attempts,status,result FROM tasks ORDER BY rowid').fetchall()
    finally: database.close()
    pending = [r for r in stored if r[2] != 'completed']
    if len(pending) != 1 or pending[0][0] != task_id: raise ValueError('Expected one saved pending boundary')
    attempt = pending[0][1]
    state_path = source/'train/checkpoints'/task_id/str(attempt)/'state.json'
    state = read(state_path)
    if state['stage'] != 'task_execution_finished' or not state['trace'].get('score'):
        raise ValueError('Recovery requires a saved independently scored solve')
    prefix = [r[0] for r in stored]
    if prefix != [t['task_id'] for t in execution['tasks'][:len(prefix)]]:
        raise ValueError('Recovery prefix differs from execution identity')
    audit = read(source/'train/requests'/(task_id+'_attempt'+str(attempt)+'.json'))
    request_ids = [r['id'] for r in audit['requests']]
    event_ids = [u['event_id'] for u in audit['usage']]
    if len(set(request_ids)) != len(request_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError('Original requests or usage are duplicated')
    bank = Bank(source/'train/bank', readonly=True, seed=execution['config']['experiment']['seed'])
    try:
        case_keys = [c['task']['physical_key'] for c in bank.train_cases()]
        quarantine = []
        contract = {'final_submission_kind': 'files', 'publication_contract': {
            'supported': True, 'required_files': ['solution.py','case1_result.xlsx']}}
        # This explicit recovery boundary is selected by the caller; no benchmark rule enters the core.
        for job in bank.jobs():
            if any(b['case_id'] == state['trace']['task']['physical_key'] for b in job.get('case_bindings', [])):
                program = bank.get(job.get('program_id', ''))
                if program:
                    try: validate_program_declaration(program, contract)
                    except ProgramContractError as exc:
                        quarantine.append({'program_id': program['id'], 'job_id': job['id'], 'reason': exc.feedback})
        if not quarantine: raise ValueError('No demonstrated illegal pending candidate to quarantine')
        bank_digest = bank.digest()
    finally: bank.close()
    sealed = state['trace'].get('episode_result', {}).get('sealed_prediction')
    if sealed is None:
        sealed = state['trace'].get('episode_result', {}).get('submission')
    bundle = (sealed or {}).get('bundle')
    if not bundle or not Path(bundle).is_dir(): raise ValueError('Saved sealed bundle is unavailable')
    bundle_identity = tree_identity(bundle)
    source_identity = tree_identity(source)
    report = {'parent_run_id': manifest['run_id'], 'source_run': str(source), 'task_id': task_id,
        'attempt': attempt, 'parent_source': execution['code'], 'child_source': code_identity(),
        'parent_bank_digest': bank_digest, 'source_tree': source_identity,
        'checkpoint_sha256': tree_identity(state_path.parent)['sha256'],
        'sealed_bundle': {'path': bundle, **bundle_identity}, 'original_request_ids': request_ids,
        'original_usage_ids': event_ids, 'inherited_task_tokens': sum(u['total_tokens'] for u in audit['usage']),
        'inherited_completed_tasks': [r[0] for r in stored if r[2] == 'completed'],
        'inherited_successes': sum(bool(json.loads(r[3])['score']['hard']) for r in stored if r[2] == 'completed'),
        'learning_case_keys': case_keys, 'quarantine': quarantine,
        'identity': {k:v for k,v in manifest['identity'].items() if k not in ('code',)},
        'saved_score': state['trace']['score'], 'new_model_calls': 0, 'new_train_examples_consumed': 0,
        'actions': ['copy_readonly_source', 'quarantine_declaration', 'commit_saved_score', 'append_episode_revision'],
        'is_recovery': True}
    report['recovery_id'] = digest(report)
    return report


def recover(source, output, task_id, *, dry_run=False, interrupt_after=None):
    report = inspect_source(source, task_id)
    output = Path(output).resolve()
    if dry_run:
        write_json(output, report)
        return report
    if output.exists():
        receipt = read(output/'recovery_receipt.json')
        if receipt['recovery_id'] != report['recovery_id']: raise ValueError('Recovery output identity differs')
        return receipt
    staging = output.with_name(output.name + '.staging-' + report['recovery_id'][:12])
    if not staging.exists():
        shutil.copytree(source, staging)
        write_json(staging/'recovery_dry_run.json', report)
    elif read(staging/'recovery_dry_run.json')['recovery_id'] != report['recovery_id']:
        raise ValueError('Recovery staging identity differs')
    if interrupt_after == 'copy': raise InterruptedError('Recovery fixture: after copy')
    old, new = str(Path(source).resolve()), str(output)
    def remap(value):
        if isinstance(value, dict): return {k:remap(v) for k,v in value.items()}
        if isinstance(value, list): return [remap(v) for v in value]
        return new+value[len(old):] if isinstance(value, str) and (value == old or value.startswith(old+'/')) else value
    lineage = {k:report[k] for k in ('recovery_id','parent_run_id','parent_source','child_source', 'parent_bank_digest')}
    lineage.update(path_mapping={old:new}, recovery_time=utc(), initial_bank='inherited recovery continuation',
                   max_new_tasks=None, automatic_formal_resume=False,
                   inherited_completed_task_ids=report['inherited_completed_tasks'],
                   inherited_task_source=report['parent_source'])
    sealed_destination = staging/'recovery_material/sealed_bundle'
    if not sealed_destination.exists(): shutil.copytree(report['sealed_bundle']['path'], sealed_destination)
    if tree_identity(sealed_destination) != {k:v for k,v in report['sealed_bundle'].items() if k != 'path'}:
        raise ValueError('Copied sealed bundle changed')
    lineage['path_mapping'][report['sealed_bundle']['path']] = str(output/'recovery_material/sealed_bundle')
    for name in ('resolved_config.json','train/config.json','train/execution_manifest.json'):
        value = remap(read(staging/name))
        configurations = value.values() if name == 'resolved_config.json' else [value['config'] if 'config' in value else value]
        for config in configurations:
            config['experiment']['implementation_revision'] = IMPLEMENTATION_REVISION
            validate_config(config)
        if name.endswith('execution_manifest.json'): value['code'] = code_identity()
        write_json(staging/name, value)
    manifest = remap(read(staging/'run_manifest.json'))
    manifest.update(run_id=report['recovery_id'], start_time=utc(), end_time=None, run_status='recovered_paused')
    manifest['identity']['source_commit'] = code_identity()['git_sha']
    manifest['identity']['source_sha256'] = code_identity()['source_sha256']
    manifest['identity']['implementation_revision'] = IMPLEMENTATION_REVISION
    manifest['identity']['initial_bank'] = lineage['initial_bank']
    manifest['identity']['recovery_id'] = report['recovery_id']
    manifest.update(manifest['identity'])
    manifest['config_hash'] = digest(read(staging/'resolved_config.json'))
    write_json(staging/'run_manifest.json', manifest)
    write_json(staging/'continuation_manifest.json', lineage)
    # Do not infer why a parent pause marker exists; preserve it as inherited evidence.
    for marker in staging.rglob('STOP_AFTER_TASK'):
        marker.rename(marker.with_name('INHERITED_STOP_AFTER_TASK'))
    bank = Bank(staging/'train/bank', seed=read(staging/'train/config.json')['experiment']['seed'])
    try:
        for item in report['quarantine']:
            bank.quarantine_program(item['program_id'], item['reason'], report['recovery_id'])
        if [c['task']['physical_key'] for c in bank.train_cases()] != report['learning_case_keys']:
            raise RuntimeError('Recovery changed the learning sample pool')
        child_bank_digest = bank.digest()
    finally: bank.close()
    checkpoint = TaskCheckpoint(staging/'train/checkpoints'/task_id/str(report['attempt']))
    trace = deepcopy(checkpoint.state['trace'])
    trace['episode_result']['sealed_prediction']['bundle'] = str(output/'recovery_material/sealed_bundle')
    trace.update(read(staging/'train/requests'/(task_id+'_attempt'+str(report['attempt'])+'.json')))
    trace.update(solve_status='completed', learning_status='failed_engineering',
        learning_error={'code':'program_submission_kind_mismatch', 'stage':'learning', 'recovery_resolution':'quarantined'},
        knowledge_after=child_bank_digest, is_recovery=True, new_model_calls=0, new_train_examples_consumed=0,
        scoring_audit_unavailable='Original task checkpoint has score, but no saved scoring audit')
    if interrupt_after == 'before_commit': raise InterruptedError('Recovery fixture: before commit')
    db = sqlite3.connect(staging/'train/run.sqlite3')
    commit_trace(staging/'train', task_id, trace, db, checkpoint, recovery_id=report['recovery_id'])
    cases = [json.loads(r[0]) for r in db.execute("SELECT result FROM tasks WHERE status='completed' ORDER BY rowid")]
    db.close()
    path = staging/'episodes.jsonl'
    event_id = 'recovery:'+report['recovery_id']
    episodes = rows(path)
    if not any(r['event_id'] == event_id for r in episodes):
        original = next(r for r in episodes if r['task_id'] == task_id and r.get('success') is None)
        revision = {**original, **terminal_episode(original,trace,utc()), 'event_id':event_id, 'supersedes_event_id':original['event_id'],
            'episode_event_type':'recovery_resolution', 'success':trace['score']['hard'],
            'official_score':trace['score']['raw_score'], 'terminal_reason':trace['execution']['reason'],
            'solve_status':'completed','learning_status':'failed_engineering','is_recovery':True,
            'new_model_calls':0,'new_train_examples_consumed':0,'task_end_time':utc()}
        with path.open('a') as stream: stream.write(json.dumps(revision)+'\n')
    write_json(staging/'episode_projection.json', episode_projection(rows(path)))
    audits=[read(p) for p in (staging/'train/requests').glob('*.json')]
    unique_usage={u['event_id']:u for a in audits for u in a['usage']}
    unknown=sum(t.get('usage') is None for a in audits for r in a['requests'] for t in r.get('http_attempts',[]))
    known_tokens=sum(u['total_tokens'] for u in unique_usage.values())
    write_json(staging/'train/summary.json', {'schema':'empirical.summary.v1','complete':False,
        'expected_tasks':len(read(staging/'train/execution_manifest.json')['tasks']), 'tasks':len(cases),
        'successes':sum(bool(t['score']['hard']) for t in cases), 'status':'recovered_paused','frozen':None,
        'knowledge_digest':child_bank_digest, 'cases':cases,
        'known_total_tokens':known_tokens,'total_tokens':None if unknown else known_tokens,
        'unknown_billing_attempts':unknown,'new_model_calls':0,'new_train_examples_consumed':0,
        'program_invocations':sum(len(c['execution'].get('attempts',[])) for c in cases),
        'completed_task_tokens':sum(u['total_tokens'] for t in cases for u in t.get('usage',[]))})
    receipt = {**lineage, 'task_id':task_id,'committed_tasks':len(cases),
        'successes':sum(bool(t['score']['hard']) for t in cases), 'learning_cases':len(report['learning_case_keys']),
        'inherited_task_tokens':report['inherited_task_tokens'], 'original_request_ids':report['original_request_ids'],
        'child_bank_digest':child_bank_digest,'new_model_calls':0,'new_train_examples_consumed':0,'is_recovery':True}
    write_json(staging/'recovery_receipt.json', receipt)
    write_json(staging/'completion.json', {'status':'recovered_paused','is_recovery':True})
    if tree_identity(source) != report['source_tree']: raise RuntimeError('Recovery source changed')
    if interrupt_after == 'after_commit': raise InterruptedError('Recovery fixture: after commit')
    staging.rename(output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(recover(args.source_run,args.output,args.task_id,dry_run=args.dry_run),ensure_ascii=False))


if __name__ == '__main__': main()
