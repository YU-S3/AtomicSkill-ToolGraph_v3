"""One fixed LiveMath guidance diagnostic: empty Bank, at most eight physical HTTPs."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from ..empirical import POLICY_DEFAULTS, GUIDANCE_POLICY_VERSION
from ..empirical.bank_view import BankView
from ..empirical.budget_governor import BudgetGovernor
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import PublicTask, digest
from ..empirical.system import EmpiricalSystem, validate_config
from skillcompiler_bench_contracts import source_identity
from skillcompiler_bench_contracts.livemath import CHOICE_PROJECTION_VERSION, CHOICE_SEED
from .formal_log import utc
from .run_empirical import code_identity, load_env, write_json
from .canonical_manifest import sha256

PAIRS = [('H', 'livemath:202602:28', 16, 'livemath:202602:29', 19, ['on','off']),
         ('F', 'livemath:202602:38', 30, 'livemath:202511:15', 44, ['off','on'])]
LIMITS = {'token_limit': 500000, 'finish_reserve': 0, 'request_limit': 8}


def read(path): return json.loads(Path(path).read_text())


def prepare(source_run, datasets, output):
    source_run, datasets, output = map(lambda p: Path(p).resolve(), (source_run,datasets,output))
    if output.exists(): raise ValueError('Diagnostic output must be fresh; never replay a paid batch')
    code = code_identity()
    if not code['git_sha'] or code['tracked_dirty']: raise ValueError('Commit the diagnostic source before execution')
    materialized = read(datasets/'materialization.json')
    if materialized.get('livemath_choice_projection_version') != CHOICE_PROJECTION_VERSION or materialized.get('livemath_choice_seed') != CHOICE_SEED:
        raise ValueError('Diagnostic requires the fixed new choice projection')
    for name, expected in materialized['files_sha256'].items():
        if sha256(datasets/name) != expected: raise ValueError('Materialized input changed: '+name)
    original = read(source_run/'train/execution_manifest.json')['tasks']
    public = {t['task_id']:t for t in read(datasets/'livemath/train.json')['tasks']}
    declaration = []
    for pair, source, si, target, ti, arms in PAIRS:
        selected = []
        for task_id, index in [(source,si),(target,ti)]:
            if original[index-1]['task_id'] != task_id: raise ValueError('Declared source/transfer order differs')
            task = public[task_id]
            if task['physical_key'] != original[index-1]['physical_key'] or task['split'] != 'train':
                raise ValueError('Declared physical Train identity differs')
            selected.append(deepcopy(task))
        if selected[0]['physical_key'] == selected[1]['physical_key']: raise ValueError('Transfer must be independent')
        declaration.append({'pair_id':pair, 'source':selected[0], 'transfer':selected[1], 'arms':arms,
                            'original_order_indices':[si,ti]})
    base = read(source_run/'train/config.json')
    for section, defaults in POLICY_DEFAULTS.items(): base[section] = {**base.get(section, {}), **defaults}
    base['data_dir'] = str(output/'train/bank')
    base['harness'] = {'adapter':'livemath', 'evaluator_records':str(datasets/'livemath/evaluator_records_train.json')}
    base['manifest'] = str(datasets/'livemath/train.json')
    base['experiment'].update(seed=42, runtime_mode='online', output_dir=str(output/'train'),
        benchmark_contracts_sha256=source_identity(), public_materialization_sha256=sha256(datasets/'materialization.json'))
    base['llm']['max_retries'] = 0
    base['llm'].setdefault('purpose_overrides', {})['guidance_learning'] = {
        'protocol':{'thinking_type':'disabled'}, 'max_completion_tokens':4096}
    base['learning']['guidance_repair_limit'] = 0
    base = validate_config(base)
    runtime = {**base['llm'], **base['llm'].get('runtime', {})}
    if runtime['max_completion_tokens'] != 32768 or base['llm']['protocol']['thinking_type'] != 'enabled':
        raise ValueError('Keep the declared Runtime thinking/cap')
    manifest = {'schema':'cf4-r3.diagnostic.v1', 'code':code, 'created_at':utc(), 'run_seed':42,
        'generation_seed':42 if base['llm'].get('generation_seed_supported') else None,
        'generation_seed_status':'supported' if base['llm'].get('generation_seed_supported') else 'unsupported',
        'initial_bank':'fresh empty', 'choice_projection_version':CHOICE_PROJECTION_VERSION, 'choice_seed':CHOICE_SEED,
        'guidance_policy_version':GUIDANCE_POLICY_VERSION, 'budget':LIMITS, 'config_hash':digest(base),
        'public_materialization_sha256':sha256(datasets/'materialization.json'), 'pairs':declaration,
        'stop':'one fixed batch; no Val/Test, replacements, retries, sweep or formal resume'}
    write_json(output/'diagnostic_manifest.json', manifest)
    write_json(output/'config.json', base)
    return base, manifest


def episode(system, task, root, identity, *, learn):
    root.mkdir(parents=True, exist_ok=False)
    task = PublicTask(**task)
    system.checkpoint = TaskCheckpoint(root/'checkpoint')
    system.audit_path = root/'requests.json'
    system.request_attribution = dict(identity)
    write_json(root/'identity.json', {**identity, 'task':asdict(task), 'config_hash':digest(system.config)})
    system.checkpoint.advance('task_started', diagnostic_identity=identity)
    try:
        trace = system.run_task(task, learn=learn, attempt_id=identity['attempt_id'])
        write_json(root/'trace.json', trace)
        system.checkpoint.advance('task_committed', trace=trace)
        return {'status':'completed', 'score':trace['score'], 'execution':trace['execution'],
            'learning':trace.get('learning'), 'learning_status':trace['learning_status'],
            'guidance_retrieval_audit':trace.get('guidance_retrieval_audit', []),
            'tokens':sum(u['total_tokens'] for u in trace['usage']),
            'bank_before':trace['knowledge_before'], 'bank_after':trace['knowledge_after']}
    except Exception as exc:
        value = {'status':'error', 'error':{'type':type(exc).__name__, 'code':getattr(exc,'code',None), 'message':str(exc)},
            'checkpoint_stage':system.checkpoint.state['stage'], 'bank_after':system.bank.digest()}
        write_json(root/'error.json', value)
        return value


def run(source_run, datasets, output, env_file=None):
    base, manifest = prepare(source_run,datasets,output)
    output = Path(output).resolve()
    if env_file: load_env(env_file)
    governor = BudgetGovernor(output/'http_ledger.json', **LIMITS)
    system = EmpiricalSystem(base, budget_governor=governor)
    result = {'schema':'cf4-r3.diagnostic.result.v1','start':utc(),'pairs':[], 'status':'running'}
    try:
        if any(system.bank.all(kind) for kind in ('skill','program','workflow','implementation')) or system.bank.train_cases():
            raise ValueError('Diagnostic Bank must be empty')
        for declared in manifest['pairs']:
            pair = declared['pair_id']
            row = {'pair_id':pair,'arms':{}, 'source':episode(system,declared['source'],output/pair/'source',
                {'pair_id':pair,'arm':'source','attempt_id':pair+'-source','diagnostic_phase':'source',
                 'projection_version':CHOICE_PROJECTION_VERSION,'learning_policy':GUIDANCE_POLICY_VERSION},learn=True)}
            result['pairs'].append(row)
            learning = row['source'].get('learning') or {}
            guide = learning.get('persisted_skill_id') or learning.get('reused_skill_id')
            if governor.state['unknown_billing']:
                row['transfer_status']='skipped_unknown_billing'; result['status']='stopped_unknown_billing'; break
            if row['source'].get('error', {}).get('code') == 'diagnostic_budget_exhausted':
                row['transfer_status']='skipped_batch_limit'; result['status']='stopped_budget_limit'; break
            if not guide:
                row['transfer_status']='skipped_no_source_guidance'
                write_json(output/'result.json', result)
                continue
            frozen = output/pair/'frozen_bank'
            snapshot = system.bank.freeze(frozen)
            row['snapshot'] = snapshot
            for arm in declared['arms']:
                config = deepcopy(base)
                config['data_dir'] = str(frozen)
                config['experiment'].update(runtime_mode='frozen', output_dir=str(output/pair/arm))
                readonly = EmpiricalSystem(config, budget_governor=governor,
                    bank_view_factory=(lambda b:BankView(b,'guidance_off')) if arm=='off' else None)
                try:
                    if readonly.bank.digest() != snapshot['digest']: raise ValueError('Frozen snapshot differs')
                    identity = {'pair_id':pair,'arm':arm,'attempt_id':pair+'-'+arm,'diagnostic_phase':'transfer',
                        'snapshot_digest':snapshot['digest'],'projection_version':CHOICE_PROJECTION_VERSION,
                        'learning_policy':GUIDANCE_POLICY_VERSION,'view':'guidance_off' if arm=='off' else 'original_retrieval'}
                    row['arms'][arm] = episode(readonly, declared['transfer'],output/pair/arm,identity,learn=False)
                    if readonly.bank.digest() != snapshot['digest']: raise RuntimeError('Frozen Bank changed')
                finally: readonly.close()
                write_json(output/'result.json', result)
                if governor.state['unknown_billing'] or len(governor.state['attempts']) >= LIMITS['request_limit'] or row['arms'][arm].get('error', {}).get('code') == 'diagnostic_budget_exhausted': break
            if len(row['arms'])==2 and all(a['status']=='completed' for a in row['arms'].values()):
                row['outcome_off_to_on'] = ('correct' if row['arms']['off']['score']['hard'] else 'wrong')+'→'+(
                    'correct' if row['arms']['on']['score']['hard'] else 'wrong')
            row['transfer_status'] = 'completed' if len(row['arms'])==2 else 'stopped'
            if governor.state['unknown_billing']:
                result['status']='stopped_unknown_billing'; break
            if any(a.get('error', {}).get('code') == 'diagnostic_budget_exhausted' for a in row['arms'].values()):
                result['status']='stopped_budget_limit'; break
            if len(governor.state['attempts']) >= LIMITS['request_limit']:
                result['status']='stopped_request_limit'; break
        else: result['status']='completed_fixed_batch'
    finally:
        result.update(end=utc(), http_attempts=len(governor.state['attempts']), unknown_billing=governor.state['unknown_billing'],
            tokens=sum(a.get('accounted_tokens',0) for a in governor.state['attempts'].values()),
            final_bank_digest=system.bank.digest(), method_claim='local mechanism evidence only')
        write_json(output/'result.json', result)
        system.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',required=True)
    parser.add_argument('--datasets',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--env-file')
    args=parser.parse_args()
    print(json.dumps(run(args.source_run,args.datasets,args.output,args.env_file),ensure_ascii=False))


if __name__=='__main__': main()
