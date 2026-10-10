"""One preregistered LM1 recompile and interleaved Val batch; no source Runtime replay."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from ..core.errors import BudgetExhausted
from ..empirical import (CHOICE_GUIDANCE_POLICY_VERSION, CHOICE_GUIDANCE_MATERIAL_VERSION,
                         CHOICE_GUIDANCE_SELECTION_VERSION, SINGLE_ANSWER_PROMPT_VERSION)
from ..empirical.bank import Bank
from ..empirical.bank_view import BankView
from ..empirical.budget_governor import BudgetGovernor
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.choice_guidance import build_verified_public_source, valid_choices, checked_asset
from ..empirical.contracts import PublicTask, digest
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.benchmarks import AnswerAdapter
from .canonical_manifest import sha256
from .formal_log import utc
from .run_empirical import code_identity, load_env, write_json

LIMITS = {'token_limit': 2500000, 'finish_reserve': 0, 'request_limit': 87, 'proposal_repair_limit': 4}
ARM_RULES = {'A': {'guidance': 'off', 'effort': 'high'}, 'B': {'guidance': 'on', 'effort': 'high'},
             'C': {'guidance': 'on', 'effort': 'low'}}
MODE = 'offline_recompile_from_completed_train'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def experience(row):
    # No evaluator record, gold, original guidance or inherited usage enters learning materials.
    return {'submission': row['raw_final_content'], 'score': row['original_score'],
            'provider_finish_reason': row['finish_reason'],
            **{k: row[k] for k in ('answer_status', 'empty_answer', 'completion_truncated')}}


def prepare(review_root, base_config, output):
    review, output = Path(review_root).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('prepare requires a fresh output directory; paid outputs cannot be replaced')
    code = code_identity()
    if not code['git_sha'] or code['tracked_dirty']: raise ValueError('Commit the validation source first')
    paths = ['train/train60_learning_summary.jsonl', 'resources/public/train.json', 'resources/public/val.json',
             'resources/private/evaluator_records_val.json', 'identity/original/run_manifest.json',
             'identity/original/train/execution_manifest.json', 'identity/actual_run_identity.json',
             'resources/projection_integrity_original.json', 'identity/original/val/execution_manifest.json']
    inventory = read(review/'SHA256SUMS.json')
    # The review package declares every source file before deriving anything.
    if isinstance(inventory, dict) and 'files' in inventory: inventory = inventory['files']
    if not isinstance(inventory, dict): raise ValueError('Unsupported review checksum manifest')
    hashes = {}
    for relative in paths:
        actual = sha256(review/relative)
        expected = inventory.get(relative)
        if isinstance(expected, dict): expected = expected.get('sha256')
        if actual != expected: raise ValueError('Review input hash mismatch: ' + relative)
        hashes[str(review/relative)] = actual
    base_path = Path(base_config).resolve()
    hashes[str(base_path)] = sha256(base_path)
    base = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    rows = lines(review/paths[0])
    public = {t['task_id']: t for t in read(review/paths[1])['tasks']}
    original = read(review/'identity/original/train/execution_manifest.json')['tasks']
    source_run = read(review/'identity/original/run_manifest.json')['run_id']
    projection_record = read(review/'resources/projection_integrity_original.json')
    projection = projection_record['choice_projection_version']
    if projection != 'livemath.choices.v1' or projection_record['choice_seed'] != 42:
        raise ValueError('Source choice projection identity differs')
    identity = read(review/'identity/actual_run_identity.json')
    sources = []
    for index, row in enumerate(rows):
        task = public[row['task_id']]
        if (row['split_order_index'] != index or task['physical_key'] != row['physical_key'] or
            original[index]['task_id'] != row['task_id'] or original[index]['physical_key'] != row['physical_key']):
            raise ValueError('Original Train order or physical identity changed')
        bound = {'source_run_id': source_run, 'source_record_hash': digest(row), 'projection_version': projection,
                 'source_order_index': index, 'source_record_path': str(review/paths[0])}
        source = build_verified_public_source(PublicTask(**task), experience(row), bound)
        sources.append({'task': task, 'experience': experience(row), 'identity': bound, 'host_source': source})
    val = read(review/'resources/public/val.json')['tasks']
    original_val = read(review/'identity/original/val/execution_manifest.json')['tasks']
    if val != original_val: raise ValueError('Original fixed Val public tasks or order changed')
    if len(sources) != 60 or sum(s['host_source']['eligible'] for s in sources) != 16 or len(val) != 17:
        raise ValueError('LM1 fixes exactly Train60/eligible16 and Val17')
    train_keys = {s['task']['physical_key'] for s in sources}
    if (not all(s['task']['split'] == 'train' for s in sources) or len(train_keys) != 60 or
        len({t['physical_key'] for t in val}) != 17 or train_keys & {t['physical_key'] for t in val}):
        raise ValueError('Physical split boundary differs')
    if not all(t['split'] == 'val' and valid_choices(PublicTask(**t)) for t in val): raise ValueError('Invalid public Val choices')
    base['data_dir'] = str(output/'train/bank')
    base['experiment'].update(seed=42, runtime_mode='online', output_dir=str(output/'train'),
                             source_run_id=source_run, choice_projection_version=projection)
    base['harness'] = {'adapter': 'livemath', 'evaluator_records': str(review/'resources/private/evaluator_records_val.json')}
    base['manifest'] = str(review/'resources/public/train.json')
    base['learning'].update(guidance_repair_limit=1, choice_guidance={
        'enabled': True, 'policy_version': CHOICE_GUIDANCE_POLICY_VERSION,
        'material_version': CHOICE_GUIDANCE_MATERIAL_VERSION, 'max_related_assets': 2, 'source_check_required': True})
    base['runtime']['choice_guidance'] = {'enabled': True, 'selection_version': CHOICE_GUIDANCE_SELECTION_VERSION,
                                        'prompt_version': SINGLE_ANSWER_PROMPT_VERSION, 'max_items': 2, 'max_total_chars': 2400}
    base['llm']['max_retries'] = 0
    base['llm']['runtime'].update(reasoning_effort='high', max_completion_tokens=32768)
    base['llm']['protocol']['thinking_type'] = 'enabled'
    overrides = base['llm'].setdefault('purpose_overrides', {})
    for purpose, cap in [('guidance_learning', 2048), ('guidance_grounding', 1536)]:
        overrides[purpose] = {'protocol': {'thinking_type': 'disabled'}, 'max_completion_tokens': cap}
    base = validate_config(base)
    write_json(output/'config.json', base)
    write_json(output/'source_records.json', sources)
    write_json(output/'val_tasks.json', val)
    hashes.update({str(output/name): sha256(output/name) for name in ['config.json', 'source_records.json', 'val_tasks.json']})
    manifest = {'schema': 'livemath.lm1.fixed-validation.v1', 'created_at': utc(), 'code': code,
        'output': str(output), 'input_hashes': hashes, 'config_hash': digest(base), 'source_run_id': source_run,
        'source_records_hash': digest(sources), 'derivation_mode': MODE, 'initial_bank': 'fresh empty',
        'run_seed': 42, 'choice_seed': 42, 'projection_version': projection,
        'public_projection_hash': digest({'train': list(public.values()), 'val': val}),
        'source_frozen_identity': identity.get('frozen_manifest'),
        'parent_train_runtime_tokens': 1247149, 'parent_total_train_tokens': 1411603,
        'budget': LIMITS, 'arms': ARM_RULES, 'val_ids': [t['task_id'] for t in val],
        'order': [{'task_id': t['task_id'], 'arms': list('ABC'[i % 3:] + 'ABC'[:i % 3])} for i, t in enumerate(val)],
        'no_exposure_branch': 'A guidance_off/high and C guidance_off/low only; no method comparison',
        'low_selection_rule': 'tokens <= 0.75 * high, correct >= high, empty <= high; otherwise retain high',
        'request_model_alias': base['llm']['model'], 'endpoint': base['llm']['base_url'].rstrip('/') +
            ('' if base['llm']['base_url'].rstrip('/').endswith('/chat/completions') else '/chat/completions'),
        'response_models': [], 'generation_seed_status': 'unsupported', 'generation_seed': None,
        'provider_serving_statement': {'checked_date': '2026-10-10', 'served_model': 'DeepSeek-V4.1-Flash',
            'source': 'https://api-docs.deepseek.com/quick_start/pricing-details-cny/',
            'note': 'Official mapping of legacy alias; remote weight snapshot is unavailable'},
        'stop': 'one fixed batch; no Test, extra pilot, replacement, formal resume, or budget increase'}
    write_json(output/'manifest.json', manifest)
    return manifest


def check_offline(manifest_path):
    manifest = read(manifest_path)
    if manifest['schema'] != 'livemath.lm1.fixed-validation.v1' or manifest['budget'] != LIMITS or manifest['arms'] != ARM_RULES:
        raise ValueError('Fixed batch policy changed')
    code = code_identity()
    if code != manifest['code'] or code['tracked_dirty']: raise ValueError('Pinned diagnostic code changed')
    for name, expected in manifest['input_hashes'].items():
        if sha256(Path(name)) != expected: raise ValueError('Bound input changed: ' + name)
    output = Path(manifest['output'])
    config = validate_config(read(output/'config.json'))
    if digest(config) != manifest['config_hash']: raise ValueError('Resolved config changed')
    if config['llm']['max_retries'] != 0 or config['llm'].get('generation_seed_supported'):
        raise ValueError('No retries or unsupported generation seed may enter the batch')
    for purpose, cap in [('guidance_learning', 2048), ('guidance_grounding', 1536)]:
        settings = config['llm']['purpose_overrides'][purpose]
        if settings['max_completion_tokens'] != cap or settings['protocol']['thinking_type'] != 'disabled':
            raise ValueError('Learning call settings differ')
    if config['llm']['runtime']['max_completion_tokens'] != 32768 or config['llm']['runtime']['reasoning_effort'] != 'high':
        raise ValueError('Runtime call settings differ')
    sources, val = read(output/'source_records.json'), read(output/'val_tasks.json')
    if digest(sources) != manifest['source_records_hash'] or [t['task_id'] for t in val] != manifest['val_ids']:
        raise ValueError('Source or Val identity changed')
    replay = [build_verified_public_source(PublicTask(**s['task']), s['experience'], s['identity']) for s in sources]
    if replay != [s['host_source'] for s in sources] or len(replay) != 60 or sum(s['eligible'] for s in replay) != 16:
        raise ValueError('Host source binding differs')
    expected_order = [{'task_id': t['task_id'], 'arms': list('ABC'[i % 3:] + 'ABC'[:i % 3])} for i, t in enumerate(val)]
    if len(val) != 17 or manifest['order'] != expected_order: raise ValueError('Interleaved order changed')
    report = {'status': 'passed', 'new_http_calls': 0, 'manifest_hash': digest(manifest), 'config_hash': digest(config),
              'source_count': 60, 'eligible': 16, 'host_skip': 44, 'val_count': 17, 'limits': LIMITS}
    write_json(output/'offline_check.json', report)
    return report


def guard(governor):
    if governor.state['unknown_billing']:
        raise BudgetExhausted('diagnostic_unknown_billing', 'Whole batch stopped: paid or pending request requires review')
    if sum(a.get('accounted_tokens', a['reserved_tokens']) for a in governor.state['attempts'].values()) > LIMITS['token_limit']:
        raise BudgetExhausted('diagnostic_budget_exhausted', 'Whole batch stopped: metered tokens exceed budget')


def episode(system, task, root, identity):
    root.mkdir(parents=True, exist_ok=True)
    result_path = root/'trace.json'
    if result_path.exists(): return read(result_path)
    task = PublicTask(**task)
    system.checkpoint = TaskCheckpoint(root/'checkpoint')
    system.audit_path = root/'requests.json'
    system.request_attribution = identity
    write_json(root/'identity.json', {'task': asdict(task), 'config_hash': digest(system.config), **identity})
    try:
        trace = system.run_task(task, learn=False, attempt_id=identity['attempt_id'])
    except BudgetExhausted:
        raise
    except Exception as exc:
        guard(system.budget_governor)
        # One physical answer only. Never replace an error, timeout, or malformed answer.
        trace = {'task': asdict(task), 'score': {'hard': False}, 'error': {'type': type(exc).__name__, 'message': str(exc)},
                 'execution': {'prediction': None, 'empty_answer': True, 'reason': 'single_answer_error'},
                 'knowledge_before': system.bank.digest(), 'knowledge_after': system.bank.digest(),
                 **(read(system.audit_path) if system.audit_path.exists() else {'requests': [], 'usage': []})}
    guard(system.budget_governor)
    write_json(result_path, trace)
    system.checkpoint.advance('task_committed', trace=trace)
    return trace


def summarize(output, result):
    arms = {}
    val = read(output/'val_tasks.json')
    for arm in 'ABC':
        traces = [read(p) for p in sorted((output/'val'/arm).glob('*/trace.json'))]
        arms[arm] = {'completed': len(traces), 'correct': sum(t['score']['hard'] for t in traces),
                     'empty': sum(bool(t['execution'].get('empty_answer')) for t in traces),
                     'tokens': sum(u['total_tokens'] for t in traces for u in t.get('usage', [])),
                     'exposed_tasks': sum(bool(t.get('injected_guidance_ids')) for t in traces)}
    high_arm = 'A' if result.get('no_method_exposure') else 'B'
    high, low = arms[high_arm], arms['C']
    cost_complete = high['completed'] == low['completed'] == 17
    low_selected = (cost_complete and low['tokens'] <= .75 * high['tokens'] and low['correct'] >= high['correct'] and low['empty'] <= high['empty'])
    paired, strata = [], {'meta_option': [], 'ordinary': []}
    for i, task in enumerate(val):
        ap, bp = output/'val/A'/str(i)/'trace.json', output/'val/B'/str(i)/'trace.json'
        if ap.exists() and bp.exists():
            a, b = read(ap), read(bp)
            row = {'task_id': task['task_id'], 'A_correct': a['score']['hard'], 'B_correct': b['score']['hard'],
                   'B_guidance_ids': b.get('injected_guidance_ids', [])}
            paired.append(row)
            # Descriptive reporting only; never used in selection, prompt or method choice.
            import re
            meta = any(re.search(r'\b(?:option|statement|conclusion)s?\b', c['text'], re.I) for c in task['inputs']['choices'])
            strata['meta_option' if meta else 'ordinary'].append(row)
    b_only = sum(r['B_correct'] and not r['A_correct'] for r in paired)
    a_only = sum(r['A_correct'] and not r['B_correct'] for r in paired)
    exposed_gain = any(r['B_correct'] and not r['A_correct'] and r['B_guidance_ids'] for r in paired)
    signal = ('no_effective_method_intervention' if result.get('no_method_exposure') else
              'incomplete_fixed_batch' if len(paired) != 17 else
              'positive_small_sample_signal' if b_only > a_only and exposed_gain else
              'degradation' if b_only < a_only else 'no_observed_net_benefit')
    comparison = 'not_run_no_exposure' if result.get('no_method_exposure') else 'complete' if len(paired) == 17 else 'incomplete'
    return {'arms': arms, 'method_comparison_status': comparison, 'paired_n': len(paired),
            'B_only_correct': b_only if comparison == 'complete' else None,
            'A_only_correct': a_only if comparison == 'complete' else None,
            'net_gain': b_only - a_only if comparison == 'complete' else None,
            'observed_pairs': {'B_only_correct': b_only, 'A_only_correct': a_only},
            'paired': paired, 'descriptive_strata': strata, 'method_signal': signal,
            'cost_reference_arm': high_arm, 'cost_comparison_complete': cost_complete,
            'candidate_effort': 'low' if low_selected else 'high',
            'interpretation': 'Fixed Val17 diagnostic; not statistical superiority/equivalence or fresh online Train'}


def run(manifest_path, env_file):
    check_offline(manifest_path)
    manifest = read(manifest_path)
    output, batch_hash = Path(manifest['output']), digest(manifest)
    binding = output/'batch_binding.json'
    if binding.exists():
        if read(binding)['manifest_hash'] != batch_hash: raise ValueError('Paid output manifest changed')
    else:
        if (output/'budget.json').exists() or (output/'train/bank').exists(): raise ValueError('Unbound paid directory')
        write_json(binding, {'manifest_hash': batch_hash, 'started_at': utc()})
    result_path = output/'result.json'
    result = read(result_path) if result_path.exists() else {'status': 'running', 'train': [], 'val': [], 'manifest_hash': batch_hash}
    if result['status'] == 'stopped_user_interrupt': result['status'] = 'running'
    if result['status'] != 'running': return result
    load_env(env_file)
    governor = BudgetGovernor(output/'budget.json', **LIMITS)
    base = read(output/'config.json')
    system = None
    try:
        guard(governor)
        system = EmpiricalSystem(base, harness=AnswerAdapter('livemath', {}), budget_governor=governor)
        sources = read(output/'source_records.json')
        for i, source in enumerate(sources):
            root = output/'train/events'/str(i)
            system.checkpoint, system.audit_path = TaskCheckpoint(root/'checkpoint'), root/'requests.json'
            event = system.learn_from_completed_record(PublicTask(**source['task']), source['experience'], source['identity'])
            guard(governor)
            write_json(root/'learning.json', event)
            row = {'index': i, 'task_id': source['task']['task_id'], 'source_record_hash': source['identity']['source_record_hash'],
                   'status': event['learning_status'], 'learning': event['learning'],
                   'incremental_tokens': sum(u['total_tokens'] for u in event['usage'])}
            result['train'] = [*result['train'][:i], row]
            write_json(result_path, result)
        frozen_path = output/'train/frozen_bank'
        if frozen_path.exists():
            freeze = read(frozen_path/'freeze.json')
            if freeze['source_digest'] != system.bank.digest(): raise ValueError('Frozen source identity mismatch')
        else: freeze = system.bank.freeze(frozen_path)
        result['frozen'] = {'freeze_time': result.get('frozen', {}).get('freeze_time') or utc(), **freeze}
        system.close(); system = None
        frozen = Bank(frozen_path, readonly=True, seed=42)
        try:
            if frozen.digest() != freeze['digest']: raise ValueError('Frozen digest mismatch')
            val = read(output/'val_tasks.json')
            selections = [frozen.select_guidance(PublicTask(**t), base['runtime']['choice_guidance']) for t in val]
            result['no_method_exposure'] = not any(s['selected'] for s in selections)
            result['qualified_assets'] = sum(checked_asset(a) for a in frozen.all('skill'))
            write_json(output/'scope_precheck.json', {'frozen_digest': frozen.digest(), 'tasks': selections,
                       'no_method_exposure': result['no_method_exposure']})
        finally: frozen.close()
        write_json(result_path, result)
        for i, (task, order) in enumerate(zip(val, manifest['order'])):
            for arm in order['arms']:
                if result['no_method_exposure'] and arm == 'B': continue
                cfg = deepcopy(base)
                off = arm == 'A' or result['no_method_exposure']
                cfg['data_dir'] = str(frozen_path)
                cfg['experiment'].update(runtime_mode='frozen', output_dir=str(output/'val'/arm))
                cfg['manifest'] = str(output/'val_tasks.json')
                cfg['llm']['runtime']['reasoning_effort'] = ARM_RULES[arm]['effort']
                write_json(output/'val'/arm/'config.json', cfg)
                system = EmpiricalSystem(cfg, budget_governor=governor,
                    bank_view_factory=(lambda b: BankView(b, 'guidance_off')) if off else None)
                try:
                    before = system.bank.digest()
                    if before != freeze['digest']: raise ValueError('Arm Frozen identity mismatch')
                    trace = episode(system, task, output/'val'/arm/str(i), {'arm': arm, 'guidance': 'off' if off else 'on',
                        'effort': ARM_RULES[arm]['effort'], 'attempt_id': 'LM1-' + arm + '-' + str(i),
                        'projection_version': manifest['projection_version'], 'batch_manifest_hash': batch_hash})
                    after = system.bank.digest()
                    if after != before: raise RuntimeError('Frozen changed during evaluation')
                    key = arm + ':' + str(i)
                    row = {'key': key, 'task_id': task['task_id'], 'arm': arm, 'guidance': 'off' if off else 'on',
                           'score': trace['score'], 'frozen_before': before, 'frozen_after': after,
                           'injected_guidance_ids': trace.get('injected_guidance_ids', [])}
                    result['val'] = [r for r in result['val'] if r['key'] != key] + [row]
                    write_json(result_path, result)
                finally: system.close(); system = None
        result['status'] = 'completed_fixed_batch'
    except (BudgetExhausted, KeyboardInterrupt) as exc:
        result.update(status='stopped_' + (getattr(exc, 'code', None) or 'user_interrupt'),
                      error={'type': type(exc).__name__, 'message': str(exc)})
    except Exception as exc:
        result.update(status='stopped_unknown_billing' if governor.state['unknown_billing'] else 'stopped_engineering',
                      error={'type': type(exc).__name__, 'message': str(exc)})
        if not governor.state['unknown_billing']: raise
    finally:
        if system: system.close()
        audits = [read(p) for p in output.glob('train/events/*/requests.json')] + [read(p) for p in output.glob('val/*/*/requests.json')]
        response_models = sorted({u['provider_metadata']['response_model'] for a in audits for u in a['usage']
                                  if u.get('provider_metadata', {}).get('response_model')})
        result.update(ended_at=utc(), http_attempts=len(governor.state['attempts']),
                      unknown_billing=governor.state['unknown_billing'], response_models=response_models,
                      incremental_tokens=sum(a.get('accounted_tokens', 0) for a in governor.state['attempts'].values()))
        result['incremental_recompile_tokens'] = sum(u['total_tokens'] for p in output.glob('train/events/*/requests.json') for u in read(p)['usage'])
        result.update(parent_train_runtime_tokens=manifest['parent_train_runtime_tokens'],
                      parent_total_train_tokens=manifest['parent_total_train_tokens'], derivation_mode=MODE)
        result['decision'] = summarize(output, result)
        write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare')
    for name in ('review-root', 'base-config', 'output'): p.add_argument('--' + name, required=True)
    p = subs.add_parser('check-offline'); p.add_argument('--manifest', required=True)
    p = subs.add_parser('run'); p.add_argument('--manifest', required=True); p.add_argument('--env-file', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': value = prepare(args.review_root, args.base_config, args.output)
    elif args.command == 'check-offline': value = check_offline(args.manifest)
    else: value = run(args.manifest, args.env_file)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__': main()
