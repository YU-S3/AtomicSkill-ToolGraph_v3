"""Explicit R3 stages; no Train300 re-solving, implicit paid authorization or Test entry."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from ..core.errors import BudgetExhausted
from ..empirical import POLICY_DEFAULTS
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, input_token_bound
from ..empirical.contracts import PublicTask, digest
from ..empirical.searchqa_guidance_r3 import VERSION, MATERIAL, POLICY, RuntimeView, FAMILIES
from ..empirical.searchqa_trace_learning import verify_source, read, sha
from ..empirical.searchqa_trace_learning_r3 import partition, public_case, learning_units, induce, specification, validate_candidate
from ..empirical import searchqa_program_probe_r3 as probe
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.benchmarks import AnswerAdapter
from .benchmark_config import benchmark_config
from .formal_log import FormalLog, tree_identity, utc
from .run_empirical import code_identity, load_env, run, write_json
from .run_searchqa_trace_r2 import jsonl, rows, unit, model_failure, usage_totals

STAGES = ('verify', 'prepare', 'reuse_spec', 'induce', 'build', 'worker', 'measure_train', 'freeze', 'val', 'report')
PROPOSED = {'D1': {'tokens': 250000, 'http': 12}, 'D2': {'tokens': 750000, 'http': 60},
            'D3a': {'tokens': 2400000, 'http': 50}, 'D3': {'tokens': 5800000, 'http': 110},
            'D4': {'tokens': 7000000, 'http': 150}}


def immutable(path, value):
    path = Path(path)
    if path.exists() and read(path) != value: raise ValueError('Immutable artifact changed: ' + str(path))
    if not path.exists(): write_json(path, value)
    return value


def verify(args):
    root = Path(args.output); identity, cases, old = verify_source(args.source_run, args.datasets)
    immutable(root / 'verification_paths.json', {**{key: str(Path(getattr(args, key)).resolve()) for key in
        ('source_run', 'r2_run', 'datasets')}, 'denied_paths': [str(Path(p).resolve()) for p in args.deny_path]})
    exposed = read(root / 'r2_exposure.json')['source_ids']
    groups = partition([PublicTask(**r['task']) for r in cases], exposed)
    source = {r['physical_key']: public_case(r) for r in cases if r['physical_key'] in groups['induction']}
    immutable(root / 'source_identity.json', identity)
    immutable(root / 'source_config.json', old)
    immutable(root / 'train_partition.json', groups)
    immutable(root / 'train_tasks.json', {r['physical_key']: r['task'] for r in cases})
    immutable(root / 'learning_sources.json', source)
    immutable(root / 'induction_units.json', learning_units(source))
    result = {'complete': True, 'http': 0, 'source_hash': digest(identity), 'partition_hash': digest(groups),
        'counts': {k: len(groups[k]) for k in ('induction', 'worker_probe', 'utility_probe', 'r2_exposed')},
        'historical_outcomes_in_new_probe_material': False, 'historical_operations': 0}
    return immutable(root / 'verify.json', result)


def configuration(old, root, *, bank=None):
    config = deepcopy(old)
    for section, values in POLICY_DEFAULTS.items(): config.setdefault(section, {}).update(deepcopy(values))
    config['experiment'].update(searchqa_method_version=VERSION, output_dir=str(root), runtime_mode='frozen' if bank else 'online')
    config['data_dir'] = str(bank or Path(root) / 'train/bank')
    config['budget']['eval_task_tokens'] = 96000
    config['llm']['max_retries'] = 0
    config = benchmark_config(config, 'searchqa')
    return validate_config(config)


def prepare(args):
    root = Path(args.output)
    if (root / 'method_lock.json').exists(): return checked(root)[0]
    if not read(root / 'verify.json')['complete']: raise ValueError('D0 verification required')
    code = code_identity()
    if not code['git_sha'] or code['tracked_dirty']: raise ValueError('Prepare requires committed clean source')
    r2 = Path(args.r2_run); ledger = read(r2 / 'budget.json')
    if ledger['unknown_billing'] or any(a['status'] != 'metered' for a in ledger['attempts'].values()):
        raise ValueError('Unknown historical billing blocks R3')
    # Concurrent manual Test owns its entire allocation until it is settled. Reserve it without calling it actual spend.
    reservation = read(args.external_reservation) if args.external_reservation else {'tokens': 0, 'http': 0, 'runs': []}
    if set(reservation) != {'tokens', 'http', 'runs'} or any(type(reservation[k]) is not int or reservation[k] < 0 for k in ('tokens', 'http')):
        raise ValueError('External allocation needs exact tokens/http and identified runs')
    limits = deepcopy(ledger['limits'])
    limits['token_limit'] -= reservation['tokens']; limits['request_limit'] -= reservation['http']
    billed = sum(a['accounted_tokens'] for a in ledger['attempts'].values())
    if limits['token_limit'] < billed + 13800000 or limits['request_limit'] < len(ledger['attempts']) + 332:
        raise ValueError('Remaining cumulative authority cannot cover the proposed R3 allocation')
    source = read(root / 'source_identity.json'); config = configuration(read(root / 'source_config.json'), root)
    config['budget'].update(limits)
    if (root / 'train/bank').exists(): raise ValueError('R3 preparation must start from an empty Bank')
    bank = Bank(root / 'train/bank', seed=42, environment=config['program_environment']); bank.bind_searchqa_method(VERSION)
    groups, tasks = read(root / 'train_partition.json'), read(root / 'train_tasks.json')
    for role in ('induction', 'worker_probe', 'utility_probe'):
        for key in groups[role]: bank.save_case(PublicTask(**tasks[key]), {'searchqa_role': role, 'historical_operation_count': 0})
    initial = bank.digest(); bank.close()
    immutable(root / 'inherited_budget.json', {**ledger, 'limits': limits})
    immutable(root / 'budget.json', {**ledger, 'limits': limits})
    immutable(root / 'resolved_config.json', config)
    immutable(root / 'cost_carryforward_manifest.json', {'parent': str(r2), 'parent_budget_hash': sha(r2 / 'budget.json'),
        'inherited_tokens': billed, 'inherited_http': len(ledger['attempts']), 'original_limits': ledger['limits'],
        'external_reserved_allocation': reservation, 'child_limits': limits, 'source_train_tokens': source['old_train_tokens'],
        'source_train_http': source['old_train_http'], 'old_cost_charged_again': False})
    immutable(root / 'phase_budget.json', {'proposed': PROPOSED, 'authorized': {}, 'D3a_included_in_D3': True})
    names = ['source_identity.json', 'source_config.json', 'train_partition.json', 'train_tasks.json', 'learning_sources.json',
             'induction_units.json', 'r2_exposure.json', 'r2_reuse_material.json', 'resolved_config.json',
             'inherited_budget.json', 'cost_carryforward_manifest.json', 'verify.json', 'd0_identity.json', 'verification_paths.json']
    plan = {'method_version': VERSION, 'run_id': root.name, 'code': code, 'created_at': utc(),
        'source_run': str(Path(args.source_run).resolve()), 'r2_run': str(r2.resolve()), 'datasets': str(Path(args.datasets).resolve()),
        'files': {name: sha(root / name) for name in names}, 'initial_bank_digest': initial, 'initial_assets': 0,
        'selector': POLICY, 'proposed_phase_caps': PROPOSED, 'generation_seed_supported': False,
        'denied_paths': [str(Path(p).resolve()) for p in args.deny_path],
        'model_first': True, 'answer_review': False, 'test_authorized': False,
        'development_disclosure': 'Val24 has repeated historical development use; Test1400 also has prior runs.'}
    immutable(root / 'method_lock.json', plan)
    immutable(root / 'plan_identity.json', {'sha256': sha(root / 'method_lock.json')})
    return plan


def checked(root):
    root = Path(root); plan = read(root / 'method_lock.json')
    if plan['method_version'] != VERSION or sha(root / 'method_lock.json') != read(root / 'plan_identity.json')['sha256'] or plan['code'] != code_identity():
        raise ValueError('R3 code/run identity changed; cross-version resume forbidden')
    for name, expected in plan['files'].items():
        if sha(root / name) != expected: raise ValueError('R3 locked material changed: ' + name)
    return plan, read(root / 'resolved_config.json')


def authorize(root, path):
    approved = read(path)
    if (approved.get('method_lock_sha256') != sha(Path(root) / 'method_lock.json') or not approved.get('authorization')
            or approved.get('approved') is not True or set(approved.get('phases', {})) - set(PROPOSED)
            or not approved.get('phases') or any(v != PROPOSED[k] for k, v in approved['phases'].items())):
        raise ValueError('Paid phase needs explicit human authorization tied to this method lock and exact phase caps')
    for name in approved['phases']:
        immutable(Path(root) / 'authorizations' / (name + '.json'), {**approved, 'phases': {name: approved['phases'][name]}})
    phases = {p.stem: read(p)['phases'][p.stem] for p in (Path(root) / 'authorizations').glob('*.json')}
    write_json(Path(root) / 'phase_budget.json', {'proposed': PROPOSED, 'authorized': phases, 'D3a_included_in_D3': True})


class PhaseGovernor(BudgetGovernor):
    def __init__(self, root, phase):
        self.root, self.phase = Path(root), phase
        self.plan = read(self.root / 'method_lock.json')
        auth = self.root / 'authorizations' / (phase + '.json')
        if not auth.exists(): raise ValueError('R3 paid stage is not authorized: ' + phase)
        approved = read(auth)
        if approved.get('method_lock_sha256') != sha(self.root / 'method_lock.json') or approved.get('approved') is not True or approved['phases'].get(phase) != PROPOSED[phase]:
            raise ValueError('R3 phase authorization identity changed')
        self.cap = PROPOSED[phase]
        super().__init__(self.root / 'budget.json', **read(self.root / 'inherited_budget.json')['limits'])

    def owns(self, row):
        phase = row.get('context', {}).get('searchqa_phase')
        return row.get('context', {}).get('searchqa_run_id') == self.plan['run_id'] and (phase == self.phase or self.phase == 'D3' and phase == 'D3a')

    def headroom(self, context):
        room = super().headroom(context); own = [a for a in self.state['attempts'].values() if self.owns(a)]
        room['tokens'] = min(room['tokens'], self.cap['tokens'] - sum(a.get('accounted_tokens', a['reserved_tokens']) for a in own) - context.get('parent_finish_reserve', 0))
        room['requests'] = min(room['requests'], self.cap['http'] - len(own))
        return room

    def admit(self, key, payload, context):
        context = {**context, 'searchqa_run_id': self.plan['run_id'], 'searchqa_phase': self.phase, 'searchqa_method': VERSION}
        room = self.headroom(context)
        if input_token_bound(payload) + payload.get('max_tokens', payload.get('max_completion_tokens', 0)) > room['tokens'] or room['requests'] <= context.get('finish_request_reserve', 0):
            raise BudgetExhausted('diagnostic_budget_exhausted', 'R3 phase exhausted; no borrowing or automatic retry')
        return super().admit(key, payload, context)


@contextmanager
def learning_system(root, phase, *, offline=False):
    if not offline and (Path(root) / 'probe_frozen_bank').exists():
        raise ValueError('R3 learning is frozen before the first utility probe; no outcome-driven relearning')
    _, config = checked(root)
    system = EmpiricalSystem(config, harness=AnswerAdapter('searchqa', {}), budget_governor=None if offline else PhaseGovernor(root, phase))
    try: yield system
    finally: system.close()


def failed(root):
    if (Path(root) / 'STOP_AFTER_TASK').exists(): raise SystemExit(75)


def induction(root):
    root = Path(root); checked(root); sources = read(root / 'learning_sources.json'); results = []
    with learning_system(root, 'D2') as system:
        for u in read(root / 'induction_units.json'):
            failed(root); folder = root / 'induce' / u['id']; path = folder / 'result.json'
            if path.exists(): results.append(read(path)); continue
            unit(system, folder, VERSION + ':' + u['id']); started = utc()
            try: result = induce(system.learner, u, sources)
            except (ValueError, SyntaxError) as exc:
                if not model_failure(exc): raise
                result = {'decision': 'rejected', 'reason': str(exc)}
            audit = read(folder / 'requests.json')
            result.update(unit=u, start_time=started, end_time=utc(),
                response_hashes=[digest(r.get('response')) for r in audit['requests']],
                repair_count=sum(r.get('repair', 0) > 0 for r in audit['requests']))
            write_json(path, result); results.append(result)
    jsonl(root / 'atomic_candidates.jsonl', results)
    return immutable(root / 'induce/summary.json', {'complete': True, 'results': results,
        'examples_consumed': len({k for r in results for k in r['unit']['case_ids']})})


def select_builds(bank):
    selected, audit = [], []
    for family in FAMILIES:
        candidates = [a for a in bank.all('skill') if a['searchqa_metadata']['candidate']['family'] == family
                      and not a['searchqa_metadata'].get('legacy_parent')]
        candidates.sort(key=lambda a: (-len(a['searchqa_metadata']['candidate']['source_case_ids']), a['id']))
        for i, asset in enumerate(candidates):
            audit.append({'id': asset['id'], 'family': family, 'selected': i == 0,
                'reason': 'one_per_operation_family_then_source_count_then_id' if i == 0 else 'family_already_covered'})
        selected.extend(candidates[:1])
    return selected, audit


def build_one(system, root, asset, spec, sources, phase, legacy=None):
    folder = Path(root) / ('reuse_spec' if phase == 'D1' else 'build') / asset['id']
    result_path = folder / 'result.json'
    if result_path.exists(): return read(result_path)
    immutable(folder / 'spec.json', spec); previous = None
    for generation in range(2):
        failed(root); gen = folder / ('generation_' + str(generation)); unit(system, gen, VERSION + ':' + phase + ':' + asset['id'] + ':' + str(generation))
        rejected = gen / 'rejection.json'
        if rejected.exists(): previous = read(rejected); continue
        if (gen / 'program.json').exists(): program = system.bank.get(read(gen / 'program.json')['id'])
        else:
            try:
                program = probe.build(system.learner, spec, asset, sources, generation=generation, previous=previous)
                program = system.bank.put('program', program); immutable(gen / 'program.json', program)
            except (ValueError, SyntaxError) as exc:
                if not model_failure(exc): raise
                audit = read(gen / 'requests.json')
                previous = {'reason': str(exc), 'program_id': None, 'generation': generation,
                    'original_response': audit['requests'][-1].get('response'), 'request_audit_hash': sha(gen / 'requests.json')}
                write_json(rejected, previous); continue
        trial = run_workers(system, root, program, gen / 'worker')
        if trial['passed']:
            return immutable(result_path, {'decision': 'qualified', 'program_id': program['id'], 'generation': generation,
                'qualification': trial, 'legacy_parent': legacy})
        previous = {'reason': 'worker_not_qualified', 'program_id': program['id'], 'generation': generation,
            'worker_feedback': trial, 'worker_report_hash': sha(gen / 'worker/summary.json')}
        write_json(rejected, previous)
    return immutable(result_path, {'decision': 'not_qualified', 'previous': previous, 'legacy_parent': legacy})


def run_workers(system, root, program, folder):
    root = Path(root); tasks = read(root / 'train_tasks.json'); groups = read(root / 'train_partition.json'); receipts = []
    spec = program['entry_constraints']['spec']; first = PublicTask(**tasks[spec['source_case_ids'][0]])
    for role, keys in [('source', spec['source_case_ids']), ('transfer', groups['worker_probe'])]:
        for key in keys:
            failed(root)
            receipts.append(probe.execute(system.bank, system.worker, program, PublicTask(**tasks[key]), folder, role=role))
    for role, arguments in probe.boundary_inputs(spec['kind']).items():
        receipts.append(probe.execute(system.bank, system.worker, program, first, folder, role=role, arguments=arguments))
    result = probe.qualify(system.bank, program, receipts)
    result['checks'] = [{'id': r['id'], 'role': r['role'], 'task_key': r['task_key'], **r['check']} for r in receipts]
    return immutable(Path(folder) / 'summary.json', result)


def reuse_spec(root):
    root = Path(root); sources = read(root / 'learning_sources.json'); results = []
    with learning_system(root, 'D1') as system:
        for parent in read(root / 'r2_reuse_material.json'):
            # Preserve the complete old spec/candidate and failed responses as parent material, never successful R3 evidence.
            old = parent['old_asset']['searchqa_metadata']['candidate']; c = deepcopy(old)
            c.update(method_version=VERSION, family='evidence_locate', operation='locate_question_term_evidence', output_role='intermediate',
                applicability={'family': 'evidence_locate', 'public_evidence_required': ['question_term_overlap']}, parent_ids=[])
            asset = validate_candidate(c, sources)
            asset['searchqa_metadata']['legacy_parent'] = deepcopy(parent)
            asset['id'] = 'skill_' + digest({k: v for k, v in asset.items() if k != 'id'})
            asset = system.bank.put('skill', asset); spec = specification(asset)
            results.append(build_one(system, root, asset, spec, sources, 'D1', legacy=parent['parent_spec_hash']))
    return immutable(root / 'reuse_spec/summary.json', {'complete': True, 'results': results})


def build(root):
    root = Path(root); checked(root)
    if not read(root / 'induce/summary.json')['complete']: raise ValueError('Induction incomplete')
    sources = read(root / 'learning_sources.json'); results = []
    with learning_system(root, 'D2') as system:
        selected, audit = select_builds(system.bank); immutable(root / 'build/selection.json', audit)
        for asset in selected: results.append(build_one(system, root, asset, specification(asset), sources, 'D2'))
    return immutable(root / 'build/summary.json', {'complete': True, 'results': results})


def worker(root):
    root = Path(root); results = []
    with learning_system(root, None, offline=True) as system:
        for program in system.bank.all('program'):
            results.append({'program_id': program['id'], 'usable': system.bank.program_eligible(program)})
    return immutable(root / 'worker/summary.json', {'complete': True, 'results': results,
        'note': 'Source/transfer/negative Workers execute inside each bounded build job before its one repair.'})


def snapshot(root, name):
    _, config = checked(root); root = Path(root); path = root / name
    bank = Bank(root / 'train/bank', seed=42, environment=config['program_environment']); bank.bind_searchqa_method(VERSION)
    try:
        if path.exists():
            if read(path / 'freeze.json')['source_digest'] != bank.digest(): raise ValueError('Bank changed after snapshot')
        else: bank.freeze(path)
    finally: bank.close()
    return tree_identity(path)['sha256']


def episode_group(root, directory, arm, tasks, split, phase):
    root = Path(root); plan, base = checked(root)
    frozen = root / ('train/frozen_bank' if split == 'val' else 'probe_frozen_bank'); target = root / directory / arm
    config = configuration(base, target, bank=frozen)
    config['manifest'] = str(Path(read(root / 'source_config.json')['manifest']).with_name(split + '.json'))
    config['harness']['evaluator_records'] = str(Path(plan['datasets']) / ('evaluator_records_' + split + '.json'))
    before = tree_identity(frozen)['sha256']
    log = FormalLog(target, {'method': VERSION, 'benchmark': 'searchqa', 'run_seed': 42, 'phase': phase, 'arm': arm,
        'frozen_hash': before, 'method_lock_sha256': sha(root / 'method_lock.json')}, config, resume=(target / 'run_manifest.json').exists())
    try:
        result = run(config, tasks, target, resume=(target / 'execution_manifest.json').exists(), readonly=True,
            formal_log=log, bank_view_factory=lambda b: RuntimeView(b, arm), budget_governor=PhaseGovernor(root, phase), freeze_on_complete=False)
        log.finish('completed' if result['complete'] else 'paused')
        if not result['complete']: raise RuntimeError('Incomplete episode group; downstream stage stopped')
        return result
    except BaseException:
        log.finish('interrupted'); raise
    finally:
        after = tree_identity(frozen)['sha256']; write_json(target / 'frozen_check.json', {'before': before, 'after': after, 'unchanged': before == after})
        if after != before: raise RuntimeError('Frozen changed during evaluation')


def paired(cases_by_arm):
    result = {'arms': {}, 'pairs': {}, 'generation_seed_supported': False, 'method': VERSION}
    reference = next(iter(cases_by_arm.values()))
    for arm, cases in cases_by_arm.items():
        if [c['task'] for c in cases] != [c['task'] for c in reference]: raise ValueError('Paired IDs/order/public inputs differ')
        attempts = [a for c in cases for a in c['execution'].get('attempts', [])]
        usage = usage_totals([u for c in cases for u in c['usage']])
        result['arms'][arm] = {**usage, 'tasks': len(cases), 'correct': sum(c['score']['hard'] for c in cases),
            'em': sum(c['score']['hard'] for c in cases) / len(cases), 'f1': sum(c['score']['soft'] for c in cases) / len(cases),
            'http': sum(len(r.get('http_attempts', [])) for c in cases for r in c['requests']),
            'program_executed': len(attempts), 'result_exposed': sum(bool(a.get('exposed_request_ids')) for a in attempts),
            'result_read': sum(bool(a.get('read_request_ids')) for a in attempts),
            'result_used_declared': sum(bool(a.get('declared_used_request_ids')) for a in attempts),
            'submission_by_program': sum(bool(a.get('submission_by_program')) for a in attempts),
            'worker_seconds': sum(a.get('worker_seconds') or 0 for a in attempts),
            'empty': sum(bool(c['execution'].get('empty_answer')) for c in cases),
            'truncated': sum(bool(c['execution'].get('completion_truncated')) for c in cases)}
    for arm, cases in list(cases_by_arm.items())[1:]:
        result['pairs'][arm] = [{'task_id': a['task']['task_id'], 'off': a['score'], 'on': b['score'],
            'delta_correct': int(b['score']['hard']) - int(a['score']['hard']),
            'off_tokens': sum(u['total_tokens'] for u in a['usage']), 'on_tokens': sum(u['total_tokens'] for u in b['usage'])}
            for a, b in zip(reference, cases)]
    return result


def measure_train(root, size):
    root = Path(root); checked(root)
    if size not in {12, 30}: raise ValueError('Only the locked prefix12 or full30 is allowed')
    if not any((root / s / 'summary.json').exists() for s in ('reuse_spec', 'build')): raise ValueError('Complete bounded generation before measurement')
    snapshot(root, 'probe_frozen_bank')
    tasks = read(root / 'train_tasks.json'); keys = read(root / 'train_partition.json')['utility_probe']
    if size == 30 and not (root / 'measure_train/prefix/report.json').exists(): raise ValueError('D3a prefix12 review must precede D3 continuation')
    phase = 'D3a' if size == 12 else 'D3'; segment = 'prefix' if size == 12 else 'suffix'
    selected = [PublicTask(**tasks[k]) for k in (keys[:12] if size == 12 else keys[12:])]
    by_arm = {}
    for arm in ('NoSkill', 'Atomic-full'):
        result = episode_group(root, 'measure_train/' + segment, arm, selected, 'train', phase)
        by_arm[arm] = result['cases']
    immutable(root / 'measure_train' / segment / 'report.json', paired(by_arm))
    if size == 30:
        by_arm = {arm: read(root / 'measure_train/prefix' / arm / 'summary.json')['cases'] + by_arm[arm] for arm in by_arm}
        immutable(root / 'measure_train/report.json', paired(by_arm))
    return paired(by_arm)


def freeze(root):
    root = Path(root); checked(root)
    if not (root / 'measure_train/report.json').exists(): raise ValueError('Full independent Train30 observation is required before final freeze')
    before = snapshot(root, 'probe_frozen_bank'); final = snapshot(root, 'train/frozen_bank')
    if (root / 'final_frozen_manifest.json').exists():
        saved = read(root / 'final_frozen_manifest.json')
        if saved['probe_tree_hash'] != before or saved['frozen_tree_hash'] != final: raise ValueError('Final freeze changed')
        return saved
    return immutable(root / 'final_frozen_manifest.json', {'frozen_tree_hash': final, 'probe_tree_hash': before,
        'report_sha256': sha(root / 'measure_train/report.json'), 'selection': POLICY, 'freeze_time': utc()})


def val(root):
    root = Path(root); plan, _ = checked(root); frozen = read(root / 'final_frozen_manifest.json')
    if tree_identity(root / 'probe_frozen_bank')['sha256'] != frozen['probe_tree_hash'] or tree_identity(root / 'train/frozen_bank')['sha256'] != frozen['frozen_tree_hash']:
        raise ValueError('Frozen identity changed')
    data = read(Path(plan['datasets']) / 'val.json'); authority_path = Path(read(root / 'source_config.json')['manifest']).with_name('val.json')
    authority = read(authority_path.parent.parent / 'manifest.json'); canonical = read(authority_path)
    if (len(data['tasks']) != 24 or data['split_sha256'] != sha(authority_path)
            or data['authority_sha256'] != sha(authority_path.parent.parent / 'manifest.json')
            or sha(authority_path) != authority['benchmarks']['searchqa']['splits']['val']['sha256']
            or [t['task_id'] for t in data['tasks']] != [t['task_id'] for t in canonical['tasks']]):
        raise ValueError('Val canonical identity/order changed')
    tasks = [PublicTask(**t) for t in data['tasks']]; by_arm = {}
    immutable(root / 'val/evaluation_lock.json', {'arms': ['NoSkill', 'Guidance-only', 'Atomic-full'], 'tasks_sha256': digest(data['tasks']),
        'frozen_hash': frozen['frozen_tree_hash'], 'runtime_policy': 'model_first', 'development_history': plan['development_disclosure']})
    for arm in ('NoSkill', 'Guidance-only', 'Atomic-full'):
        by_arm[arm] = episode_group(root, 'val', arm, tasks, 'val', 'D4')['cases']
    return immutable(root / 'val/report.json', paired(by_arm))


def report(root):
    root = Path(root); plan, _ = checked(root); ledger = read(root / 'budget.json'); inherited = read(root / 'inherited_budget.json')
    new = [a for k, a in ledger['attempts'].items() if k not in inherited['attempts']]
    bank = Bank(root / 'train/bank', readonly=True, seed=42)
    try:
        skills, programs = bank.all('skill'), bank.all('program')
        assets = {'active_skill_count': len(skills), 'program_spec_count': len(list(root.glob('**/spec.json'))),
            'usable_program_count': sum(bank.program_eligible(p) for p in programs), 'generated_program_count': len(programs)}
    finally: bank.close()
    build_dirs = [p for name in ('reuse_spec', 'build') for p in (root / name).glob('*/generation_*')]
    assets['builder_valid_rate'] = (sum((p / 'program.json').exists() for p in build_dirs) / len(build_dirs)) if build_dirs else None
    jsonl(root / 'builder_attempts.jsonl', [{'path': str(p.relative_to(root)), 'generated_valid_code': (p / 'program.json').exists(),
        'requests_sha256': sha(p / 'requests.json') if (p / 'requests.json').exists() else None,
        'failure': read(p / 'rejection.json') if (p / 'rejection.json').exists() else None} for p in build_dirs])
    receipts = []
    for folder in build_dirs:
        for path in (folder / 'worker').glob('*.json'):
            row = read(path)
            if row.get('policy') == probe.QUALIFICATION:
                receipts.append({'path': str(path.relative_to(root)), 'sha256': sha(path), 'id': row['id'], 'role': row['role'],
                    'check': row['check'], 'wall_seconds': row['end_time'] - row['start_time'], 'cpu_seconds': row['result'].get('cpu_seconds')})
    jsonl(root / 'worker_receipts.jsonl', receipts)
    outcome = {**assets, 'run_id': plan['run_id'], 'code': plan['code'], 'method': VERSION,
        'incremental_known_tokens': sum(a.get('accounted_tokens', 0) for a in new), 'incremental_http': len(new),
        'unknown_billing': ledger['unknown_billing'], 'source_cost': read(root / 'cost_carryforward_manifest.json'),
        'worker_cost': {'invocations': len(receipts), 'wall_seconds': sum(r['wall_seconds'] for r in receipts),
            'cpu_seconds': sum(r['cpu_seconds'] or 0 for r in receipts), 'cpu_missing': sum(r['cpu_seconds'] is None for r in receipts)},
        'phases': {phase: {'http': sum(a['context'].get('searchqa_phase') == phase for a in new),
            'reserved_or_accounted_tokens': sum(a.get('accounted_tokens', a['reserved_tokens']) for a in new if a['context'].get('searchqa_phase') == phase)} for phase in PROPOSED},
        'train_probe': read(root / 'measure_train/report.json') if (root / 'measure_train/report.json').exists() else None,
        'val': read(root / 'val/report.json') if (root / 'val/report.json').exists() else None,
        'test_authorized': False, 'natural_learning_or_utility_claim': False if not new else 'see_actual_receipts_and_paired_results'}
    write_json(root / 'final_review.json', outcome); return outcome


def isolation_command(args):
    root, source, r2, datasets = map(lambda p: Path(p).resolve(), (args.output, args.source_run, args.r2_run, args.datasets))
    if root == source or source in root.parents or root == r2 or r2 in root.parents: raise ValueError('Independent R3 output required')
    root.mkdir(parents=True, exist_ok=True); blocked = root / '.denied_dir'; blocked.mkdir(exist_ok=True); blocked.chmod(0)
    forbidden = root / '.denied_file'; forbidden.touch(exist_ok=True); forbidden.chmod(0)
    original = read(source / 'train/config.json'); authority = Path(original['manifest']).parent
    dirs = [source / 'val', source / 'test', r2 / 'val', r2 / 'train_probe', datasets.parent / '_resource_pool']
    dirs.extend(Path(p).resolve() for p in args.deny_path)
    files = [datasets / (s + '.json') for s in ('val', 'test', 'reserve')]
    files += [authority / (s + '.json') for s in ('val', 'test', 'reserve')]
    files += list(datasets.glob('evaluator_records_*.json'))
    files += [root / 'd0_host_audit.json']
    if args.stage not in {'verify', 'prepare'}:
        dirs += [source / 'train', r2]
        if (root / 'measure_train').exists(): dirs.append(root / 'measure_train')
        if (root / 'val').exists(): dirs.append(root / 'val')
    paths = [p for p in files + dirs if p.exists()]
    command = ['bwrap', '--die-with-parent', '--ro-bind', '/', '/', '--unshare-pid', '--proc', '/proc', '--dev-bind', '/dev', '/dev',
        '--bind', str(root), str(root), '--bind', '/tmp', '/tmp', '--setenv', 'GIT_OPTIONAL_LOCKS', '0',
        '--setenv', 'SEARCHQA_R3_SANDBOX', '1', '--setenv', 'SEARCHQA_R3_DENIED', json.dumps([str(p) for p in paths])]
    for p in paths: command += ['--ro-bind', str(blocked if p.is_dir() else forbidden), str(p)]
    return command + [sys.executable, '-m', __spec__.name, *sys.argv[1:]]


def validate_isolation():
    if os.environ.get('SEARCHQA_R3_SANDBOX') != '1': raise RuntimeError('R3 Train derivation requires permission isolation')
    for name in json.loads(os.environ['SEARCHQA_R3_DENIED']):
        try:
            path = Path(name)
            if path.is_dir(): list(path.iterdir())
            else: path.read_bytes()
        except PermissionError: continue
        raise RuntimeError('R3 read denial failed: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, default='verify'); parser.add_argument('--output', required=True)
    parser.add_argument('--source-run'); parser.add_argument('--r2-run'); parser.add_argument('--datasets')
    parser.add_argument('--historical-manifest'); parser.add_argument('--external-reservation')
    parser.add_argument('--deny-path', action='append', default=[])
    parser.add_argument('--authorization'); parser.add_argument('--env-file'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-size', type=int, choices=(12, 30), default=12)
    args = parser.parse_args(); root = Path(args.output).resolve(); args.output = str(root)
    plan = read(root / 'method_lock.json') if (root / 'method_lock.json').exists() else None
    paths = plan or (read(root / 'verification_paths.json') if (root / 'verification_paths.json').exists() else None)
    if paths:
        if args.deny_path and [str(Path(p).resolve()) for p in args.deny_path] != paths['denied_paths']:
            raise ValueError('Learning denied paths changed')
        args.deny_path = paths['denied_paths']
    for key in ('source_run', 'r2_run', 'datasets'):
        supplied = getattr(args, key)
        if paths and supplied and str(Path(supplied).resolve()) != paths[key]: raise ValueError('Source path changed')
        setattr(args, key, supplied or (paths or {}).get(key))
        if not getattr(args, key): parser.error('--' + key.replace('_', '-') + ' is required before prepare')
    root.mkdir(parents=True, exist_ok=True)
    if not os.environ.get('SEARCHQA_R3_SANDBOX'):
        if args.stage == 'verify':
            from .searchqa_r3_audit import audit
            if not args.historical_manifest: parser.error('--historical-manifest is required for D0')
            audit(args.r2_run, root, args.historical_manifest)
        elif (root / 'd0_host_audit.json').exists():
            if sha(root / 'd0_host_audit.json') != read(root / 'd0_identity.json')['host_audit_sha256']:
                raise ValueError('D0 Host audit identity changed')
            record = read(root / 'd0_host_audit.json')
            for name, expected in {**record['historical_files'], **record['r2_files']}.items():
                if sha(name) != expected: raise ValueError('Historical artifact changed: ' + name)
        if args.stage in {'verify', 'prepare', 'reuse_spec', 'induce', 'build', 'worker'}:
            raise SystemExit(subprocess.call(isolation_command(args)))
    if args.stage in {'verify', 'prepare', 'reuse_spec', 'induce', 'build', 'worker'}:
        validate_isolation()
        write_json(root / 'isolation' / (args.stage + '.json'), {'passed': True, 'http': 0,
            'denied_paths': json.loads(os.environ['SEARCHQA_R3_DENIED']), 'source_mount': 'readonly'})
    if args.env_file: load_env(args.env_file)
    with (root / '.run.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another R3 stage owns this run')
        failed(root)
        if args.authorization: authorize(root, args.authorization)
        stage_id = args.stage + ('_' + str(args.probe_size) if args.stage == 'measure_train' else '')
        receipt = root / 'stages' / (stage_id + '.json')
        if receipt.exists() and args.stage not in {'verify', 'report'}:
            if not args.resume: raise ValueError('Completed stage requires --resume to verify, not replay')
            checked(root); print(json.dumps(read(receipt), ensure_ascii=False)); return
        write_json(root / 'status.json', {'stage': stage_id, 'state': 'running', 'pid': os.getpid(), 'updated_at': utc()})
        try:
            if args.stage in {'verify', 'prepare'}: result = globals()[args.stage](args)
            elif args.stage == 'induce': result = induction(root)
            elif args.stage == 'measure_train': result = measure_train(root, args.probe_size)
            else: result = globals()[args.stage](root)
            if args.stage not in {'verify', 'report'}: immutable(receipt, result)
        except BaseException as exc:
            write_json(root / 'status.json', {'stage': stage_id, 'state': 'stopped', 'type': type(exc).__name__, 'error': str(exc), 'updated_at': utc()}); raise
        write_json(root / 'status.json', {'stage': stage_id, 'state': 'completed', 'updated_at': utc()})
        print(json.dumps({'stage': stage_id, 'result': result}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
