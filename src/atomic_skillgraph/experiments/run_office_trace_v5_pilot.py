"""Isolated V5 Train-only pipeline; paid stages require a new manifest-specific authorization."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json

import yaml

from ..core.errors import BudgetExhausted
from ..empirical.budget_governor import BudgetGovernor, input_token_bound
from ..empirical.checkpoint import save_json
from ..empirical.contracts import digest, validate_program
from ..empirical.office_operation_mining import read, sha
from ..empirical.office_operation_mining_v5 import mine, material, select, inventory
from ..empirical.office_program_contract_v5 import VERSION, MATERIAL_VERSION, PROBE_VERSION, ContractError, validate_candidate, field_contract
from ..empirical.office_trace_program_induction_v5 import (EXTRACTOR, BUILDER, RequestStore, ModelOutputError,
    program_from_response, builder_material, repair_material, bundle)
from ..empirical.office_program_probe_v5 import probe_cases, execute_probe
from .formal_log import utc, tree_identity
from .run_empirical import code_identity, load_env
from .run_office_trace_v4_pilot import jsonl, rows, assets, permission, provider_for, exclusive

PHASE_LIMITS = {'learn': {'tokens': 500000, 'http': 60}, 'consume': {'tokens': 250000, 'http': 20}}


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if (cfg['method_version'] != VERSION or cfg['seed'] != 42 or cfg['phase_limits'] != PHASE_LIMITS
            or cfg['repairs_per_candidate'] != 1 or not 2 <= cfg['paid_candidates'] <= 4
            or not 6 <= cfg['source_task_count'] <= 10 or not 8 <= cfg['max_opportunities'] <= 12
            or len(set(cfg['transfer_tasks'])) != 2 or cfg['focus_v4_opportunities'] != [2, 3]
            or cfg['authorization'] != 'pending_explicit_user_approval' or cfg['stop_after'] != 'train_only_pilot_review'
            or cfg['source_window_chars'] != 12000 or cfg['wire_max_bytes'] != 65536
            or cfg['paired_arm_limits'] != {'tokens': 62500, 'http': 5, 'native_calls': 48}):
        raise ValueError('V5 configuration differs from the reviewed finite Train-only protocol')
    return cfg


def prepare(args):
    root, prior = Path(args.output).resolve(), Path(args.v4_run).resolve()
    if root.exists(): raise ValueError('V5 requires a fresh output directory')
    cfg = settings(args.config); old = read(prior / 'pilot_manifest.json')
    if old.get('method_version') != 'office-trace-v4-pilot.v1': raise ValueError('Expected the actual V4 pilot')
    source, parent, corpus = (Path(old[k]).resolve() for k in ('source', 'parent_bank', 'corpus_root'))
    if any(p == root or p in root.parents or root in p.parents for p in (source, parent, corpus, prior, Path(old['parent_run']).resolve())):
        raise ValueError('V5 must not overlap historical runs or corpus')
    code = code_identity()
    if not code['git_sha'] or code['tracked_dirty']: raise ValueError('Prepare only clean committed source')
    if tree_identity(parent) != old['parent_frozen']: raise ValueError('V3 Frozen changed')
    for path, expected_sha in old['parent_files'].items():
        if sha(path) != expected_sha: raise ValueError('V4 parent identity changed: ' + path)
    sources = read(prior / 'source_records.json')
    canonical_path = Path(read(source / 'config.json')['manifest'])
    canonical = read(canonical_path)['tasks']
    if (len(sources) != 120 or len({s['task']['physical_key'] for s in sources}) != 120
            or any(s['task']['split'] != 'train' for s in sources)
            or {s['task']['task_id'] for s in sources} != {s['task_id'] for s in canonical}):
        raise ValueError('Expected the exact canonical Train120')
    ledger = read(prior / 'budget/usage.json')
    if (ledger['unknown_billing'] or any(a['status'] != 'metered' for a in ledger['attempts'].values())
            or ledger['limits'] != {'token_limit': 90000000, 'request_limit': 480000, 'finish_reserve': 0}):
        raise ValueError('Historical V4 ledger must be fully reconciled under the existing 90M authority')
    # Reserve other ongoing Office experiments at their full remaining ceiling. This is accounting
    # metadata only: no authored Bank/source/content is admitted to the learning inputs.
    reservations = []
    for name in args.reserved_run or []:
        sibling = Path(name).resolve(); carry = read(sibling / 'cost_carryforward_manifest.json')
        sibling_config = read(sibling / 'config.json'); tasks = read(sibling / 'tasks.json')
        count = len(tasks['tasks'] if isinstance(tasks, dict) else tasks)
        limits = read(sibling / 'budget.json')['limits']
        reservations.append({'run_root': str(sibling), 'plan_sha256': sha(sibling / 'plan.json'),
            'tokens': count * sibling_config['budget']['eval_task_tokens'],
            'http': limits['request_limit'] - carry['historical_http'], 'reason': 'full sibling ceiling reserved; no sibling assets used'})
    limits = deepcopy(ledger['limits'])
    limits['token_limit'] -= sum(r['tokens'] for r in reservations)
    limits['request_limit'] -= sum(r['http'] for r in reservations)
    inherited_tokens = sum(a['accounted_tokens'] for a in ledger['attempts'].values())
    if inherited_tokens + sum(v['tokens'] for v in PHASE_LIMITS.values()) > limits['token_limit'] or len(ledger['attempts']) + 80 > limits['request_limit']:
        raise ValueError('Proposed V5 phases and concurrent reservations exceed cumulative authority')
    inherited = {**deepcopy(ledger), 'limits': limits}
    runtime = deepcopy(read(prior / 'runtime_config.json'))
    runtime['experiment'].update(output_dir=str(root / 'worker'), method_version=VERSION)
    existing = read(prior / 'parent_programs.json')
    if [p for k, p in assets(parent) if k == 'program'] != existing: raise ValueError('Parent Program catalog identity changed')
    spans, opportunities = [], []
    for record in sources:
        if sha(record['path']) != record['sha256']: raise ValueError('Original Train trace changed')
        trace = read(record['path'])
        if trace['task'] != record['task']: raise ValueError('Original task identity changed')
        ss, oo = mine(trace, record['sha256'], runtime['program_environment'], existing)
        spans.extend(ss); opportunities.extend(oo)
    old_selection = read(prior / 'selection.json')['opportunities']
    selection = select(opportunities, focus_ids=[old_selection[i] for i in cfg['focus_v4_opportunities']],
        transfer_tasks=cfg['transfer_tasks'], task_count=cfg['source_task_count'], maximum=cfg['max_opportunities'], paid_maximum=cfg['paid_candidates'])
    root.mkdir(parents=True)
    jsonl(root / 'source_span_catalog.jsonl', spans); jsonl(root / 'opportunity_catalog.jsonl', opportunities)
    save_json(root / 'source_records.json', sources); save_json(root / 'selection.json', selection)
    save_json(root / 'inventory.json', inventory(opportunities, spans)); save_json(root / 'runtime_config.json', runtime)
    save_json(root / 'parent_programs.json', existing); save_json(root / 'field_contract.json', field_contract())
    save_json(root / 'budget/inherited.json', inherited); save_json(root / 'budget/usage.json', inherited)
    save_json(root / 'budget/sibling_reservations.json', reservations)
    immutable = ['source_span_catalog.jsonl', 'opportunity_catalog.jsonl', 'source_records.json', 'selection.json',
                 'inventory.json', 'runtime_config.json', 'parent_programs.json', 'field_contract.json',
                 'budget/inherited.json', 'budget/sibling_reservations.json']
    audit_paths = [*prior.rglob('*.json'), canonical_path, Path(old['parent_run']) / 'val/summary.json']
    plan = {'method_version': VERSION, 'material_version': MATERIAL_VERSION, 'probe_version': PROBE_VERSION,
        'run_id': root.name, 'created_at': utc(), 'code': code, 'config': cfg, 'source': str(source),
        'v4_run': str(prior), 'parent_bank': str(parent), 'parent_frozen': tree_identity(parent),
        'source_projection': 'public Train task/observed operations only; no final answers, gold, scores or authored assets',
        'parent_files': {str(p): sha(p) for p in audit_paths if p.is_file()},
        'corpus_root': str(corpus), 'corpus_identity': tree_identity(corpus),
        'files': {n: sha(root / n) for n in immutable}, 'inherited_tokens': inherited_tokens,
        'inherited_http': len(ledger['attempts']), 'effective_cumulative_limits': limits,
        'permissions_sha256': digest(permission(runtime)), 'paid_authorization': 'pending',
        'stage_order': ['prepare', 'check_zero_http', 'new_explicit_authorization', 'learn', 'probe', 'optional_consume', 'review_stop']}
    save_json(root / 'pilot_manifest.json', plan); save_json(root / 'manifest_identity.json', {'sha256': sha(root / 'pilot_manifest.json')})
    (root / 'candidate_bank').mkdir(); (root / 'programs').mkdir()
    return report(root)


def checked(root, *, corpus=False):
    plan = read(root / 'pilot_manifest.json')
    if plan['method_version'] != VERSION or plan['material_version'] != MATERIAL_VERSION or plan['probe_version'] != PROBE_VERSION:
        raise RuntimeError('V5 method identity changed; no cross-version resume')
    if sha(root / 'pilot_manifest.json') != read(root / 'manifest_identity.json')['sha256'] or plan['code'] != code_identity():
        raise RuntimeError('Code/manifest identity changed; no cross-version resume')
    for name, expected_sha in plan['files'].items():
        if sha(root / name) != expected_sha: raise RuntimeError('V5 immutable input changed: ' + name)
    for name, expected_sha in plan['parent_files'].items():
        if sha(name) != expected_sha: raise RuntimeError('Historical audit changed: ' + name)
    for record in read(root / 'source_records.json'):
        if sha(record['path']) != record['sha256']: raise RuntimeError('Original Train trace changed')
    if tree_identity(plan['parent_bank']) != plan['parent_frozen']: raise RuntimeError('Original Frozen changed')
    if corpus and tree_identity(plan['corpus_root']) != plan['corpus_identity']: raise RuntimeError('Corpus changed')
    for reservation in read(root / 'budget/sibling_reservations.json'):
        if sha(Path(reservation['run_root']) / 'plan.json') != reservation['plan_sha256']: raise RuntimeError('Sibling accounting identity changed')
    config = read(root / 'runtime_config.json')
    if digest(permission(config)) != plan['permissions_sha256']: raise RuntimeError('Tool ABI identity changed')
    return plan, config


class PhaseGovernor(BudgetGovernor):
    def __init__(self, root, phase):
        self.root, self.phase = Path(root), phase
        self.plan = read(self.root / 'pilot_manifest.json')
        auth = read(self.root / 'authorization.json')
        if (auth.get('method_version') != VERSION or auth.get('approved') is not True or not auth.get('user_authorization')
                or auth.get('manifest_sha256') != sha(self.root / 'pilot_manifest.json')
                or auth.get('phase_limits') != PHASE_LIMITS or phase not in PHASE_LIMITS
                or phase not in auth.get('approved_phases', [])):
            raise ValueError('V5 requires new explicit matching paid authorization; V4 approval cannot be reused')
        self.cap = PHASE_LIMITS[phase]
        super().__init__(self.root / 'budget/usage.json', **self.plan['effective_cumulative_limits'])

    def headroom(self, context):
        room = super().headroom(context)
        owned = [a for a in self.state['attempts'].values() if a['context'].get('office_v5_run') == self.plan['run_id']
                 and a['context'].get('office_v5_phase') == self.phase]
        room['tokens'] = min(room['tokens'], self.cap['tokens'] - sum(a.get('accounted_tokens', a['reserved_tokens']) for a in owned))
        room['requests'] = min(room['requests'], self.cap['http'] - len(owned))
        return room

    def admit(self, audit_id, payload, context):
        with self.lock:
            context = {**context, 'office_v5_run': self.plan['run_id'], 'office_v5_phase': self.phase}
            room = self.headroom(context)
            reserve = input_token_bound(payload) + payload.get('max_tokens', payload.get('max_completion_tokens', 0))
            if reserve > room['tokens'] or room['requests'] <= 0:
                raise BudgetExhausted('diagnostic_budget_exhausted', 'Frozen V5 phase/cumulative allowance exhausted')
            super().admit(audit_id, payload, context)


def verify_candidates(root):
    values = [read(p) for p in sorted((root / 'candidate_bank').glob('*.json'))]
    selected = read(root / 'selection.json')['paid_opportunities']
    for value in values:
        index = value['opportunity_index']; decision = read(root / 'decisions' / (str(index) + '.json'))
        if (value['method_version'] != VERSION or value['formal_publication'] is not False
                or value['generation_basis']['id'] != selected[index] or digest(value['candidate']) != digest(decision['candidate'])
                or value['id'] != 'v5_program_' + validate_program(value['program'])):
            raise RuntimeError('Candidate/source identity changed')
        request = Path(value['request_path']).resolve()
        if request.parent != (root / 'requests' / ('op' + str(index))).resolve() or request.name not in {'build.json', 'repair.json'}:
            raise RuntimeError('Candidate response must belong to this V5 generation')
        record = read(request)
        if (record['payload_sha256'] != digest(record['payload']) or record['phase'] != 'learn'
                or record['payload']['messages'][0]['content'] != BUILDER):
            raise RuntimeError('Candidate Builder request identity changed')
        response = RequestStore.decode(record)
        if digest(response) != value['response_sha256'] or response.get('source') != value['program']['source']:
            raise RuntimeError('Candidate lacks matching natural model response')
    return values


def learn(root):
    plan, config = checked(root, corpus=True)
    gate = read(root / 'zero_http_check.json')
    if not gate['passed'] or gate['manifest_sha256'] != sha(root / 'pilot_manifest.json'): raise RuntimeError('Zero HTTP gate required')
    governor = PhaseGovernor(root, 'learn'); cfg = plan['config']; permissions = permission(config)
    verify_candidates(root)
    clients = {role: RequestStore(root / 'requests', provider_for(config, role, governor), phase='learn',
                                 wire_max_bytes=cfg['wire_max_bytes']) for role in ('extractor', 'tool_builder')}
    spans = rows(root / 'source_span_catalog.jsonl'); opportunities = {o['id']: o for o in rows(root / 'opportunity_catalog.jsonl')}
    existing = read(root / 'parent_programs.json')
    for index, oid in enumerate(read(root / 'selection.json')['paid_opportunities']):
        path = root / 'decisions' / (str(index) + '.json'); key = 'op' + str(index)
        state = read(path) if path.exists() else {'opportunity_id': oid, 'status': 'pending', 'repair_used': False, 'diagnostics': []}
        if state['status'] == 'complete': continue
        op = opportunities[oid]; source = material(op, spans, cfg['source_window_chars'])
        data = {'source': source, 'existing_programs': [{k: p.get(k) for k in ('id', 'input_schema', 'output_schema', 'allowed_tools')} for p in existing],
                'permissions': permissions}
        def ask(role, name, prompt, payload, validate):
            request = root / 'requests' / key / (name + '.json')
            try:
                response, receipt = clients[role].ask(key + '/' + name, prompt, payload)
                return validate(response), response, receipt, request
            except (ContractError, ModelOutputError) as exc:
                diag = exc.diagnostic if isinstance(exc, ContractError) else {'code': 'model_response_invalid', 'field_path': '$',
                    'expected': str(exc), 'actual_type': 'response', 'repairable': True, 'evidence_hash': digest(source)}
                save_json(root / 'diagnostics' / key / (name + '.diagnostic.json'), diag)
                state['diagnostics'].append(diag)
                repair_request = root / 'requests' / key / 'repair.json'
                if state['repair_used'] and state.get('repair_stage') != name: raise
                state.update(repair_used=True, repair_stage=name); save_json(path, state)
                repaired = repair_material(read(request), diag, payload)
                response, receipt = clients[role].ask(key + '/repair', prompt, repaired, repair=True)
                return validate(response), response, receipt, repair_request
        try:
            if 'candidate' not in state:
                candidate, _, _, _ = ask('extractor', 'proposal', EXTRACTOR, data,
                    lambda p: validate_candidate(p, op, spans, permissions['allowed_names'], existing, source))
                state['candidate'] = candidate; save_json(path, state)
            candidate = state['candidate']
            if candidate['decision'] != 'propose_program': state.update(status='complete', decision=candidate['decision'])
            else:
                program, response, receipt, request = ask('tool_builder', 'build', BUILDER, builder_material(candidate, permissions),
                    lambda p: program_from_response(p, candidate, config['program_environment'], permissions['allowed_names']))
                value = bundle(candidate, program, request, response)
                value.update(opportunity_index=index, repair_used=state['repair_used'], response_metadata=receipt['turn'].get('provider_metadata'))
                target = root / 'candidate_bank' / (program['id'] + '.json')
                if not target.exists():
                    save_json(target, value); (root / 'programs' / (program['id'] + '.py')).write_text(program['source'], encoding='utf-8')
                state.update(status='complete', decision='candidate_created', program_id=program['id'])
        except (ContractError, ModelOutputError) as exc:
            diag = exc.diagnostic if isinstance(exc, ContractError) else {'code': 'model_response_invalid', 'expected': str(exc), 'evidence_hash': digest(source)}
            state['diagnostics'].append(diag); state.update(status='complete', decision='candidate_rejected')
            save_json(root / 'diagnostics' / key / 'final.diagnostic.json', diag)
        # Host TypeError/KeyError are deliberately NOT converted into model failures or repaired.
        save_json(path, state)
    return report(root)


def probe(root):
    plan, config = checked(root, corpus=True); candidates = verify_candidates(root)
    transfer = [s for s in rows(root / 'source_span_catalog.jsonl') if s['source_task']['task_id'] in plan['config']['transfer_tasks']]
    sources = read(root / 'source_records.json')
    for value in candidates:
        path = root / 'candidate_bank' / (value['id'] + '.json')
        if value.get('probe_complete'): continue
        folder = root / 'probes' / value['id']; plan_path = folder / 'plan.json'
        if not plan_path.exists(): save_json(plan_path, probe_cases(value, transfer))
        probes = read(plan_path)
        if probes['status'] != 'ready':
            value.update(probe_complete=True, coverage=probes['status'], probe_plan=probes)
            save_json(path, value); continue
        receipts = []
        for case in probes['cases']:
            output = folder / (case['label'] + '.json'); intent = output.with_suffix('.intent.json')
            if output.exists():
                receipt = read(output)
                if receipt['case'] != case or receipt['source_sha256'] != digest(value['program']['source']): raise RuntimeError('Worker receipt identity changed')
            else:
                if intent.exists(): raise RuntimeError('Worker started without receipt; manual review required, no silent retry')
                save_json(intent, {'case_sha256': digest(case), 'program_sha256': digest(value['program']), 'started_at': utc()})
                task = next(s['task'] for s in sources if s['task']['physical_key'] == case['source_case_id'])
                receipt = execute_probe(value, case, config, task); save_json(output, receipt)
            receipts.append(receipt)
        local = [r for r in receipts if not r['case']['label'].startswith('transfer')]
        cross = [r for r in receipts if r['case']['label'].startswith('transfer')]
        passed = len(local) == 3 and all(r['passed'] for r in local)
        value.update(probe_complete=True, status='worker_checked' if passed else 'worker_failed',
            probe_receipts=[str(folder / (c['label'] + '.json')) for c in probes['cases']], transfer_coverage=probes['transfer_coverage'])
        if passed and len(cross) == 2 and all(r['passed'] for r in cross): value['status'] = 'transfer_checked'
        save_json(path, value)
    return report(root)


def report(root):
    plan = read(root / 'pilot_manifest.json'); ledger = read(root / 'budget/usage.json')
    bundles = [read(p) for p in sorted((root / 'candidate_bank').glob('*.json'))]
    decisions = [read(p) for p in sorted((root / 'decisions').glob('*.json'))]
    own = [a for a in ledger['attempts'].values() if a['context'].get('office_v5_run') == plan['run_id']]
    selected = read(root / 'selection.json')['paid_opportunities']
    complete = len(decisions) == len(selected) and all(d['status'] == 'complete' for d in decisions)
    status = ('completed_no_candidate' if not bundles else 'learn_complete') if complete else 'prepared' if not decisions else 'learning'
    if bundles and complete and all(b.get('probe_complete') for b in bundles):
        status = 'completed_candidates_checked' if any(b['status'] == 'transfer_checked' for b in bundles) else 'completed_no_qualified_candidate'
    consumption = [read(p) for p in sorted((root / 'consumption').glob('*/*/trace.json'))]
    result = {'method_version': VERSION, 'run_id': plan['run_id'], 'created_at': utc(),
        'status': status,
        'decisions': dict(Counter(d.get('decision', 'pending') for d in decisions)),
        'candidate_statuses': dict(Counter(b['status'] for b in bundles)),
        'coverage': dict(Counter(b.get('coverage', 'probed' if b.get('probe_complete') else 'pending') for b in bundles)),
        'new_http': len(own), 'new_metered_tokens': sum(a.get('accounted_tokens', 0) for a in own),
        'new_reserved_tokens': sum(a['reserved_tokens'] for a in own if a['status'] != 'metered'),
        'phase_usage': {phase: {'tokens': sum(a.get('accounted_tokens', a['reserved_tokens']) for a in own if a['context'].get('office_v5_phase') == phase),
            'http': sum(a['context'].get('office_v5_phase') == phase for a in own)} for phase in PHASE_LIMITS},
        'inherited_tokens': plan['inherited_tokens'], 'inherited_http': plan['inherited_http'],
        'sibling_reservations': read(root / 'budget/sibling_reservations.json'), 'unknown_billing': ledger['unknown_billing'],
        'parent_frozen_unchanged': tree_identity(plan['parent_bank']) == plan['parent_frozen'],
        'consumption': [{k:t[k] for k in ('arm', 'metrics', 'task_correct', 'tokens', 'http')} | {'task_id':t['task']['task_id']} for t in consumption],
        'formal_promotions': 0, 'val_test_started': False, 'paid_authorization': 'pending' if not (root / 'authorization.json').exists() else 'record_present'}
    jsonl(root / 'program_bundles.jsonl', bundles)
    save_json(root / 'pilot_review.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'check', 'learn', 'probe', 'consume', 'report'])
    parser.add_argument('--v4-run'); parser.add_argument('--reserved-run', action='append')
    parser.add_argument('--config', default='configs/officeqa_trace_v5_seed42.yaml')
    parser.add_argument('--output', required=True); parser.add_argument('--env-file'); parser.add_argument('--zero-http', action='store_true')
    args = parser.parse_args(); root = Path(args.output).resolve()
    if args.stage == 'prepare': result = prepare(args)
    else:
        if not root.is_dir(): raise ValueError('Prepare V5 first')
        with exclusive(root):
            try:
                if args.stage in {'learn', 'consume'} and args.env_file: load_env(args.env_file)
                if args.stage == 'check':
                    if not args.zero_http: raise ValueError('check requires --zero-http')
                    from .office_v5_offline import check
                    result = check(root)
                elif args.stage == 'consume':
                    from .office_v5_consumption import consume
                    result = consume(root)
                else: result = globals()[args.stage](root)
            except Exception as exc:
                save_json(root / 'last_error.json', {'stage': args.stage, 'time': utc(), 'error': str(exc),
                    'error_code': getattr(exc, 'code', type(exc).__name__), 'http_retried': False,
                    'classified_as': 'budget' if isinstance(exc, BudgetExhausted) else 'infrastructure_or_identity'})
                raise
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__': main()
