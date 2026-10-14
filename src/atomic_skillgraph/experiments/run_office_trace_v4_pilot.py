"""Isolated Train-only V4 pilot. No formal Bank publication, Val or Test entry point."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3

import yaml

from ..agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from ..core.errors import BudgetExhausted
from ..empirical.budget_governor import BudgetGovernor, input_token_bound
from ..empirical.checkpoint import save_json
from ..empirical.contracts import digest
from ..empirical.office_operation_mining import VERSION, read, sha, mine, select, material
from ..empirical.office_trace_program_induction import (EXTRACTOR, BUILDER, ModelOutputError, RequestStore,
    validate_proposal, program_from_response, bundle)
from ..empirical.office_program_probe import probe_cases, execute_probe
from ..empirical.program_worker import program_permission_view
from ..harness.benchmarks import OfficeAdapter
from .formal_log import utc, tree_identity
from .run_empirical import code_identity, load_env

PHASE_LIMITS = {'learn': {'tokens': 1000000, 'http': 80}, 'consume': {'tokens': 800000, 'http': 40}}


def jsonl(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records), encoding='utf-8')
    os.replace(temporary, path)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def assets(parent):
    with sqlite3.connect((Path(parent) / 'bank.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        return [(kind, json.loads(payload)) for kind, payload in db.execute('SELECT kind,payload FROM assets ORDER BY id')]


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if (cfg['method_version'] != VERSION or cfg['seed'] != 42 or cfg['phase_limits'] != PHASE_LIMITS
            or cfg['max_opportunities'] != 8 or cfg['max_builder_generations'] != 6 or cfg['repairs_per_candidate'] != 1
            or cfg['authorization'] != 'pending_explicit_user_approval' or cfg['stop_after'] != 'train_only_pilot_review'
            or cfg['wire_max_bytes'] != 65536 or cfg['source_window_chars'] != 12000
            or cfg['paired_arm_limits'] != {'tokens': 200000, 'http': 10, 'native_calls': 48}
            or len(cfg['source_tasks']) != 6 or len(cfg['transfer_tasks']) != 2
            or len(set(cfg['source_tasks'] + cfg['transfer_tasks'])) != 8):
        raise ValueError('V4 preregistration/caps changed; prepare a separately reviewed configuration')
    return cfg


def prepare(args):
    root, source, parent, corpus = map(lambda p: Path(p).resolve(), (args.output, args.source, args.parent_bank, args.corpus_root))
    if root.exists() or any(p == root or p in root.parents or root in p.parents for p in (source, parent, corpus)):
        raise ValueError('Use an independent new V4 output directory')
    cfg = settings(args.config); code = code_identity()
    if not code['git_sha'] or code['tracked_dirty']: raise ValueError('Prepare only clean committed code')
    parent_run = parent.parents[1]
    old = read(source / 'config.json'); execution = read(source / 'execution_manifest.json')
    tasks = execution['tasks']; canonical_path = Path(old['manifest']); canonical = read(canonical_path)['tasks']
    if (old['experiment'].get('benchmark') != 'officeqa' or len(tasks) != 120
            or len({t['task_id'] for t in tasks}) != 120 or any(t['split'] != 'train' for t in tasks)
            or {t['task_id'] for t in tasks} != {t['task_id'] for t in canonical}):
        raise ValueError('Expected canonical Office Train120, no held-out material')
    parent_plan = read(parent_run / 'plan.json')
    parent_sources = {r['task']['task_id']: r for r in parent_plan['source_records']}
    ledger = read(parent_run / 'budget.json')
    if (ledger['limits'] != {'token_limit': 90000000, 'request_limit': 480000, 'finish_reserve': 0}
            or ledger['unknown_billing'] or any(a['status'] != 'metered' for a in ledger['attempts'].values())):
        raise ValueError('Expected fully metered historical 90M ledger')
    bank_assets = assets(parent); counts = dict(Counter(k for k, _ in bank_assets))
    if any(counts.get(k, 0) != n for k, n in {'skill': 39, 'program': 12, 'implementation': 12, 'workflow': 0}.items()):
        raise ValueError('V3 Frozen asset identity differs from reviewed 39/12/12/0')
    runtime = deepcopy(read(parent_run / 'resolved_config.json')['val'])
    runtime['harness']['corpus_root'] = str(corpus)
    runtime['experiment']['output_dir'] = str(root / 'worker')
    runtime['experiment']['runtime_mode'] = 'frozen'
    spans, opportunities, sources, outcomes = [], [], [], []
    for task in tasks:
        path = source / 'traces' / (task['task_id'] + '.json'); trace = read(path); actual_sha = sha(path)
        if trace['task'] != task or parent_sources[task['task_id']]['sha256'] != actual_sha:
            raise ValueError('Original Train trace/task identity changed: ' + str(path))
        ss, oo = mine(trace, actual_sha, runtime['program_environment']); spans.extend(ss); opportunities.extend(oo)
        sources.append({'task': task, 'path': str(path), 'sha256': actual_sha})
        outcomes.append({'task_id': task['task_id'], 'score': trace.get('score'), 'source_trace_sha256': actual_sha})
    selection = select(opportunities, spans, cfg['source_tasks'], cfg['max_opportunities'])
    root.mkdir(parents=True)
    jsonl(root / 'source_span_catalog.jsonl', spans); jsonl(root / 'opportunity_catalog.jsonl', opportunities)
    save_json(root / 'source_records.json', sources); save_json(root / 'private_audit/source_outcomes.json', outcomes)
    save_json(root / 'runtime_config.json', runtime)
    programs = [p for k, p in bank_assets if k == 'program']
    save_json(root / 'parent_programs.json', programs)
    save_json(root / 'budget/inherited.json', ledger); save_json(root / 'budget/usage.json', ledger)
    save_json(root / 'selection.json', {'opportunities': selection, 'transfer_tasks': cfg['transfer_tasks'],
        'rule': 'one earliest actual read/chain/operation per preset task, then earliest actual failures; no score filtering'})
    # Record actual public corpus identity; no corpus content is sent to the Provider at prepare time.
    corpus_identity = tree_identity(corpus)
    immutable = ['source_span_catalog.jsonl', 'opportunity_catalog.jsonl', 'source_records.json',
                 'runtime_config.json', 'parent_programs.json', 'budget/inherited.json', 'selection.json']
    plan = {'method_version': VERSION, 'run_id': root.name, 'created_at': utc(), 'code': code, 'config': cfg,
        'source': str(source), 'parent_bank': str(parent), 'parent_run': str(parent_run),
        'parent_frozen': tree_identity(parent), 'parent_asset_counts': counts,
        'parent_files': {str(p): sha(p) for p in (parent_run / 'budget.json', parent_run / 'plan.json',
            source / 'config.json', source / 'execution_manifest.json', canonical_path)},
        'corpus_root': str(corpus), 'corpus_identity': corpus_identity,
        'files': {name: sha(root / name) for name in immutable},
        'inherited_tokens': sum(a['accounted_tokens'] for a in ledger['attempts'].values()),
        'inherited_http': len(ledger['attempts']), 'paid_authorization': 'pending',
        'source_projection': 'allowlisted public Train task and actual native arguments/results; no final prediction/gold/score',
        'stage_order': ['prepare', 'check_zero_http', 'explicit_authorization', 'learn', 'probe', 'optional_train_pairs', 'review_stop']}
    save_json(root / 'pilot_manifest.json', plan); save_json(root / 'manifest_identity.json', {'sha256': sha(root / 'pilot_manifest.json')})
    for name in ('atomic_specs', 'program_bundles', 'worker_probe_receipts'): jsonl(root / (name + '.jsonl'), [])
    (root / 'pilot_bank').mkdir(); (root / 'programs').mkdir()
    return {'run_id': root.name, 'source_count': len(sources), 'opportunities': len(opportunities), 'selected': len(selection), 'http': 0}


def checked(root, *, corpus=False):
    root = Path(root); plan = read(root / 'pilot_manifest.json')
    if sha(root / 'pilot_manifest.json') != read(root / 'manifest_identity.json')['sha256'] or plan['code'] != code_identity():
        raise RuntimeError('Code/manifest identity changed; no cross-version resume')
    for name, expected in plan['files'].items():
        if sha(root / name) != expected: raise RuntimeError('Immutable pilot input changed: ' + name)
    for path, expected in plan['parent_files'].items():
        if sha(path) != expected: raise RuntimeError('Immutable parent changed: ' + path)
    for record in read(root / 'source_records.json'):
        if sha(record['path']) != record['sha256']: raise RuntimeError('Original Train trace changed')
    if tree_identity(plan['parent_bank']) != plan['parent_frozen']: raise RuntimeError('Original Frozen changed')
    if corpus and tree_identity(plan['corpus_root']) != plan['corpus_identity']: raise RuntimeError('Public corpus identity changed')
    return plan, read(root / 'runtime_config.json')


class PhaseGovernor(BudgetGovernor):
    def __init__(self, root, phase):
        self.root, self.phase = Path(root), phase
        self.plan = read(self.root / 'pilot_manifest.json')
        authorization = read(self.root / 'authorization.json')
        if (authorization.get('approved') is not True or not authorization.get('user_authorization')
                or authorization.get('manifest_sha256') != sha(self.root / 'pilot_manifest.json')
                or authorization.get('phase_limits') != PHASE_LIMITS or phase not in PHASE_LIMITS):
            raise ValueError('V4 paid phases require explicit matching user approval; suggested caps are not approval')
        self.cap = PHASE_LIMITS[phase]
        super().__init__(self.root / 'budget/usage.json', **read(self.root / 'budget/inherited.json')['limits'])

    def headroom(self, context):
        room = super().headroom(context)
        owned = [a for a in self.state['attempts'].values() if a['context'].get('office_v4_run') == self.plan['run_id']
                 and a['context'].get('office_v4_phase') == self.phase]
        room['tokens'] = min(room['tokens'], self.cap['tokens'] - sum(a.get('accounted_tokens', a['reserved_tokens']) for a in owned))
        room['requests'] = min(room['requests'], self.cap['http'] - len(owned))
        return room

    def admit(self, audit_id, payload, context):
        with self.lock:
            context = {**context, 'office_v4_run': self.plan['run_id'], 'office_v4_phase': self.phase}
            room = self.headroom(context)
            reserve = input_token_bound(payload) + payload.get('max_tokens', payload.get('max_completion_tokens', 0))
            if reserve > room['tokens'] or room['requests'] <= 0:
                raise BudgetExhausted('diagnostic_budget_exhausted', 'Frozen V4 child/parent allowance exhausted; no borrowing')
            super().admit(audit_id, payload, context)


def provider_for(config, role, governor):
    llm = config['llm']; role_config = llm[role]
    protocol = {**llm.get('protocol', {}), **role_config.get('protocol', {})}
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig(base_url=llm['base_url'], model=llm['model'],
        api_key_env=llm['api_key_env'], max_completion_tokens=role_config['max_completion_tokens'],
        dialect=llm['dialect'], thinking_type=protocol.get('thinking_type', 'enabled'),
        reasoning_effort=role_config['reasoning_effort'], request_timeout_seconds=role_config['request_timeout_seconds'],
        token_limit_field=llm.get('token_limit_field', 'max_tokens'), capability_profile=llm.get('capability_profile'),
        max_retries=0))
    provider.budget_governor = governor
    return provider


def permission(config):
    from ..empirical.task_context import TaskContext
    adapter = OfficeAdapter({}, config)
    try: return program_permission_view(adapter.tool_definitions(), [], [TaskContext(config['runtime']).tool()],
                                        environment=config['program_environment'])
    finally: adapter.close()


def index_outputs(root):
    bundles = [read(p) for p in sorted((root / 'pilot_bank').glob('*.json'))]
    jsonl(root / 'program_bundles.jsonl', bundles)
    jsonl(root / 'atomic_specs.jsonl', [b['atomic'] for b in bundles])
    receipts = [read(p) for p in sorted((root / 'probes').glob('*/revision*/*.json')) if p.name in {'positive.json', 'variation.json', 'negative.json'}]
    jsonl(root / 'worker_probe_receipts.jsonl', receipts)


def rejected_response(path):
    """Original response/error only; its full sent payload remains in the immutable request file."""
    record = read(path)
    return {**{k:record[k] for k in ('status','error','error_code','payload_sha256') if k in record},
            'model_response': {k:record.get('turn',{}).get(k) for k in ('content','tool_calls','finish_reason')}}


def learn(root):
    plan, config = checked(root, corpus=True)
    gate = read(root / 'zero_http_check.json')
    if not gate['passed'] or gate['manifest_sha256'] != sha(root / 'pilot_manifest.json'): raise RuntimeError('Zero HTTP gate must pass first')
    governor = PhaseGovernor(root, 'learn'); cfg = plan['config']; permissions = permission(config)
    clients = {role: RequestStore(root / 'requests', provider_for(config, role, governor), phase='learn',
        wire_max_bytes=cfg['wire_max_bytes']) for role in ('extractor', 'tool_builder')}
    opportunities = {r['id']: r for r in rows(root / 'opportunity_catalog.jsonl')}
    spans = rows(root / 'source_span_catalog.jsonl'); existing = read(root / 'parent_programs.json')
    existing_view = [{k: p.get(k) for k in ('id', 'input_schema', 'output_schema', 'allowed_tools', 'entry_constraints')} for p in existing]
    for index, oid in enumerate(read(root / 'selection.json')['opportunities']):
        path = root / 'decisions' / (str(index) + '.json')
        if path.exists() and read(path).get('status') == 'complete': continue
        op = opportunities[oid]; key = f'op{index}'
        state = read(path) if path.exists() else {'opportunity_id': oid, 'repair_used': False, 'status': 'pending', 'errors': []}
        data = {**material(op, spans, cfg['source_window_chars']), 'existing_programs': existing_view,
                'permissions': permissions}
        try:
            if 'candidate' not in state:
                try:
                    proposal, _ = clients['extractor'].ask(key + '/proposal', EXTRACTOR, data)
                    candidate = validate_proposal(proposal, op, spans, permissions['allowed_names'], existing)
                except (ModelOutputError, ValueError) as exc:
                    repair_path = root / 'requests' / key / 'proposal_repair.json'
                    if state['repair_used'] and not repair_path.exists(): raise
                    state['repair_used'] = True; save_json(path, state)
                    proposal, _ = clients['extractor'].ask(key + '/proposal_repair', EXTRACTOR,
                        {**data, 'original_response': rejected_response(root / 'requests' / key / 'proposal.json'),
                         'actual_error': str(exc)}, repair=True)
                    candidate = validate_proposal(proposal, op, spans, permissions['allowed_names'], existing)
                state['candidate'] = candidate; save_json(path, state)
            candidate = state['candidate']
            if candidate['decision'] != 'propose_program':
                state.update(status='complete', decision=candidate['decision']); save_json(path, state); continue
            # Count durable first-generation intents, including rejected or interrupted responses.
            generated = list((root / 'requests').glob('op*/build.json'))
            if not (root / 'requests' / key / 'build.json').exists() and len(generated) >= cfg['max_builder_generations']:
                state.update(status='complete', decision='builder_cap_reached'); save_json(path, state); continue
            build_material = {'candidate': candidate, 'source': material(op, spans), 'permissions': permissions}
            try:
                response, receipt = clients['tool_builder'].ask(key + '/build', BUILDER, build_material)
                program = program_from_response(response, candidate, config['program_environment'], permissions['allowed_names'])
            except (ModelOutputError, ValueError, SyntaxError) as exc:
                state['errors'].append({'stage': 'builder', 'error': str(exc)})
                if state['repair_used'] and not (root / 'requests' / key / 'repair.json').exists(): raise
                state['repair_used'] = True; save_json(path, state)
                repair_data = {**build_material, 'original_response': rejected_response(root / 'requests' / key / 'build.json'),
                               'actual_error': str(exc)}
                response, receipt = clients['tool_builder'].ask(key + '/repair', BUILDER, repair_data, repair=True)
                program = program_from_response(response, candidate, config['program_environment'], permissions['allowed_names'])
            value = bundle(candidate, program, root / 'requests' / key, response)
            value.update(opportunity_index=index, repair_used=state['repair_used'], response_metadata=receipt['turn'].get('provider_metadata'))
            # A repeated identical implementation is not a new asset/generation success.
            prior = root / 'pilot_bank' / (program['id'] + '.json')
            if prior.exists(): state.update(status='complete', decision='duplicate_candidate', program_id=program['id'])
            else:
                save_json(prior, value); (root / 'programs' / (program['id'] + '.py')).write_text(program['source'], encoding='utf-8')
                state.update(status='complete', decision='candidate_created', program_id=program['id'])
        except (ModelOutputError, ValueError, SyntaxError) as exc:
            state.update(status='complete', decision='candidate_rejected')
            state['errors'].append({'stage': 'proposal_or_builder', 'error': str(exc)})
        save_json(path, state); index_outputs(root)
    index_outputs(root)
    return report(root)


def probe(root):
    plan, config = checked(root, corpus=True)
    transfer = [s for s in rows(root / 'source_span_catalog.jsonl') if s['source_task']['task_id'] in plan['config']['transfer_tasks']]
    for path in sorted((root / 'pilot_bank').glob('*.json')):
        value = read(path)
        if value.get('probe_complete'): continue
        folder = root / 'probes' / value['id'] / ('revision1' if value.get('probe_repair') else 'revision0')
        plan_path = folder / 'plan.json'
        if not plan_path.exists(): save_json(plan_path, probe_cases(value, transfer))
        probes = read(plan_path)
        if probes['status'] != 'ready':
            value.update(probe_complete=True, coverage='coverage_insufficient', probe_plan=probes)
            save_json(path, value); continue
        source_records = read(root / 'source_records.json')
        receipts = []
        for case in probes['cases']:
            output = folder / (case['label'] + '.json'); intent = output.with_suffix('.intent.json')
            if output.exists(): receipts.append(read(output)); continue
            if intent.exists(): raise RuntimeError('Unresolved Worker probe; preserve receipt and inspect before retry')
            save_json(intent, {'case_hash': digest(case), 'started_at': utc(), 'program_hash': digest(value['program'])})
            task = next(s['task'] for s in source_records if s['task']['physical_key'] == case['source_case_id'])
            receipt = execute_probe(value, case, config, task)
            save_json(output, receipt); receipts.append(receipt)
        passed = all(r['passed'] for r in receipts)
        value.update(status=('cross_train_passed' if 'transfer_span_id' in probes['cases'][1] else 'worker_probe_passed')
                     if passed else 'worker_probe_failed', probe_complete=True,
                     probe_receipts=[str(folder / (c['label'] + '.json')) for c in probes['cases']])
        save_json(path, value)
    index_outputs(root)
    return report(root)


def repair(root):
    """Explicit paid continuation, sharing the one repair slot with initial Builder errors."""
    plan, config = checked(root, corpus=True); governor = PhaseGovernor(root, 'learn'); permissions = permission(config)
    client = RequestStore(root / 'requests', provider_for(config, 'tool_builder', governor), phase='learn')
    for path in sorted((root / 'pilot_bank').glob('*.json')):
        value = read(path)
        pending_repair = value.get('worker_repair_pending', False)
        if value['status'] != 'worker_probe_failed' or value.get('repair_used') and not pending_repair: continue
        failures = [read(p) for p in value['probe_receipts'] if not read(p)['passed']]
        # Only actual execution/schema errors authorize repair; ordinary not_found/oracle failures are outcomes.
        error = next((r for r in failures if r['result'].get('status') == 'execution_error'), None)
        if error is None: continue
        key = f"op{value['opportunity_index']}/worker_repair"
        original = path.with_suffix('.original')
        if not original.exists(): save_json(original, value)
        value.update(repair_used=True, worker_repair_pending=True); save_json(path, value)
        data = {'candidate': value['candidate'], 'original_source': value['program']['source'],
                'actual_error': error, 'permissions': permissions}
        try:
            response, receipt = client.ask(key, BUILDER, data, repair=True)
            program = program_from_response(response, value['candidate'], config['program_environment'], permissions['allowed_names'])
            if program['id'] == value['id']:
                value.update(repair_error='unchanged_source_after_error',worker_repair_pending=False)
                save_json(path,value);continue
            replacement = bundle(value['candidate'], program, root / 'requests' / key, response)
            replacement.update(opportunity_index=value['opportunity_index'], repair_used=True, probe_repair=True,
                               parent_program_id=value['id'], response_metadata=receipt['turn'].get('provider_metadata'))
            value.update(superseded_by=program['id'], worker_repair_pending=False); save_json(path, value)
            save_json(root / 'pilot_bank' / (program['id'] + '.json'), replacement)
            (root / 'programs' / (program['id'] + '.py')).write_text(program['source'], encoding='utf-8')
        except (ModelOutputError, ValueError, SyntaxError) as exc:
            value.update(repair_error=str(exc), worker_repair_pending=False); save_json(path, value)
    index_outputs(root)
    return probe(root)


def report(root):
    plan = read(root / 'pilot_manifest.json'); ledger = read(root / 'budget/usage.json')
    index_outputs(root); bundles = rows(root / 'program_bundles.jsonl')
    decisions = [read(p) for p in sorted((root / 'decisions').glob('*.json'))]
    own = [a for a in ledger['attempts'].values() if a['context'].get('office_v4_run') == plan['run_id']]
    consumption = [read(p) for p in sorted((root / 'consumption').glob('*/*/trace.json'))]
    outcome_path = root / 'private_audit/source_outcomes.json'
    outcomes = {r['task_id']: r.get('score', {}).get('hard') for r in read(outcome_path)} if outcome_path.exists() else {}
    source_ids = {b['generation_basis']['source_task']['task_id'] for b in bundles}
    if consumption: jsonl(root / 'pilot_consumption_traces.jsonl', consumption)
    result = {'method_version': VERSION, 'run_id': plan['run_id'], 'created_at': utc(),
        'decisions': dict(Counter(d.get('decision', 'pending') for d in decisions)),
        'candidate_statuses': dict(Counter(b['status'] for b in bundles if not b.get('superseded_by'))),
        'candidate_source_tasks': sorted(source_ids),
        'source_correctness_counts': dict(Counter('correct' if outcomes[t] is True else 'incorrect' if outcomes[t] is False else 'unknown'
            for t in source_ids if t in outcomes)),
        'coverage': dict(Counter(b.get('coverage','probed' if b.get('probe_complete') else 'pending') for b in bundles)),
        'phase_usage': {phase: {'tokens':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in own if a['context'].get('office_v4_phase')==phase),
            'http':sum(a['context'].get('office_v4_phase')==phase for a in own)} for phase in PHASE_LIMITS},
        'new_tokens': sum(a.get('accounted_tokens', a['reserved_tokens']) for a in own), 'new_http': len(own),
        'cumulative_tokens': sum(a.get('accounted_tokens', a['reserved_tokens']) for a in ledger['attempts'].values()),
        'cumulative_http': len(ledger['attempts']), 'unknown_billing': ledger['unknown_billing'],
        'paired_arms': len(consumption), 'outputs_consumed': sum(len(r.get('consumed', [])) for r in consumption),
        'parent_frozen_unchanged': tree_identity(plan['parent_bank']) == plan['parent_frozen'],
        'formal_promotions': 0, 'val_test_started': False,
        'next': 'explicit_paid_authorization_required' if not (root / 'authorization.json').exists() else 'train_only_review'}
    save_json(root / 'pilot_review.json', result)
    (root / 'pilot_review.md').write_text('# Office Trace V4 pilot\n\n' +
        'Train-only 候选试验；旧 Frozen 只读。Worker 通过不等于正式 usable；无真实消费不声称降本。\n\n' +
        '```json\n' + json.dumps(result, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    return result


@contextmanager
def exclusive(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.pilot.lock').open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'check', 'learn', 'probe', 'repair', 'consume', 'report'])
    parser.add_argument('--source'); parser.add_argument('--parent-bank'); parser.add_argument('--corpus-root')
    parser.add_argument('--config', default='configs/officeqa_trace_v4_program_probe_seed42.yaml')
    parser.add_argument('--output', required=True); parser.add_argument('--env-file'); parser.add_argument('--zero-http', action='store_true')
    args = parser.parse_args(); root = Path(args.output).resolve()
    if args.stage == 'prepare':
        result = prepare(args)
    else:
        with exclusive(root):
            try:
                if args.stage in {'learn', 'repair', 'consume'}:
                    if args.env_file: load_env(args.env_file)
                if args.stage == 'check':
                    if not args.zero_http: raise ValueError('check requires --zero-http')
                    from .office_v4_offline import check
                    result = check(root)
                elif args.stage == 'consume':
                    from .office_v4_consumption import consume
                    result = consume(root)
                else: result = globals()[args.stage](root)
            except Exception as exc:
                save_json(root / 'last_error.json', {'stage': args.stage, 'timestamp': utc(), 'error': str(exc),
                    'error_code': getattr(exc, 'code', type(exc).__name__), 'classified_as':
                    'budget' if isinstance(exc, BudgetExhausted) else 'infrastructure_or_identity', 'http_retried': False})
                raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
