"""R3 production-path checks; all model responses here are explicitly intercepted fixtures."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint, save_json
from atomic_skillgraph.empirical.contracts import PublicTask, digest, validate_schema_instance
from atomic_skillgraph.empirical.searchqa_guidance_r3 import (VERSION, POLICY, MATERIAL, NECESSARY, FAMILIES,
    IMAGE, RuntimeView, select, normalize_finish, validate_step)
from atomic_skillgraph.empirical.searchqa_trace_learning_r3 import (partition, public_case, validate_candidate,
    specification, full_validator, induce, learning_units)
from atomic_skillgraph.empirical import searchqa_program_probe_r3 as probe
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.empirical.answer_executor import STEP
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.public_results import SEARCHQA_R3_EXECUTION, PROGRAM_EXECUTION
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from atomic_skillgraph.experiments.run_searchqa_trace_r3 import PhaseGovernor, PROPOSED, paired, immutable
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_empirical import config_for
from test_cf2_contracts import http, response


def task(i=1, split='train', family='evidence_locate'):
    example = probe.boundary_inputs(family)['boundary:unicode_parameter']
    return PublicTask('fixture:' + str(i), 'physical:' + str(i), example['question'].replace('Ζήτα', 'Station'),
        {'context': example['context'].replace('Ζήτα', 'Station') + ' Detail ' + str(i) + '.'}, split)


def candidate(family='evidence_locate'):
    tasks = [task(1, family=family), task(2, family=family)]
    sources = {t.physical_key: public_case({'task': asdict(t), 'physical_key': t.physical_key,
        'prediction': '<answer>advisory</answer>', 'correct': i == 0, 'trace_sha256': digest(asdict(t))}) for i, t in enumerate(tasks)}
    refs = []
    for key, source in sources.items():
        doc = source['documents'][0]
        refs.append({'case_id': key, **{k: doc[k] for k in ('doc_index', 'start', 'end', 'quote_sha')}, 'basis': 'posthoc_public_check'})
    c = {'kind': 'atomic_procedure', 'method_version': VERSION, 'family': family, 'operation': family,
        'goal': 'Extract bounded public evidence', 'applicability': {'family': family, 'public_evidence_required': [NECESSARY[family]]},
        'procedure': ['Apply this typed public operation only when its explicit source cues are present.'],
        'stop_condition': 'Abstain when evidence is missing or ambiguous.', 'source_case_ids': list(sources),
        'evidence_refs': refs, 'parent_ids': [], 'execution_intent': 'guidance_only', 'output_role': 'intermediate'}
    return c, sources


def config(root):
    c = config_for(root / 'bank')
    c['experiment'].update(benchmark='searchqa', searchqa_method_version=VERSION, output_dir=str(root / 'out'))
    c['harness'] = {'adapter': 'searchqa'}
    c['program_environment'].update(adapter_abi='simple.v2', image_digest=IMAGE)
    c['budget'] = {'eval_task_tokens': 96000, 'token_limit': 1000000, 'request_limit': 100, 'finish_reserve': 0}
    c['llm']['runtime'].update(max_completion_tokens=32768, reasoning_effort='high')
    c['llm']['protocol'] = {'thinking_type': 'enabled'}
    return validate_config(benchmark_config(c, 'searchqa'))


SOURCE = '''import re
def run(ctx, inputs):
    text = inputs['context']; question = inputs['question']
    if not text.strip() or not question.strip():
        return {'status':'needs_input','outputs':{}}
    markers = list(re.finditer(re.escape('[DOC]'), text))
    spans = [(m.end(), markers[i+1].start() if i+1<len(markers) else len(text)) for i,m in enumerate(markers)]
    if not markers: spans = [(0,len(text))]
    words = set(re.findall(r'\\w+',question.casefold())) - {'what','where','which','is','the','was','in','for','a','of'}
    found=[]
    for i,(start,end) in enumerate(spans):
        quote=text[start:end]
        if words & set(re.findall(r'\\w+',quote.casefold())) and len(quote)<=600:
            found.append({'doc_index':i+1,'start':start,'end':end,'quote':quote,'rank_reason':'question_term_overlap'})
    return {'status':'ok','outputs':{'matches':found[:8]}} if found else {'status':'not_found','outputs':{}}
'''


def make_program(tmp_path, family='evidence_locate'):
    c, sources = candidate(family); asset = validate_candidate(c, sources)
    return probe.prepare(SOURCE, specification(asset), asset, config(tmp_path)['program_environment']), asset, sources


def test_S03_partition_hides_all_new_probe_history():
    tasks = [task(i) for i in range(300)]; exposed = [t.physical_key for t in tasks[:19]]
    group = partition(tasks, exposed)
    assert group == partition(list(reversed(tasks)), list(reversed(exposed)))
    sets = [set(group[k]) for k in ('induction', 'worker_probe', 'utility_probe')]
    assert list(map(len, sets)) == [240, 30, 30]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert not set(exposed) & (sets[1] | sets[2])


@pytest.mark.parametrize('bad', ['source', 'offset', 'hash', 'parent', 'length', 'one_source', 'no_change'])
def test_S04_S05_full_validator_retains_original_for_one_repair(bad):
    c, sources = candidate(); value = {'decision': 'candidate', 'candidate': deepcopy(c), 'reason': 'fixture'}
    if bad == 'source': value['candidate']['source_case_ids'][0] = 'val:forbidden'
    if bad == 'offset': value['candidate']['evidence_refs'][0]['end'] += 99999
    if bad == 'hash': value['candidate']['evidence_refs'][0]['quote_sha'] = 'bad'
    if bad == 'parent': value['candidate']['parent_ids'] = ['human-bank']
    if bad == 'length': value['candidate']['procedure'] = ['x' * 300, 'y' * 300]
    if bad == 'one_source': value['candidate']['source_case_ids'] = ['physical:1'] * 2
    if bad == 'no_change': value['decision'] = 'no_change'
    original = deepcopy(value)
    with pytest.raises(ValueError) as error: full_validator(sources, {'immutable': True})(value)
    assert error.value.repair_material['original_proposal'] == original
    assert error.value.repair_material['legal_source_ids'] == sorted(sources)
    assert error.value.repair_material['original_material_hash'] == digest({'immutable': True})


def test_S04_S07_actual_repair_loop_and_builder_wire(tmp_path, monkeypatch):
    c, sources = candidate(); broken = deepcopy(c); broken['procedure'] = ['x' * 300, 'y' * 300]
    seen = http(monkeypatch, [response([{'decision': 'candidate', 'candidate': broken, 'reason': 'fixture'}], name='submit_searchqa_candidate'),
        response([{'decision': 'candidate', 'candidate': c, 'reason': 'fixture repair'}], name='submit_searchqa_candidate'),
        response([{'source': SOURCE}], name='submit_searchqa_code')])
    system = EmpiricalSystem(config(tmp_path), harness=AnswerAdapter('searchqa', {}))
    for i in (1, 2): system.bank.save_case(task(i), {'searchqa_role': 'induction'})
    system.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint'); system.audit_path = tmp_path / 'requests.json'
    try:
        result = induce(system.learner, {'id': 'fixture', 'family': c['family'], 'case_ids': list(sources)}, sources)
        assert result['decision'] == 'candidate' and len(seen) == 2
        repair = json.loads(seen[1]['messages'][1]['content'])
        assert repair['original_proposal']['candidate'] == broken and repair['legal_source_ids'] == sorted(sources)
        asset = system.bank.get(result['asset_id'])
        program = probe.build(system.learner, specification(asset), asset, sources)
        assert program['state'] == 'candidate' and len(seen) == 3
        assert seen[-1]['thinking']['type'] == 'disabled' and seen[-1]['max_tokens'] == 4096
        assert system.requests[-1]['effective_call_settings']['max_retries'] == 0
        assert seen[-1]['tool_choice']['function']['name'] == 'submit_searchqa_code'
        assert system.requests[-1]['payload_max_bytes'] == 65536
        assert probe.build(system.learner, specification(asset), asset, sources) == program and len(seen) == 3
    finally: system.close()


@pytest.mark.parametrize('kind', FAMILIES)
def test_S08_typed_contracts_have_real_positive_examples(tmp_path, kind):
    program, _, _ = make_program(tmp_path, kind)
    args = probe.boundary_inputs(kind)['boundary:unicode_parameter']
    from atomic_skillgraph.empirical.searchqa_guidance import documents
    spans = [{k: d[k] for k in ('doc_index', 'start', 'end', 'quote')} for d in documents(args['context'])]
    fact = {'relation': 'born_in', 'subject': 'Ζήτα', 'candidate': 'Theta', 'source_ref': spans[0], 'ambiguity': False}
    values = {'evidence_locate': {'matches': [{**s, 'rank_reason': 'question_term_overlap'} for s in spans]},
        'explicit_alias': {'aliases': [{'alias_pair': ['Ζήτα', 'Theta'], 'cue': 'also known as', 'source_ref': spans[0]}]},
        'typed_numeric_compare': {'values': [{'subject': 'Ζήτα', 'value': v, 'unit': 'kg', 'time': '', 'span': s} for v, s in zip(['12', '18'], spans)],
                                  'comparison': {'left': 0, 'right': 1, 'relation': 'less'}},
        'explicit_relation': {'facts': [fact]}, 'multi_doc_join': {'entity_keys': ['a-42'], 'evidence_refs': spans, 'conflicts': []},
        'source_check': {'candidate': 'Theta', 'public_support_refs': [fact], 'abstain': False}}
    assert probe.check(program, args, {'status': 'ok', 'outputs': values[kind]})['passed']
    with pytest.raises(ValueError): probe.check(program, {**args, 'context': 'x' + args['context']}, {'status': 'ok', 'outputs': values[kind]})
    with pytest.raises(ValueError): probe.check(program, args, {'status': 'ok', 'outputs': {'matches': []}})


def test_S09_ambiguity_unit_and_alias_are_not_semantic_guesses(tmp_path):
    from atomic_skillgraph.empirical.searchqa_guidance import documents
    p, _, _ = make_program(tmp_path, 'explicit_relation')
    args = probe.boundary_inputs('explicit_relation')['boundary:ambiguous']; d = documents(args['context'])[0]
    fact = {'subject': 'Ζήτα', 'relation': 'born_in', 'candidate': 'Theta', 'ambiguity': False,
            'source_ref': {k: d[k] for k in ('doc_index', 'start', 'end', 'quote')}}
    with pytest.raises(ValueError): probe.check(p, args, {'status': 'ok', 'outputs': {'facts': [fact]}})
    p, _, _ = make_program(tmp_path, 'typed_numeric_compare')
    args = probe.boundary_inputs('typed_numeric_compare')['boundary:unit_conflict']; docs = documents(args['context'])
    output = {'values': [{'subject': 'Ζήτα', 'value': v, 'unit': u, 'time': '', 'span': {k: d[k] for k in ('doc_index', 'start', 'end', 'quote')}}
        for v, u, d in zip(['12', '18'], ['kg', 'miles'], docs)], 'comparison': {'left': 0, 'right': 1, 'relation': 'less'}}
    with pytest.raises(ValueError, match='units'): probe.check(p, args, {'status': 'ok', 'outputs': output})


def test_S10_S11_private_receipt_cannot_be_read_or_bound():
    ctx = TaskContext({}); raw = {'status': 'ok', 'outputs': {'evidence': 'visible'}, 'execution_receipt': {'secret': 'host-only'}}
    rid = ctx.register('test', raw, origin=SEARCHQA_R3_EXECUTION)
    assert 'execution_receipt' not in json.dumps(ctx.view(rid)) and ctx.results[rid] == raw
    assert ctx.resolve_public({'result_id': rid, 'path': ['outputs', 'evidence']}) == 'visible'
    assert ctx.read(rid, path=['outputs', 'evidence'])['data'] == 'visible'
    for path in (['execution_receipt'], ['execution_receipt', 'secret']):
        with pytest.raises(ValueError): ctx.resolve_public({'result_id': rid, 'path': path})
        with pytest.raises(ValueError): ctx.read(rid, path=path)
    with pytest.raises((ValueError, KeyError)): ctx.read('forged', path=['outputs'])
    old = ctx.register('r2', raw, origin=PROGRAM_EXECUTION)
    assert 'execution_receipt' in ctx.view(old)  # Scoped change; old version remains reproducible.


@pytest.mark.parametrize('change', ['valid', 'unknown', 'wrong_action', 'read_result', 'missing_answer', 'unrelated'])
def test_S12_finish_normalization_is_conservative(change):
    raw = {'action': 'finish', 'answer': '<answer>x</answer>', 'unit': 'characters', 'end': 3}
    if change == 'unknown': raw['offset'] = 0
    if change == 'wrong_action': raw['action'] = 'call_program'
    if change == 'read_result': raw.update(action='read_result', result_id='result', path=['outputs'])
    if change == 'missing_answer': raw['answer'] = ''
    if change == 'unrelated': raw['source'] = 'unrelated'
    normalized, audit = normalize_finish(raw, STEP)
    if change == 'valid':
        validate_step(normalized, STEP); assert audit['raw_arguments'] == raw and normalized['answer'] == raw['answer']
    else:
        with pytest.raises(ValueError): validate_step(normalized, STEP)


@pytest.mark.parametrize('finish', ['length', 'stop'])
def test_S13_length_only_finalization_retains_complete_public_input(tmp_path, monkeypatch, finish):
    seen = http(monkeypatch, [response(finish=finish), response(finish='stop', content='<answer>correct</answer>')])
    c = config(tmp_path); t = task(9)
    b = Bank(tmp_path / 'bank'); b.bind_searchqa_method(VERSION); b.freeze(tmp_path / 'frozen'); b.close()
    c['data_dir'] = str(tmp_path / 'frozen'); c['experiment']['runtime_mode'] = 'frozen'
    system = EmpiricalSystem(c, harness=AnswerAdapter('searchqa', {t.task_id: {'answers': ['correct']}}), readonly=True,
        bank_view_factory=lambda b: RuntimeView(b, 'NoSkill'))
    try:
        result = system.run_task(t, learn=False)
        assert len(seen) == (2 if finish == 'length' else 1), (result['execution'], [(r.get('purpose'), r.get('error')) for r in system.requests])
        if finish == 'length':
            final = json.loads(seen[1]['messages'][1]['content'])
            assert final['goal'] == t.goal and final['inputs'] == t.inputs
            assert seen[1]['thinking']['type'] == 'disabled' and seen[1]['max_tokens'] == 2048 and not seen[1].get('tools')
            assert result['score']['hard']
    finally: system.close()


def test_S15_S16_S21_empty_selection_and_manual_contamination(tmp_path):
    c, sources = candidate(); asset = validate_candidate(c, sources)
    bank = Bank(tmp_path / 'bank'); bank.bind_searchqa_method(VERSION)
    for i in (1, 2): bank.save_case(task(i), {'searchqa_role': 'induction'})
    bank.put('skill', asset)
    assert not select([asset], PublicTask('x', 'x', 'Who?', {'context': '[DOC] one document'}), POLICY)['selected']
    assert len(select([asset, {**asset, 'id': 'skill_other'}], task(), POLICY)['selected']) == 1
    with pytest.raises(ValueError, match='external/manual'): bank.put('skill', {'id': 'manual', 'goal': 'human', 'guidance': 'human'})
    bank.close()
    legacy = Bank(tmp_path / 'legacy'); legacy.put('skill', {'goal': 'human'})
    with pytest.raises(ValueError, match='fresh'): legacy.bind_searchqa_method(VERSION)
    legacy.close()


def test_S18_terminal_builder_failure_never_issues_again(tmp_path, monkeypatch):
    c, sources = candidate(); asset = validate_candidate(c, sources)
    seen = http(monkeypatch, [response(finish='length')])
    system = EmpiricalSystem(config(tmp_path), harness=AnswerAdapter('searchqa', {}))
    system.checkpoint = TaskCheckpoint(tmp_path / 'checkpoint'); system.audit_path = tmp_path / 'requests.json'
    try:
        for _ in range(2):
            with pytest.raises(ValueError): probe.build(system.learner, specification(asset), asset, sources)
        assert len(seen) == 1 and len(list((tmp_path / 'checkpoint/responses').glob('*.json'))) == 1
        assert len(system.usage.events) == 1
    finally: system.close()


def test_S20_stage_caps_unknown_billing_and_D3_prefix_accounting(tmp_path):
    from atomic_skillgraph.empirical.searchqa_trace_learning import sha
    from atomic_skillgraph.core.errors import BudgetExhausted
    save_json(tmp_path / 'method_lock.json', {'run_id': 'fixture'})
    limits = {'token_limit': 10000000, 'request_limit': 500, 'finish_reserve': 0}
    state = {'limits': limits, 'attempts': {'prefix': {'context': {'searchqa_run_id': 'fixture', 'searchqa_phase': 'D3a'},
        'status': 'metered', 'accounted_tokens': 2000000, 'reserved_tokens': 2000000}}, 'unknown_billing': False}
    save_json(tmp_path / 'budget.json', state); save_json(tmp_path / 'inherited_budget.json', {**state, 'attempts': {}})
    with pytest.raises(ValueError, match='not authorized'): PhaseGovernor(tmp_path, 'D3')
    save_json(tmp_path / 'authorizations/D3.json', {'approved': True, 'method_lock_sha256': sha(tmp_path / 'method_lock.json'), 'phases': {'D3': PROPOSED['D3']}})
    g = PhaseGovernor(tmp_path, 'D3'); assert g.headroom({})['tokens'] == 3800000
    with pytest.raises(BudgetExhausted): g.admit('too-large', {'max_tokens': 4000000}, {})
    g.admit('unknown', {'max_tokens': 100}, {})
    resumed = PhaseGovernor(tmp_path, 'D3'); assert resumed.billing_blocked
    with pytest.raises(BudgetExhausted): resumed.admit('next', {'max_tokens': 1}, {})


@pytest.mark.skipif(not os.environ.get('PROGRAM_IMAGE_DIGEST'), reason='requires real locked Docker')
def test_S14_S17_S19_real_worker_bank_freeze_runtime_and_read_consumption(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY', 'intercepted-only')
    c = config(tmp_path); program, asset, _ = make_program(tmp_path)
    b = Bank(tmp_path / 'bank', seed=42, environment=c['program_environment']); b.bind_searchqa_method(VERSION)
    for i, role in [(1, 'induction'), (2, 'induction'), (3, 'worker_probe')]: b.save_case(task(i), {'searchqa_role': role})
    b.put('skill', asset); program = b.put('program', program)
    worker = ProgramWorker({'record_execution_receipt': True, 'wall_timeout_seconds': 30})
    receipts = [probe.execute(b, worker, program, task(i), tmp_path / 'receipts', role='source' if i < 3 else 'transfer') for i in (1, 2, 3)]
    for role, args in probe.boundary_inputs('evidence_locate').items():
        receipts.append(probe.execute(b, worker, program, task(), tmp_path / 'receipts', role=role, arguments=args))
    assert not probe.qualify(b, program, [r for r in receipts if not (r['role'] == 'source' and r['task_key'] == 'physical:2')])['passed']
    assert probe.qualify(b, program, receipts)['passed']
    program = b.get(program['id']); assert b.program_eligible(program) and len(b.all('implementation')) == 1
    unknown = task(99); claim = digest([program['id'], unknown.physical_key, 'transfer', probe.inputs(unknown)])
    save_json(tmp_path / 'receipts' / (claim + '.started.json'), {})
    with pytest.raises(RuntimeError, match='Unresolved'): probe.execute(b, worker, program, unknown, tmp_path / 'receipts', role='transfer')
    b.freeze(tmp_path / 'frozen'); b.close()
    c['data_dir'] = str(tmp_path / 'frozen'); c['experiment']['runtime_mode'] = 'frozen'; calls = []
    def post(*args, **kwargs):
        payload = kwargs['json']; calls.append(payload); material = json.loads(payload['messages'][1]['content'])
        assert 'execution_receipt' not in json.dumps(material)
        if len(calls) == 1: action = {'action': 'call_program', 'name': program['id'], 'arguments': {}}
        elif len(calls) == 2: action = {'action': 'read_result', 'result_id': material['local_results'][0]['result_id'], 'path': ['outputs', 'matches']}
        else: return SimpleNamespace(status_code=200, ok=True, headers={}, json=lambda: response(finish='stop', content='<answer>correct</answer>'))
        return SimpleNamespace(status_code=200, ok=True, headers={}, json=lambda: response([action], name='answer_step'))
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
    t = task(4, 'val'); system = EmpiricalSystem(c, harness=AnswerAdapter('searchqa', {t.task_id: {'answers': ['correct']}}),
        readonly=True, bank_view_factory=lambda b: RuntimeView(b, 'Atomic-full'))
    system.worker = worker; before = system.bank.digest()
    try:
        trace = system.run_task(t, learn=False); a = trace['execution']['attempts'][0]
        assert trace['score']['hard'] and a['program_executed'] and len(a['exposed_request_ids']) == 2 and len(a['read_request_ids']) == 1
        assert not a['outputs_consumed'] and not a['declared_used_request_ids'] and not a['submission_by_program']
        assert system.bank.digest() == before and trace['method_identity']['searchqa_method_version'] == VERSION
    finally: system.close()


def test_S01_S02_S04_S06_S10_real_D0_archive_when_available():
    root = os.environ.get('SEARCHQA_R3_D0')
    if not root: pytest.skip('requires actual immutable R2 audit, not invented historical records')
    root = Path(root); audit = json.loads((root / 'd0_host_audit.json').read_text())
    assert audit['http'] == 0 and audit['historical_files_verified'] == 8700
    assert len(audit['exposed_ids']) == 19 and len(audit['induction_rejections']) == 4
    assert sum(r['classification'] == 'completion_length' for r in audit['builder_responses']) == 6
    assert len(audit['receipt_projections']) == 7 and all(r['receipt_removed'] and r['outputs_unchanged'] for r in audit['receipt_projections'])
    exact = [r for r in audit['protocol_replays'] if 'raw_call' in r]
    assert exact and all(r['raw_call']['arguments']['action'] == 'read_result' and r['replay_error'] for r in exact)
    assert any(r.get('r3_eligible_for_one_finish') for r in audit['protocol_replays'])
    sources = json.loads((root / 'learning_sources.json').read_text()); groups = json.loads((root / 'train_partition.json').read_text())
    assert not set(sources) & set(groups['utility_probe'])
    assert any(s['feedback_qualification'] == 'failed_task_demand' for s in sources.values())
    assert all(not {'gold', 'answers', 'score', 'correct'} & set(s) for s in sources.values())


def test_S22_pairs_reject_changed_public_input():
    t = asdict(task()); c = {'task': t, 'score': {'hard': True, 'soft': 1}, 'execution': {}, 'usage': [], 'requests': []}
    result = paired({'NoSkill': [c], 'Guidance-only': [c], 'Atomic-full': [c]})
    assert all(v['em'] == 1 for v in result['arms'].values())
    bad = {**c, 'task': {**t, 'goal': 'changed'}}
    with pytest.raises(ValueError, match='differ'): paired({'NoSkill': [c], 'Atomic-full': [bad]})


def test_S04_S07_real_rejections_and_failed_spec_material_use_production_contracts(tmp_path, monkeypatch):
    root = os.environ.get('SEARCHQA_R3_D0')
    if not root: pytest.skip('requires real R2 D0 artifacts')
    from atomic_skillgraph.experiments.searchqa_r3_audit import material
    from atomic_skillgraph.empirical.searchqa_trace_learning import read
    from atomic_skillgraph.experiments.run_searchqa_trace_r2 import unit
    root = Path(root); sources = read(root / 'learning_sources.json'); audit = read(root / 'd0_host_audit.json')
    for row in audit['induction_rejections']:
        c = deepcopy(row['old_raw_candidate'])
        c.update(method_version=VERSION, family='evidence_locate', operation='locate_question_term_evidence', output_role='intermediate',
                 applicability={'family': 'evidence_locate', 'public_evidence_required': ['question_term_overlap']})
        with pytest.raises(ValueError) as error:
            full_validator({k: sources[k] for k in c['source_case_ids']}, {'old_response_hash': row['response_hash']})(
                {'decision': 'candidate', 'candidate': c, 'reason': 'schema migration replay only'})
        assert '600' in str(error.value) or 'maxLength' in str(error.value)
        assert error.value.repair_material['original_proposal']['candidate'] == c
    seen = http(monkeypatch, [response([{'source': SOURCE}], name='submit_searchqa_code') for _ in range(3)])
    system = EmpiricalSystem(config(tmp_path), harness=AnswerAdapter('searchqa', {}))
    try:
        for i, parent in enumerate(read(root / 'r2_reuse_material.json')):
            c = deepcopy(parent['old_asset']['searchqa_metadata']['candidate'])
            c.update(method_version=VERSION, family='evidence_locate', operation='locate_question_term_evidence', output_role='intermediate',
                applicability={'family': 'evidence_locate', 'public_evidence_required': ['question_term_overlap']}, parent_ids=[])
            asset = validate_candidate(c, sources); asset['searchqa_metadata']['legacy_parent'] = parent
            unit(system, tmp_path / 'build' / str(i), 'real_material_fixture:' + str(i))
            program = probe.build(system.learner, specification(asset), asset, sources)
            assert program['state'] == 'candidate'
            sent = material(system.requests[-1])
            assert sent['historical_failed_spec_parent']['old_spec'] == parent['old_spec']
        assert len(seen) == 3 and max(len(json.dumps(p).encode()) for p in seen) < 65536
    finally: system.close()


def test_S22_full_canonical_probe_and_val_with_intercepted_transport(tmp_path, monkeypatch):
    root = os.environ.get('SEARCHQA_R3_D0')
    if not root: pytest.skip('requires real canonical D0 material')
    import shutil
    import atomic_skillgraph.experiments.run_searchqa_trace_r3 as runner
    import atomic_skillgraph.experiments.run_empirical as empirical
    from atomic_skillgraph.empirical.searchqa_trace_learning import read, sha
    original = Path(root)
    for name in ('verify.json', 'source_identity.json', 'source_config.json', 'train_partition.json', 'train_tasks.json',
                 'learning_sources.json', 'induction_units.json', 'r2_exposure.json', 'r2_reuse_material.json', 'd0_identity.json', 'verification_paths.json'):
        shutil.copyfile(original / name, tmp_path / name)
    # This test's separate directory is never a run/Bank used for model evaluation.
    identity = {'git_sha': 'offline-fixture-only', 'tracked_dirty': False, 'source_sha256': 'offline-fixture-only'}
    monkeypatch.setattr(runner, 'code_identity', lambda: identity); monkeypatch.setattr(empirical, 'code_identity', lambda: identity)
    args = SimpleNamespace(output=str(tmp_path), source_run=read(original / 'source_identity.json')['source_run'],
        r2_run='/home/yangchengyu/searchqa_trace_r2_seed42_20261014_a418866',
        datasets='/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1/searchqa', external_reservation=None, deny_path=[])
    runner.prepare(args)
    for phase, caps in PROPOSED.items():
        save_json(tmp_path / 'authorizations' / (phase + '.json'), {'approved': True, 'authorization': 'offline interception fixture only',
            'method_lock_sha256': sha(tmp_path / 'method_lock.json'), 'phases': {phase: caps}})
    n = len(read(tmp_path / 'induction_units.json'))
    seen = http(monkeypatch, [*[response([{'decision': 'no_change', 'reason': 'offline fixture'}], name='submit_searchqa_candidate') for _ in range(n)],
        *[response(finish='stop', content='<answer>offline fixture</answer>') for _ in range(132)]])
    runner.induction(tmp_path); runner.build(tmp_path); runner.worker(tmp_path)
    runner.measure_train(tmp_path, 12); prefix_count = len(seen)
    prefix_hash = sha(tmp_path / 'measure_train/prefix/NoSkill/summary.json')
    runner.measure_train(tmp_path, 30)
    assert len(seen) - prefix_count == 36 and sha(tmp_path / 'measure_train/prefix/NoSkill/summary.json') == prefix_hash
    runner.freeze(tmp_path); result = runner.val(tmp_path)
    assert len(seen) == n + 132 and all(a['tasks'] == 24 for a in result['arms'].values())
    before = len(seen); runner.val(tmp_path); assert len(seen) == before
    final = runner.report(tmp_path)
    assert final['usable_program_count'] == 0 and not final['test_authorized'] and not final['unknown_billing']
    assert all(read(p)['unchanged'] for p in (tmp_path / 'val').glob('*/frozen_check.json'))
