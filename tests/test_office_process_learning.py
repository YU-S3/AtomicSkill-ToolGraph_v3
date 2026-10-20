"""Full-Train contract and real Docker learning/parameter/source replay; no model HTTP."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical import process_learning as process, public_pages
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.model_view import expand_material
from atomic_skillgraph.empirical.source_catalog import catalog_material, bind_proposal
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import office, http, response

spec = importlib.util.spec_from_file_location('process_entry', Path(__file__).parents[1]/'experiments/run_office_process_learning.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def make_system(tmp_path):
    old = office(tmp_path)
    base = deepcopy(old.config)
    old.close()
    base['budget'] = {'token_limit': 5000000, 'request_limit': 200, 'finish_reserve': 0}
    cfg = entry.configure(base, tmp_path/'run', tmp_path/'data', 'train')
    cfg['llm']['api_key_env'] = 'MODEL_API_KEY'
    cfg['experiment']['benchmark'] = 'officeqa'
    cfg['program_environment']['adapter_abi'] = 'simple.v2'
    cfg['program_environment']['image_digest'] = os.environ.get('PROGRAM_IMAGE_DIGEST', 'sha256:'+'a'*64)
    text = '# Public table, amounts in dollars\n| year | amount |\n| 2020 | 2 |\n| 2021 | 4 |\n| 2022 | 6 |\n' + '公开页尾\n'*500
    page = {'document': 'a.txt', 'page': '1', 'source_url': 'https://example.org/public',
            'text': text, 'text_digest': digest(text)}
    task = PublicTask('office', 'office-physical', 'Compute the mean of the amounts.',
        {'office_public_input_version': public_pages.VERSION, 'source_pages': [page]}, 'train')
    factory = lambda: OfficeAdapter({'office': {'answer': '4'}, 'next': {'answer': '20'}}, cfg)
    s = EmpiricalSystem(cfg, harness=factory(), adapter_factory=factory)
    s.adapter.reset(task)
    s.task_context = TaskContext(cfg['runtime'])
    public_pages.seed(task, s.task_context)
    s.phase = 'train'
    return s, task


def actual_case(s, task, *, answer=False):
    broker = Broker(s.adapter, 48, context=s.task_context)
    source = 'import json\nvalues=[2,4,6]\nprint(json.dumps({"mean":sum(values)/len(values)' + (', "answer":str(sum(values)/len(values))' if answer else '') + '}))'
    result = broker.call('execute_python', {'source': source})
    assert result['accepted'], result
    return {'tools': broker.events, 'score': {'hard': False},
        'execution': {'attempts': [], 'prediction': 'wrong', 'reason': 'finish'},
        'result_store': {'program_results': deepcopy(s.task_context.results)}}


def proposal_for(s, task, trace):
    experience = s.learner._experience(task, trace)
    material = catalog_material({'experience': process.case_view(experience, task.physical_key)})
    card = next(c for c in material['experience']['local_evidence'] if c['operation_name'] == 'execute_python')
    source = next(r['id'] for r in card['input_sources'] if r['value'] == [2, 4, 6])
    target = next(r['id'] for r in card['output_references'] if r.get('path') == ['data', 'computed', 'mean'])
    proposal = {'decision': 'propose_skill_and_program_spec', 'generate_program': True, 'skill': {
        'goal': 'Compute mean of a numerical series', 'guidance': 'Use the supplied numerical series.',
        'execution_intent': 'program_requested', 'result_role': 'intermediate',
        'entry_constraints': 'Nonempty numeric series',
        'input_schema': {'type': 'object', 'properties': {'values': {'type': 'array', 'items': {'type': 'number'}}}, 'required': ['values']},
        'output_schema': {'type': 'object', 'properties': {'mean': {'type': 'number'}}, 'required': ['mean']}},
        'realization_request': {'skill_id': '$new', 'action': 'build', 'case_bindings': [{
            'operation_id': card['operation_id'], 'input_ids': {'values': source}, 'output_ids': {'mean': target}}]}}
    return proposal, material, experience


def test_frozen_inputs_are_shared_and_no_train_holdout(tmp_path):
    s, task = make_system(tmp_path)
    try:
        for phase in ('train', 'val'):
            config = entry.configure(s.config, tmp_path/'other', tmp_path/'data', phase)
            assert config['budget']['eval_task_tokens'] == 216576
            assert config['runtime']['global_action_budget'] == 48
            assert config['experiment']['runtime_mode'] == ('online' if phase == 'train' else 'frozen')
        bad = deepcopy(s.config)
        bad['experiment']['choice_selection_physical_keys'] = ['held-out']
        with pytest.raises(ValueError, match='all Train'): validate_config(bad)
        bad = deepcopy(s.config); bad['experiment']['office_human_bank_version'] = 'human'
        with pytest.raises(ValueError, match='all Train'): validate_config(bad)
        wire = public_pages.runtime_material(s, {'task': {'inputs': {'source_pages': 'truncated'}}})
        assert wire['task']['inputs'] == task.inputs
        original = deepcopy(s.task_context.results)
        trial = TaskContext(s.config['runtime'])
        assert public_pages.seed(task, trial, recorded=original) == wire['provided_source_page_refs']
        assert trial.results == original
        original[next(iter(original))]['data']['text'] += 'tampered'
        with pytest.raises(ValueError, match='identity'):
            public_pages.seed(task, TaskContext(s.config['runtime']), recorded=original)
        assert {'read_context', 'grep_page'} <= {t['name'] for t in s.adapter.tool_definitions()}
    finally: s.close()


def test_actual_full_case_wire_preserved(tmp_path, monkeypatch):
    s, task = make_system(tmp_path)
    trace = {'tools': [], 'score': {'hard': False, 'gold': 'PRIVATE GOLD'},
             'execution': {'attempts': [], 'prediction': 'wrong', 'reason': 'finish'}}
    seen = http(monkeypatch, [response([{'decision': 'no_change'}], name='submit_learning')])
    try:
        outcome = s.learner.learn(task, trace)
        assert outcome['decision'] == 'no_change'
        payload = expand_material(json.loads(seen[0]['messages'][1]['content']))
        assert payload['experience']['task']['inputs'] == task.inputs
        assert payload['experience']['submission'] == 'wrong'
        assert 'PRIVATE GOLD' not in json.dumps(seen)
        assert s.bank.train_cases()[0]['task']['physical_key'] == task.physical_key
        assert s.config['learning']['min_distinct_train_cases_before_first_build'] == 1
    finally: s.close()


@pytest.mark.parametrize('wrong_constant', [False, True])
def test_learner_builder_real_docker_then_next_task(tmp_path, monkeypatch, wrong_constant):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'): pytest.skip('Requires locked Docker image')
    s, task = make_system(tmp_path)
    trace = actual_case(s, task)
    proposal, _, _ = proposal_for(s, task, trace)
    source = ("def run(ctx, inputs):\n values=inputs['values']\n return {'status':'ok','outputs':{'mean':"
              + ('4.0+0*len(values)' if wrong_constant else 'sum(values)/len(values)') + '}}')
    replies = [response([proposal], name='submit_learning'),
               response([{'source': source}], name='submit_program')]
    if wrong_constant: replies.append(response([{'source': source}], name='submit_program'))
    seen = http(monkeypatch, replies)
    try:
        result = s.learner.learn(task, trace)
        assert not result['errors'], result
        tests = result['tests']
        assert tests and tests[0]['validation']['variation']['passed'] is (not wrong_constant)
        asset = s.bank.get(result['program'])
        assert s.bank.program_eligible(asset) is (not wrong_constant)
        assert len(seen) == (3 if wrong_constant else 2)
        built = expand_material(json.loads(seen[1]['messages'][1]['content']))
        assert built['examples'][0]['public_task']['inputs'] == task.inputs
        if not wrong_constant:
            options = s.bank.program_options('mean series')
            assert asset['id'] in {p['id'] for p in options}
            other = PublicTask('next', 'next-physical', 'Mean of another numerical series', task.inputs, 'train')
            adapter = s.adapter_factory(); adapter.reset(other)
            try:
                broker = Broker(adapter, 48, context=TaskContext(s.config['runtime']))
                public_pages.seed(other, broker.context)
                actual = s.worker.execute(asset, {'values': [10, 20, 30]}, broker)
                assert actual['status'] == 'ok' and actual['outputs']['mean'] == 20
            finally: adapter.close()
            skill = s.bank.all('skill')[0]
            assert skill['evidence_source']['physical_key'] == task.physical_key
    finally: s.close()


def test_real_resource_membership_and_val_policy():
    path = os.environ.get('OFFICE_EXTERNAL_RESOURCES')
    if not path: pytest.skip('Set approved external resource snapshot')
    _, manifest, tasks = entry.resources(path)
    assert len(tasks['train']) == 120 and len(tasks['val']) == 24
    assert 'val_policy' in manifest
    # The approved legacy Val remains honestly identified; do not manufacture public pages.
    assert all(t.inputs['source_pages'] == [] for t in tasks['val'])


def test_actual_runtime_input_and_learning_boundary(tmp_path, monkeypatch):
    s, task = make_system(tmp_path)
    seen = http(monkeypatch, [response([{'action': 'finish', 'answer': '4'}]),
                             response([{'decision': 'no_change'}], name='submit_learning')])
    try:
        trace = s.run_task(task, learn=True)
        assert trace['score']['hard'] is True
        assert trace['learning']['decision'] == 'no_change'
        runtime = expand_material(json.loads(seen[0]['messages'][1]['content']))
        assert runtime['task']['inputs'] == task.inputs
        assert len(runtime['provided_source_page_refs']) == 1
        assert seen[0]['messages'][0]['content'].startswith(public_pages.ANSWER_CONTRACT)
        assert len(seen) == 2  # No separate planner, selection or verification model turn.
    finally: s.close()


def test_finish_output_contract_is_included_before_capacity_measurement(tmp_path, monkeypatch):
    s, task = make_system(tmp_path)
    seen = http(monkeypatch, [response(content='4', finish='stop')])
    try:
        packet = s.prepare_finish({'original_goal': task.goal}, 'Use the acquired evidence.')
        s.agent('runtime', 'Use the acquired evidence.', packet['materials'], None, None,
                decision_purpose='finish_only')
        assert seen[0]['messages'][0]['content'].startswith(public_pages.ANSWER_CONTRACT)
        assert seen[0]['max_tokens'] == 32768 and seen[0]['thinking']['type'] == 'disabled'
        from atomic_skillgraph.empirical.budget_governor import input_token_bound
        assert packet['serialized_input_bytes'] == input_token_bound(seen[0])
    finally: s.close()


def test_failed_task_cannot_qualify_a_final_answer(tmp_path):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'): pytest.skip('Requires locked Docker image')
    s, task = make_system(tmp_path)
    try:
        trace = actual_case(s, task, answer=True)
        proposal, material, case = proposal_for(s, task, trace)
        card = material['experience']['local_evidence'][0]
        answer_id = next(r['id'] for r in card['output_references']
                         if r.get('path') == ['data', 'computed', 'answer'])
        proposal['skill'].update(result_role='final_answer',
            output_schema={'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer']})
        proposal['realization_request']['case_bindings'][0]['output_ids'] = {'answer': answer_id}
        cases = {task.physical_key: (task, case)}
        bound = bind_proposal(proposal, material, cases, s.bank)
        with pytest.raises(ValueError, match='failed source task'):
            s.learner.validate_learning_proposal(bound, cases)
        case['score']['hard'] = True
        assert s.learner.validate_learning_proposal(bound, cases)['canonical_bindings']
    finally: s.close()


def test_source_page_program_trial_restores_exact_scope(tmp_path):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'): pytest.skip('Requires locked Docker image')
    from atomic_skillgraph.empirical.local_validation import canonical_binding, validate_on_source
    from atomic_skillgraph.empirical.model_view import operation_directory
    s, task = make_system(tmp_path)
    try:
        ref = public_pages.seed(task, s.task_context)[0]
        source = ("import json\nref=" + repr(ref) +
                  "\nr=ctx.read_result(ref['result_id'],path=ref['path'],limit=80)\nprint(json.dumps({'text':r['data']}))")
        broker = Broker(s.adapter, 48, context=s.task_context)
        assert broker.call('execute_python', {'source': source})['accepted']
        trace = {'tools': broker.events, 'score': {'hard': False}, 'execution': {'attempts': []},
                 'result_store': {'program_results': deepcopy(s.task_context.results)}}
        case = s.learner._experience(task, trace)
        evidence = next(e for e in case['local_evidence'] if e['operation'].get('name') == 'execute_python')
        inputs = next(r for r in operation_directory(evidence, task.physical_key)['input_sources'] if r['value'] == ref)
        program = s.bank.put('program', {'entry': 'run', 'environment': s.config['program_environment'],
            'allowed_tools': ['read_result'], 'result_role': 'intermediate',
            'input_schema': {'type': 'object', 'properties': {'ref': {'type': 'object'}}, 'required': ['ref']},
            'output_schema': {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']},
            'source': "def run(ctx, inputs):\n r=ctx.read_result(inputs['ref']['result_id'],path=inputs['ref']['path'],limit=80)\n return {'status':'ok','outputs':{'text':r['data']}}"})
        binding = canonical_binding({'case_id': task.physical_key, 'local_evidence_ref': evidence['id'],
            'inputs': {'ref': ref}, 'input_refs': {'ref': {k: v for k, v in inputs.items() if k != 'value'}},
            'reference_fields': {'text': ['data', 'computed', 'text']}}, case, program)
        result = validate_on_source(s, program, binding, task, case, 'source-page-trial')
        assert result['validation']['passed'], result
        assert result['result']['outputs']['text'] == task.inputs['source_pages'][0]['text'][:80]
    finally: s.close()


def test_val_uses_readonly_learned_bank_without_learning(tmp_path, monkeypatch):
    from atomic_skillgraph.experiments.formal_log import tree_identity
    s, task = make_system(tmp_path)
    skill = s.bank.put('skill', {'goal': 'Compute mean', 'guidance': 'Use a nonempty numerical series.',
        'execution_intent': 'guidance_only', 'entry_constraints': 'Numerical series is available',
        'input_schema': {'type': 'object'}, 'output_schema': {'type': 'object'}})
    frozen = tmp_path/'frozen'
    s.bank.freeze(frozen)
    cfg = deepcopy(s.config); cfg['experiment']['runtime_mode'] = 'frozen'; cfg['data_dir'] = str(frozen)
    s.close()
    before = tree_identity(frozen)
    val = PublicTask(task.task_id, task.physical_key, task.goal, task.inputs, 'val')
    factory = lambda: OfficeAdapter({'office': {'answer': '4'}}, cfg)
    s = EmpiricalSystem(cfg, readonly=True, harness=factory(), adapter_factory=factory)
    seen = http(monkeypatch, [response([{'action': 'finish', 'answer': '4'}])])
    try:
        trace = s.run_task(val)
        assert trace['learning_status'] == 'frozen'
        assert len(seen) == 1
        material = expand_material(json.loads(seen[0]['messages'][1]['content']))
        assert skill['id'] in material['recovery']['active_guidance_ids']
        assert material['memory']['guidance'][0]['entry_constraints'] == skill['entry_constraints']
    finally: s.close()
    assert tree_identity(frozen) == before


def test_complete_real_train_materials(tmp_path, monkeypatch):
    root = os.environ.get('OFFICE_PROCESS_SOURCE')
    if not root: pytest.skip('Set immutable Train observation')
    paths = sorted(Path(root).glob('episodes/*/off/traces/*.json'))
    assert len(paths) == 12
    base = json.loads((Path(root)/'base_config.json').read_text())
    cfg = entry.configure(base, tmp_path/'run', Path(os.environ['OFFICE_EXTERNAL_RESOURCES']), 'train')
    cfg['llm']['api_key_env'] = 'MODEL_API_KEY'
    seen = http(monkeypatch, [response([{'decision': 'no_change'}], name='submit_learning') for _ in paths])
    s = EmpiricalSystem(cfg)
    receipts = []
    try:
        for path in paths:
            raw = path.read_bytes()
            trace = json.loads(raw); task = PublicTask(**trace['task'])
            s.adapter.reset(task); s.task_context = TaskContext(cfg['runtime']); s.phase = 'train'
            public_pages.seed(task, s.task_context)
            s._task_start = len(s.usage.events)
            s.budget_scope = task.task_id
            outcome = s.learner.learn(task, trace)
            assert outcome['decision'] == 'no_change', outcome
            shown = expand_material(json.loads(seen[-1]['messages'][1]['content']))
            assert shown['experience']['task']['inputs'] == task.inputs
            assert path.read_bytes() == raw
            receipts.append({'task_id': task.task_id, 'source_hash': digest(trace),
                'wire_bytes': s.requests[-1]['input_wire_bytes'] if 'input_wire_bytes' in s.requests[-1] else
                    len(json.dumps(seen[-1]).encode()),
                'complete_public_pages': True, 'model_http': 0})
        assert len(seen) == 12 and len(s.bank.train_cases()) == 12
        out = os.environ.get('PROCESS_GATE_DIR')
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out)/'train12_material_replay.json').write_text(json.dumps(receipts, indent=2))
    finally: s.close()


def test_formal_plan_all_train_then_val_and_cumulative_admission(tmp_path, monkeypatch):
    data = os.environ.get('OFFICE_EXTERNAL_RESOURCES')
    source = os.environ.get('OFFICE_PROCESS_SOURCE')
    if not data or not source: pytest.skip('Requires the approved immutable resources')
    monkeypatch.setattr(entry, 'code_identity', lambda: {'git_sha': 'offline-fixture', 'tracked_dirty': False})
    parent = tmp_path/'parent'; parent.mkdir()
    ledger = {'limits': {'token_limit': 155000000, 'request_limit': 1000000, 'finish_reserve': 0},
              'attempts': {}, 'unknown_billing': False}
    entry.write_json(parent/'budget.json', ledger)
    entry.write_json(parent/'launch_status.json', {'status': 'completed'})
    root = tmp_path/'formal'
    plan = entry.prepare(root, Path(source)/'base_config.json', data, parent)
    assert list(plan['tasks']) == ['train', 'val']
    assert len(plan['identity']['learning_members']) == 120
    assert plan['identity']['train_selection_members'] == []
    assert len(plan['identity']['validation_members']) == 24
    assert plan['new_token_cap'] == 53197824
    assert plan['authorized_cumulative_ceiling'] == 190000000
    assert entry.read(root/'cost_carryforward_manifest.json')['authorization'] == entry.AUTHORIZATION
    assert not (root/'train/bank').exists() and plan['identity']['human_bank_loaded'] is False
    original = entry.read(Path(data)/'train.json')['tasks']
    assert set(plan['identity']['learning_members']) == {t['task_id'] for t in original}
    assert plan['identity']['learning_members'] != [t['task_id'] for t in original]
    assert entry.read(parent/'budget.json') == ledger
    ledger['attempts']['prior'] = {'status': 'metered', 'accounted_tokens': 130000000, 'reserved_tokens': 130000000}
    entry.write_json(parent/'budget.json', ledger)
    approved = entry.prepare(tmp_path/'within190m', Path(source)/'base_config.json', data, parent)
    assert approved['limits']['token_limit'] == 183197824
    assert entry.read(tmp_path/'within190m/budget.json')['attempts'] == ledger['attempts']
    spent = entry.MAX_CUMULATIVE - 1000000
    ledger['attempts']['prior'] = {'status': 'metered', 'accounted_tokens': spent, 'reserved_tokens': spent}
    entry.write_json(parent/'budget.json', ledger)
    with pytest.raises(ValueError, match='approved ceiling'):
        entry.prepare(tmp_path/'overbudget', Path(source)/'base_config.json', data, parent)
    assert not (tmp_path/'overbudget').exists()


def test_repeat_readiness_requires_both_completed_and_settled(tmp_path):
    campaign = tmp_path/'campaign'
    entry.write_json(campaign/'status.json', {'status': 'running', 'pid': os.getpid()})
    assert entry.repeats_settled(campaign) is False
    entry.write_json(campaign/'status.json', {'status': 'stopped'})
    with pytest.raises(ValueError, match='stopped'):
        entry.repeats_settled(campaign)
    entry.write_json(campaign/'status.json', {'status': 'completed'})
    for index in (2, 3):
        parent = campaign/f'repeat_{index}'
        entry.write_json(parent/'launch_status.json', {'status': 'completed'})
        entry.write_json(parent/'budget_prefix.json', {'attempts': {}})
        entry.write_json(parent/'budget.json', {'attempts': {}, 'unknown_billing': False})
    assert entry.repeats_settled(campaign) is True
    entry.write_json(campaign/'repeat_3/budget.json', {
        'attempts': {'inflight': {'status': 'admitted'}}, 'unknown_billing': False})
    with pytest.raises(ValueError, match='settle'):
        entry.repeats_settled(campaign)
    entry.write_json(campaign/'repeat_3/launch_status.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='Both'):
        entry.repeats_settled(campaign)


def test_waiting_launch_allocates_only_after_repeats_and_never_duplicates(tmp_path, monkeypatch):
    root, campaign = tmp_path/'natural', tmp_path/'campaign'
    calls = []
    monkeypatch.setattr(entry, 'code_identity', lambda: {'git_sha': 'test', 'tracked_dirty': False})
    readiness = iter((False, True))
    monkeypatch.setattr(entry, 'repeats_settled', lambda _: next(readiness))
    def sleep(seconds):
        assert seconds == 30 and not root.exists() and not calls
        assert entry.status(root)['launch']['status'] == 'waiting_for_repeats'
    monkeypatch.setattr(entry.time, 'sleep', sleep)
    def prepare(actual_root, base, data, parent):
        assert parent == campaign/'repeat_3' and actual_root == root
        plan = {'inherited_used_or_reserved': 130000000, 'limits': {'token_limit': 183197824}}
        entry.write_json(root/'plan.json', plan)
        calls.append('prepare')
        return plan
    monkeypatch.setattr(entry, 'prepare', prepare)
    monkeypatch.setattr(entry, 'execute', lambda *args: calls.append('execute') or {'train': {}, 'val': {}})
    entry.after_repeats(root, 'base', 'data', campaign, 'env')
    assert calls == ['prepare', 'execute']
    assert entry.status(root)['launch']['status'] == 'completed_train_val'
    with pytest.raises(ValueError, match='already exists'):
        entry.after_repeats(root, 'base', 'data', campaign, 'env')
    assert calls == ['prepare', 'execute']
