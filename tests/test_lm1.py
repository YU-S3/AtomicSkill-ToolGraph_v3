"""LM1 production contracts, intercepted transport and immutable historical replay."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.agents.protocol import NativeToolSpec
from atomic_skillgraph.agents.provider import OpenAICompatibleProvider, OpenAICompatibleConfig
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical import (CHOICE_GUIDANCE_POLICY_VERSION, CHOICE_GUIDANCE_MATERIAL_VERSION,
                                        CHOICE_GUIDANCE_SELECTION_VERSION, SINGLE_ANSWER_PROMPT_VERSION)
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.bank_view import BankView
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.choice_guidance import (build_verified_public_source, validate_check,
                                                       validate_proposal, words, render, checked_asset)
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.prompts import CHOICE_PROPOSAL, GUIDANCE_CHECK, single_answer_prompt
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.experiments.run_livemath_lm1_validation import LIMITS, experience, guard
from test_cf2_contracts import http, response
from test_empirical import config_for


def task(key='source', split='train', goal='Analyze compact manifold embeddings and integral cohomology.'):
    return PublicTask(key, key + '-physical', goal,
                      {'choices': [{'label': 'A', 'text': 'Compact manifold embeddings under the stated premises.'},
                                   {'label': 'B', 'text': 'A universal conclusion without these premises.'}]}, split)


def completed():
    return {'submission': '<answer>A</answer>', 'score': {'hard': True}, 'answer_status': 'valid',
            'empty_answer': False, 'completion_truncated': False, 'provider_finish_reason': 'stop'}


def proposal():
    return {'decision': 'upsert_guidance', 'goal': 'Compare manifold embeddings',
            'guidance': 'Check compact manifold premises before comparing the embedding conclusions.',
            'scope_terms': ['manifold', 'embeddings'], 'applicability': 'Compact manifold embedding questions under matching premises.'}


def source(t=None):
    return build_verified_public_source(t or task(), completed(),
                                       {'projection_version': 'choices.fixture.v1', 'source_run_id': 'parent', 'source_record_hash': 'record'})


def check(p=None, s=None, status='supported'):
    p, s = p or proposal(), s or source()
    return {'status': status, 'policy_version': CHOICE_GUIDANCE_POLICY_VERSION, 'proposal_hash': digest(p),
            'proposal_quote': p['guidance'], 'source_quote': s['selected_choice_text'], 'reason': 'Fixture source compatibility.'}


def asset(p=None, s=None):
    p, s = p or proposal(), s or source()
    return {**{k: p[k] for k in ('goal', 'guidance', 'scope_terms', 'applicability')}, 'evidence_source': s,
            'grounding_check': {**check(p, s), 'source_hash': digest(s)}, 'grounded_proposal': p,
            'guidance_policy_version': CHOICE_GUIDANCE_POLICY_VERSION, 'execution_intent': 'guidance_only',
            'input_schema': {'type': 'object'}, 'output_schema': {'type': 'object'}}


def config(root):
    c = config_for(root/'bank')
    c['learning']['choice_guidance'] = {'enabled': True}
    c['runtime']['choice_guidance'] = {'enabled': True}
    c['llm']['max_retries'] = 0
    c['llm']['api_key_env'] = 'MODEL_API_KEY'
    c['llm']['purpose_overrides'] = {purpose: {'protocol': {'thinking_type': 'disabled'}, 'max_completion_tokens': cap}
                                    for purpose, cap in [('guidance_learning', 2048), ('guidance_grounding', 1536)]}
    return c


def recompile(system, t=None, identity=None):
    t = t or task()
    root = Path(system.config['experiment']['output_dir'])/'events'/t.task_id
    system.checkpoint, system.audit_path = TaskCheckpoint(root/'checkpoint'), root/'requests.json'
    return system.learn_from_completed_record(t, completed(), identity or
        {'projection_version': 'choices.fixture.v1', 'source_run_id': 'parent', 'source_record_hash': 'record-' + t.task_id})


@pytest.mark.parametrize('status', ['supported', 'contradicted', 'insufficient_evidence', 'bad_quote', 'empty', 'length'])
def test_only_complete_supported_check_publishes(tmp_path, monkeypatch, status):
    p, s = proposal(), source()
    c = check(p, s, status if status in {'supported', 'contradicted', 'insufficient_evidence'} else 'supported')
    if status == 'bad_quote': c['source_quote'] = 'not in public source'
    seen = http(monkeypatch, [response([p], name='submit_learning'),
        response([] if status == 'empty' else [c], name='submit_guidance_check', finish='length' if status == 'length' else 'tool_calls')])
    cfg = config(tmp_path); cfg['experiment']['output_dir'] = str(tmp_path/'output')
    system = EmpiricalSystem(cfg, harness=AnswerAdapter('livemath', {}))
    try:
        result = recompile(system)
        assert len(seen) == 2
        assert [r['tool_choice']['function']['name'] for r in seen] == ['submit_learning', 'submit_guidance_check']
        assert [r['max_tokens'] for r in seen] == [2048, 1536]
        assert 'existing_skill_id' not in seen[0]['tools'][0]['function']['parameters']['properties']
        assert all(r['thinking']['type'] == 'disabled' for r in seen)
        assert bool(system.bank.all('skill')) == (status == 'supported')
        assert all(u['bucket'] == 'extractor_e1' for u in result['usage'])
        assert {u['provider_metadata']['decision_purpose'] for u in result['usage']} == {'proposal', 'grounding'}
        before = system.bank.digest()
        assert recompile(system) == result and len(seen) == 2 and system.bank.digest() == before
        with pytest.raises(ValueError, match='source identity changed'):
            recompile(system, identity={'source_run_id': 'other'})
    finally: system.close()


def test_structural_json_text_repair_is_counted_and_not_accepted(tmp_path, monkeypatch):
    bad = {**proposal(), 'goal': 'x'*161, 'guidance': 'y'*1001}
    seen = http(monkeypatch, [response([], content=json.dumps(bad), finish='stop'),
                             response([{'decision': 'no_change'}], name='submit_learning')])
    cfg = config(tmp_path); cfg['experiment']['output_dir'] = str(tmp_path/'output')
    governor = BudgetGovernor(tmp_path/'budget.json', **LIMITS)
    s = EmpiricalSystem(cfg, harness=AnswerAdapter('livemath', {}), budget_governor=governor)
    try:
        result = recompile(s)
        assert result['learning']['decision'] == 'no_change' and not s.bank.all('skill') and len(seen) == 2
        repair_text = seen[1]['messages'][-1]['content']
        assert 'ToolCall' in repair_text and 'goal' in repair_text and 'guidance' in repair_text
        assert [a['context']['decision_purpose'] for a in governor.state['attempts'].values()] == ['proposal', 'repair']
    finally: s.close()


def test_repair_batch_limit_persists_and_later_proposals_reject(tmp_path, monkeypatch):
    bad = {'decision': 'upsert_guidance'}
    seen = http(monkeypatch, [response([bad], name='submit_learning') for _ in range(9)])
    cfg = config(tmp_path); cfg['experiment']['output_dir'] = str(tmp_path/'output')
    gov = BudgetGovernor(tmp_path/'budget.json', **LIMITS)
    s = EmpiricalSystem(cfg, harness=AnswerAdapter('livemath', {}), budget_governor=gov)
    try:
        for i in range(5):
            assert recompile(s, task(str(i)))['learning']['decision'] == 'rejected'
        assert len(seen) == 9 and sum(a['context']['decision_purpose'] == 'repair' for a in gov.state['attempts'].values()) == 4
        restored = BudgetGovernor(tmp_path/'budget.json', **LIMITS)
        assert not restored.repair_available() and not restored.state['unknown_billing']
    finally: s.close()


def test_source_binding_rejects_ambiguous_choices_and_never_reads_gold():
    t = task(); s = source(t)
    assert s['selected_choice_text'] == t.inputs['choices'][0]['text'] and 'gold' not in json.dumps(s)
    exp = completed(); exp['private_gold'] = {'label': 'B'}
    assert build_verified_public_source(t, exp, {})['submitted_label'] == 'A'
    exp['score']['hard'] = False
    assert not build_verified_public_source(t, exp, {})['eligible']
    t.inputs['choices'][1]['label'] = 'A'
    assert source(t)['reason'] == 'source_choice_unresolvable'
    for term in ['strongest', 'statement', 'question', 'options', 'x', '123', r'\mathbb']:
        assert not words(term)
    p = proposal(); p['scope_terms'] = ['invented', 'topics']
    with pytest.raises(ValueError, match='source_goal'): validate_proposal(p, s, [], CHOICE_PROPOSAL)


def test_scope_stats_frozen_view_and_parent_filter(tmp_path):
    b = Bank(tmp_path/'bank')
    try:
        t = task(); b.save_case(t, completed()); stats = b.update_choice_statistics()
        b.save_case(t, completed()); assert b.update_choice_statistics() == stats and stats['N'] == 1
        old = b.put('skill', {'goal': t.goal, 'guidance': 'Old unchecked guidance', 'execution_intent': 'guidance_only'})
        parent = b.put('skill', asset())
        child = b.put('skill', {**asset({**proposal(), 'guidance': 'Verify compact manifold premises before using embedding conclusions.'}), 'parent_skill_id': parent['id']})
        policy = {'max_items': 2, 'max_total_chars': 2400}
        sel = b.select_guidance(t, policy)
        assert sel['injected_ids'] == [child['id']] and len(sel['selected']) == 1
        assert {a['asset_id']: a['reason'] for a in sel['audit']}[old['id']] == 'unchecked_asset'
        assert not b.select_guidance(task(goal='The strongest statement question options.'), policy)['selected']
        assert not b.select_guidance(task(goal='Investigate probability random distributions.'), policy)['selected']
        assert not b.select_guidance(t, {**policy, 'max_total_chars': 20})['selected']
        before = b.digest(); b.db.execute("DELETE FROM metadata WHERE key='choice_guidance_retrieval_stats'"); b.db.commit()
        with pytest.raises(ValueError, match='statistics'): b.select_guidance(t, policy)
        b.update_choice_statistics(); assert b.digest() == before
        frozen = tmp_path/'frozen'; manifest = b.freeze(frozen)
        f = Bank(frozen, readonly=True)
        try:
            assert not f.train_cases() and f.select_guidance(t, policy)['injected_ids'] == [child['id']]
            assert BankView(f, 'guidance_off').select_guidance(t, policy)['selected'] == []
            assert f.digest() == manifest['digest']
        finally: f.close()
    finally: b.close()


def test_runtime_three_arms_same_public_prompt_one_answer_and_frozen(tmp_path, monkeypatch):
    seen = http(monkeypatch, [response([], content='<answer>A</answer>', finish='stop') for _ in range(3)])
    cfg = config(tmp_path); b = Bank(cfg['data_dir']); b.save_case(task(), completed()); b.update_choice_statistics(); b.put('skill', asset())
    frozen = tmp_path/'frozen'; snap = b.freeze(frozen); b.close()
    t = task('val', 'val')
    for arm, effort in [('A', 'high'), ('B', 'high'), ('C', 'low')]:
        c = deepcopy(cfg); c['data_dir'] = str(frozen); c['llm']['runtime']['reasoning_effort'] = effort
        c['experiment'].update(runtime_mode='frozen', output_dir=str(tmp_path/arm))
        s = EmpiricalSystem(c, harness=AnswerAdapter('livemath', {t.task_id: {'correct_choice': t.inputs['choices'][0], 'choices': t.inputs['choices']}}),
                            bank_view_factory=(lambda b: BankView(b, 'guidance_off')) if arm == 'A' else None)
        try:
            trace = s.run_task(t, learn=False)
            assert len(trace['requests']) == 1 and trace['learning_status'] == 'frozen'
            assert trace['knowledge_before'] == trace['knowledge_after'] == snap['digest']
            assert bool(trace['injected_guidance_ids']) == (arm != 'A')
        finally: s.close()
    assert len({r['messages'][0]['content'] for r in seen}) == 1
    assert all('tool_choice' not in r and r['max_tokens'] == 32768 for r in seen)
    assert [r['reasoning_effort'] for r in seen] == ['high', 'high', 'low']
    materials = [json.loads(r['messages'][1]['content']) for r in seen]
    assert all(m['inputs'] == t.inputs and m['goal'] == t.goal for m in materials)
    assert materials[1] == materials[2] and materials[0]['guidance'] == []


def test_provider_optional_choice_preserves_old_payload_and_refuses_invalid_names():
    tools = [NativeToolSpec('submit_learning', 'submit', CHOICE_PROPOSAL)]
    p = OpenAICompatibleProvider(OpenAICompatibleConfig(base_url='https://example.test', model='model', api_key_env='KEY', thinking_type='disabled', max_completion_tokens=2048))
    messages = [{'role': 'user', 'content': 'public'}]
    assert 'tool_choice' not in p._build_payload(messages, tools)
    with pytest.raises(ValueError): p._build_payload(messages, tools, {'type': 'function', 'function': {'name': 'missing'}})
    from dataclasses import replace
    p.config = replace(p.config, thinking_type='enabled')
    with pytest.raises(ValueError, match='disabled'): p._build_payload(messages, tools, {'type': 'function', 'function': {'name': 'submit_learning'}})


def test_budget_reservations_pending_unknown_billing_and_identity(tmp_path):
    gov = BudgetGovernor(tmp_path/'budget.json', token_limit=100, finish_reserve=0, request_limit=2)
    payload = {'max_tokens': 90, 'messages': [{'content': 'a'*20}]}
    with pytest.raises(BudgetExhausted): gov.admit('too_big', payload, {})
    assert not gov.state['attempts']
    gov.admit('pending', {'max_tokens': 1}, {})
    restored = BudgetGovernor(tmp_path/'budget.json', token_limit=100, finish_reserve=0, request_limit=2)
    assert restored.state['unknown_billing']
    with pytest.raises(BudgetExhausted): guard(restored)
    with pytest.raises(ValueError, match='identity'): BudgetGovernor(tmp_path/'budget.json', token_limit=101, finish_reserve=0, request_limit=2)


def test_config_versions_budgets_strict_and_old_defaults_unchanged(tmp_path):
    old = config_for(tmp_path/'old'); old_valid = validate_config(old)
    assert 'choice_guidance' not in old_valid['runtime']
    for section, field, value in [('runtime', 'max_items', 3), ('learning', 'source_check_required', False), ('learning', 'material_version', 'old')]:
        c = config(tmp_path); c[section]['choice_guidance'][field] = value
        with pytest.raises(ValueError): validate_config(c)
    c = config(tmp_path); c['learning']['min_distinct_train_cases_before_first_build'] = 2
    with pytest.raises(ValueError, match='policy|one real'): validate_config(c)


def fixed_review(tmp_path, monkeypatch, *, no_change=False):
    """A complete synthetic source package, with the same fixed batch cardinalities."""
    from atomic_skillgraph.experiments import run_livemath_lm1_validation as runner
    from atomic_skillgraph.experiments.canonical_manifest import sha256
    from atomic_skillgraph.experiments.run_empirical import write_json
    review = tmp_path/'review'; base_path = tmp_path/'base.json'
    train = [task('train-' + str(i)) for i in range(60)]
    val = [task('val-' + str(i), 'val') for i in range(17)]
    from dataclasses import asdict
    rows = []
    for i, t in enumerate(train):
        exp = completed(); exp['score'] = {'hard': i < 16}
        rows.append({'split_order_index': i, 'task_id': t.task_id, 'physical_key': t.physical_key,
                     'raw_final_content': exp['submission'], 'original_score': exp['score'], 'finish_reason': 'stop',
                     **{k: exp[k] for k in ('answer_status', 'empty_answer', 'completion_truncated')}})
    values = {
        'resources/public/train.json': {'tasks': [asdict(t) for t in train]},
        'resources/public/val.json': {'tasks': [asdict(t) for t in val]},
        'resources/private/evaluator_records_val.json': {t.task_id: {'choices': t.inputs['choices'], 'correct_choice': t.inputs['choices'][0]} for t in val},
        'identity/original/run_manifest.json': {'run_id': 'parent'},
        'identity/original/train/execution_manifest.json': {'tasks': [asdict(t) for t in train]},
        'identity/original/val/execution_manifest.json': {'tasks': [asdict(t) for t in val]},
        'identity/actual_run_identity.json': {'frozen_manifest': {'digest': 'old'}},
        'resources/projection_integrity_original.json': {'choice_projection_version': 'livemath.choices.v1', 'choice_seed': 42}}
    for name, value in values.items(): write_json(review/name, value)
    row_path = review/'train/train60_learning_summary.jsonl'; row_path.parent.mkdir(parents=True)
    row_path.write_text('\n'.join(json.dumps(r) for r in rows))
    write_json(review/'SHA256SUMS.json', {p.relative_to(review).as_posix(): sha256(p) for p in review.rglob('*') if p.is_file()})
    write_json(base_path, config(tmp_path))
    monkeypatch.setattr(runner, 'code_identity', lambda: {'git_sha': 'fixture', 'tracked_dirty': False, 'source_sha256': 'fixture-source'})
    output = tmp_path/'batch'
    runner.prepare(review, base_path, output)
    env = tmp_path/'fixture.env'; env.write_text('MODEL_API_KEY=fixture-key\n')
    sent = []
    from types import SimpleNamespace
    def post(*args, **kwargs):
        payload = deepcopy(kwargs['json']); sent.append(payload)
        materials = json.loads(payload['messages'][1]['content'])
        names = [t['function']['name'] for t in payload.get('tools', [])]
        if names == ['submit_learning']:
            body = response([{'decision': 'no_change'} if no_change else proposal()], name='submit_learning')
        elif names == ['submit_guidance_check']:
            body = response([check(materials['proposal'], materials['source'])], name='submit_guidance_check')
        else: body = response([], content='<answer>A</answer>', finish='stop')
        return SimpleNamespace(status_code=200, ok=True, headers={}, json=lambda: body)
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
    return runner, output, env, sent


@pytest.mark.parametrize('no_change', [False, True])
def test_whole_fixed_batch_restarts_without_http_and_declared_fallback(tmp_path, monkeypatch, no_change):
    runner, output, env, sent = fixed_review(tmp_path, monkeypatch, no_change=no_change)
    report = runner.check_offline(output/'manifest.json')
    assert report['new_http_calls'] == 0 and not sent
    result = runner.run(output/'manifest.json', env)
    assert result['status'] == 'completed_fixed_batch'
    assert result['http_attempts'] == len(sent) == (50 if no_change else 83)
    assert len(result['train']) == 60 and len(result['val']) == (34 if no_change else 51)
    assert result['no_method_exposure'] == no_change
    assert result['qualified_assets'] == (0 if no_change else 16)
    assert result['incremental_recompile_tokens'] == (80 if no_change else 160)
    assert [r['arm'] for r in result['val'][:6]] == (['A', 'C', 'C', 'A', 'C', 'A'] if no_change else ['A', 'B', 'C', 'B', 'C', 'A'])
    assert all(r['frozen_before'] == r['frozen_after'] == result['frozen']['digest'] for r in result['val'])
    before = len(sent)
    assert runner.run(output/'manifest.json', env) == result and len(sent) == before
    with pytest.raises(ValueError, match='fresh'): runner.prepare(tmp_path/'review', tmp_path/'base.json', output)


def test_paid_proposal_recovers_before_apply_without_another_http(tmp_path, monkeypatch):
    p = proposal(); s = source()
    seen = http(monkeypatch, [response([p], name='submit_learning'), response([check(p, s)], name='submit_guidance_check')])
    cfg = config(tmp_path); cfg['experiment']['output_dir'] = str(tmp_path/'output')
    system = EmpiricalSystem(cfg, harness=AnswerAdapter('livemath', {}))
    original = TaskCheckpoint.commit_decision
    stopped = []
    def interrupt(self, key, status, **values):
        if status == 'applied' and not stopped:
            stopped.append(True); raise InterruptedError('after paid response, before apply')
        return original(self, key, status, **values)
    monkeypatch.setattr(TaskCheckpoint, 'commit_decision', interrupt)
    try:
        with pytest.raises(InterruptedError): recompile(system)
        assert len(seen) == 1
        result = recompile(system)
        assert result['learning']['persisted_skill_id'] and len(seen) == 2
        assert len(system.bank.all('skill')) == 1
        assert recompile(system) == result and len(seen) == 2
    finally: system.close()


def test_receive_identity_binds_source_schema_prompt_purpose_version_and_choice(tmp_path, monkeypatch):
    cfg = config(tmp_path); s = EmpiricalSystem(cfg, harness=AnswerAdapter('livemath', {}))
    s.checkpoint = TaskCheckpoint(tmp_path/'checkpoint')
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs)); return {'decision': 'no_change'}
    monkeypatch.setattr(s, 'agent', fake)
    def receive(material=None, schema=None, prompt='fixed', purpose='guidance_learning', version=CHOICE_GUIDANCE_MATERIAL_VERSION):
        return s.learner._receive('choice', 'extractor', prompt, material or {'source_hash': 'one'},
            'submit_learning', schema or CHOICE_PROPOSAL, decision_purpose=purpose, material_version=version)
    try:
        assert receive() == receive() and len(calls) == 1
        for change in [{'material': {'source_hash': 'two'}}, {'schema': {**CHOICE_PROPOSAL, 'required': ['decision', 'goal']}},
                       {'prompt': 'changed'}, {'purpose': 'guidance_grounding'}]:
            with pytest.raises(ValueError, match='changed'): receive(**change)
        assert len(calls) == 1
        s.config['llm']['purpose_overrides']['guidance_learning']['protocol']['thinking_type'] = 'enabled'
        with pytest.raises(ValueError, match='semantics changed'): receive()
        receive(version='choice-guidance.material.future')
        assert len(calls) == 2
    finally: s.close()


def test_whole_batch_unknown_billing_is_terminal_without_resend(tmp_path, monkeypatch):
    runner, output, env, sent = fixed_review(tmp_path, monkeypatch)
    from types import SimpleNamespace
    def unmetered(*args, **kwargs):
        sent.append(kwargs['json'])
        body = response([{'decision': 'no_change'}], name='submit_learning'); body.pop('usage')
        return SimpleNamespace(status_code=200, ok=True, headers={}, json=lambda: body)
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post', unmetered)
    result = runner.run(output/'manifest.json', env)
    assert result['status'].startswith('stopped_') and result['unknown_billing'] and len(sent) == 1
    assert runner.run(output/'manifest.json', env) == result and len(sent) == 1


@pytest.fixture
def review():
    value = os.environ.get('LM1_REVIEW_ROOT')
    if not value: pytest.skip('Historical review replay requires LM1_REVIEW_ROOT')
    return Path(value)


def test_original_60_source_records_and_100_original_scores(review):
    from atomic_skillgraph.experiments.run_livemath_lm1_validation import read, lines
    from atomic_skillgraph.harness.scorers.livemath import evaluate
    public = {t['task_id']: t for t in read(review/'resources/public/train.json')['tasks']}
    rows = lines(review/'train/train60_learning_summary.jsonl')
    sources = [build_verified_public_source(PublicTask(**public[r['task_id']]), experience(r), {}) for r in rows]
    assert len(sources) == 60 and sum(s['eligible'] for s in sources) == 16
    test = lines(review/'test/test100_light.jsonl')
    records = read(review/'resources/private/evaluator_records_test.json')
    correct = sum(evaluate(r['raw_final_content'], records[r['task_id']]['correct_choice'], records[r['task_id']]['choices'])['em'] for r in test)
    assert len(test) == 100 and correct == 20
    assert sum(r['completion_truncated'] and r['empty_answer'] for r in test) == 18
    assert sum(not r['original_score']['hard'] and not r['completion_truncated'] for r in test) == 62


def test_two_original_conflicts_are_excluded_and_contradicted_control_rejects(review, tmp_path):
    from atomic_skillgraph.experiments.run_livemath_lm1_validation import read, lines
    all_assets = read(review/'train/final_guidance_assets.json')['assets']
    public = {t['task_id']: t for t in read(review/'resources/public/train.json')['tasks']}
    rows = {r['task_id']: r for r in lines(review/'train/train60_learning_summary.jsonl')}
    ids = ['skill_31c1afc651f0e08f541046d84eef2be621d8fbf9664b7864420a8abff7e23d05',
           'skill_4f0312202e4cc6f0757898031aa1bccad7f99d45573882bc7d3eb3c4c45330e7']
    b = Bank(tmp_path/'bank')
    try:
        for a in all_assets: b.put('skill', a)
        for a in [a for a in all_assets if a['id'] in ids]:
            t = PublicTask(**public[a['evidence_source']['source_task_id']])
            s = build_verified_public_source(t, experience(rows[t.task_id]), {})
            assert s['eligible']
            p = {'goal': a['goal'], 'guidance': a['guidance'], 'applicability': 'Source premises only.'}
            c = check(p, s, 'contradicted'); c['proposal_quote'] = a['guidance'].splitlines()[0]
            assert not validate_check(c, p, s, GUIDANCE_CHECK)
            assert not checked_asset(a)
            assert not b.select_guidance(t, {'max_items': 2, 'max_total_chars': 2400})['selected']
    finally: b.close()
