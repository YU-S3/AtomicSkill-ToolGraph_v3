"""Execute the real prior Learner for known CF3 failures; no AST/string rewrites."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

from test_cf4_contracts import seed_job, binding, raw, ROOT
from test_cf2_contracts import office
from test_cf2_realization import skill, learning_trace
from test_cf3_takeover import transport
from atomic_skillgraph.empirical.contracts import object_schema
from skillcompiler_bench_contracts.livemath import normalize_livemath_item


def original_learner(tmp_path):
    source = subprocess.check_output(['git', 'show', 'bc63d02bc3bb0988cdbd15a203a068fc162f0b9b:src/atomic_skillgraph/empirical/learner.py'], cwd=ROOT, text=True)
    path = tmp_path / 'original_learner.py'; path.write_text(source)
    spec = importlib.util.spec_from_file_location('atomic_skillgraph.empirical.cf3_original_learner', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.Learner


def test_t01_real_raw_old_candidates_lack_correct_label():
    before = raw()['mcq']
    assert before['correct_choice']['label'].strip().upper().rstrip('.):') not in {c['label'] for c in before['choices']}
    assert normalize_livemath_item(raw())['correct_choice'] in normalize_livemath_item(raw())['choices']


def test_t16_original_production_learner_overflows_usable_fixed_slots(tmp_path, monkeypatch):
    s = office(tmp_path); sk, p, job = seed_job(s)
    proposal = {'decision': 'reuse_existing', 'existing_skill_id': sk['id'], 'realization_request': {
        'skill_id': sk['id'], 'action': 'trial', 'case_bindings': [binding('office-physical')]}}
    transport(monkeypatch, [('submit_learning', proposal)])
    old = original_learner(tmp_path)(s)
    with pytest.raises(ValueError, match='at most two fixed slots'): old.learn(s.adapter.task, learning_trace())
    assert s.bank.jobs()[0]['case_bindings'] == job['case_bindings']
    s.close()


def test_t20_original_production_learner_saves_invalid_skill_before_preflight(tmp_path, monkeypatch):
    s = office(tmp_path)
    proposal = {'decision': 'propose_skill_and_program_spec', 'skill': skill('find target', inputs=object_schema({'resource': {'type': 'string'}}, ['resource'])),
        'realization_request': {'skill_id': '$new', 'action': 'build', 'case_bindings': [binding('office-physical')]}}
    proposal['realization_request']['case_bindings'][0]['inputs'] = {}
    sent = transport(monkeypatch, [('submit_learning', proposal)])
    result = original_learner(tmp_path)(s).learn(s.adapter.task, learning_trace())
    assert result['inapplicable'] and len(s.bank.all('skill')) == len(s.bank.jobs()) == len(sent) == 1
    s.close()
