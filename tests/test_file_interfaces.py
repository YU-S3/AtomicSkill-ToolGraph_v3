"""Recorded Train proposals, strict publication paths and completed-task stops; no live LLM."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.contracts import PublicTask, object_schema
from atomic_skillgraph.empirical.prompts import BUILD, LEARNING
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments.run_empirical import run
from atomic_skillgraph.harness.benchmarks import AnswerAdapter, SpreadsheetAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_cf2_contracts import http, office, response
from test_cf2_realization import learning_trace
from test_empirical import config_for, program, worker
from test_provider_transport import Response


RECORDED = json.loads((Path(__file__).parent / 'fixtures/cf2_file_interfaces.json').read_text())


def current_node_protocol(proposal):
    proposal = deepcopy(proposal)
    for node in proposal.get('workflow', {}).get('nodes', []):
        node.pop('dynamic', None)
        node['execution_mode'] = 'skill' if node.get('skill_id') else 'dynamic'
        if node['execution_mode'] == 'skill': node['purpose'] = node.pop('goal')
    return proposal


def test_recorded_unknown_skill_name_uses_existing_repair_limit_without_partial_asset(tmp_path, monkeypatch):
    seen = http(monkeypatch, [response([current_node_protocol(p)], name='submit_learning') for p in RECORDED['office_proposals']])
    system = office(tmp_path)
    try:
        result = system.learner.learn(system.adapter.task, learning_trace())
        assert result['decision'] == 'rejected' and result['errors']
        assert len(seen) == 2
        assert system.bank.all('skill') == system.bank.jobs() == []
        schema = seen[-1]['tools'][0]['function']['parameters']
        assert schema['properties']['realization_request']['properties']['skill_id']['enum'] == ['$new']
        assert 'enum' not in schema['properties']['skill']['properties']['goal']
        assert 'enum' not in LEARNING['properties']['realization_request']['properties']['skill_id']
    finally:
        system.close()


def test_new_and_existing_skill_references_are_distinct_and_reach_http(tmp_path, monkeypatch):
    proposal = current_node_protocol(RECORDED['office_proposals'][-1])
    proposal['realization_request'].update(skill_id='$new', action='defer', case_bindings=[])
    seen = http(monkeypatch, [response([proposal], name='submit_learning')])
    system = office(tmp_path)
    try:
        system.learner.learn(system.adapter.task, learning_trace())
        existing = system.bank.all('skill')[0]['id']
        reuse = {'decision': 'reuse_existing', 'existing_skill_id': existing,
                 'realization_request': {'skill_id': existing, 'action': 'defer', 'case_bindings': []}}
        system.learner._skill_references(reuse)
        reuse['realization_request']['skill_id'] = 'treasury_monthly_series_geometric_mean'
        with pytest.raises(ValueError, match='Skill reference'):
            system.learner._skill_references(reuse)
        with pytest.raises(ValueError, match='Skill reference'):
            system.learner._skill_references({'decision': 'no_change', 'workflow': {
                'nodes': [{'skill_id': '$new'}]}})
        assert len(seen) == 1 and 'never an invented name or alias' in seen[0]['messages'][0]['content']
    finally:
        system.close()


def test_recorded_absolute_file_declaration_remains_execution_failure(tmp_path, worker):
    import openpyxl
    path = tmp_path / 'input.xlsx'
    workbook = openpyxl.Workbook()
    workbook.active.title = 'Sheet1'
    for row, value in enumerate(['prefix', 'ABC', None, None, 'XYZ', None], 1):
        workbook.active.cell(row=row, column=1, value=value)
    workbook.save(path)
    workbook.close()
    config = config_for(tmp_path / 'bank')
    adapter = SpreadsheetAdapter({'sheet': {'public_files': {'input.xlsx': str(path)}}}, config)
    adapter.reset(PublicTask('sheet', 'physical', 'recorded file interface'))
    proposal = RECORDED['spreadsheet_program']
    binding = proposal['trial_inputs'][0]['inputs']
    artifact = program(proposal['source'], inputs=RECORDED['spreadsheet_input_schema'],
                       outputs=RECORDED['spreadsheet_output_schema'])
    artifact.update(id='recorded-interface-fixture', allowed_tools=[])
    try:
        result = worker.execute(artifact, binding, Broker(adapter, 1))
        assert result['status'] == 'execution_error' and 'Output path must be relative' in result['detail']
        assert not (adapter.workspace.root / 'manifest.json').exists()
        assert not (tmp_path / 'bank').exists()
    finally:
        adapter.close()


def test_program_path_and_publication_name_remain_separate(tmp_path, worker):
    config = config_for(tmp_path / 'bank')
    adapter = SpreadsheetAdapter({'file': {}}, config)
    adapter.reset(PublicTask('file', 'file-physical', 'interface regression only'))
    artifact = program("def run(ctx, inputs):\n"
        "    from pathlib import Path\n"
        "    Path(inputs['output_path']).write_text('regression-only')\n"
        "    return {'status':'ok','outputs':{'output_path':inputs['output_path'],'files':['report.txt']}}",
        inputs=object_schema({'output_path': {'type': 'string'}}, ['output_path']),
        outputs=object_schema({'output_path': {'type': 'string'}, 'files': {'type': 'array'}}, ['output_path', 'files']))
    artifact.update(id='publication-interface-fixture', allowed_tools=[])
    try:
        result = worker.execute(artifact, {'output_path': '/workspace/report.txt'}, Broker(adapter, 1))
        assert result['status'] == 'ok'
        assert result['outputs']['output_path'] == '/workspace/report.txt'
        assert result['workspace']['outputs'] == ['report.txt']
        before = (adapter.workspace.root / 'manifest.json').read_bytes()
        for invalid in ['/workspace/report.txt', '/tmp/report.txt', '../report.txt', 'inputs/input.xlsx']:
            stage = adapter.workspace.stage()
            with pytest.raises(ValueError, match='relative and outside inputs'):
                adapter.workspace.publish(stage, [invalid])
            adapter.workspace.discard(stage)
            assert (adapter.workspace.root / 'manifest.json').read_bytes() == before
        assert not adapter.submission_ready() and not (tmp_path / 'bank').exists()
    finally:
        adapter.close()


def test_publication_contract_reaches_builder_http(tmp_path, monkeypatch):
    seen = http(monkeypatch, [response([{'source': "def run(ctx, inputs): pass", 'trial_inputs': []}], name='submit_program')])
    system = office(tmp_path)
    try:
        from atomic_skillgraph.empirical.prompts import BUILDER_PROMPT
        system.agent('tool_builder', BUILDER_PROMPT, {}, 'submit_program', BUILD)
        assert 'output_path' in seen[0]['messages'][0]['content']
        assert 'files=["result.xlsx"]' in seen[0]['messages'][0]['content']
        tool = SpreadsheetAdapter({}, system.config).available_tools()[0]
        assert 'relative to /workspace' in tool['description']
        assert 'not OUTPUT_PATH' in tool['input_schema']['properties']['files']['description']
    finally:
        system.close()


def test_completed_task_boundary_preserves_full_membership_and_does_not_freeze(tmp_path, monkeypatch):
    import atomic_skillgraph.experiments.run_empirical as runner
    monkeypatch.setattr(runner, 'code_identity', lambda: {'git_sha': 'fixture', 'tracked_dirty': False, 'source_sha256': 'fixture'})
    monkeypatch.setenv('MODEL_API_KEY', 'fixture-key')
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs['json'])
        if kwargs['json'].get('tools'):
            return Response(response([{'decision': 'no_change'}], name='submit_learning'))
        return Response(response(content='correct', finish='stop'))
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post', post)
    tasks = [PublicTask('train:' + str(i), 'physical:' + str(i), 'question', split='train') for i in range(3)]
    records = {t.task_id: {'answers': ['correct']} for t in tasks}
    output = tmp_path / 'train'
    config = config_for(output / 'bank')
    partial = run(config, tasks, output, adapter=AnswerAdapter('searchqa', records), stop_after_tasks=2)
    identity = json.loads((output / 'execution_manifest.json').read_text())
    assert [t['task_id'] for t in identity['tasks']] == [t.task_id for t in tasks]
    assert not partial['complete'] and partial['tasks'] == 2 and partial['expected_tasks'] == 3
    assert partial['frozen'] is None and not (output / 'frozen_bank').exists()
    assert (output / 'STOP_AFTER_TASK').exists() and len(sent) == 4
    (output / 'STOP_AFTER_TASK').unlink()
    complete = run(config, tasks, output, adapter=AnswerAdapter('searchqa', records), resume=True)
    assert complete['complete'] and complete['tasks'] == 3 and len(sent) == 6
    assert (output / 'frozen_bank/freeze.json').exists()
    assert json.loads((output / 'execution_manifest.json').read_text()) == identity
