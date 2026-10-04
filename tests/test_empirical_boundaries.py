"""Behavior retained from the native, recovery and source-support tests."""
import json
import time
from pathlib import Path
import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import object_schema
from atomic_skillgraph.empirical.workspace import Workspace
from atomic_skillgraph.harness.simple_protocol import Broker, UnknownSideEffect
from test_empirical import Adapter, program


def test_rpc_duplicate_conflict_rejection_and_terminal():
    adapter = Adapter()
    broker = Broker(adapter, 3)
    broker.open_lease('one')
    arguments = {'name': 'act', 'arguments': {'value': 'x'}}
    def rpc(sequence, values):
        return broker.rpc('one', sequence, 'call', values, deadline=time.monotonic()+2, allowed_tools=['act'])
    assert rpc(1, arguments) == rpc(1, arguments)
    assert adapter.steps == 1 and len(broker.events) == 1
    with pytest.raises(ValueError, match='Conflicting'):
        rpc(1, {'name': 'act', 'arguments': {'value': 'different'}})
    assert not rpc(2, {'name': 'act', 'arguments': {}})['accepted']
    assert len(broker.events) == 2 and adapter.steps == 1
    broker.close_lease('one')
    with pytest.raises(RuntimeError, match='lease'):
        rpc(3, arguments)
    broker.done = True
    with pytest.raises(RuntimeError): broker.call('act', {'value': 'x'})
    assert adapter.steps == 1


def test_native_deadline_stops_attempt_without_recovery():
    class Slow(Adapter):
        def call(self, *args):
            time.sleep(.1)
            return super().call(*args)
    broker = Broker(Slow(), 10)
    with pytest.raises(UnknownSideEffect):
        broker.call('act', {'value': 'x'}, deadline=time.monotonic()+.01)
    assert broker.events[0]['state'] == 'unknown' and broker.remaining_calls() == 0
    with pytest.raises(UnknownSideEffect): broker.call('act', {'value': 'x'})


def test_freeze_omits_unqualified_code_and_marks_workflow_dynamic(tmp_path):
    bank = Bank(tmp_path / 'bank')
    asset = bank.put('program', program("def run(ctx, inputs): return {'status':'ok','outputs':{}}"))
    workflow = bank.put('workflow', {'nodes': [{'id': 'p', 'goal': 'prepare', 'program_id': asset['id'], 'args': {}}]})
    manifest = bank.freeze(tmp_path / 'frozen')
    frozen = Bank(tmp_path / 'frozen', readonly=True)
    assert frozen.all('program') == [] and manifest['programs'] == []
    node = frozen.get(workflow['id'])['nodes'][0]
    assert node['dynamic'] and 'program_id' not in node
    frozen.close()
    bank.close()


def test_workspace_complete_publish_discard_and_path_boundary(tmp_path):
    public = tmp_path / 'input.xlsx'
    public.write_text('public')
    workspace = Workspace(tmp_path / 'workspace', {'input.xlsx': public})
    stage = workspace.stage()
    (stage / 'result.xlsx').write_text('complete')
    with pytest.raises(ValueError): workspace.publish(stage, ['../../input.xlsx'])
    (stage / 'link').symlink_to(public)
    with pytest.raises(ValueError): workspace.publish(stage, ['link'])
    (stage / 'link').unlink()
    manifest = workspace.publish(stage, ['result.xlsx'])
    assert json.loads((workspace.root / 'manifest.json').read_text()) == manifest
    stage = workspace.stage()
    (stage / 'partial.xlsx').write_text('partial')
    workspace.discard(stage)
    assert json.loads((workspace.root / 'manifest.json').read_text()) == manifest
    assert public.read_text() == 'public'


def test_image_message_is_real_and_capability_checked():
    from atomic_skillgraph.agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
    config = dict(base_url='https://example.invalid', model='locked-model', api_key_env='TEST_KEY', max_completion_tokens=32)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'question'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aW1hZ2U=', 'detail': 'high'}}]}]
    with pytest.raises(TypeError): OpenAICompatibleProvider(OpenAICompatibleConfig(**config))._build_payload(messages, [])
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig(**config, dialect='openai_chat', input_modalities=('text','image')))
    assert provider._build_payload(messages, [])['messages'] == messages


def test_unauthorized_rpc_counts_once_without_native_execution():
    adapter = Adapter()
    broker = Broker(adapter, 3)
    broker.open_lease('p')
    values = {'name': 'act', 'arguments': {'value': 'x'}}
    result = broker.rpc('p', 1, 'call', values, deadline=time.monotonic()+2, allowed_tools=[])
    assert not result['accepted'] and adapter.steps == 0 and len(broker.events) == 1
    assert broker.rpc('p', 1, 'call', values, deadline=time.monotonic()+2, allowed_tools=[]) == result
    assert len(broker.events) == 1


def test_same_program_version_is_validated_once_and_schema_fields_exist(tmp_path, monkeypatch):
    from atomic_skillgraph.empirical import bank as module
    from atomic_skillgraph.empirical.contracts import validate_workflow
    original = module.validate_program
    calls = []
    def validate(asset):
        calls.append(asset)
        return original(asset)
    monkeypatch.setattr(module, 'validate_program', validate)
    bank = Bank(tmp_path)
    asset = program("def run(ctx, inputs): return {'status':'ok','outputs':{'value':'x'}}",
                    outputs=object_schema({'value': {'type':'string'}}, ['value']))
    registered = bank.put('program', asset)
    assert bank.put('program', asset) == registered and len(calls) == 1
    workflow = {'nodes':[{'id':'p','goal':'produce','program_id':registered['id'],'args':{}},
        {'id':'c','goal':'consume','args':{'x':{'from':'p','field':'missing'}}}]}
    with pytest.raises(ValueError, match='Unknown result field'):
        validate_workflow(workflow, {registered['id']: registered})
    bank.close()
