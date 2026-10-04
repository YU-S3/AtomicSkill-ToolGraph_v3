"""Retained authentication, metering and response-order boundaries."""
import json
import pytest
import requests
from atomic_skillgraph.agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider, AgentProviderError
from atomic_skillgraph.agents.protocol import AgentTurn, NativeToolCall
from atomic_skillgraph.empirical.system import EmpiricalSystem
from test_empirical import Adapter, config_for


class Response:
    status_code = 200
    ok = True
    headers = {'x-request-id': 'fixed-provider-id'}
    text = ''
    content = b''
    def __init__(self, payload): self.payload = payload
    def json(self): return self.payload


def test_auth_stays_in_header_and_real_usage_is_recorded(monkeypatch):
    monkeypatch.setenv('TEST_KEY', 'fixture-private-key')
    requests_seen = []
    def post(*args, **kwargs):
        requests_seen.append(kwargs)
        return Response({'choices': [{'finish_reason': 'stop', 'message': {'content':'answer', 'reasoning_content':'private-reasoning'}}],
                         'usage': {'prompt_tokens':11,'completion_tokens':7,'total_tokens':18}})
    monkeypatch.setattr(requests, 'post', post)
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig('https://example.invalid','actual-model','TEST_KEY',32,max_retries=0))
    turn = provider.complete([{'role':'user','content':'question'}], tools=[])
    assert turn.total_tokens == 18 and turn.reasoning_tokens is None
    assert requests_seen[0]['headers']['Authorization'] == 'Bearer fixture-private-key'
    assert 'fixture-private-key' not in json.dumps(requests_seen[0]['json'])
    assert 'private-reasoning' not in json.dumps(provider.request_records)


def test_missing_billing_is_unknown_not_free(monkeypatch):
    monkeypatch.setenv('TEST_KEY', 'fixture-private-key')
    monkeypatch.setattr(requests,'post',lambda *a,**k: Response({'choices':[{'finish_reason':'stop','message':{'content':'answer'}}]}))
    provider = OpenAICompatibleProvider(OpenAICompatibleConfig('https://example.invalid','actual-model','TEST_KEY',32,max_retries=0))
    with pytest.raises(AgentProviderError): provider.complete([{'role':'user','content':'question'}],tools=[])
    record = provider.request_records[0]
    assert record['usage_status'] == 'unavailable' and record['usage'] is None


def test_repair_replies_to_every_tool_call_before_next_request(tmp_path):
    class Provider:
        calls = []
        def complete(self, messages, tools):
            self.calls.append(messages.copy())
            calls = [NativeToolCall('first','submit',{}),NativeToolCall('second','submit',{})] if len(self.calls)==1 else [NativeToolCall('fixed','submit',{})]
            assistant = {'role':'assistant','content':'','reasoning_content':'private',
                         'tool_calls':[{'id':c.call_id,'type':'function','function':{'name':c.name,'arguments':'{}'}} for c in calls]}
            return AgentTurn('',calls,'tool_calls',1,1,2,None,1,replay_assistant_message=assistant)
    provider = Provider()
    system = EmpiricalSystem(config_for(tmp_path),harness=Adapter(),provider=provider)
    assert system.agent('runtime','prompt',{},'submit',{'type':'object'},repair_limit=1)=={}
    assert [m['tool_call_id'] for m in provider.calls[1] if m['role']=='tool'] == ['first','second']
    assert 'private' not in json.dumps(system.requests)
    system.bank.close()
