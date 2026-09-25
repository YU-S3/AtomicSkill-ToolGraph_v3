import json
from types import SimpleNamespace
import pytest
from atomic_skillgraph.harness.scienceworld_actions import catalog
from experiments.baselines.scienceworld.protocol import ScienceWorldTextPolicyProtocol as Protocol, PolicyRejection
from experiments.baselines.scienceworld.runner import ScienceWorldTextEpisodeRunner

def boundary():
    items = catalog([{'action': 'wait1', 'template_id': 0, 'obj_ids': []},
        {'action': 'move red cup to blue box', 'template_id': 1, 'obj_ids': [2,3]},
        {'action': 'move green cup to white box', 'template_id': 1, 'obj_ids': [4,5]}], 3)
    return SimpleNamespace(action_catalog=lambda: items, validator_channel=lambda: SimpleNamespace(revision=3))

def test_policy_does_not_invent_catalog_cartesian_product():
    h = boundary()
    assert Protocol.parse('{"action_type":"WAIT1","arguments":{}}', harness=h, revision=3).raw_action == 'wait1'
    with pytest.raises(PolicyRejection, match='illegal_or_stale'):
        Protocol.parse(json.dumps({'action_type':'MOVE', 'arguments':{'entity':'red cup','container':'white box'}}), harness=h, revision=3)
    with pytest.raises(PolicyRejection, match='stale_revision'):
        Protocol.parse('{"action_type":"WAIT1","arguments":{}}', harness=h, revision=2)
    with pytest.raises(PolicyRejection) as caught:
        Protocol.parse('{"intent":"explore","candidate_outputs":{},"action_type":"WAIT1"}', harness=h, revision=3)
    assert len(caught.value.errors) == 3

def test_one_repair_and_no_environment_action_for_invalid_response(tmp_path):
    calls = []
    class Harness:
        def __init__(self, **kwargs):
            self.channel = SimpleNamespace(revision=0, done=False, score=0, won=False)
            self._info = {'moves':0}
            self._frame = {'task_description':'Example','look':'public', 'observation':'public','inventory':'empty'}
        def reset(self, task): pass
        def validator_channel(self): return self.channel
        def action_catalog(self): return catalog([{'action':'wait1','template_id':0,'obj_ids':[]}], 0)
        def execute_action(self, *args): raise AssertionError('Invalid tuple reached environment')
        def _close_backend(self): pass
    def chat(**kwargs):
        calls.append(kwargs)
        return '{"action_type":"WAIT1", "arguments":{}, "extra":true}'
    task = {'task_id':'example','task_type':'1-1','macro_type':'1','task_name':'example',
            'variation_idx':0,'source_split':'train'}
    result = ScienceWorldTextEpisodeRunner(chat, harness_factory=Harness).run(task, '', tmp_path)
    assert len(calls) == result['protocol_rejections'] == 2
    assert result['environment_actions'] == 0
    assert result['termination_reason'] == 'protocol_repair_exhausted'
    assert not result['infrastructure_failure']


def test_interrupted_attempt_is_not_part_of_resumed_canonical_path(tmp_path):
    class Harness:
        def __init__(self, **kwargs):
            self.channel = SimpleNamespace(revision=0, done=False, score=0, won=False)
            self._info = {'moves': 0}
            self._frame = {'task_description':'Example', 'look':'public', 'observation':'public', 'inventory':'empty'}
        def reset(self, task): pass
        def validator_channel(self): return self.channel
        def action_catalog(self):
            return catalog([{'action':'wait1','template_id':0,'obj_ids':[]}], self.channel.revision)
        def execute_action(self, *args):
            self.channel.revision += 1
            self._info['moves'] += 1
            return SimpleNamespace(accepted=True, benchmark_score=0, benchmark_reward=0,
                done=False, metadata={'environment_moves':self._info['moves']}, observation='waited')
        def _close_backend(self): pass
    calls = 0
    def failing_chat(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError('credentials must not be copied to an error receipt')
        return '{"action_type":"WAIT1","arguments":{}}'
    task = {'task_id':'example','task_type':'1-1','macro_type':'1','task_name':'example',
            'variation_idx':0,'source_split':'train'}
    with pytest.raises(ConnectionError):
        ScienceWorldTextEpisodeRunner(failing_chat, harness_factory=Harness).run(task, '', tmp_path)
    failed = list((tmp_path / 'attempts').iterdir())[0]
    original = (failed / 'environment_actions.jsonl').read_bytes()
    assert not (tmp_path / 'result.json').exists()
    assert 'credentials' not in (failed / 'infrastructure_failure.json').read_text()
    result = ScienceWorldTextEpisodeRunner(lambda **kw:'{"action_type":"WAIT1","arguments":{}}',
        max_actions=1, harness_factory=Harness).run(task, '', tmp_path)
    assert result['environment_actions'] == 1
    assert len(list((tmp_path / 'attempts').iterdir())) == 2
    assert (failed / 'environment_actions.jsonl').read_bytes() == original
    assert result['selected_attempt_id'] != failed.name
    assert len(__import__('pathlib').Path(result['canonical_actions_path']).read_text().splitlines()) == 1


def test_report_preserves_retry_costs_reasoning_overlap_and_missing_capture(tmp_path):
    from experiments.baselines.scienceworld.report import summarize
    rows = [{'task_id':name, 'task_type':'1-1', 'macro_type':'1',
        'official_score':100, 'normalized_score':1, 'perfect_success':True,
        'environment_actions':2, 'environment_moves':2, 'wall_time_ms':10}
        for name in ('a', 'b')]
    events = [{'provider_attempt_id':str(i), 'logical_call_id':'call', 'task_id':'a',
        'role':'target', 'attempt':i+1, 'status':'failed' if i==0 else 'succeeded',
        'prompt_tokens':10, 'completion_tokens':5, 'reasoning_tokens':4}
        for i in range(2)]
    (tmp_path/'provider_calls.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    result = summarize(rows, tmp_path)
    assert result['usage']['recorded_total_tokens'] == 30
    assert result['usage']['recorded_reasoning_tokens'] == 8
    assert result['usage']['logical_requests'] == 1
    assert result['usage']['retries'] == 1
    assert result['efficiency']['token_quantile_unknown_tasks'] == 1
    assert result['efficiency']['p50_tokens'] == 30
    assert result['per_task'][1]['usage_capture_present'] is False
    assert result['per_task'][1]['selected_route'] is None
