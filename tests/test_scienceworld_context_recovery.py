import json
from experiments.baselines.scienceworld.protocol import ScienceWorldTextPolicyProtocol as Protocol


def test_retired_catalog_preserves_observations_actions_and_current_choices():
    old = {'task': 'goal', 'observation': 'important fact', 'look': 'room',
           'inventory': 'object', 'valid_actions_compact': {'MOVE': ['a']}}
    messages = [{'role': 'system', 'content': 'instructions'},
                {'role': 'user', 'content': json.dumps(old)},
                {'role': 'assistant', 'content': '{"action_type":"MOVE","arguments":{"to":"a"}}'}]
    action = dict(messages[-1])
    Protocol.retire_action_catalogs(messages)
    history = json.loads(messages[1]['content'])
    assert history['observation'] == old['observation']
    assert history['inventory'] == old['inventory']
    assert 'valid_actions_compact' not in history
    assert messages[-1] == action
    current = {**old, 'valid_actions_compact': {'MOVE': ['b', 'c']}}
    messages.append({'role': 'user', 'content': json.dumps(current)})
    assert json.loads(messages[-1]['content']) == current
    assert old['valid_actions_compact'] == {'MOVE': ['a']}


def test_skillopt_resume_skips_only_identical_completed_episode(tmp_path, monkeypatch):
    import pytest
    from experiments.baselines.scienceworld.skillopt_adapter import ScienceWorldSkillOptAdapter
    from experiments.baselines.scienceworld.runner import ScienceWorldTextEpisodeRunner
    entry = {'task_id':'t','task_type':'type','source_split':'train'}
    adapter = ScienceWorldSkillOptAdapter([], [], lambda *a: None)
    episode=tmp_path/'predictions/t'
    attempt=episode/'attempts/a'
    attempt.mkdir(parents=True)
    result={'source_identity':entry,'selected_attempt_id':'a','perfect_success':True,
            'normalized_score':1.0,'official_score':100}
    (episode/'result.json').write_text(json.dumps(result))
    (attempt/'messages.json').write_text(json.dumps([{'role':'system','content':Protocol.instruction+'\n\nReusable skill knowledge:\nskill'}]))
    monkeypatch.setattr(ScienceWorldTextEpisodeRunner,'run',lambda *a:pytest.fail('completed task replayed'))
    assert adapter.rollout([entry],'skill',tmp_path)[0]['hard']==1
    with pytest.raises(ValueError,match='skill mismatch'):
        adapter.rollout([entry],'different',tmp_path)
