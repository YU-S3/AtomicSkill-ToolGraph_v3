import copy
from atomic_skillgraph.agents.node_context import project_node_context
from atomic_skillgraph.agents.protocol import NativeToolSpec


def test_exact_goal_duplicate_only_and_all_obligations_unchanged():
    raw = {'task_goal': 'Move the same object.', 'current_state_snapshot': {
        'current_atomic': {'summary': 'Move the same object.',
            'outputs': [{'name': 'object', 'required_resolution': 'concrete'}]},
        'confirmed_bindings': {'object': 'item_1'}, 'missing_bindings': ['destination'],
        'downstream_obligations': {'consumer': {'object': 'item_1'}}},
        'execution_frame': {'repeat': {'distinct': ['item_2']}},
        'current_observation': 'New unknown feedback: preserve this verbatim.',
        'current_action_catalog': [{'action_id': 'a1'}]}
    original = copy.deepcopy(raw)
    tool = NativeToolSpec('environment_action', 'Act', {'type': 'object'})
    result, audit = project_node_context(raw, native_tool_specs=[tool])
    assert raw == original
    assert result == {k: v for k, v in raw.items() if k != 'task_goal'}
    assert not audit['context_projection_fallback']
    assert audit['after_body_utf8_bytes'] < audit['before_body_utf8_bytes']
    assert audit['native_tools_utf8_bytes'] > 0
    raw['task_goal'] += ' Twice, using different objects.'
    result, audit = project_node_context(raw, native_tool_specs=[tool])
    assert result == raw and audit['context_projection_fallback']


def test_production_builder_retains_unproven_goal_and_records_audit():
    from atomic_skillgraph.agents.context_builder import ContextBuilder
    audit = {}
    rendered = ContextBuilder().runtime_node(task_goal='Observe two distinct objects.',
        atomic_contract={'summary': 'Observe one object.', 'inputs': [], 'outputs': [],
                         'preconditions': [], 'effects': [], 'guideline': {}},
        observation='Unrecognized observation!', action_catalog=[], relevant_action_history=[],
        remaining_budget={}, implementation_invocations=[], projection_audit=audit)
    assert 'Observe two distinct objects.' in rendered
    assert 'Unrecognized observation!' in rendered
    assert audit['node_context']['context_projection_fallback']
