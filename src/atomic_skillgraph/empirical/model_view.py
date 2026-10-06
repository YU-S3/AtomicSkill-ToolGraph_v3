"""Readable, public stage projections. Execution and scoring use original values."""
from copy import deepcopy
from . import POLICY_DEFAULTS
from .contracts import PublicTask

MODEL_VIEW_VERSION = POLICY_DEFAULTS['runtime']['model_view_version']


def model_task(task, adapter):
    if getattr(adapter, 'task', None) is not None and hasattr(adapter, 'model_task'):
        return deepcopy(adapter.model_task(task))
    return {'goal': task.goal, 'inputs': deepcopy(task.inputs)}


def project(stage, material, *, task=None, adapter=None, context=None):
    value = deepcopy(material)
    if 'original_task' in value and task is not None:
        value['original_task'] = model_task(task, adapter)
    if stage == 'runtime' and 'node_interface' in value:
        public_task = value.pop('original_task')
        interface = value.pop('node_interface')
        node_goal = value.pop('node_goal')
        if interface.get('node_goal') == node_goal: interface.pop('node_goal')
        if node_goal == public_task['goal']: node_goal = {'task_field': 'goal'}
        inputs = value.pop('inputs')
        if context and task:
            source = context.reference({'goal': task.goal, 'inputs': task.inputs}, 'task_inputs')
            inputs = {k: {**source, 'path': ['inputs', k]} if k in task.inputs and v == task.inputs[k]
                      else context.preview(v) for k, v in inputs.items()}
        state = value.pop('public_state')
        if state.get('inputs') == public_task.get('inputs'): state.pop('inputs')
        memory = value.pop('working_memory')
        recent = value.pop('recent')
        shown = {r['result_id'] for r in recent if 'result_id' in r}
        for row in memory:
            if row.get('result_id') in shown: row['result'] = {'result_id': row['result_id'], 'path': []}
        return {'model_view_version': MODEL_VIEW_VERSION, 'task': public_task,
            'node': {'id': value.pop('node_id'), 'goal': node_goal, 'interface': interface},
            'bindings': {'inputs': inputs, 'missing': value.pop('missing'),
                         'completed_results': value.pop('completed_results')},
            'handoff': {k: value.pop(k) for k in ('required_handoff_fields', 'handoff_consumers', 'return_example', 'pending_outputs')},
            'state': state,
            'calls': {k: value.pop(k) for k in ('tools', 'programs', 'remaining_calls', 'allowed_calls')},
            'memory': {'guidance': value.pop('guidance'), 'operations': memory, 'recent_results': recent},
            'recovery': value}
    if stage == 'extractor' and task is not None:
        value['experience']['task'] = model_task(task, adapter)
        if context:
            for case in value.get('completed_train_cases', []):
                original = case['task']
                case['task'] = context.preview(model_task(PublicTask('', '', original['goal'], original.get('inputs', {})), adapter))
                case['action_prefix'] = context.preview(case['action_prefix'])
    return value


def callable_tools(material):
    return material.get('calls', {}).get('tools', material.get('tools', []))
