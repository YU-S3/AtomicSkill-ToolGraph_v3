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
        contract = getattr(adapter, 'answer_contract', lambda: '')()
        if contract: value['original_task']['answer_contract'] = contract
        if context and adapter.capabilities.interaction != 'single_answer':
            value['original_task']['inputs'] = {k: context.preview(v) for k, v in value['original_task']['inputs'].items()}
    if stage == 'runtime' and 'node_interface' in value:
        public_task = value.pop('original_task')
        interface = value.pop('node_interface')
        node_goal = value.pop('node_goal')
        if interface.get('node_goal') == node_goal: interface.pop('node_goal')
        if node_goal == public_task['goal']: node_goal = {'task_field': 'goal'}
        inputs = value.pop('inputs')
        if context and task and inputs:
            source = context.reference({'goal': task.goal, 'inputs': task.inputs}, 'task_inputs')
            inputs = {k: {**source, 'path': ['inputs', k]} if k in task.inputs and v == task.inputs[k]
                      else context.preview(v) for k, v in inputs.items()}
        state = value.pop('public_state')
        if task and state.get('inputs') == task.inputs: state.pop('inputs')
        initial = public_task.get('inputs', {}).get('environment_task', {})
        if context and isinstance(initial, dict) and state.get('observation') and state['observation'] == initial.get('initial_observation'):
            state['observation'] = context.reference(state['observation'], 'initial_observation')
        memory = value.pop('working_memory')
        recent = value.pop('recent')
        completed = value.pop('completed_results')
        if context:
            for row in recent:
                rid = row.get('result_id')
                if not rid: continue
                original = context.results[rid]
                if original.get('observation') and original.get('observation') == state.get('observation'):
                    row['observation'] = {'result_id': rid, 'path': ['observation']}
                for fields in completed.values():
                    for key, field in fields.items():
                        if key in original.get('outputs', {}) and field == original['outputs'][key]:
                            fields[key] = {'result_id': rid, 'path': ['outputs', key]}
        # Memory stores query/scope/acceptance and the result ID; previews live in recent_results.
        return {'model_view_version': MODEL_VIEW_VERSION, 'task': public_task,
            'node': {'id': value.pop('node_id'), 'goal': node_goal, 'interface': interface},
            'bindings': {'inputs': inputs, 'missing': value.pop('missing'),
                         'completed_results': completed},
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
