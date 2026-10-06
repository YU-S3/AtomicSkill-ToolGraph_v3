"""One planning decision and one bounded structural repair."""
from .contracts import validate_workflow, normalize_workflow
from .prompts import PLANNER_PROMPT, PLAN
from .task_context import public_view
from copy import deepcopy
from .model_view import project


def dynamic(task):
    return {'interface_version': 'empirical.workflow.v2', "goal": task.goal,
            "nodes": [{"id": "task", 'execution_mode': 'dynamic', "goal": task.goal, "args": {}}]}


class Planner:
    def __init__(self, bank, agent):
        self.bank, self.agent = bank, agent

    def validate(self, workflow, completed=()):
        return validate_workflow(normalize_workflow(workflow, self.bank), completed=completed, bank=self.bank)

    def plan(self, task, adapter, feedback=None, completed_results=None):
        self.last_decision_id = None
        related = self.bank.retrieve(task.goal)
        if adapter.capabilities.interaction == "single_answer" or not related:
            return dynamic(task)
        workflows = {a['id']: a for a in related if 'nodes' in a}
        def instantiate(value):
            if value.get('mode') == 'select':
                if set(value) - {'mode','workflow_id','node_args','node_modes'} or 'workflow_id' not in value:
                    raise ValueError('select requires workflow_id and optional node_args/node_modes')
                if value.get('workflow_id') not in workflows: raise ValueError('Workflow was not retrieved')
                workflow = deepcopy(workflows[value['workflow_id']])
                nodes = {n['id']: n for n in workflow['nodes']}
                for node_id, overrides in value.get('node_args', {}).items():
                    if node_id not in nodes: raise ValueError('Unknown selected workflow node')
                    nodes[node_id].setdefault('args', {}).update(overrides)
            else:
                if value.get('mode') == 'compose' and set(value) != {'mode','workflow'}:
                    raise ValueError('compose requires a workflow only')
                workflow = value.get('workflow', value)
            workflow = normalize_workflow(workflow, self.bank, value.get('node_modes'))
            return self.validate(workflow, completed_results or ())
        materials = {'original_task': {"goal": task.goal, "inputs": task.inputs}, 'related_interfaces': self.bank.planning_cards(task.goal),
                     'program_options': self.bank.program_options(task.goal),
                     'tool_definitions': getattr(adapter, 'tool_definitions', adapter.available_tools)(),
                     'current_tools': adapter.available_tools(), 'public_state': public_view(adapter)}
        if feedback or completed_results:
            materials.update(feedback=feedback or [], completed_results=completed_results or {})
        materials = project('planner', materials, task=task, adapter=adapter)
        try:
            response = self.agent("planner", PLANNER_PROMPT, materials, "submit_plan", PLAN,
                              validator=instantiate, repair_limit=1,
                              owner_state_version=getattr(self, 'checkpoint', None).state.get('executor_state', {}).get('owner_version', 0)
                              if getattr(self, 'checkpoint', None) else 0)
            self.last_decision_id = getattr(getattr(self.agent, '__self__', None), 'last_decision_id', None)
            return instantiate(response)
        except ValueError:
            return dynamic(task)
