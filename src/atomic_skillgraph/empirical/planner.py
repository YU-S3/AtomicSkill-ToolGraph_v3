"""One planning decision and one bounded structural repair."""
from .contracts import validate_workflow
from .prompts import PLANNER_PROMPT, PLAN
from .task_context import public_view
from copy import deepcopy


def dynamic(task):
    return {"goal": task.goal, "nodes": [{"id": "task", "goal": task.goal, "args": {}}]}


class Planner:
    def __init__(self, bank, agent):
        self.bank, self.agent = bank, agent

    def validate(self, workflow, completed=()):
        return validate_workflow(workflow, {p['id']: p for p in self.bank.all('program')},
                                 {s['id']: s for s in self.bank.all('skill')}, completed)

    def plan(self, task, adapter, feedback=None, completed_results=None):
        related = self.bank.retrieve(task.goal)
        if adapter.capabilities.interaction == "single_answer" or not related:
            return dynamic(task)
        workflows = {a['id']: a for a in related if 'nodes' in a}
        def instantiate(value):
            if value.get('mode') == 'select':
                if set(value) != {'mode','workflow_id','node_args'}:
                    raise ValueError('select requires workflow_id and node_args only')
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
            return self.validate(workflow, completed_results or ())
        materials = {"task": {"goal": task.goal, "inputs": task.inputs}, "interfaces": related,
                     "programs": self.bank.program_options(task.goal),
                     "public_state": public_view(adapter), "tools": adapter.available_tools(), "feedback": feedback,
                     "completed_results": completed_results or {}}
        try:
            return instantiate(self.agent("planner", PLANNER_PROMPT, materials, "submit_plan", PLAN,
                              validator=instantiate, repair_limit=1))
        except ValueError:
            return dynamic(task)
