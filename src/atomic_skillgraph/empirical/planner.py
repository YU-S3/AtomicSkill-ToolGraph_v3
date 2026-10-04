"""One planning decision and one bounded structural repair."""
from .contracts import validate_workflow
from .prompts import PLANNER_PROMPT, WORKFLOW


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
        materials = {"task": {"goal": task.goal, "inputs": task.inputs}, "interfaces": related,
                     "programs": [{k: p[k] for k in ("id", "state", "input_schema", "output_schema")}
                                  for p in self.bank.all("program") if p["state"] == "usable"][:8],
                     "public_state": adapter.observe(), "tools": adapter.available_tools(), "feedback": feedback,
                     "completed_results": completed_results or {}}
        try:
            return self.agent("planner", PLANNER_PROMPT, materials, "submit_plan", WORKFLOW,
                              validator=lambda value: self.validate(value, completed_results or ()), repair_limit=1)
        except ValueError:
            return dynamic(task)
