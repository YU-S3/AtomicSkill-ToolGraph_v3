"""Thin public ALFWorld bridge. No raw action parser in the empirical core."""
from collections import defaultdict
from dataclasses import asdict, is_dataclass

from ..empirical.contracts import PublicTask, object_schema
from .protocol import HarnessTask
from .simple_protocol import Capabilities


class SimpleAlfWorld:
    capabilities = Capabilities(tool_surface="exact_catalog", checkpoint_mode="replay", local_check="available")

    def __init__(self, harness):
        self.harness = harness
        self.last = None
        self.task = None
        self.held = set()
        self.visited = []
        self.location = None

    def reset(self, task):
        self.task, self.held = task, set()
        self.visited, self.location = [], None
        context = dict(task.inputs.get("environment_task", {}))
        original = HarnessTask(task.task_id, task.goal, "alfworld", context.get("task_type", ""),
                               context.get("context", {}), context.get("metadata", {}))
        self.last = self.harness.reset(original)
        return self.observe()

    def observe(self):
        if self.last is None:
            return {}
        frame = self.harness.public_discovery_frame()
        return {"observation": self.last.observation, "done": self.last.done,
                "held_objects": sorted(self.held),
                "visited_locations": self.visited, "current_location": self.location,
                "discovery": asdict(frame) if is_dataclass(frame) else {}}

    def available_tools(self):
        if self.last and self.last.done:
            return []
        groups = defaultdict(list)
        for action in self.harness.action_catalog():
            groups[action.action_type].append(action.arguments)
        return [{"name": name, "description": "ALFWorld native " + name,
                 "input_schema": object_schema({key: {"type": "string"} for key in options[0]}, options[0]),
                 "current_arguments": options, "output_schema": {"type": "object"}}
                for name, options in sorted(groups.items()) if name != "UNKNOWN"]

    def tool_definitions(self):
        # Stable primitive interfaces, including operations unavailable at the
        # current location. The broker still resolves each call against the
        # actual public catalog, never against this static documentation.
        return [{"name": row["action_type"], "description": "ALFWorld native " + row["action_type"],
                 "input_schema": object_schema({key: {"type": "string"} for key in row["argument_roles"]}, row["argument_roles"])}
                for row in self.harness.primitive_action_schema()]

    def call(self, name, arguments):
        matches = [action for action in self.harness.action_catalog()
                   if action.action_type == name and action.arguments == arguments]
        if len(matches) != 1:
            return {"accepted": False, "status": "blocked", "error": "No current exact action matches these arguments",
                    "observation": self.last.observation, "done": bool(self.last.done)}
        action = matches[0]
        self.last = self.harness.execute_action(action.action_id, action.revision)
        if self.last.accepted:
            obj = arguments.get("object")
            if name == "GO_TO":
                self.location = arguments.get('destination')
                if self.location not in self.visited:
                    self.visited.append(self.location)
            elif name == "TAKE" and obj:
                self.held.add(obj)
            elif name == "PUT" and obj:
                self.held.discard(obj)
        return {"accepted": self.last.accepted, "status": "ok" if self.last.accepted else "blocked",
                "observation": self.last.observation, "outputs": {}, "done": self.last.done}

    def check_local(self, inputs, outputs, events):
        # A cheap independent result check based exclusively on public accepted
        # native feedback. No hidden object tree, model assertions or witnesses.
        returned_ids = [value for value in outputs.values() if isinstance(value, str) and value in self.held]
        acquired = {e["arguments"].get("object") for e in events
                    if e["name"] == "TAKE" and e.get("result", {}).get("accepted")}
        query = inputs.get('target_query', inputs.get('query'))
        if isinstance(query, str) and any(value in acquired and
                (value == query or value.rsplit('_', 1)[0] == query.casefold()) for value in returned_ids):
            return "passed"
        return "unavailable"

    def submit(self, final_output):
        return {"task_id": self.task.task_id, "prediction": final_output}

    def evaluate(self, sealed_output):
        if sealed_output["task_id"] != self.task.task_id:
            raise ValueError("Submission task mismatch")
        won = bool(self.last.won)
        return {"raw_score": float(won), "hard": won, "soft": float(won), "scorer": "alfworld.official-won"}

    def close(self):
        self.harness._close_backend()
