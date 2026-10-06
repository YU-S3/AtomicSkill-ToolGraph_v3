"""Thin public ALFWorld bridge. No raw action parser in the empirical core."""
from collections import defaultdict
from dataclasses import asdict, is_dataclass
import re

from ..empirical.contracts import PublicTask, object_schema
from .simple_protocol import HarnessTask
from .simple_protocol import Capabilities
from .tool_spec import ToolSpec, result_schema
from .alfworld import normalize_entity
from ..empirical.contracts import digest


class SimpleAlfWorld:
    capabilities = Capabilities(tool_surface="exact_catalog", checkpoint_mode="replay", local_check="available")

    def __init__(self, harness):
        self.harness = harness
        self.last = None
        self.task = None
        self.held = set()
        self.visited = []
        self.location = None
        self.inventory_status = 'known'
        self.discovered, self.inspected, self.object_locations = {}, {}, {}
        self.specs = {row['action_type']: ToolSpec(row['action_type'],
            row.get('public_semantics', 'Execute only exact arguments in the current public action catalog.'),
            object_schema({key: {'type': 'string'} for key in row['argument_roles']}, row['argument_roles']),
            result_schema({'type': 'object'}),
            effect='read_only' if row['action_type'] in {'LOOK', 'INVENTORY', 'EXAMINE'} else 'stateful')
            for row in harness.primitive_action_schema() if row['action_type'] != 'UNKNOWN'}

    def reset(self, task):
        self.task, self.held = task, set()
        self.visited, self.location = [], None
        self.inventory_status = 'known'
        self.discovered, self.inspected, self.object_locations = {}, {}, {}
        context = dict(task.inputs.get("environment_task", {}))
        original = HarnessTask(context.get('native_task_id', task.task_id), task.goal, "alfworld", context.get("task_type", ""),
                               context.get("context", {}), context.get("metadata", {}))
        self.last = self.harness.reset(original)
        self._apply_public_result('reset', {}, self.last)
        self.initial_observation = self.last.observation
        return self.observe()

    def model_task(self, task=None):
        task = task or self.task
        environment = task.inputs.get('environment_task', {})
        inputs = {'goal_roles': environment.get('context', {}).get('goal_roles', {})}
        if self.task and task.goal == self.task.goal:
            inputs['initial_observation'] = self.initial_observation
        elif 'initial_observation' in environment.get('context', {}):
            inputs['initial_observation'] = environment['context']['initial_observation']
        return {'goal': task.goal, 'inputs': {'environment_task': inputs}}

    def inherit_discovery(self, source):
        self.harness.inherit_discovery(source.harness)

    def observe(self):
        if self.last is None:
            return {}
        frame = self.harness.public_discovery_frame()
        return {"observation": self.last.observation, "done": self.last.done,
                "held_objects": sorted(self.held),
                "inventory_status": self.inventory_status,
                "visited_locations": self.visited, "current_location": self.location,
                "discovery": asdict(frame) if is_dataclass(frame) else {}}

    def available_tools(self):
        if self.last and (self.last.done or self.last.won):
            return []
        groups = defaultdict(list)
        for action in self.harness.action_catalog():
            groups[action.action_type].append(action.arguments)
        return [self.specs[name].view(current_arguments=options)
                for name, options in sorted(groups.items()) if name in self.specs]

    def tool_definitions(self):
        # Stable primitive interfaces, including operations unavailable at the
        # current location. The broker still resolves each call against the
        # actual public catalog, never against this static documentation.
        return [spec.view() for spec in self.specs.values()]

    def model_state(self):
        result = {k: v for k, v in self.observe().items() if k != 'discovery'}
        if self.inventory_status != 'known': result['held_objects'] = 'unknown'
        result.update(discovered=list(self.discovered.values()), inspected_scopes=list(self.inspected.values()),
                      object_locations=self.object_locations)
        return result

    def progress_key(self):
        return digest([{k: v for k,v in self.model_state().items() if k != 'observation'}, [{k: t[k] for k in ('name', 'current_arguments')}
                                           for t in self.available_tools()]])

    def _inventory(self, observation):
        text = str(observation).strip()
        if re.fullmatch(r'You are (?:not carrying anything|carrying nothing)[.!]?', text, re.I):
            return set()
        match = re.fullmatch(r'You are carrying:\s*(.*?)[.!]?\s*', text, re.I | re.S)
        if not match: return None
        content = match.group(1).strip().rstrip('.')
        if content.casefold() in {'nothing', 'none', 'empty'}: return set()
        items = re.split(r',\s*(?:and\s+)?|\s+and\s+', content)
        if not all(re.fullmatch(r'(?:a |an |the )?[\w -]+ \d+', item.strip(), re.I) for item in items):
            return None
        return {normalize_entity(re.sub(r'^(?:a|an|the)\s+', '', item.strip(), flags=re.I)) for item in items}

    def _apply_public_result(self, name, arguments, result):
        before = {'held_objects': sorted(self.held), 'inventory_status': self.inventory_status,
                  'current_location': self.location, 'object_locations': dict(self.object_locations)}
        self.public_update = {'changed_fields': [], 'inventory_status': self.inventory_status}
        if not result.accepted and name != 'reset': return
        obj = arguments.get('object')
        if name == 'GO_TO':
            self.location = arguments['destination']
            if self.location not in self.visited: self.visited.append(self.location)
        elif name == 'TAKE' and obj:
            self.held.add(obj)
            self.object_locations[obj] = 'inventory'
        elif name in {'PUT', 'MOVE'} and obj:
            self.held.discard(obj)
            self.object_locations[obj] = arguments['destination']
        elif name == 'INVENTORY':
            inventory = self._inventory(result.observation)
            self.inventory_status = 'unknown' if inventory is None else 'known'
            for entity, location in list(self.object_locations.items()):
                if location == 'inventory' and (inventory is None or entity not in inventory):
                    self.object_locations.pop(entity)
            if inventory is not None:
                self.held = inventory
                self.object_locations.update({entity: 'inventory' for entity in inventory})
        elif name in {'SLICE', 'USE'}:
            self.inventory_status = 'unknown'
            self.object_locations = {entity: location for entity, location in self.object_locations.items() if location != 'inventory'}
        frame = self.harness.public_discovery_frame()
        frame = asdict(frame) if is_dataclass(frame) else {}
        for row in frame.get('records', []):
            self.discovered[(row['entity'], row['location'])] = {k: row[k] for k in ('entity', 'location', 'relation_kind')}
        for row in frame.get('inspected_scopes', []):
            self.inspected[row['location']] = {k: row[k] for k in ('location', 'status')}
        after = {'held_objects': sorted(self.held), 'inventory_status': self.inventory_status,
                 'current_location': self.location, 'object_locations': self.object_locations}
        self.public_update = {'changed_fields': [k for k in before if before[k] != after[k]],
                              'inventory_status': self.inventory_status}

    def call(self, name, arguments):
        matches = [action for action in self.harness.action_catalog()
                   if action.action_type == name and action.arguments == arguments]
        if len(matches) != 1:
            self.public_update = {'changed_fields': [], 'inventory_status': self.inventory_status}
            return {"accepted": False, "data": {}, "error": "No current exact action matches these arguments",
                    "observation": self.last.observation, "done": bool(self.last.done), "environment_step": 0}
        action = matches[0]
        self.last = self.harness.execute_action(action.action_id, action.revision)
        self._apply_public_result(name, arguments, self.last)
        return {"accepted": self.last.accepted, "data": {}, "error": None if self.last.accepted else "Nothing happens",
                "observation": self.last.observation, "done": self.last.done or self.last.won, "environment_step": 1}

    def check_local(self, inputs, outputs, events):
        # A cheap independent result check based exclusively on public accepted
        # native feedback. No hidden object tree, model assertions or witnesses.
        if self.inventory_status != 'known': return 'unavailable'
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
        from importlib.metadata import version
        self.score_audit = {'scorer_version': 'alfworld@' + version('alfworld'),
            'raw_scorer_output': {'won': self.last.won, 'done': self.last.done,
                'benchmark_score': self.last.benchmark_score, 'benchmark_reward': self.last.benchmark_reward}}
        return {"raw_score": float(won), "hard": won, "soft": float(won), "scorer": "alfworld.official-won"}

    def close(self):
        self.harness._close_backend()

    def abort(self):
        self.close()
