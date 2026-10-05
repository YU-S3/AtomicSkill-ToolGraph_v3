"""Ordinary interfaces and explicit, task-local value references."""
import ast
import hashlib
import json
from dataclasses import dataclass, field

from ..agents.protocol import validate_schema_instance


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def object_schema(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    physical_key: str
    goal: str
    inputs: dict = field(default_factory=dict)
    split: str = "train"


@dataclass(frozen=True)
class TrialCase:
    case_id: str
    physical_task_key: str
    public_task_ref: str
    inputs: dict
    start_mode: str = 'reset'
    prefix: tuple = ()
    check: str = 'task_outcome'
    continuation: str = 'dynamic'

    def __post_init__(self):
        if self.start_mode not in {'reset', 'prefix_replay'} or self.check not in {'local', 'task_outcome'}:
            raise ValueError('Invalid TrialCase mode')
        if self.start_mode == 'reset' and self.prefix:
            raise ValueError('Reset TrialCase cannot include a prefix')


def program_digest(program):
    keys = ['source', 'entry', 'input_schema', 'output_schema', 'allowed_tools', 'environment']
    keys += [key for key in ('result_role', 'entry_constraints') if key in program]
    return digest({key: sorted(set(program[key])) if key == 'allowed_tools' else program[key] for key in keys})


def validate_program(program):
    for key in ("source", "entry", "input_schema", "output_schema", "allowed_tools", "environment"):
        if key not in program:
            raise ValueError("Program missing " + key)
    if program["entry"] != "run" or program.get("backend", "sandbox_python_v1") != "sandbox_python_v1":
        raise ValueError("Program requires sandbox_python_v1/run")
    if not isinstance(program["allowed_tools"], list) or not all(
            isinstance(name, str) and name for name in program["allowed_tools"]):
        raise ValueError("allowed_tools must be tool names")
    for schema in (program["input_schema"], program["output_schema"]):
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("Program interfaces must be object schemas")
    environment = program['environment']
    if program.get('result_role', 'intermediate') not in {'intermediate','final_answer','final_files'}:
        raise ValueError('Unknown Program result_role')
    if not isinstance(environment, dict) or not environment.get('adapter_abi') or not environment.get('image_digest'):
        raise ValueError('Program environment needs Adapter ABI and locked image digest')
    tree = ast.parse(program["source"])
    entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    if len(entries) != 1 or [arg.arg for arg in entries[0].args.args] != ["ctx", "inputs"]:
        raise ValueError("Program entry must be def run(ctx, inputs)")
    compile(tree, "<program>", "exec")  # Syntax only; user code never runs here.
    return program_digest(program)


def resolve_node_interface(node, bank):
    """Read one execution binding; references never authorize an automatic route."""
    mode = node.get('execution_mode')
    if mode is None:
        if node.get('program_id'):
            mode = 'program'  # Read-only projection of an unambiguous legacy node.
        elif not node.get('skill_id'):
            mode = 'dynamic'
        else:
            raise ValueError('Legacy Skill node requires an explicit execution_mode')
    if mode not in {'dynamic', 'skill', 'program'}:
        raise ValueError('Unknown execution_mode')
    references = node.get('reference_skill_ids', [])
    if not isinstance(references, list) or len(references) > 3 or len(set(references)) != len(references):
        raise ValueError('reference_skill_ids needs at most three distinct Skill IDs')
    skills = {s['id']: s for s in bank.all('skill')}
    if any(ref not in skills for ref in references):
        raise ValueError('Unknown reference Skill')
    skill_id, program_id = node.get('skill_id'), node.get('program_id')
    if (mode == 'dynamic' and (skill_id or program_id) or mode == 'skill' and program_id or
            mode == 'program' and skill_id):
        raise ValueError('Execution bindings must be mutually exclusive')
    asset = None
    if mode == 'skill':
        if not skill_id or skill_id not in skills:
            raise ValueError('Unknown bound Skill')
        asset = skills[skill_id]
    elif mode == 'program':
        asset = bank.get(program_id or '')
        if not asset or 'source' not in asset:
            raise ValueError('Unknown bound Program')
        linked = sorted(i['skill_id'] for i in bank.all('implementation') if i['program_id'] == program_id)
        skill_id = next((i for i in linked if i in skills), None)
    if mode == 'dynamic':
        goal = node.get('goal')
        if not isinstance(goal, str) or not goal:
            raise ValueError('Dynamic node needs a result goal')
    else:
        if node.get('execution_mode') and 'goal' in node:
            raise ValueError('Bound node uses the asset goal; put parent intent in purpose')
        goal = (skills.get(skill_id) or asset).get('goal')
        if not goal:
            if mode == 'skill':
                raise ValueError('Bound Skill needs its stored goal')
            goal = 'Execute Program ' + program_id
    return {'execution_mode': mode, 'node_goal': goal, 'purpose': node.get('purpose', ''),
            'reference_skill_ids': list(references), 'bound_skill_id': skill_id,
            'bound_program_id': program_id, 'input_schema': asset.get('input_schema') if asset else None,
            'output_schema': asset.get('output_schema') if asset else None,
            'result_role': (asset or {}).get('result_role', 'intermediate'),
            'args': node.get('args', {}), 'after': node.get('after', [])}


def normalize_workflow(workflow, bank, node_modes=None):
    from copy import deepcopy
    result = deepcopy(workflow)
    modes = node_modes or {}
    if set(modes) - {n['id'] for n in result['nodes']}:
        raise ValueError('Unknown node_modes node')
    for node in result['nodes']:
        if 'execution_mode' in node and node['id'] in modes:
            raise ValueError('node_modes cannot override a native v2 binding')
        if 'execution_mode' not in node:
            mode = modes.get(node['id'])
            if node.get('program_id'):
                mode = 'program'
            elif not node.get('skill_id'):
                mode = 'dynamic'
            if mode not in {'dynamic', 'skill', 'program'}:
                raise ValueError('Legacy Skill node needs node_modes dynamic/skill')
            if mode == 'dynamic' and node.get('skill_id'):
                node.setdefault('reference_skill_ids', []).append(node.pop('skill_id'))
            node['execution_mode'] = mode
            if mode != 'dynamic':
                node.setdefault('purpose', node.pop('goal', ''))
        node.pop('dynamic', None)
        resolve_node_interface(node, bank)
    result['interface_version'] = 'empirical.workflow.v2'
    return result


def output_view(outputs, aliases=None):
    if not isinstance(outputs, dict):
        raise ValueError('outputs must be ordinary values in an object')
    aliases = aliases or {}
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()):
        raise ValueError('output_aliases must map actual fields to field names')
    if len(set(aliases.values())) != len(aliases):
        raise ValueError('Duplicate output alias target')
    view = dict(outputs)
    for source, target in aliases.items():
        if source not in outputs:
            raise ValueError('Unknown output alias source ' + source)
        if target in outputs and (type(outputs[target]) is not type(outputs[source]) or outputs[target] != outputs[source]):
            raise ValueError('Output alias would overwrite a different original value')
        view[target] = outputs[source]
    return view


def handoff_requirements(workflow, node_id, completed=()):
    consumers = []
    for node in workflow['nodes']:
        if node['id'] in completed or node['id'] == node_id:
            continue
        for name, ref in node.get('args', {}).items():
            if set(ref) == {'from', 'field'} and ref['from'] == node_id:
                consumers.append({'node_id': node['id'], 'input': name, 'field': ref['field']})
    for name, ref in workflow.get('outputs', {}).items():
        if set(ref) == {'from', 'field'} and ref['from'] == node_id:
            consumers.append({'node_id': '$outputs', 'input': name, 'field': ref['field']})
    return {'required_fields': sorted({r['field'] for r in consumers}), 'consumers': consumers}


class HandoffError(ValueError):
    def __init__(self, feedback):
        self.feedback = feedback
        super().__init__(json.dumps(feedback, ensure_ascii=False))


def validate_workflow(workflow, known_programs=(), known_skills=(), completed=(), *, bank=None):
    if bank is None:
        class Interfaces:
            def all(self, kind):
                return list((known_programs if kind == 'program' else known_skills if kind == 'skill' else {}).values())
            def get(self, key):
                return known_programs.get(key) if isinstance(known_programs, dict) else None
        if not isinstance(known_programs, dict) or not isinstance(known_skills, dict):
            # ID-only callers can check references, but cannot resolve execution interfaces.
            if not isinstance(known_programs, dict):
                known_programs = {key: {'id': key, 'source': '', 'input_schema': {}, 'output_schema': {}} for key in known_programs}
            if not isinstance(known_skills, dict):
                known_skills = {key: {'id': key, 'goal': key, 'input_schema': {}, 'output_schema': {}} for key in known_skills}
        bank = Interfaces()
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 16:
        raise ValueError("Workflow needs 1..16 ordered nodes")
    seen = set(completed)
    fields = {node_id: set(outputs) for node_id, outputs in completed.items()} if isinstance(completed, dict) else {}
    def check_field(ref):
        if ref['from'] in fields and ref['field'] not in fields[ref['from']]:
            raise ValueError('Unknown result field ' + ref['field'])
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in seen:
            raise ValueError("Node needs a unique id")
        interface = resolve_node_interface(node, bank)
        if not isinstance(node.get("after", []), list) or not set(node.get("after", [])).issubset(seen):
            raise ValueError("Order must reference completed predecessors")
        for ref in node.get("args", {}).values():
            if not isinstance(ref, dict):
                raise ValueError("Arguments must be explicit references")
            keys = set(ref)
            if keys == {"from", "field"}:
                if ref["from"] not in seen or not isinstance(ref["field"], str):
                    raise ValueError("Result reference needs a preceding producer and field")
                check_field(ref)
            elif keys not in ({"task"}, {"literal"}, {"unresolved"}):
                raise ValueError("Unsupported parameter reference")
        seen.add(node_id)
        schema = interface['output_schema']
        aliases = node.get('output_aliases', {})
        if len(set(aliases.values())) != len(aliases):
            raise ValueError('Duplicate output alias target')
        if schema and schema.get('additionalProperties') is False:
            actual = set(schema.get('properties', {}))
            if set(aliases) - actual:
                raise ValueError('Unknown output alias source')
            fields[node_id] = actual | set(aliases.values())
    for ref in workflow.get("outputs", {}).values():
        if not isinstance(ref, dict) or set(ref) != {"from", "field"} or ref["from"] not in seen:
            raise ValueError("Final output must reference a declared node field")
        check_field(ref)
    return workflow


class ValueStore:
    def __init__(self, task, context=None):
        self.task = {**task.inputs, "goal": task.goal}
        self.context = context
        self.results = {}
        self.history = []

    def resolve(self, references):
        values, missing = {}, {}
        for name, ref in references.items():
            if "literal" in ref:
                values[name] = ref["literal"]
            elif "task" in ref and ref["task"] in self.task:
                values[name] = self.task[ref["task"]]
            elif "from" in ref and ref["field"] in self.results.get(ref["from"], {}):
                value = self.results[ref["from"]][ref["field"]]
                values[name] = self.context.resolve(value) if self.context and isinstance(value, ResultRef) else value
            else:
                missing[name] = ref
        return values, missing

    def publish(self, node_id, outputs, origin, output_refs=None, aliases=None):
        if node_id in self.results:
            raise ValueError("Completed results cannot be overwritten")
        refs = output_refs or {}
        if refs:
            self.context.bind(outputs, refs)
        actual = {**outputs, **{k: ResultRef(v) for k, v in refs.items()}}
        self.results[node_id] = output_view(actual, aliases)
        self.history.append({"node": node_id, "origin": origin, "outputs": actual,
                             'handoff_view': self.results[node_id]})

    def model_view(self):
        return {node: {key: dict(value) if isinstance(value, ResultRef) else self.context.preview(value)
                       if self.context else value for key, value in fields.items()}
                for node, fields in self.results.items()}


class ResultRef(dict):
    """Tagged in process so an ordinary user dictionary is never unwrapped."""


@dataclass(frozen=True)
class RuntimeDecision:
    actions: tuple
    call_ids: tuple
    response_id: str
