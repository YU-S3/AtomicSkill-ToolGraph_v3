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


def validate_workflow(workflow, known_programs=(), known_skills=(), completed=()):
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
        if not isinstance(node_id, str) or not node_id or node_id in seen or not node.get("goal"):
            raise ValueError("Node needs a unique id and goal")
        if node.get("program_id") and node["program_id"] not in known_programs:
            raise ValueError("Unknown Program " + node["program_id"])
        if node.get("skill_id") and node["skill_id"] not in known_skills:
            raise ValueError("Unknown Skill " + node["skill_id"])
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
        producer = known_programs.get(node.get('program_id')) if isinstance(known_programs, dict) else None
        producer = producer or (known_skills.get(node.get('skill_id')) if isinstance(known_skills, dict) else None)
        if producer and producer['output_schema'].get('additionalProperties') is False:
            fields[node_id] = set(producer['output_schema'].get('properties', {}))
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

    def publish(self, node_id, outputs, origin, output_refs=None):
        if node_id in self.results:
            raise ValueError("Completed results cannot be overwritten")
        refs = output_refs or {}
        if refs:
            self.context.bind(outputs, refs)
        self.results[node_id] = {**outputs, **{k: ResultRef(v) for k, v in refs.items()}}
        self.history.append({"node": node_id, "origin": origin, "outputs": self.results[node_id]})

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
