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
    tree = ast.parse(program["source"])
    entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    if len(entries) != 1 or [arg.arg for arg in entries[0].args.args] != ["ctx", "inputs"]:
        raise ValueError("Program entry must be def run(ctx, inputs)")
    compile(tree, "<program>", "exec")  # Syntax only; user code never runs here.
    return digest({key: program[key] for key in (
        "source", "entry", "input_schema", "output_schema", "allowed_tools", "environment")})


def validate_workflow(workflow, known_programs=(), known_skills=(), completed=()):
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 16:
        raise ValueError("Workflow needs 1..16 ordered nodes")
    seen = set(completed)
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
            elif keys not in ({"task"}, {"literal"}, {"unresolved"}):
                raise ValueError("Unsupported parameter reference")
        seen.add(node_id)
    for ref in workflow.get("outputs", {}).values():
        if not isinstance(ref, dict) or set(ref) != {"from", "field"} or ref["from"] not in seen:
            raise ValueError("Final output must reference a declared node field")
    return workflow


class ValueStore:
    def __init__(self, task):
        self.task = {**task.inputs, "goal": task.goal}
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
                values[name] = self.results[ref["from"]][ref["field"]]
            else:
                missing[name] = ref
        return values, missing

    def publish(self, node_id, outputs, origin):
        if node_id in self.results:
            raise ValueError("Completed results cannot be overwritten")
        self.results[node_id] = dict(outputs)
        self.history.append({"node": node_id, "origin": origin, "outputs": dict(outputs)})
