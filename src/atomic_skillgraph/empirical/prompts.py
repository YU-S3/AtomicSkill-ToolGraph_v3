"""One readable protocol per role, with no historical witness language."""
from .contracts import object_schema

ANY_OBJECT = {"type": "object"}
TEXT = {"type": "string"}
REF = {"oneOf": [object_schema({"task": TEXT}, ["task"]),
    object_schema({"from": TEXT, "field": TEXT}, ["from", "field"]),
    object_schema({"literal": {}}, ["literal"]), object_schema({"unresolved": TEXT}, ["unresolved"])]}
NODE = object_schema({"id": TEXT, "goal": TEXT, "skill_id": TEXT, "program_id": TEXT,
    'dynamic': {'type': 'boolean'},
    "args": {"type": "object", "additionalProperties": REF},
    "after": {"type": "array", "items": TEXT}}, ["id", "goal", "args"])
WORKFLOW = object_schema({"goal": TEXT, "nodes": {"type": "array", "items": NODE, "minItems": 1, "maxItems": 16},
    "outputs": {"type": "object", "additionalProperties": REF}}, ["goal", "nodes"])
SKILL = object_schema({"goal": TEXT, "guidance": TEXT, "input_schema": ANY_OBJECT,
    "output_schema": ANY_OBJECT}, ["goal", "input_schema", "output_schema"])
LEARNING = object_schema({"decision": {"enum": ["no_change", "reuse_existing", "propose_skill_and_program_spec",
    "propose_or_revise_workflow"]}, "skill": SKILL, "generate_program": {"type": "boolean"},
    "existing_skill_id": TEXT, "workflow": WORKFLOW, "rationale": TEXT}, ["decision"])
TRIAL_INPUT = object_schema({'case_id': TEXT, 'inputs': ANY_OBJECT,
    'start_mode': {'enum': ['reset', 'prefix_replay']},
    'prefix': {'type': 'array', 'items': object_schema({'name': TEXT, 'arguments': ANY_OBJECT}, ['name', 'arguments'])}},
    ['case_id', 'inputs', 'start_mode', 'prefix'])
BUILD = object_schema({"source": TEXT, "example_inputs": ANY_OBJECT,
    'trial_inputs': {'type': 'array', 'items': TRIAL_INPUT, 'maxItems': 2}}, ['source', 'trial_inputs'])
STEP = object_schema({"action": {"enum": ["call_tool", "call_program", "complete_node", "revise_plan", "finish"]},
    "name": TEXT, "arguments": ANY_OBJECT, "outputs": ANY_OBJECT, "workflow": WORKFLOW,
    "used_inputs": {"type": "array", "items": TEXT}, "replaced_inputs": {"type": "array", "items": TEXT},
    "answer": {}, "detail": TEXT}, ["action"])

PLANNER_PROMPT = """Choose a short executable workflow for the original task. Use the supplied Skills/Workflows as
experience, not proof. Omit useless exploration and express dependencies only as explicit task/from/literal/unresolved
references. Preserve the user's quantities and identities. Nodes without a program are dynamic. Do not invent asset IDs.
Return one submit_plan ToolCall. A from reference must name an earlier node's actual output field."""
LEARNER_PROMPT = """Learn a reusable capability or workflow from this real Train experience, including its failures.
Choose no_change, reuse_existing, propose_skill_and_program_spec, or propose_or_revise_workflow. At most one new
program specification. Prefer a useful local goal including necessary search/preparation, ordinary I/O and concise
guidance; category queries and concrete object IDs are different fields. Examples are hypotheses, not proofs: no
witnesses/owners/occurrence certificates. Workflows may trim exploration, merge preparation, or retain dynamic gaps.
Declare all value dependencies explicitly. Do not claim historical success that is not in the supplied record.
If proposing a new skill, an accompanying workflow may reference it as $new. Return one submit_learning ToolCall."""
BUILDER_PROMPT = """Generate a reusable restricted Python program: def run(ctx, inputs) -> dict. Use normal variables,
branches and loops. ctx.observe(), ctx.available_tools(), ctx.call(name, arguments), ctx.remaining_calls() are the only
public task interfaces. No LLM, evaluator, network, host paths, external installation or private state. Do not call
execute_python recursively: use Python directly in the authorized /workspace with its public inputs. Standard library
is available. Calls consume the same task budget. Return status ok/not_found/needs_input/blocked and an outputs dict
matching the supplied schema. ok is not official success. Consult actual public feedback and current tool availability;
search can return not_found. Return exact resource IDs from the current tool arguments for subsequent calls; display
labels are not tool IDs. Do not hardcode example object instances/locations. Generate source plus ordinary
trial_inputs with one submit_program ToolCall. Provide one independent binding for each supplied case_id.
Use each case's public task to construct its own inputs; never copy concrete objects or file paths from another case.
Prefer start_mode reset and prefix []; intermediate-state skills require the exact real public action prefix
from that case's recorded experience. Only use the tool names supplied to you."""
RUNTIME_PROMPT = """Execute the current node toward the original user's task, keeping exact resource identities and
quantity constraints. Prefer an existing program when inputs are ready; otherwise use a public tool, call a preparation
program, or supply missing values. Only call currently provided tools with their normal arguments. Return one runtime_step
ToolCall: call_tool(name,arguments), call_program(name=program_id,arguments), complete_node(outputs), revise_plan(workflow),
or finish(answer). Node completion and a program's ok do not establish official success. On plan/feedback conflict revise
the unfinished plan once; never change the user's task or overwrite completed results. Avoid repeated unchanged failures.
When consuming supplied node inputs, list their field names in used_inputs. If local recovery redoes a preparation
and replaces its supplied result, list that field in replaced_inputs; do not credit the earlier preparation."""
