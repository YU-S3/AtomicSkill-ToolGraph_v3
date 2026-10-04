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
    "output_schema": ANY_OBJECT, 'execution_intent': {'enum': ['guidance_only', 'program_requested']},
    'result_role': {'enum': ['intermediate', 'final_answer', 'final_files']}, 'entry_constraints': TEXT},
    ["goal", "input_schema", "output_schema"])
LEARNING = object_schema({"decision": {"enum": ["no_change", "reuse_existing", "propose_skill_and_program_spec",
    "propose_or_revise_workflow"]}, "skill": SKILL, "generate_program": {"type": "boolean"},
    "existing_skill_id": TEXT, "workflow": WORKFLOW, "rationale": TEXT,
    'realization_request': object_schema({'skill_id': TEXT, 'action': {'enum': ['build','repair','trial','defer']},
        'case_bindings': {'type': 'array', 'maxItems': 2, 'items': object_schema({'case_id': TEXT,
            'inputs': ANY_OBJECT, 'start_mode': {'enum': ['reset','prefix_replay']},
            'prefix': {'type': 'array', 'items': object_schema({'name': TEXT, 'arguments': ANY_OBJECT}, ['name','arguments'])}},
            ['case_id','inputs','start_mode','prefix'])}}, ['skill_id','action','case_bindings'])}, ["decision"])
TRIAL_INPUT = object_schema({'case_id': TEXT, 'inputs': ANY_OBJECT,
    'start_mode': {'enum': ['reset', 'prefix_replay']},
    'prefix': {'type': 'array', 'items': object_schema({'name': TEXT, 'arguments': ANY_OBJECT}, ['name', 'arguments'])}},
    ['case_id', 'inputs', 'start_mode', 'prefix'])
BUILD = object_schema({"source": TEXT, "example_inputs": ANY_OBJECT,
    'trial_inputs': {'type': 'array', 'items': TRIAL_INPUT, 'maxItems': 2}}, ['source', 'trial_inputs'])
STEP = object_schema({"action": {"enum": ["call_tool", "call_program", "complete_node", "revise_plan", "finish"]},
    "name": TEXT, "arguments": ANY_OBJECT, "outputs": ANY_OBJECT, "workflow": WORKFLOW,
    "used_inputs": {"type": "array", "items": TEXT}, "replaced_inputs": {"type": "array", "items": TEXT},
    "answer": {}, "detail": TEXT, 'output_mapping': {'type': 'object', 'additionalProperties': TEXT},
    'argument_refs': {'type': 'object', 'additionalProperties': object_schema({'result_id': TEXT,
        'path': {'type': 'array', 'items': {'oneOf': [TEXT, {'type': 'integer','minimum':0}]}}}, ['result_id','path'])},
    'output_refs': {'type': 'object', 'additionalProperties': object_schema({'result_id': TEXT,
        'path': {'type': 'array', 'items': {'oneOf': [TEXT, {'type': 'integer','minimum':0}]}}}, ['result_id','path'])}}, ["action"])
PLAN = {'oneOf': [object_schema({'mode': {'enum': ['select']}, 'workflow_id': TEXT,
    'node_args': {'type': 'object', 'additionalProperties': {'type': 'object', 'additionalProperties': REF}}}, ['mode','workflow_id','node_args']),
    object_schema({'mode': {'enum': ['compose']}, 'workflow': WORKFLOW}, ['mode','workflow'])]}
FINISH = object_schema({'action': {'enum': ['finish']}, 'answer': {}, 'detail': TEXT}, ['action','answer'])

PLANNER_PROMPT = """Choose a short executable workflow for the original task. Use the supplied Skills/Workflows as
experience, not proof. Omit useless exploration and express dependencies only as explicit task/from/literal/unresolved
references. Preserve the user's quantities and identities. Nodes without a program are dynamic. Do not invent asset IDs.
Return one submit_plan ToolCall: mode=select with a retrieved workflow_id and explicit node_args overrides, or
mode=compose with a complete short workflow. Selecting a workflow preserves its node goals and structure.
A from reference must name an earlier node's actual output field."""
LEARNER_PROMPT = """Learn a reusable capability or workflow from this real Train experience, including its failures.
Choose no_change, reuse_existing, propose_skill_and_program_spec, or propose_or_revise_workflow. At most one new
program specification. Prefer a useful local goal including necessary search/preparation, ordinary I/O and concise
guidance; category queries and concrete object IDs are different fields. Examples are hypotheses, not proofs: no
witnesses/owners/occurrence certificates. Workflows may trim exploration, merge preparation, or retain dynamic gaps.
Declare all value dependencies explicitly. Do not claim historical success that is not in the supplied record.
If selected_local_goal is supplied, it defines the requested reusable capability. Preserve that scope in the Skill,
its inputs and its workflow. The full example task is context, not permission to add extra required objects or
conditions from that task. Leave subsequent task-specific work to continuation. Do not broaden the selected goal
while repairing a response's structure. If proposing a new skill, an accompanying workflow may reference it as $new.
execution_intent is guidance_only for single-answer QA; tool capabilities can be program_requested.
guidance_only does not request a Program job: set generate_program=false and omit realization_request.
Declare result_role intermediate/final_answer/final_files in the Skill. A missing Program is pending work, not
completion. reuse_existing with generate_program=true still requests a build. Use realization_request for at most
one related pending job and 0-2 actually applicable completed Train case_ids, fixing each input and reset or real
prefix start. Required fields and needed public tools must exist in that case; do not select unrelated examples
to fill two slots. If none applies, defer. Preserve requested scope and use result references for large resources.
Return one submit_learning ToolCall."""
BUILDER_PROMPT = """Generate a reusable restricted Python program: def run(ctx, inputs) -> dict. Use normal variables,
branches and loops. ctx.observe(), ctx.available_tools(), ctx.call(name, arguments), ctx.remaining_calls(), ctx.read_result() are the only
public task interfaces. No LLM, evaluator, network, host paths, external installation or private state. Do not call
execute_python recursively: use Python directly in the authorized /workspace with its public inputs. Standard library
is available. Calls consume the same task budget. Return status ok/not_found/needs_input/blocked and an outputs dict
matching the supplied schema. ok is not official success. Consult actual public feedback and current tool availability;
search can return not_found. Return exact resource IDs from the current tool arguments for subsequent calls; display
labels are not tool IDs. Do not hardcode example object instances/locations. Generate source plus ordinary
trial_inputs with one submit_program ToolCall. Provide one independent binding for each supplied case_id.
Use each case's public task to construct its own inputs; never copy concrete objects or file paths from another case.
ctx.read_result(result_id, offset=0, limit=None, path=None) reads only results already obtained in this episode.
Support the declared entry_constraints: for mid-episode preparation, first inspect current public state/results and
currently legal arguments instead of always restarting search. Preserve fixed trial bindings exactly.
Prefer start_mode reset and prefix []; intermediate-state skills require the exact real public action prefix
from that case's recorded experience. Implement the supplied Skill's local goal; do not expand it to solve the
complete example task or add unrelated required targets. Only use the tool names supplied to you."""
RUNTIME_PROMPT = """Execute the current node toward the original user's task, keeping exact resource identities and
quantity constraints. Prefer an existing program when inputs are ready; otherwise use a public tool, call a preparation
program, or supply missing values. Only call currently provided tools with their normal arguments. The materials state
the allowed call count; batch only independent read_only/batchable tools. Otherwise return one runtime_step
ToolCall: call_tool(name,arguments), call_program(name=program_id,arguments,output_mapping), complete_node(outputs,output_refs), revise_plan(workflow),
or finish(answer). Node completion and a program's ok do not establish official success. On plan/feedback conflict revise
the unfinished plan once; never change the user's task or overwrite completed results. Avoid repeated unchanged failures.
When consuming supplied node inputs, list their field names in used_inputs. If local recovery redoes a preparation
and replaces its supplied result, list that field in replaced_inputs; do not credit the earlier preparation.
Runtime outputs are ordinary values, never plan literal wrappers. Use output_refs/argument_refs with result_id and
field/index path for large acquired values; do not copy whole documents or lists. Fields cannot appear in both values
and references. output_mapping maps preparation output fields to current node inputs; replacing known inputs must be
declared in replaced_inputs. No references to future results or other calls in the same batch."""
