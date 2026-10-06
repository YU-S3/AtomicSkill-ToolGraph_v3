"""One readable protocol per role, with no historical witness language."""
from .contracts import object_schema

ANY_OBJECT = {"type": "object"}
TEXT = {"type": "string"}
REF = {"oneOf": [object_schema({"task": TEXT}, ["task"]),
    object_schema({"from": TEXT, "field": TEXT}, ["from", "field"]),
    object_schema({"literal": {}}, ["literal"]), object_schema({"unresolved": TEXT}, ["unresolved"])]}
ALIASES = {'type': 'object', 'additionalProperties': TEXT}
NODE = object_schema({"id": TEXT, 'execution_mode': {'enum': ['dynamic', 'skill', 'program']},
    "goal": TEXT, "skill_id": TEXT, "program_id": TEXT, 'purpose': TEXT,
    'reference_skill_ids': {'type': 'array', 'items': TEXT, 'maxItems': 3}, 'output_aliases': ALIASES,
    "args": {"type": "object", "additionalProperties": REF},
    "after": {"type": "array", "items": TEXT}}, ["id", 'execution_mode', "args"])
WORKFLOW = object_schema({"goal": TEXT, 'interface_version': {'enum': ['empirical.workflow.v2']},
    "nodes": {"type": "array", "items": NODE, "minItems": 1, "maxItems": 16},
    "outputs": {"type": "object", "additionalProperties": REF}}, ["goal", "nodes"])
SKILL = object_schema({"goal": TEXT, "guidance": TEXT, "input_schema": ANY_OBJECT,
    "output_schema": ANY_OBJECT, 'execution_intent': {'enum': ['guidance_only', 'program_requested']},
    'result_role': {'enum': ['intermediate', 'final_answer', 'final_files']}, 'entry_constraints': TEXT},
    ["goal", "input_schema", "output_schema"])
LEARNING = object_schema({"decision": {"enum": ["no_change", "reuse_existing", "propose_skill_and_program_spec",
    "propose_or_revise_workflow"]}, "skill": SKILL, "generate_program": {"type": "boolean"},
    "existing_skill_id": TEXT, "workflow": WORKFLOW, "rationale": TEXT,
    'realization_request': object_schema({'skill_id': TEXT, 'action': {'enum': ['build','repair','trial','defer']},
        'case_bindings': {'type': 'array', 'items': object_schema({'case_id': TEXT,
            'inputs': ANY_OBJECT, 'start_mode': {'enum': ['reset','prefix_replay']},
            'prefix': {'type': 'array', 'items': object_schema({'name': TEXT, 'arguments': ANY_OBJECT}, ['name','arguments'])}},
            ['case_id','inputs','start_mode','prefix'])}}, ['skill_id','action','case_bindings'])}, ["decision"])
TRIAL_INPUT = object_schema({'case_id': TEXT, 'inputs': ANY_OBJECT,
    'start_mode': {'enum': ['reset', 'prefix_replay']},
    'prefix': {'type': 'array', 'items': object_schema({'name': TEXT, 'arguments': ANY_OBJECT}, ['name', 'arguments'])}},
    ['case_id', 'inputs', 'start_mode', 'prefix'])
BUILD = object_schema({"source": TEXT, "example_inputs": ANY_OBJECT,
    'trial_inputs': {'type': 'array', 'items': TRIAL_INPUT, 'maxItems': 2}}, ['source', 'trial_inputs'])
PATCH = object_schema({'target_node': TEXT, 'kind': {'enum': ['args', 'handoff', 'detach']}, 'reason': TEXT,
    'args': {'type': 'object', 'additionalProperties': REF}, 'output_aliases': ALIASES,
    'consumer_refs': {'type': 'object', 'additionalProperties': {'type': 'object', 'additionalProperties': REF}},
    'dynamic_goal': TEXT}, ['target_node', 'kind', 'reason'])
STEP = object_schema({"action": {"enum": ["call_tool", "call_program", "complete_node", "revise_plan", "patch_node", "finish"]},
    "name": TEXT, "arguments": ANY_OBJECT, "outputs": ANY_OBJECT, "workflow": WORKFLOW,
    "used_inputs": {"type": "array", "items": TEXT}, "replaced_inputs": {"type": "array", "items": TEXT},
    "answer": {}, "detail": TEXT, 'patch': PATCH, 'output_mapping': ALIASES,
    'argument_refs': {'type': 'object', 'additionalProperties': object_schema({'result_id': TEXT,
        'path': {'type': 'array', 'items': {'oneOf': [TEXT, {'type': 'integer','minimum':0}]}}}, ['result_id','path'])},
    'output_refs': {'type': 'object', 'additionalProperties': object_schema({'result_id': TEXT,
        'path': {'type': 'array', 'items': {'oneOf': [TEXT, {'type': 'integer','minimum':0}]}}}, ['result_id','path'])}}, ["action"])
PLAN = object_schema({'mode': {'enum': ['select','compose']}, 'workflow_id': TEXT,
    'node_args': {'type': 'object', 'additionalProperties': {'type': 'object', 'additionalProperties': REF}},
    'node_modes': {'type': 'object', 'additionalProperties': {'enum': ['dynamic', 'skill']}},
    'workflow': WORKFLOW}, ['mode'])
def finish_text_prompt(contract):
    return ('Return only the final answer as nonempty plain text, using information already acquired. '
            'Do not call tools, execute programs, read more results or replan. ' + contract)


GUIDANCE_LEARNING = object_schema({
    'decision': {'enum': ['no_change', 'reuse_existing', 'upsert_guidance']},
    'existing_skill_id': TEXT,
    'guidance_skill': object_schema({'goal': TEXT, 'guidance': TEXT}, ['goal', 'guidance']),
    'rationale': TEXT}, ['decision'])
GUIDANCE_LEARNER_PROMPT = """Learn concise reusable guidance from this completed public Train experience.
Use no_change if no useful general guidance was learned; reuse_existing only with a real supplied guidance Skill ID.
For new or revised guidance, submit upsert_guidance with guidance_skill containing nonempty goal and guidance;
an optional existing_skill_id identifies the real parent to revise. Do not invent IDs, workflows, Programs, jobs,
trial cases, private gold answers or per-question answer lookup tables. Preserve the task's original scope.
Only one submit_learning ToolCall is permitted; this request does not solve the task again."""

PLANNER_PROMPT = """Choose a short executable workflow for the original task. Prefer compatible usable Programs or
short compositions of them. execution_summary reports actual bindings; many dynamic nodes do not imply a mature
executable workflow. Do not replace a dynamic reference with an execution binding without explicitly selecting it.
Use the supplied Skills/Workflows as
experience, not proof. Omit useless exploration and express dependencies only as explicit task/from/literal/unresolved
references. Preserve the user's quantities and identities. Do not invent asset IDs. execution_mode=dynamic has a
result goal and optional reference_skill_ids, but no skill_id/program_id execution binding. Reference guidance never
grants an automatic route or completion schema. execution_mode=skill selects skill_id; execution_mode=program
selects program_id. Bound nodes have no goal: their full goal and I/O come from the actual asset. Put parent intent
in optional purpose. Unknown locations/values are unresolved or producer references, not discovered facts.
Stable tool_definitions explain semantics; only current_tools are callable now. Legal different operations may achieve
the same result goal. New combinations of usable Programs execute without whole-workflow trials or node confirmation.
Return one submit_plan ToolCall: mode=select with a retrieved workflow_id and explicit node_args overrides, or
mode=compose with a complete short workflow. Selecting a workflow preserves its node goals and structure.
A from reference must name an earlier node's actual output field. output_aliases maps actual fields to handoff aliases
while retaining originals. Selecting an old ambiguous Skill-only Workflow also supplies node_modes dynamic/skill in
this same response; selecting a native v2 Workflow needs no node_modes."""
LEARNER_PROMPT = """Learn a reusable capability or workflow from this real Train experience, including its failures.
Choose no_change, reuse_existing, propose_skill_and_program_spec, or propose_or_revise_workflow. At most one new
program specification. Prefer a useful local goal including necessary search/preparation, ordinary I/O and concise
guidance; category queries and concrete object IDs are different fields. Examples are hypotheses, not proofs: no
witnesses/owners/occurrence certificates. Workflows may trim exploration, merge preparation, or retain dynamic gaps.
Declare all value dependencies explicitly. Do not claim historical success that is not in the supplied record.
New workflows use execution_mode dynamic/skill/program. Reference Skills are reference_skill_ids, never execution
bindings. Dynamic nodes have a result goal; bound nodes omit goal and use the asset's full I/O, with optional purpose.
Reuse real program calls and actual handoffs; a new composition or task-local patch is not a validated asset.
If selected_local_goal is supplied, it defines the requested reusable capability. Preserve that scope in the Skill,
its inputs and its workflow. The full example task is context, not permission to add extra required objects or
conditions from that task. Leave subsequent task-specific work to continuation. Do not broaden the selected goal
while repairing a response's structure. A new Skill has no assigned ID yet: use $new in realization_request.skill_id
and workflow node skill_id. Existing references must be real supplied Skill IDs, never an invented name or alias.
For file workspaces, program paths may be absolute inside /workspace; files/deleted_files are publication names
relative to /workspace, without a leading slash, inputs prefix or '..'. Keep output_path separate from files.
execution_intent is guidance_only for single-answer QA; tool capabilities can be program_requested.
guidance_only does not request a Program job: set generate_program=false and omit realization_request.
Declare result_role intermediate/final_answer/final_files in the Skill. A missing Program is pending work, not
completion. reuse_existing with generate_program=true still requests a build. Use realization_request for at most
one related pending job and 0-2 actually applicable completed Train case_ids, fixing each input and reset or real
prefix start. Required fields and needed public tools must exist in that case; do not select unrelated examples
to fill two slots. If none applies, defer. Preserve requested scope and use result references for large resources.
Return one submit_learning ToolCall."""
BUILDER_PROMPT = """Your current response must be exactly one submit_program ToolCall with the supplied submission_contract.
Do not directly call grep, read, glob or execute_python during generation, including recovery.
build_request declares the Skill and fixed bindings; future_program_api describes the generated Program's future RPC
permissions, not tools for your current response. workspace_capabilities describes authorized local Python execution.
Historical example calls describe Runtime Agent actions, not additional Program permissions.
Generate a reusable restricted Python program: def run(ctx, inputs) -> dict. Use normal variables,
branches and loops. Follow future_program_api.public_program_abi: exact_catalog ToolView supplies a list of complete current_arguments
dictionaries; named_tools supplies input_schema without exhaustive arguments. Do not discard tools because top-level
arguments/current_arguments is absent. Return the explicit status/outputs envelope, not a bare task output object.
ctx.observe(), ctx.available_tools(), ctx.call(name, arguments), ctx.remaining_calls(), ctx.read_result() are the only
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
complete example task or add unrelated required targets. Future RPC calls must use future_program_api.allowed_names.
For a durable run(ctx, inputs), INPUT_PATH and OUTPUT_PATH globals are not supplied; use inputs or the authorized
workspace. Those globals belong only to an independent solution.py replay contract, when explicitly requested.
For file workspaces, use /workspace/... paths inside Python as needed, but return outputs.files/deleted_files as
relative publication names, e.g. write /workspace/result.xlsx and declare files=["result.xlsx"]. An output_path
field can retain the program path; do not copy that absolute value into files. Absolute publication paths,
inputs/... and '..' paths are rejected; do not bypass those checks."""
RUNTIME_PROMPT = """Materials are grouped as task, node, bindings, handoff, state, calls, memory, recovery.
node.interface is the real I/O contract. A task_field points to the single displayed task field. Result references
contain result_id/path: use argument_refs/output_refs, or bounded read_result when the preview is insufficient;
do not pass a reference object as an ordinary business value. Follow task.answer_contract when supplied.
Execute the current node toward the original user's task, keeping exact resource identities and
quantity constraints. Prefer an existing program when inputs are ready; otherwise use a public tool, call a preparation
program, or supply missing values. Only call currently provided tools with their normal arguments. The materials state
the allowed call count; batch only independent read_only/batchable tools. Otherwise return one runtime_step
ToolCall: call_tool(name,arguments), call_program(name=program_id,arguments,output_mapping), complete_node(outputs,output_refs), revise_plan(workflow),
patch_node(patch), or finish(answer). Node completion and a program's ok do not establish official success. Use resolved
node_interface: dynamic requires only required_handoff_fields; bound nodes retain the asset's full I/O. Reference
guidance and purpose cannot override actual feedback or impose an execution schema. complete_node outputs are ordinary
values, e.g. {"resource_id":"resource_1"}, not plan references. Keep pending_outputs after handoff errors: supplement
them, explicitly alias actual fields, or patch unfinished consumer refs instead of searching again. patch_node kind=args
updates future arguments; handoff maps actual outputs/references; detach releases an inapplicable task-local binding with
dynamic_goal. No node additions/deletions/reordering or Bank changes. Local patches spend normal decisions but no full
replan. Explicit revise_plan and system recovery share one full replan, then at most one remaining-task Dynamic escape,
retaining state/results/budgets and usable Program options. Never change the user's task or overwrite completed results.
Avoid repeated unchanged failures and public no-progress loops.
When consuming supplied node inputs, list their field names in used_inputs. If local recovery redoes a preparation
and replaces its supplied result, list that field in replaced_inputs; do not credit the earlier preparation.
Runtime outputs are ordinary values, never plan literal wrappers. Use output_refs/argument_refs with result_id and
field/index path for large acquired values; do not copy whole documents or lists. Fields cannot appear in both values
and references. output_mapping maps preparation output fields to current node inputs; replacing known inputs must be
declared in replaced_inputs. No references to future results or other calls in the same batch."""
