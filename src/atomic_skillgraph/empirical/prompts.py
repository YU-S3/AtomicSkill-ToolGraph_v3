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
SOURCE_REF = object_schema({'kind':{'enum':['public_json','public_span','operation_json','source_literal']},
    'path':{'type':'array'},'start':{'type':'integer'},'end':{'type':'integer'},'value':{}}, ['kind','path'])
OUTPUT_REF = {'oneOf':[{'type':'array'},object_schema({'kind':{'enum':['json','publication']},
    'path':{'type':'array'},'field':TEXT,'name':TEXT},['kind'])]}
LEARNING = object_schema({"decision": {"enum": ["no_change", "reuse_existing", "propose_skill_and_program_spec",
    "propose_or_revise_workflow"]}, "skill": SKILL, "generate_program": {"type": "boolean"},
    "existing_skill_id": TEXT, "workflow": WORKFLOW, "rationale": TEXT,
    'realization_request': object_schema({'skill_id': TEXT, 'action': {'enum': ['build','repair','trial','defer']},
        'case_bindings': {'type': 'array', 'items': object_schema({'case_id': TEXT,
            'inputs': ANY_OBJECT, 'input_refs':{'type':'object','additionalProperties':SOURCE_REF}, 'local_evidence_ref': TEXT,
            'reference_fields': {'type':'object','additionalProperties':OUTPUT_REF}}, ['case_id','inputs','input_refs','local_evidence_ref','reference_fields'])}},
        ['skill_id','action','case_bindings'])}, ["decision"])
TRIAL_INPUT = object_schema({'case_id': TEXT, 'inputs': ANY_OBJECT, 'local_evidence_ref':TEXT,
    'reference_fields':{'type':'object','additionalProperties':{'type':'array'}},
    'start_mode': {'enum': ['reset', 'prefix_replay']},
    'prefix': {'type': 'array', 'items': object_schema({'name': TEXT, 'arguments': ANY_OBJECT}, ['name', 'arguments'])}},
    ['case_id', 'inputs', 'start_mode', 'prefix'])
BUILD = object_schema({'source': TEXT, 'binding_hash': TEXT}, ['source'])
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
    'guidance_skill': object_schema({'goal': {'type': 'string', 'minLength': 1, 'maxLength': 256},
        'guidance': {'type': 'string', 'minLength': 1, 'maxLength': 1600}}, ['goal', 'guidance']),
    'rationale': {'type': 'string', 'minLength': 1, 'maxLength': 512}}, ['decision'])
GUIDANCE_LEARNER_PROMPT = """Learn concise reusable guidance from this completed public Train experience.
Use no_change if no useful general guidance was learned; reuse_existing only with a real supplied guidance Skill ID.
For new or revised guidance, submit upsert_guidance with guidance_skill containing nonempty goal and guidance;
an optional existing_skill_id identifies the real parent to revise. Do not invent IDs, workflows, Programs, jobs,
trial cases, private gold answers or per-question answer lookup tables. Preserve the task's original scope.
guidance_skill contains only goal (<=256 characters) and guidance (<=1600 characters); existing_skill_id stays
outside it and must be a real supplied ID. rationale is <=512 characters. State applicability, an executable
approach and limits; a successful label is evidence for that submission, not proof of every mathematical claim.
Do not infer a replacement answer or theorem from scalar failure feedback. Necessary evidence and any material
tables are included in this request; you cannot call read_result here.
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
Stable tool_definitions and current_tools describe future environment execution; you cannot call them now.
Your only current response tool is submit_plan. Legal different operations may achieve
the same result goal. New combinations of usable Programs execute without whole-workflow trials or node confirmation.
Return one submit_plan ToolCall: mode=select with a retrieved workflow_id and explicit node_args overrides, or
mode=compose with a complete short workflow. Selecting a workflow preserves its node goals and structure.
A from reference must name an earlier node's actual output field. output_aliases maps actual fields to handoff aliases
while retaining originals. Selecting an old ambiguous Skill-only Workflow also supplies node_modes dynamic/skill in
this same response; selecting a native v2 Workflow needs no node_modes.
REF examples: {"task":"input_field"}, {"from":"earlier_node","field":"output_field"},
{"literal":3}, {"unresolved":"location not yet known"}. Bare strings are invalid REF values."""
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
execution_intent depends on real local evidence, not the final answer interface. A single-answer operation can be program_requested.
guidance_only does not request a Program job: set generate_program=false and omit realization_request.
Declare result_role intermediate/final_answer/final_files in the Skill. A missing Program is pending work, not
completion. reuse_existing with generate_program=true still requests a build. Use realization_request for at most
one related pending job and 0-2 actually applicable completed Train case_ids, selecting existing local_evidence_ref.
Host freezes prefix, start_mode, environment and trace identity; never submit those fields. Required inputs and tools must exist; do not select unrelated examples
to fill slots. One real local binding is sufficient. input_refs maps each input to a catalog public_json/operation_json/source_literal path,
or a public_span path with exact start/end. A span proves provenance, not parameter semantics. Optional branch defaults are not validated facts.
reference_fields maps outputs to typed json path or publication files/file selectors from the directory.
final_files requires files (array) and deleted_files (array); every other required output also needs a real reference.
An inspection has no published files; a declared filename is not publication evidence. If none applies, defer.
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
labels are not tool IDs. Do not hardcode example object instances/locations. Generate source with one submit_program
ToolCall; an optional binding_hash must equal build_request.binding_hash. Host alone supplies fixed trial inputs and replay start.
Use each case's public task to construct its own inputs; never copy concrete objects or file paths from another case.
ctx.read_result(result_id, offset=0, limit=None, path=None) reads only results already obtained in this episode.
Support the declared entry_constraints: for mid-episode preparation, first inspect current public state/results and
currently legal arguments instead of always restarting search. Do not emit trial_inputs or replace a selected source.
Implement the supplied Skill's local goal; do not expand it to solve the
complete example task or add unrelated required targets. Future RPC calls must use future_program_api.allowed_names.
For a durable run(ctx, inputs), INPUT_PATH and OUTPUT_PATH globals are not supplied; use inputs or the authorized
workspace. Those globals belong only to an independent solution.py replay contract, when explicitly requested.
For file workspaces, use /workspace/... paths inside Python as needed, but return outputs.files/deleted_files as
relative publication names, e.g. write /workspace/result.xlsx and declare files=["result.xlsx"]. An output_path
field can retain the program path; do not copy that absolute value into files. Absolute publication paths,
inputs/... and '..' paths are rejected; do not bypass those checks."""
CHOICE_PROPOSAL = object_schema({
    'decision': {'type': 'string', 'enum': ['no_change', 'reuse_existing', 'upsert_guidance']},
    'existing_skill_id': {'type': 'string'},
    'goal': {'type': 'string', 'maxLength': 160},
    'guidance': {'type': 'string', 'maxLength': 1000},
    'scope_terms': {'type': 'array', 'minItems': 2, 'maxItems': 4, 'uniqueItems': True,
                    'items': {'type': 'string', 'minLength': 2}},
    'applicability': {'type': 'string', 'maxLength': 240},
    'rationale': {'type': 'string', 'maxLength': 240}}, ['decision'])
GUIDANCE_CHECK = object_schema({
    'status': {'type': 'string', 'enum': ['supported', 'contradicted', 'insufficient_evidence']},
    'policy_version': {'type': 'string'}, 'proposal_hash': {'type': 'string'},
    'proposal_quote': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
    'source_quote': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
    'reason': {'type': 'string', 'minLength': 1, 'maxLength': 400}},
    ['status', 'policy_version', 'proposal_hash', 'proposal_quote', 'source_quote', 'reason'])
CHOICE_LEARNER_PROMPT = """Submit one flat submit_learning ToolCall using the supplied schema.
The host source is a verified submission under the source question's premises, not a general mathematical proof.
Extract a short reusable conditional observation or operation compatible with that source; do not reverse its
conclusion, broaden premises, infer failed examples' answers, or preserve option letters as lookup instructions.
scope_terms must be 2-4 distinct topic words actually occurring in source_goal after the supplied normalization.
Use no_change if there is no reusable operation beyond the correct answer, or reuse_existing only for an offered
checked related ID. For upsert_guidance supply goal, guidance, scope_terms, applicability at the top level.
Respect all lengths without truncating conditions or formulas. Related cards are advisory, not proofs."""
GUIDANCE_CHECK_PROMPT = """Return exactly one submit_guidance_check ToolCall. Independently check the proposal
against the host-bound source_goal and verified selected_choice_text under their explicit premises. Do not solve
the mathematical question anew or use other options as verified facts. Reject reversal or unsupported extension
of the source. Choose supported, contradicted, or insufficient_evidence. Quote exact nonempty substrings from
the proposal and the source; return the supplied policy_version and proposal_hash unchanged. This is a bounded
source compatibility check, not a theorem proof. Do not rewrite or repair the proposal."""


def single_answer_prompt(contract, *, semantic=False):
    if not semantic:
        return 'Answer the question once using only the public input and any supplied guidance. ' + contract
    return ('Answer once from the public question and the actual meaning of every option. Check quantifiers and '
            'premises; distinguish a true statement from the strongest conclusion that can be proved. Examine '
            'options that assess other options\' sufficiency or strength on the same terms as other candidates. '
            'Consult guidance only when its applicability matches; it cannot override the question or establish '
            'facts by authority. Each guidance item is a limited source-checked observation, not a general proof. ' + contract)


RUNTIME_PROMPT = """Materials are grouped as task, node, bindings, handoff, state, calls, memory, recovery.
node.interface is the real I/O contract. A task_field points to the single displayed task field. Result references
contain result_id/path: use argument_refs/output_refs, or bounded read_result when the preview is insufficient;
do not pass a reference object as an ordinary business value. Follow task.answer_contract when supplied.
Execute the current node toward the original user's task, keeping exact resource identities and
quantity constraints. Prefer an existing program when inputs are ready; otherwise use a public tool, call a preparation
program, or supply missing values. Your callable response tool is runtime_step; calls.tools describes environment
operations to put inside runtime_step, not direct response ToolCalls. The materials state
the allowed call count; batch independent read_only/batchable operations as multiple runtime_step ToolCalls.
Otherwise return one runtime_step
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
declared in replaced_inputs. No references to future results or other calls in the same batch.
read_result is wrapped as a runtime_step with
{"action":"call_tool","name":"read_result","arguments":{"result_id":"EXISTING_RESULT_ID","path":[]}}.
EXISTING_RESULT_ID is a placeholder: replace it with a real supplied result ID, never invent one.
Plan REF values are objects, e.g. {"task":"field"}, {"from":"node","field":"output"},
{"literal":3}, {"unresolved":"reason"}; never bare strings."""
