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
            'Prefer acquired calculation outputs; check the requested scope, units and precision against the evidence. '
            'Public notes are model-reported working notes, not independent proof. '
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
A from reference must name an earlier node's actual output field. Prefer one dynamic node when no executable
binding is available; split when distinct reusable operations need an explicit handoff. For new dynamic nodes,
use the same field names in outputs and downstream references instead of inventing aliases.
output_aliases maps actual fields to handoff aliases
while retaining originals. Selecting an old ambiguous Skill-only Workflow also supplies node_modes dynamic/skill in
this same response; selecting a native v2 Workflow needs no node_modes.
REF examples: {"task":"input_field"}, {"from":"earlier_node","field":"output_field"},
{"literal":3}, {"unresolved":"location not yet known"}. Bare strings are invalid REF values."""
LEARNER_PROMPT = """Learn real Train, including failures. One flat submit_learning ToolCall.
Judge local work, not task success. Compile one useful search/read/compute procedure;
batch search with locators/counts can qualify as intermediate without final figures.
Parameterize recorded arrays/scalars. related=Bank assets; local_evidence=temporary/native work,
not Bank coverage. no_change for unsupported/nonreusable/covered work.
Never invent evidence, owners, success, outputs or retrieval.
Use source_excerpt/input_sources/output_references; Builder gets full source.
Arrays use aggregate AST paths, not observation/elements. Only executed
outputs/parameters. Computation is intermediate; final_answer requires answer:string;
final_files requires files:array/deleted_files:array. observation is text; data.computed is JSON.
Typed json/publication files/file refs for all required outputs; inspection/filenames are not publication.
Python allows /workspace absolutes; publication names exclude leading slash, inputs/ and '..'; separate output_path.
Skill refs: $new or supplied real IDs, never aliases. Workflow: dependencies; dynamic goal/reference_skill_ids;
bound asset I/O/no goal/optional purpose; compositions/patches are not proof. Repair keeps selected_local_goal/scope.
guidance_only: generate_program=false/no realization_request. generate_program=true requests pending builds,
even reuse_existing. Bind current experience.local_evidence or completed_train_cases;
no prior job/second case needed. Use real case_id/local_evidence_ref and inputs/tools. input_refs: exact catalog
public_json/operation_json/source_literal paths or public_span start/end. Spans prove provenance, not semantics.
Host owns prefix/start_mode/environment/source trace; omit them."""
BUILDER_PROMPT = """Return exactly one submit_program ToolCall using submission_contract. During generation/recovery,
do not call grep/read/glob/execute_python. build_request fixes the Skill/bindings; future_program_api authorizes
future RPC, workspace_capabilities authorizes local Python. Historical Runtime calls grant no permissions.
Generate reusable def run(ctx, inputs) -> dict with ordinary variables/branches/loops.
Follow future_program_api.public_program_abi: exact_catalog ToolView lists complete current_arguments;
named_tools supplies input_schema. Missing top-level arguments/current_arguments does not remove a tool.
Return status/outputs, not a bare task output.
ctx.observe(), ctx.available_tools(), ctx.call(name, arguments), ctx.remaining_calls(), ctx.read_result() are the only
public task interfaces. No LLM, evaluator, network, host paths, external installation or private state. Do not call
execute_python recursively: use Python directly in the authorized /workspace with its public inputs. Standard library
is available. Calls share the task budget. Return ok/not_found/needs_input/blocked with schema-matching outputs;
ok is not official success. Use real public feedback/current tools; search can return not_found.
Use exact current resource IDs, not display labels. Never hardcode example objects/locations or copy them
across cases. Construct inputs from each public case. Optional binding_hash equals build_request.binding_hash;
Host fixes trial inputs/replay start.
ctx.read_result(result_id, offset=0, limit=None, path=None) reads only results already obtained in this episode.
Follow entry_constraints; mid-episode preparation first inspects current public state/results/legal arguments.
Do not emit trial_inputs or replace selected source. Implement only the Skill's local goal, without adding
whole-task goals/unrelated targets. RPC uses future_program_api.allowed_names.
For a durable run(ctx, inputs), INPUT_PATH and OUTPUT_PATH globals are not supplied; use inputs or the authorized
workspace. Those globals belong only to an independent solution.py replay contract, when explicitly requested.
Python may use /workspace/...; outputs.files/deleted_files require relative publication names:
write /workspace/result.xlsx, declare files=["result.xlsx"]. output_path may keep the program path;
files cannot. Absolute publication paths, inputs/... and '..' are rejected; do not bypass checks."""
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
PUBLIC_CHOICE_LEARNER_PROMPT=CHOICE_LEARNER_PROMPT+"""
Every upsert_guidance must supply knowledge_type. Choose procedural_advisory only from actual
source.public_process_evidence: give goal, guidance, applicability, operation_intent, prerequisites and an exact
source_process_quote at the top level; omit scope_terms for this type. Keep the entire rendered advisory under
512 characters: prefer guidance under 180 and applicability under 80 characters. It is a conditional operation,
not a theorem or answer guarantee. No process evidence means no_change. domain_fact retains the narrow
2-4 source topic words and source premises; do not turn a correct label into a reusable proof."""
GUIDANCE_CHECK_PROMPT = """Return exactly one submit_guidance_check ToolCall. Independently check the proposal
against the host-bound source_goal and verified selected_choice_text under their explicit premises. Do not solve
the mathematical question anew or use other options as verified facts. Reject reversal or unsupported extension
of the source. Choose supported, contradicted, or insufficient_evidence. Quote exact nonempty substrings from
the proposal and the source; return the supplied policy_version and proposal_hash unchanged. This is a bounded
source compatibility check, not a theorem proof. Do not rewrite or repair the proposal."""
PUBLIC_GUIDANCE_CHECK_PROMPT=GUIDANCE_CHECK_PROMPT+"""
For procedural_advisory, quote the actual public_process, and check the operation/prerequisite boundary.
A self_reported_process is not independent execution or a mathematical proof: reject theorem claims or
invented operations. Your result means only quote_matched/source_compatible, never mathematically_proved."""


def single_answer_prompt(contract, *, semantic=False):
    if not semantic:
        return 'Answer the question once using only the public input and any supplied guidance. ' + contract
    return ('Answer once from the public question and the actual meaning of every option. Check quantifiers and '
            'premises; distinguish a true statement from the strongest conclusion that can be proved. Examine '
            'options that assess other options\' sufficiency or strength on the same terms as other candidates. '
            'Consult guidance only when its applicability matches; it cannot override the question or establish '
            'facts by authority. Each guidance item is a limited source-checked observation, not a general proof. ' + contract)


RUNTIME_PROMPT = """Materials are grouped as task, node, bindings, handoff, state, calls, memory, recovery.
Execution priority: reuse a compatible usable Program when its inputs are ready. Otherwise, when
calls.task_local_program is supplied and the next work repeats search, filtering or reading, write and run
one bounded temporary script through its tool instead of taking a separate model turn for each operation.
Choose task-specific queries and stopping conditions yourself from public evidence. A temporary script works
with an empty Bank and does not require a saved Skill, a Builder request or whole-task success. It creates no
persistent asset; later Train learning still requires actual source/output evidence and independent validation.
Use a single native call for an isolated inspection, unknown input or feedback that cannot yet be handled in
code. Stop searching once the required evidence is acquired, then compute from it and submit the answer.
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
Use a narrow query from the task's explicit constraints before listing the whole corpus. Read at actual acquired
offsets; a search hit is a locator, not the surrounding table. Continue a truncated result at its next_offset and
business path, e.g. ["data"] for a list or ["data","text"] for acquired text; do not read a dictionary root.
For numerical tasks, obtain the relevant rows and use an available compute tool to calculate from those public
values, preserving their units and provenance. Submit the answer once ready instead of continuing exploration.
When using execute_python for a computation, print one JSON object containing the acquired input values and
computed results with clear field names, using json.dumps. This preserves typed actual outputs for later learning.
calls.task_local_program gives the actual temporary context API and invocation limits. Submit top-level
source in runtime_step(action=call_tool,name=its tool,arguments={"source":"..."}); do not merely define run.
Keep dependent calls and loops inside that script. Check ctx.remaining_calls() and each result's accepted/error
before consuming data; cap iterations and output size. The launch and internal calls spend the same task budget.
Reuse acquired paths, offsets and result IDs; print compact JSON with real locators, needed rows, inputs and
computed results. Do not dump whole documents, assume a corpus mount/network, or invent missing values.
Keep the entire printed JSON within calls.task_local_program.max_stdout_bytes, measured after UTF-8 encoding;
larger stdout is truncated and cannot provide typed JSON results. Select needed rows or smaller text windows.
ctx.call returns public tool data with its actual business types. Runtime memory may display source_ref/content
preview windows; do not copy those preview wrappers into RPC field access. Inspect returned value types first.
Check existing memory results before repeating a query. Acquired text pages and calculations remain visible
after later searches; a repeated query without a changed scope adds no evidence.
Keep a concise working note in detail with acquired locators, values or completed calculations needed next.
memory.public_notes contains actual public Runtime text, labelled model_reported; verify it against source results.
return_example names the actual fields to emit before output_aliases are applied. A handoff patch targets the
producer holding pending_outputs, and consumer_refs names that producer's unfinished consumers.
read_result is wrapped as a runtime_step with
{"action":"call_tool","name":"read_result","arguments":{"result_id":"EXISTING_RESULT_ID","path":["data","text"]}}.
EXISTING_RESULT_ID is a placeholder: replace it with a real supplied result ID, never invent one.
Each ToolCall arguments is one flat JSON object, never concatenated JSON or a direct environment-tool ToolCall.
Plan REF values are objects, e.g. {"task":"field"}, {"from":"node","field":"output"},
{"literal":3}, {"unresolved":"reason"}; never bare strings."""
