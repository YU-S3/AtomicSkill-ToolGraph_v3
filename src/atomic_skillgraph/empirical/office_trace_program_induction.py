"""V4-only candidate induction and durable physical request receipts."""
import ast
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json

from ..agents.protocol import parse_json_strict, validate_schema_instance
from ..agents.provider import ProviderAgentProtocolError
from .checkpoint import save_json
from .contracts import digest, validate_program
from .office_operation_mining import QUALIFICATION, bind_inputs, read
from .program_worker import validate_context_source

EXTRACTOR = '''The task's final answer may be incorrect while an intermediate operation genuinely occurred.
Identify one small parameterized operation from the public Train events, including failures. Observed outputs
are not verified values. Prefer scoped search/acquire, table-row or series extraction and evidence checking.
Reuse an equivalent existing program instead of building another calculator. Never invent a past event,
source ID, corpus path, number or final answer. No complete handoff is required to propose a candidate.
Return ONE JSON object, no markdown: decision=propose_program|reuse_existing|no_change,
source_opportunity_ids (only supplied IDs), evidence_limitations and reason.
For reuse_existing include existing_program_id from the supplied list.
For propose_program include atomic={goal,applicability,preconditions:[...],result_role:"intermediate",
stop_condition} and program_spec={input_schema,output_schema,allowed_tools,result_fields,input_bindings,
probe_contract}. Schemas are closed object schemas. Each input binding is
{span_id,section:"arguments"|"result"|"task",path:[...]}; task refers to source_task.
To parameterize a row/column/unit/period actually present in supplied public text, optionally add
slice:{start:<zero-based character index>,end:<exclusive index>}. Host copies exactly this observed substring;
no invented replacement value or inferred number is permitted. A uniquely observed
argument with the same name can be defaulted by Host. A missing/hypothesized value cannot be bound.
probe_contract declares an independently checkable local operation:
  retrieval: {kind:"retrieval"}; outputs must contain text,source_path,source_offset and complete;
  numeric: {kind:"numeric",operation:"sum"|"mean"|"difference"|"ratio"|"log_ratio"|"sample_std",
            values_input:<bound numeric array field>,output_field:<number field>};
  table_cell: {kind:"table_cell",text_input:<text field>,row_input:<exact row label field>,
              column_input:<exact header field>,unit_input:<unit field>,output_field:<number field>}.
Other operations may be proposed with kind=unverified, but will remain candidate_only/coverage_insufficient.
Do not claim verified headers, units or completeness absent from evidence. Stop on missing/ambiguous data.'''

BUILDER = '''Implement exactly this Atomic as def run(ctx, inputs), returning {"status":...,"outputs":{...}}.
Return ONE JSON object {"source":<Python source>} with no markdown. Input/output schemas are fixed by the
candidate specification. No task IDs, dates, answers, corpus paths or data constants from the example.
Use only actual Context ABI and tool schemas: ctx.call("read_context", args), NEVER ctx.read_context(args).
No host file access, network, evaluator or recursive execute_python. Corpus text comes only from ctx.call
or already bound public text. Preserve source path/offset/units. Check accepted, handle missing/ambiguous
rows and conflicting units. Unfinished pagination/40 legacy hits is not proof of absence or completeness.
Return needs_input/not_found/blocked instead of default numbers; include a top-level detail explaining
the missing prerequisite. Retrieval outputs complete=true only for a complete search and acquired text.
For table_cell the independent oracle requires one exact header and row, a unique finite numeric cell and
the actual requested unit in the text. The supplied probe contract is a limited test, not a correctness claim.'''


class ModelOutputError(ValueError):
    pass


class RequestStore:
    """Commit paid response before decoding; ambiguous interruptions never repeat HTTP."""
    def __init__(self, root, provider, *, phase, wire_max_bytes=65536, attribution=None):
        self.root, self.provider, self.phase = Path(root), provider, phase
        self.wire_max_bytes, self.attribution = wire_max_bytes, attribution or {}

    def ask(self, key, prompt, material, *, repair=False):
        messages = [{'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': json.dumps(material, ensure_ascii=False, allow_nan=False)}]
        return self.turn(key, messages, repair=repair)

    def turn(self, key, messages, *, repair=False):
        path = self.root / (key + '.json')
        payload = self.provider._build_payload(messages, [])
        identity = digest(payload)
        if path.exists():
            record = read(path)
            if record['payload_sha256'] != identity: raise RuntimeError('request_identity_changed: refuse altered replay')
            if record['status'] == 'response_received': return self.decode(record), record
            if record['status'] in {'model_error', 'material_refusal'}: raise ModelOutputError(record['error'])
            raise RuntimeError('unresolved_http: inspect durable ledger/response; no duplicate paid request')
        record = {'payload': payload, 'payload_sha256': identity, 'phase': self.phase,
                  'provider': self.provider.snapshot(), 'repair': repair, 'status': 'prepared', 'requests': []}
        from ..experiments.formal_log import utc
        record['start_time'] = utc()
        wire = len(json.dumps(payload, allow_nan=False).encode())
        record['wire_bytes'] = wire
        if wire > self.wire_max_bytes:
            record.update(status='material_refusal', error='material_too_large: complete request exceeds frozen wire cap')
            save_json(path, record)
            raise ModelOutputError(record['error'])
        save_json(path, record)
        before = self.provider.request_record_count
        self.provider.set_request_context(session_id=str(path), stage=self.phase, repair=repair,
            decision_purpose='repair' if repair else self.phase, payload_max_bytes=self.wire_max_bytes,
            **self.attribution)
        try:
            turn = self.provider.complete(messages, tools=[])
        except Exception as exc:
            usage_turn = getattr(exc, 'usage_turn', None)
            model_error = isinstance(exc, ProviderAgentProtocolError)
            record.update(status='model_error' if model_error else 'infrastructure_error',
                          error=str(exc), error_code=getattr(exc, 'code', type(exc).__name__))
            if usage_turn: record['turn'] = asdict(usage_turn)
            record['requests'] = list(self.provider.request_records_since(before)); record['end_time'] = utc()
            save_json(path, record)
            if model_error: raise ModelOutputError(str(exc)) from exc
            raise
        record.update(status='response_received', turn=asdict(turn),
                      requests=list(self.provider.request_records_since(before)), end_time=utc())
        save_json(path, record)
        return self.decode(record), record

    @staticmethod
    def decode(record):
        turn = record['turn']
        if turn.get('finish_reason') == 'length': raise ModelOutputError('truncated_response: no silent partial JSON')
        if turn.get('tool_calls'): raise ModelOutputError('unexpected_tool_call: return the specified JSON content')
        try:
            result = parse_json_strict(turn.get('content') or '')
            if not isinstance(result, dict): raise ValueError('Expected one JSON object')
            return result
        except ValueError as exc: raise ModelOutputError('invalid_json: ' + str(exc)) from exc


def validate_proposal(proposal, opportunity, spans, tools, existing):
    allowed = {'decision', 'source_opportunity_ids', 'atomic', 'program_spec', 'evidence_limitations', 'reason', 'existing_program_id'}
    if set(proposal) - allowed: raise ValueError('extra_schema_field: remove unadvertised proposal fields')
    if proposal.get('source_opportunity_ids') != [opportunity['id']]:
        raise ValueError('invalid_source_id: select exactly the supplied opportunity ID')
    decision = proposal.get('decision')
    if decision not in {'propose_program', 'reuse_existing', 'no_change'}: raise ValueError('Invalid candidate decision')
    if not proposal.get('evidence_limitations'): raise ValueError('Missing evidence limitations')
    if decision == 'no_change': return deepcopy(proposal)
    if decision == 'reuse_existing':
        if proposal.get('existing_program_id') not in {p['id'] for p in existing}:
            raise ValueError('invalid_existing_program_id: select a real supplied Program')
        return deepcopy(proposal)
    atomic = proposal.get('atomic', {})
    if (set(atomic) != {'goal', 'applicability', 'preconditions', 'result_role', 'stop_condition'}
            or atomic['result_role'] != 'intermediate' or not isinstance(atomic['preconditions'], list)
            or not all(isinstance(atomic[k], str) and atomic[k].strip() for k in ('goal', 'applicability', 'stop_condition'))):
        raise ValueError('atomic_schema: describe one intermediate operation with explicit prerequisites and stop condition')
    spec = proposal.get('program_spec', {})
    if set(spec) - {'input_schema', 'output_schema', 'allowed_tools', 'result_fields', 'input_bindings', 'probe_contract'}:
        raise ValueError('extra_schema_field: ProgramSpec has unadvertised fields')
    for field in ('input_schema', 'output_schema'):
        schema = spec.get(field, {})
        if (schema.get('type') != 'object' or schema.get('additionalProperties') is not False
                or not isinstance(schema.get('properties'), dict)
                or set(schema.get('required', [])) - set(schema['properties'])):
            raise ValueError('schema_contract: closed object input/output schemas required')
        check_schema(schema)
    if (not isinstance(spec.get('allowed_tools'), list) or set(spec['allowed_tools']) - set(tools)
            or 'execute_python' in spec['allowed_tools']): raise ValueError('unauthorized_program_tool')
    if not isinstance(spec.get('result_fields'), list) or not set(spec['result_fields']) <= set(spec['output_schema']['properties']):
        raise ValueError('result_fields must name actual output schema fields')
    if not isinstance(spec.get('probe_contract'), dict): raise ValueError('Missing local probe contract')
    source_ids = [e['id'] for e in opportunity['events'][-2:]]
    values, bindings = bind_inputs(spec, spans, source_ids)
    return {**deepcopy(proposal), 'bound_inputs': values, 'input_provenance': bindings,
            'qualification': QUALIFICATION, 'generation_basis': opportunity}


def program_from_response(response, candidate, environment, tools):
    if set(response) != {'source'} or not isinstance(response['source'], str):
        raise ValueError('builder_schema: return exactly {source: Python source}')
    spec = candidate['program_spec']
    program = {k: deepcopy(spec[k]) for k in ('input_schema', 'output_schema', 'allowed_tools')}
    program.update(source=response['source'], entry='run', backend='sandbox_python_v1',
                   environment=deepcopy(environment), result_role='intermediate')
    if set(program['allowed_tools']) - set(tools): raise ValueError('unauthorized_program_tool')
    validate_context_source(program['source'])
    for node in ast.walk(ast.parse(program['source'])):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'ctx':
            if node.attr not in {'call', 'read_result', 'observe', 'available_tools', 'remaining_calls'}:
                raise ValueError('context_api_misuse: Context has no ' + node.attr + '; use ctx.call(name, arguments) from supplied ABI')
    program['id'] = 'pilot_program_' + validate_program(program)
    validate_schema_instance(candidate['bound_inputs'], program['input_schema'])
    return program


def check_schema(schema):
    allowed = {'type', 'properties', 'required', 'additionalProperties', 'items', 'description',
               'minimum', 'maximum', 'minItems', 'maxItems', 'minLength', 'maxLength', 'enum'}
    if not isinstance(schema, dict) or set(schema) - allowed:
        raise ValueError('schema_contract: unsupported schema keywords')
    types = schema.get('type', [])
    if isinstance(types, str): types = [types]
    if not types or set(types) - {'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'}:
        raise ValueError('schema_contract: explicit supported type required')
    for child in schema.get('properties', {}).values(): check_schema(child)
    if 'items' in schema: check_schema(schema['items'])


def bundle(candidate, program, request_path, response):
    atomic_id = 'pilot_atomic_' + digest(candidate['atomic'])
    return {'schema': 'office-program-candidate.v1', 'status': 'candidate_only',
        'id': program['id'], 'atomic': {'id': atomic_id, **candidate['atomic']}, 'program': program,
        'implementation': {'id': 'pilot_impl_' + digest([atomic_id, program['id']]),
                           'skill_id': atomic_id, 'program_id': program['id'], 'status': 'candidate_only'},
        'candidate': candidate, 'generation_basis': candidate['generation_basis'],
        'source_trace_sha256': candidate['generation_basis']['source_trace_sha256'],
        'request_path': str(request_path), 'response_sha256': digest(response),
        'environment_digest': digest(program['environment']), 'outputs_consumed': False,
        'formal_publication': False}
