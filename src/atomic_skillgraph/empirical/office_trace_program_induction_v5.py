"""V5 generation consumes the same field contract that the Host validates."""
import ast
from copy import deepcopy
import json

from .contracts import digest, validate_program
from .office_program_contract_v5 import VERSION, CANDIDATE_SCHEMA_VERSION, ContractError, field_contract, instance
from .office_trace_program_induction import RequestStore, ModelOutputError, BUILDER as ABI_PROMPT
from .program_worker import validate_context_source
from ..agents.protocol import parse_json_strict

EXTRACTOR = '''Propose one small reusable operation from the supplied public Train observations/failures.
Observed text is not verified numeric truth. Incomplete evidence can support a hypothesis whose missing
parameters are runtime_required; it cannot support fabricated values or a successful historical operation.
Reuse an equivalent existing Program. Describe intermediate acquisition/extraction/checking operations,
not the whole answer. Return exactly one JSON object matching the Host contract below. No markdown.
Every source ID and field path must be advertised. Unicode slices use original field coordinates and
must stay in the displayed window. Never infer a value from reason/applicability. A table cell needs
actual text, row, column and unit inputs. Missing values stay pending, without automatic defaulting.
This contract is generated from the actual validator:\n''' + json.dumps(field_contract(), separators=(',', ':'), ensure_ascii=False)
BUILDER = ABI_PROMPT + '''\nThe V5 input_schema and output_schema are immutable. pending_inputs are names,
not example values: generate prerequisite checks, do not fill them in. All output fields must be supported
by actual inputs or tool receipts. Reference contracts use *_input_field as parameter NAMES.
The complete program return is {status,outputs,detail?}; status=ok proves only the local operation.'''


def program_from_response(response, candidate, environment, tools):
    evidence_hash = candidate['evidence_hash']
    instance(response, {'type': 'object', 'properties': {'source': {'type': 'string', 'minLength': 1}},
        'required': ['source'], 'additionalProperties': False}, '$', evidence_hash, 'worker_contract_invalid')
    source = response['source']
    try:
        validate_context_source(source)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'ctx':
                if node.attr not in {'call', 'read_result', 'observe', 'available_tools', 'remaining_calls'}:
                    raise ValueError('context_api_misuse: use ctx.call("read_context", arguments); no ctx.' + node.attr)
    except (ValueError, SyntaxError) as exc:
        raise ContractError('context_api_misuse' if 'context_api_misuse' in str(exc) else 'worker_contract_invalid',
                            'source', str(exc), source, evidence_hash) from exc
    spec = candidate['program_spec']
    if set(spec['allowed_tools']) - set(tools): raise RuntimeError('Validated candidate permission identity changed')
    program = {k: deepcopy(spec[k]) for k in ('input_schema', 'output_schema', 'allowed_tools')}
    program.update(source=source, entry='run', backend='sandbox_python_v1', environment=deepcopy(environment), result_role='intermediate')
    try: identity = validate_program(program)
    except ValueError as exc: raise ContractError('worker_contract_invalid', 'source', str(exc), source, evidence_hash) from exc
    program['id'] = 'v5_program_' + identity
    return program


def builder_material(candidate, permissions):
    return {k: deepcopy(candidate[k]) for k in ('atomic', 'program_spec', 'bound_inputs', 'pending_inputs', 'input_provenance')} | {
        'permissions': permissions, 'evidence_limitations': candidate['evidence_limitations'], 'method_version': VERSION}


def repair_material(response_record, diagnostic, data):
    """No duplicated original payload; original request stays immutable on disk."""
    projected = deepcopy(data)
    original = response_record.get('turn', {}).get('content')
    try: proposal = parse_json_strict(original or '')
    except ValueError: proposal = {}
    spec = proposal.get('program_spec', {}) if isinstance(proposal, dict) else {}
    if not isinstance(spec, dict): spec = {}
    allowed = spec.get('allowed_tools', projected.get('program_spec', {}).get('allowed_tools', []))
    permissions = projected.get('permissions', {})
    if not isinstance(allowed, list) or not all(isinstance(s, str) for s in allowed): allowed = []
    projected['permissions'] = {'allowed_names': permissions.get('allowed_names', []),
        'public_program_abi': permissions.get('public_program_abi'),
        'runtime_tool_definitions': [t for t in permissions.get('runtime_tool_definitions', []) if t['name'] in allowed]}
    if 'existing_programs' in projected:
        projected['existing_program_ids'] = [p['id'] for p in projected.pop('existing_programs')]
    if 'source' in projected:
        source = projected['source']; bindings = spec.get('input_bindings', {})
        refs = list(bindings.values()) if isinstance(bindings, dict) else bindings if isinstance(bindings, list) else []
        mentioned = {ref.get('span_id') for ref in refs if isinstance(ref, dict) and isinstance(ref.get('span_id'), str)}
        if mentioned:
            source['spans'] = [s for s in source['spans'] if s['id'] in mentioned]
        # Retain entire advertised text windows/headers/units/rows. Only redundant envelope
        # metadata and unrelated tool schemas are removed; no new hidden fields become bindable.
        source['opportunity'] = {k:v for k,v in source['opportunity'].items() if k in {'id','origin','missing_preconditions','operation_requirements'}}
    return {'original_response': response_record.get('turn', {}).get('content'), 'diagnostic': diagnostic,
        'source_material': projected, 'original_payload_sha256': response_record['payload_sha256']}


def bundle(candidate, program, request_path, response):
    atomic_id = 'v5_atomic_' + digest(candidate['atomic'])
    return {'schema': CANDIDATE_SCHEMA_VERSION, 'method_version': VERSION, 'status': 'candidate_built',
        'id': program['id'], 'program': program, 'atomic': {'id': atomic_id, **candidate['atomic']},
        'implementation': {'id': 'v5_impl_' + digest([atomic_id, program['id']]), 'skill_id': atomic_id,
                           'program_id': program['id'], 'status': 'candidate'},
        'candidate': candidate, 'generation_basis': candidate['generation_basis'],
        'request_path': str(request_path), 'response_sha256': digest(response), 'outputs_consumed': False,
        'formal_publication': False, 'validation_stages': [*candidate['validation_stages'], 'build']}
