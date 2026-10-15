"""Single V5 contract: declarations are hypotheses; only Host fields supply values."""
from copy import deepcopy
import json
import math
import re

from ..agents.protocol import SchemaValidationError, validate_schema_instance
from .contracts import digest, object_schema
from .office_operation_mining import at

VERSION = 'office-trace-v5-program-induction.v3'
MATERIAL_VERSION = 'office.operation-fields.v5.2'
CANDIDATE_SCHEMA_VERSION = 'office.program-candidate.v5.2'
BINDING_VERSION = 'office.observed-binding.v5'
PROBE_VERSION = 'office.independent-probe.v5'
STRING = {'type': 'string', 'minLength': 1}
STRINGS = {'type': 'array', 'items': STRING, 'minItems': 1, 'uniqueItems': True}
PATH = {'type': 'array', 'items': {'oneOf': [STRING, {'type': 'integer', 'minimum': 0}]}}
BINDING_KINDS = ['observed_field', 'observed_text_slice', 'runtime_required']
PARAMETER_NAME = r'[A-Za-z_][A-Za-z0-9_]*'
SCHEMA_KEYWORDS = {kind: sorted({'type', 'description', 'enum', *extra}) for kind, extra in {
    'object': ['properties', 'required', 'additionalProperties'], 'array': ['items', 'minItems', 'maxItems', 'uniqueItems'],
    'string': ['minLength', 'maxLength'], 'integer': ['minimum', 'maximum'], 'number': ['minimum', 'maximum'],
    'boolean': [], 'null': []}.items()}


def closed(properties, required=None):
    return object_schema(properties, list(properties) if required is None else required)


OBSERVED = {'kind': {'const': 'observed_field'}, 'span_id': STRING,
            'section': {'enum': ['arguments', 'result', 'task']}, 'path': PATH}
BINDINGS = {
    'observed_field': closed(OBSERVED),
    'observed_text_slice': closed({**OBSERVED, 'kind': {'const': 'observed_text_slice'},
                                  'start': {'type': 'integer', 'minimum': 0}, 'end': {'type': 'integer', 'minimum': 1}}),
    'runtime_required': closed({'kind': {'const': 'runtime_required'}, 'source_requirement': STRING}),
}
PROBE_FIELDS = {
    'table_cell': {'text_input_field': 'string', 'row_input_field': 'string',
                   'column_input_field': 'string', 'unit_input_field': 'string'},
    'series_extract': {'text_input_field': 'string', 'rows_input_field': 'array:string',
                       'column_input_field': 'string', 'unit_input_field': 'string'},
    'numeric': {'values_input_field': 'array:number'},
    'evidence_check': {'text_input_field': 'string', 'required_input_field': 'array:string'},
    'source_locator': {'pattern_input_field': 'string'},
    'retrieval': {'query_input_field': 'string'},
}
OUTPUT_TYPES = {'table_cell': 'number', 'series_extract': 'array:number', 'numeric': 'number',
                'evidence_check': 'boolean', 'source_locator': 'array:string'}
OPTIONAL_PROBE_FIELDS = {'retrieval': {'paths_input_field': 'array:string'}}
FIXED_OUTPUTS = {'retrieval': {'text': 'string', 'source_path': 'string', 'source_offset': 'integer', 'complete': 'boolean'}}
PROBE_SEMANTICS = {
    'source_locator': 'Glob filename discovery: pattern is a glob, not a text regex. The referenced output is the exact array of paths returned by ctx.call(glob). For grep plus reading a window, use table_locator/retrieval.',
    'retrieval': 'Search the supplied regex with grep/grep_page, then read/read_context at an actual matching hit path and offset. Return the exact read text, actual source_path and window start source_offset; complete reflects search pagination. Optional paths input is an array of filenames; omit the contract reference to search without a scope. A text window is not a verified table cell.',
    'table_cell': 'Extract one numeric cell from uniquely identified observed text, row, column and unit. Text-only output is not this numeric oracle.',
    'series_extract': 'Extract a numeric array in the exact requested row order, with an independently identifiable column and unit; do not fill missing rows.',
    'numeric': 'Compute on the supplied nonempty numeric array: sum/mean; sample_std uses n-1; difference is values[1]-values[0], ratio is values[1]/values[0], log_ratio is its natural log. The last three require exactly two values. Undefined operations return an explicit gap.',
    'evidence_check': 'Case-insensitive containment of every supplied nonempty required string in actual text. Complete evidence returns true; missing text/requirements returns needs_input with a gap, not status=ok with false.',
}
OPERATION_PROBES = {'source_locator': 'source_locator', 'table_locator': 'retrieval',
                    'table_cell': 'table_cell', 'series_extract': 'series_extract',
                    'numeric_compute': 'numeric', 'evidence_check': 'evidence_check'}
PROBES = {}
for _kind, _fields in PROBE_FIELDS.items():
    _properties = {'kind': {'const': _kind}, **{name: STRING for name in _fields}}
    if _kind in OUTPUT_TYPES: _properties['output_field'] = STRING
    if _kind == 'numeric':
        _properties['operation'] = {'enum': ['sum', 'mean', 'difference', 'ratio', 'log_ratio', 'sample_std']}
    _required = list(_properties)
    _properties.update({name: STRING for name in OPTIONAL_PROBE_FIELDS.get(_kind, {})})
    PROBES[_kind] = closed(_properties, _required)

ATOMIC_SCHEMA = closed({'goal': STRING, 'applicability': STRING, 'preconditions': {'type': 'array', 'items': STRING},
    'stop_condition': STRING, 'result_role': {'const': 'intermediate'}, 'source_opportunity_ids': STRINGS})
PROGRAM_SCHEMA = closed({'contract_version': {'const': CANDIDATE_SCHEMA_VERSION},
    'operation_kind': {'enum': list(OPERATION_PROBES)}, 'input_schema': {'type': 'object'},
    'output_schema': {'type': 'object'}, 'allowed_tools': {'type': 'array', 'items': STRING, 'uniqueItems': True},
    'input_bindings': {'type': 'object', 'additionalProperties': {'oneOf': list(BINDINGS.values())}},
    'probe_contract': {'oneOf': list(PROBES.values())}, 'expected_output_fields': STRINGS})
COMMON = {'decision': STRING, 'source_opportunity_ids': STRINGS, 'reason': STRING,
          'evidence_limitations': {'oneOf': [STRING, STRINGS]}}
DECISIONS = {name: closed({**COMMON, 'decision': {'const': name}, **extra}) for name, extra in {
    'propose_program': {'atomic': ATOMIC_SCHEMA, 'program_spec': PROGRAM_SCHEMA},
    'reuse_existing': {'existing_program_id': STRING}, 'no_change': {}}.items()}
CANDIDATE_SCHEMA = {'oneOf': list(DECISIONS.values())}


class ContractError(ValueError):
    def __init__(self, code, field_path, expected, actual, evidence_hash, *, repairable=True):
        actual_type = ('null' if actual is None else 'boolean' if isinstance(actual, bool) else
                       {dict: 'object', list: 'array', str: 'string', int: 'integer', float: 'number'}.get(type(actual), type(actual).__name__))
        self.diagnostic = {'code': code, 'field_path': field_path, 'expected': expected,
            'actual_type': actual_type, 'repairable': repairable, 'evidence_hash': evidence_hash}
        if code == 'binding_shape_invalid': self.diagnostic['allowed_binding_kinds'] = BINDING_KINDS
        super().__init__(code + ': ' + field_path + ': ' + str(expected))


def fail(code, path, expected, actual, evidence_hash):
    raise ContractError(code, path, expected, actual, evidence_hash)


def instance(value, schema, path, evidence_hash, code='value_type_mismatch'):
    try: validate_schema_instance(value, schema, path=path)
    except SchemaValidationError as exc:
        fail(code, exc.path, exc.constraint or str(exc), value, evidence_hash)
    # JSON input and output values must be finite, including values read from historical records.
    try: json.dumps(value, allow_nan=False)
    except ValueError: fail(code, path, 'finite JSON values', value, evidence_hash)


def check_schema(schema, path, evidence_hash):
    """Validate the supported schema language itself before validating any instance."""
    if not isinstance(schema, dict): fail('schema_invalid', path, 'JSON Schema object', schema, evidence_hash)
    kind = schema.get('type')
    if not isinstance(kind, str) or kind not in SCHEMA_KEYWORDS:
        fail('schema_invalid', path + '.type', 'one supported JSON type', kind, evidence_hash)
    allowed = set(SCHEMA_KEYWORDS[kind])
    if kind == 'object':
        props, required = schema.get('properties'), schema.get('required')
        if not isinstance(props, dict) or not isinstance(required, list) or not all(isinstance(n, str) for n in required):
            fail('schema_invalid', path, 'properties object and required string array', schema, evidence_hash)
        if (schema.get('additionalProperties') is not False or len(set(required)) != len(required)
                or set(required) - set(props)):
            fail('schema_invalid', path, 'closed properties and unique declared required names', schema, evidence_hash)
        for name, child in props.items():
            if not isinstance(name, str) or not re.fullmatch(PARAMETER_NAME, name):
                fail('schema_invalid', path + '.properties', 'parameter identifiers', name, evidence_hash)
            check_schema(child, path + '.properties.' + name, evidence_hash)
    elif kind == 'array':
        check_schema(schema.get('items'), path + '.items', evidence_hash)
        if 'uniqueItems' in schema and type(schema['uniqueItems']) is not bool:
            fail('schema_invalid', path + '.uniqueItems', 'boolean', schema['uniqueItems'], evidence_hash)
    if set(schema) - allowed: fail('schema_invalid', path, sorted(allowed), schema, evidence_hash)
    for lo, hi in [('minimum', 'maximum'), ('minItems', 'maxItems'), ('minLength', 'maxLength')]:
        for key in (lo, hi):
            if key not in schema: continue
            value = schema[key]
            valid = (type(value) in (int, float) and math.isfinite(value)) if key in {'minimum', 'maximum'} else (type(value) is int and value >= 0)
            if not valid: fail('schema_invalid', path + '.' + key, 'valid numeric bound', value, evidence_hash)
        if lo in schema and hi in schema and schema[lo] > schema[hi]:
            fail('schema_invalid', path, 'minimum <= maximum', schema, evidence_hash)
    if 'description' in schema: instance(schema['description'], {'type': 'string'}, path + '.description', evidence_hash, 'schema_invalid')
    if 'enum' in schema:
        if not isinstance(schema['enum'], list) or not schema['enum']:
            fail('schema_invalid', path + '.enum', 'nonempty array', schema['enum'], evidence_hash)
        for value in schema['enum']: instance(value, {k:v for k,v in schema.items() if k != 'enum'}, path + '.enum', evidence_hash, 'schema_invalid')


def schema_type(schema):
    kind = schema.get('type')
    return 'array:' + schema_type(schema['items']) if kind == 'array' else kind


def require_field_type(schema, name, expected, path, evidence_hash):
    declared = schema_type(schema['properties'][name])
    required = name in schema['required']
    if declared != expected or not required:
        exc = ContractError('value_type_mismatch', path + '.properties.' + name,
            'required ' + expected, schema['properties'][name], evidence_hash)
        exc.diagnostic.update(declared_type=declared, field_required=required)
        raise exc


def validate_probe_contract(spec, evidence_hash):
    contract = spec.get('probe_contract')
    if not isinstance(contract, dict): fail('probe_contract_invalid', 'program_spec.probe_contract', 'tagged object', contract, evidence_hash)
    kind = contract.get('kind')
    if not isinstance(kind, str) or kind not in PROBES:
        fail('oracle_unavailable', 'program_spec.probe_contract.kind', list(PROBES), kind, evidence_hash)
    if not isinstance(spec.get('operation_kind'), str) or OPERATION_PROBES.get(spec['operation_kind']) != kind:
        fail('probe_contract_invalid', 'program_spec.operation_kind', 'matching operation/probe kind', spec.get('operation_kind'), evidence_hash)
    props, outputs = spec['input_schema']['properties'], spec['output_schema']['properties']
    required_fields = {**PROBE_FIELDS[kind]}
    required_fields.update({name: typ for name, typ in OPTIONAL_PROBE_FIELDS.get(kind, {}).items() if name in contract})
    for field, expected_type in required_fields.items():
        name = contract.get(field)
        if not isinstance(name, str) or name not in props:
            fail('probe_input_field_missing', 'program_spec.probe_contract.' + field, list(props), name, evidence_hash)
        require_field_type(spec['input_schema'], name, expected_type, 'program_spec.input_schema', evidence_hash)
    if kind in OUTPUT_TYPES and not isinstance(contract.get('output_field'), str):
        fail('probe_output_field_missing', 'program_spec.probe_contract.output_field', list(outputs), contract.get('output_field'), evidence_hash)
    checks = ({contract['output_field']: OUTPUT_TYPES[kind]} if kind in OUTPUT_TYPES else
              FIXED_OUTPUTS[kind])
    for name, expected_type in checks.items():
        if not isinstance(name, str) or name not in outputs:
            fail('probe_output_field_missing', 'program_spec.probe_contract.output_field', list(outputs), name, evidence_hash)
        require_field_type(spec['output_schema'], name, expected_type, 'program_spec.output_schema', evidence_hash)
    instance(contract, PROBES[kind], 'program_spec.probe_contract', evidence_hash, 'probe_contract_invalid')


def validate_binding(name, ref, schema, spans, material, evidence_hash):
    path = 'program_spec.input_bindings.' + name
    if not isinstance(ref, dict): fail('binding_shape_invalid', path, 'InputBinding object', ref, evidence_hash)
    kind = ref.get('kind')
    if not isinstance(kind, str) or kind not in BINDINGS: fail('binding_shape_invalid', path + '.kind', BINDING_KINDS, kind, evidence_hash)
    instance(ref, BINDINGS[kind], path, evidence_hash, 'binding_shape_invalid')
    if kind == 'runtime_required': return None, {'name': name, 'schema': schema, 'source_requirement': ref['source_requirement']}
    visible = next((s for s in material['spans'] if s['id'] == ref['span_id']), None)
    source = next((s for s in spans if s['id'] == ref['span_id']), None)
    if visible is None or source is None: fail('unknown_source_span', path + '.span_id', 'advertised Host span ID', ref['span_id'], evidence_hash)
    if source['source_task']['physical_key'] != material['source_task']['physical_key']:
        fail('unknown_source_span', path + '.span_id', 'same physical Train task', ref['span_id'], evidence_hash)
    observed = next((f for f in visible['fields'] if f['section'] == ref['section'] and f['path'] == ref['path']), None)
    if observed is None: fail('path_not_observed', path + '.path', 'advertised exact field path', ref['path'], evidence_hash)
    try: value = at(source['source_task'] if ref['section'] == 'task' else source[ref['section']], ref['path'])
    except ValueError: fail('path_not_observed', path + '.path', 'existing Host field', ref['path'], evidence_hash)
    if digest(value) != observed['sha256']: raise RuntimeError('Host material field identity changed')
    if kind == 'observed_text_slice':
        if not isinstance(value, str): fail('value_type_mismatch', path, 'string slice source', value, evidence_hash)
        if not 0 <= ref['start'] < ref['end'] <= len(value):
            fail('text_slice_out_of_bounds', path, {'unicode_length': len(value)}, ref, evidence_hash)
        if 'window' in observed and not observed['window']['start'] <= ref['start'] < ref['end'] <= observed['window']['end']:
            fail('path_not_observed', path, 'slice inside displayed Unicode window', ref, evidence_hash)
        value = value[ref['start']:ref['end']]
    elif 'window' in observed:
        fail('source_incomplete', path, 'explicit displayed slice or a new material projection', ref, evidence_hash)
    instance(value, schema, path, evidence_hash)
    return value, {**ref, 'binding_version': BINDING_VERSION, 'value_sha256': digest(value),
                   'source_trace_sha256': source['source_trace_sha256'], 'physical_key': source['source_task']['physical_key'],
                   'tool_name': source['name']}


def validate_candidate(proposal, opportunity, spans, tools, existing, material):
    evidence_hash = digest(material)
    if not isinstance(proposal, dict): fail('proposal_shape_invalid', '$', 'CandidateDecision object', proposal, evidence_hash)
    decision = proposal.get('decision')
    if not isinstance(decision, str) or decision not in DECISIONS:
        fail('proposal_shape_invalid', 'decision', list(DECISIONS), decision, evidence_hash)
    spec = proposal.get('program_spec')
    if decision == 'propose_program':
        if not isinstance(spec, dict): fail('proposal_shape_invalid', 'program_spec', 'ProgramSpec object', spec, evidence_hash)
        bindings = spec.get('input_bindings')
        if not isinstance(bindings, dict): fail('binding_shape_invalid', 'program_spec.input_bindings', 'object<string, InputBinding>', bindings, evidence_hash)
        for name, ref in bindings.items():
            kind = ref.get('kind') if isinstance(ref, dict) else None
            if not isinstance(kind, str) or kind not in BINDINGS:
                fail('binding_shape_invalid', 'program_spec.input_bindings.' + name, BINDING_KINDS, ref, evidence_hash)
            instance(ref, BINDINGS[kind], 'program_spec.input_bindings.' + name, evidence_hash, 'binding_shape_invalid')
        for field in ('input_schema', 'output_schema'):
            check_schema(spec.get(field), 'program_spec.' + field, evidence_hash)
            if spec[field]['type'] != 'object': fail('schema_invalid', 'program_spec.' + field, 'closed object schema', spec[field], evidence_hash)
        validate_probe_contract(spec, evidence_hash)
    instance(proposal, DECISIONS[decision], '$', evidence_hash, 'proposal_shape_invalid')
    if proposal['source_opportunity_ids'] != [opportunity['id']]:
        fail('unknown_source_span', 'source_opportunity_ids', [opportunity['id']], proposal['source_opportunity_ids'], evidence_hash)
    if decision == 'reuse_existing' and proposal['existing_program_id'] not in {p['id'] for p in existing}:
        fail('field_ref_not_declared', 'existing_program_id', [p['id'] for p in existing], proposal['existing_program_id'], evidence_hash)
    result = {**deepcopy(proposal), 'method_version': VERSION, 'evidence_hash': evidence_hash}
    if decision != 'propose_program': return result
    if proposal['atomic']['source_opportunity_ids'] != proposal['source_opportunity_ids']:
        fail('unknown_source_span', 'atomic.source_opportunity_ids', proposal['source_opportunity_ids'], proposal['atomic']['source_opportunity_ids'], evidence_hash)
    if set(spec['allowed_tools']) - set(tools) or 'execute_python' in spec['allowed_tools']:
        fail('field_ref_not_declared', 'program_spec.allowed_tools', list(tools), spec['allowed_tools'], evidence_hash)
    if set(spec['expected_output_fields']) != set(spec['output_schema']['properties']):
        fail('field_ref_not_declared', 'program_spec.expected_output_fields', list(spec['output_schema']['properties']), spec['expected_output_fields'], evidence_hash)
    props = spec['input_schema']['properties']
    if set(bindings) - set(props) or set(spec['input_schema']['required']) - set(bindings):
        fail('field_ref_not_declared', 'program_spec.input_bindings', 'exact declared bindings; explicit runtime_required for missing values', bindings, evidence_hash)
    values, provenance, pending = {}, {}, []
    for name, ref in bindings.items():
        value, record = validate_binding(name, ref, props[name], spans, material, evidence_hash)
        if ref['kind'] == 'runtime_required': pending.append(record)
        else: values[name], provenance[name] = value, record
    if not pending: instance(values, spec['input_schema'], 'bound_inputs', evidence_hash)
    return {**result, 'bound_inputs': values, 'pending_inputs': pending, 'input_provenance': provenance,
        'status': 'spec_valid_pending_inputs' if pending else 'spec_valid', 'generation_basis': deepcopy(opportunity),
        'validation_stages': ['strict_schema_parse', 'semantic_typecheck', 'source_bind', 'pending_or_bound']}


def field_contract():
    """Exactly the same schemas used by Host validation, serialized for the Learner."""
    return {'version': VERSION, 'candidate_schema': CANDIDATE_SCHEMA, 'binding_version': BINDING_VERSION,
        'schema_language': {'keywords_by_type': SCHEMA_KEYWORDS, 'parameter_name_fullmatch': PARAMETER_NAME,
            'object': 'properties and unique required array are mandatory; required must be a subset; additionalProperties must be false, recursively',
            'array': 'items must be one supported schema object, not a list; uniqueItems is boolean',
            'bounds': 'size bounds are nonnegative integers; numeric bounds are finite; minimum <= maximum',
            'enum': 'nonempty list of values that satisfy the same declared schema',
            'unsupported': 'No type unions, defaults, tuple items, schema references, or additional keywords'},
        'probe_field_types': PROBE_FIELDS, 'optional_probe_field_types': OPTIONAL_PROBE_FIELDS,
        'probe_output_types': OUTPUT_TYPES, 'probe_fixed_outputs': FIXED_OUTPUTS,
        'probe_semantics': PROBE_SEMANTICS, 'operation_probes': OPERATION_PROBES,
        'rules': ['*_input_field and output_field name declared schema properties, never table values.',
                  'source_opportunity_ids must be exactly [the current source.opportunity.id], both at the top level and in atomic. reuse_existing must name an actual advertised existing_program_id.',
                  'allowed_tools must be a subset of the advertised permissions.allowed_names, excluding execute_python; Programs cannot recursively run the native Python tool.',
                  'expected_output_fields must list EVERY output_schema.properties key exactly once, including optional outputs; it is not a subset of important fields.',
                  'output_schema describes the Program outputs object, not the outer {status,outputs,detail?} envelope or the native-tool result wrapper. Declare only outputs the Program actually produces.',
                  'Every referenced probe input (including an optional reference when present) must be required in input_schema with exactly its advertised type.',
                  'For probe_output_types, output_field names a required output_schema property with exactly that type. For probe_fixed_outputs, all listed names/types are fixed required output properties.',
                  'array:string means {type:array,items:{type:string}}; array:number means {type:array,items:{type:number}}. Neither means a string or array of objects.',
                  'input_bindings is an object mapping only declared input parameter names; every required input must have a binding. No implicit defaults or list conversion. Every bound value must satisfy its declared schema.',
                  'Bind actual advertised controls even when their operation failed: a failed search does not erase its observed query/path inputs. Only genuinely missing inputs are runtime_required. Bindings must name declared input properties.',
                  'runtime_required is a missing prerequisite, never an observed value or successful source probe.',
                  'source_path/source_offset require actual source location; pure text cannot prove a corpus offset.'],
        'field_reference_examples': {'valid_if_declared': {'column_input_field': 'column_label'},
            'invalid_literals': [{'column_input_field': 'Surplus, or deficit (-)'}, {'unit_input_field': 'millions'}]}}
