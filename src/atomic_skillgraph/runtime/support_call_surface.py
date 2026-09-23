"""One request-local, read-only Support selection boundary.

An option identifies a proved *mapping*, not a permission to execute.  The
ordinary current-state proof, resolver and transaction still own execution.
JSON snapshots prevent a prompt projection from changing the decode table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..agents.protocol import NativeToolSpec, SchemaValidationError, validate_schema_instance
from ..core.refs import canonical_json, content_hash
from ..core.serialization import to_primitive
from .negative_memory import state_signature
from .support_interface import public_arguments_schema

VERSION = 'r103.support-call-options.v1'
NODE_HELP = ('Select a support_call_id and supply that option\'s exact arguments. '
    'Its complete parent transfer is fixed; do not submit a producer reference or output_mapping. '
    'Choose search scopes and other declared controls explicitly. The helper returns control to you; '
    'A semantic category is not a concrete observed identity. A blocked candidate has no callable option. '
    'selection is not evidence that inputs are grounded or that execution will succeed.')
TASK_HELP = ('Select a support_call_id and supply that option\'s exact arguments. '
    'Task capabilities publish their own validated outputs, with no parent-role transfer. '
    'Do not submit a producer reference or output_mapping. You choose any declared search scopes '
    'and controls; ordinary grounding and execution checks still apply.')


def contract_hash(atomic):
    return content_hash(to_primitive({name: getattr(atomic, name, None) for name in
        ('ref', 'inputs', 'outputs', 'preconditions', 'effects', 'validator_spec')}))


@dataclass(frozen=True)
class SupportCallOption:
    support_call_id: str
    _json: str

    def public(self):
        return {'support_call_id': self.support_call_id, **json.loads(self._json)}

    @property
    def atomic_ref(self):
        return json.loads(self._json)['atomic_ref']

    @property
    def input_schema(self):
        return json.loads(self._json)['input_schema']


@dataclass(frozen=True)
class SupportCallSurface:
    version: str
    scope: str
    consumer_occurrence_id: str
    consumer_atomic_ref: str | None
    session_id: str
    revision: int
    surface_context_fingerprint: str
    options: tuple[SupportCallOption, ...]
    _blocked_json: str

    @property
    def blocked(self):
        return tuple(json.loads(self._blocked_json))

    def public_candidates(self):
        fields = ('support_call_id', 'atomic_ref', 'summary', 'input_schema', 'input_requirements',
                  'output_mapping', 'predicate_input_mapping', 'program_available', 'execution_readiness')
        return [{k: v for k, v in option.public().items() if k in fields} for option in self.options]

    def native_tool(self):
        if not self.options:
            return None
        return NativeToolSpec('invoke_support_atomic', NODE_HELP if self.scope == 'node' else TASK_HELP, {
            'type': 'object', 'required': ['support_call_id', 'arguments'], 'additionalProperties': False,
            'properties': {'support_call_id': {'type': 'string',
                'enum': [o.support_call_id for o in self.options]},
                'arguments': public_arguments_schema(self.options)}},
            call_kind='support', scope=self.scope, result_owner='support_executor_and_validator')


@dataclass(frozen=True)
class ResolvedSupportSelection:
    option: SupportCallOption
    _arguments_json: str

    @property
    def arguments(self):
        return json.loads(self._arguments_json)

    @property
    def output_mapping(self):
        return self.option.public()['output_mapping']

    @property
    def input_mapping(self):
        return self.option.public()['predicate_input_mapping']

    def canonical_arguments(self):
        return {'support_atomic_ref': self.option.atomic_ref, 'arguments': self.arguments,
                'output_mapping': self.output_mapping, 'predicate_input_mapping': self.input_mapping}


def context_fingerprint(ctx, occurrence, consumer):
    return content_hash({'state': state_signature(ctx, occurrence),
                         'consumer_contract': contract_hash(consumer) if consumer else None})


def build_surface(candidates, *, skills, routes, consumer, occurrence, ctx, session_id):
    scope = getattr(occurrence, 'consumer_scope', 'node')
    consumer_hash = contract_hash(consumer) if consumer else None
    options, blocked, selected = [], [], set()
    for candidate in candidates:
        producer = skills.get_atomic(candidate.atomic_ref)
        compiled = routes.get(candidate.atomic_ref, [])
        if len(compiled) != 1 or not candidate.input_schema:
            blocked.append({'atomic_ref': candidate.atomic_ref, 'reason_code': 'no_unambiguous_program'})
            continue
        previews = ([{'input_mapping': {}, 'output_mapping': None, 'mapping_evidence': []}]
                    if scope == 'task' else [p for p in candidate.mapping_previews if p['status'] == 'proven'])
        if not previews:
            blocked.append({'atomic_ref': candidate.atomic_ref, 'reason_code': 'no_proven_mapping',
                'required_anchor_or_relation': [p.get('required_anchor_or_relation', [])
                    for p in candidate.mapping_previews]})
            continue
        # Limit distinct callable Atomics, not blocked retrieval hits or mapping options.
        if candidate.atomic_ref not in selected and len(selected) >= 3:
            continue
        selected.add(candidate.atomic_ref)
        for preview in previews:
            identity = {'version': VERSION, 'scope': scope,
                'consumer_atomic_ref': str(consumer.ref) if consumer else None,
                'consumer_contract_hash': consumer_hash, 'producer_contract_hash': contract_hash(producer),
                'atomic_ref': candidate.atomic_ref, 'input_schema': candidate.input_schema,
                'output_mapping': preview['output_mapping'],
                'predicate_input_mapping': preview['input_mapping'],
                'mapping_evidence': preview.get('mapping_evidence', [])}
            option_id = 'sc_' + content_hash(identity)[:32]
            if any(o.support_call_id == option_id for o in options):
                continue
            details = {**identity, 'summary': producer.summary,
                'program_available': True, 'execution_readiness': 'validate_after_arguments',
                'anchor_inputs': preview.get('anchor_inputs', []),
                'input_requirements': {p.name: {'required': p.required, 'semantic_type': p.semantic_type,
                    'required_resolution': p.required_resolution} for p in producer.inputs},
                'route_identity': str(compiled[0].implementation.ref)}
            options.append(SupportCallOption(option_id, canonical_json(to_primitive(details))))
    return SupportCallSurface(VERSION, scope, occurrence.occurrence_id,
        str(consumer.ref) if consumer else None, session_id, ctx.world_revision,
        context_fingerprint(ctx, occurrence, consumer), tuple(options), canonical_json(blocked[:3]))


def decode(surface, call, *, session_id, occurrence, consumer, ctx, skills):
    """Resolve a raw call without rewriting it. Return structured failures."""
    def failure(code, path, expected, actual):
        return None, {'accepted': False, 'error': code, 'error_code': code, 'reason_code': code,
            'argument_path': path, 'expected_constraint': expected, 'actual_summary': actual,
            'support_call_id': call.arguments.get('support_call_id'),
            'relevant_revision': ctx.world_revision}
    if (surface.version != VERSION or surface.session_id != session_id
            or surface.scope != getattr(occurrence, 'consumer_scope', 'node')
            or surface.consumer_occurrence_id != occurrence.occurrence_id
            or surface.revision != ctx.world_revision
            or surface.surface_context_fingerprint != context_fingerprint(ctx, occurrence, consumer)):
        return failure('support_surface_stale_or_wrong_owner', '$.support_call_id',
                       'current request and consumer', call.arguments.get('support_call_id'))
    tool = surface.native_tool()
    if tool is None:
        return failure('support_option_unavailable', '$.support_call_id', [], call.arguments.get('support_call_id'))
    try:
        boundary_schema = tool.input_schema
        boundary_schema['properties']['arguments'] = {'type': 'object'}
        validate_schema_instance(call.arguments, boundary_schema)
    except SchemaValidationError as exc:
        return failure('support_call_schema_invalid', exc.path, exc.constraint, exc.actual)
    option = next(o for o in surface.options if o.support_call_id == call.arguments['support_call_id'])
    details = option.public()
    try:
        current = skills.get_atomic(option.atomic_ref)
    except KeyError:
        return failure('support_contract_changed', '$.support_call_id', details['producer_contract_hash'], None)
    if contract_hash(current) != details['producer_contract_hash']:
        return failure('support_contract_changed', '$.support_call_id', details['producer_contract_hash'], contract_hash(current))
    try:
        validate_schema_instance(call.arguments['arguments'], option.input_schema, path='$.arguments')
    except SchemaValidationError as exc:
        _, payload = failure('support_atomic_input_schema_invalid', exc.path, exc.constraint, exc.actual)
        payload.update(constraint_scope='stable_schema', support_atomic_ref=option.atomic_ref)
        return ResolvedSupportSelection(option, canonical_json(call.arguments['arguments'])), payload
    return ResolvedSupportSelection(option, canonical_json(call.arguments['arguments'])), None
