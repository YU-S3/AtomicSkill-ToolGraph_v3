"""Pure, explicit caller-control certification. Never grants world authority."""
from __future__ import annotations

import copy
import hashlib
from ..core.bindings import (BindingExpression, BindingSource, BindingStatus,
    BindingResolution, RuntimeBinding, GroundingConstraint, GroundingConstraintKind)
from ..core.serialization import json_value_key, json_values_equal

VERSION = 'r103.caller-input.v1'

def validate_declarations(atomic):
    declarations = atomic.validator_spec.get('input_authorization', {})
    if not isinstance(declarations, dict):
        raise ValueError('input_authorization must be an object')
    parameters = {p.name:p for p in atomic.inputs}
    for role, declaration in declarations.items():
        p = parameters.get(role)
        if p is None or p.required_resolution != 'semantic' or not isinstance(declaration,dict):
            raise ValueError('caller authorization requires a declared semantic input')
        if declaration == {'kind':'caller_boolean'} and p.semantic_type in {'bool','boolean'}:
            continue
        if (declaration.get('kind') == 'ordered_entity_scope' and p.semantic_type in {'list','array'}
            and set(declaration) == {'kind','element_semantic_type','min_items','max_items','unique_items'}
            and declaration['element_semantic_type'] == 'entity'
            and type(declaration['min_items']) is int and type(declaration['max_items']) is int
            and 1 <= declaration['min_items'] <= declaration['max_items'] <= 8
            and declaration['unique_items'] is True):
            continue
        raise ValueError(f'invalid caller-control declaration: {role}')
    return declarations

def authorize(atomic, role, value, *, call_id, evidence_store, revision, fixed_anchor=None):
    """Return an uncommitted binding; only InvocationTransaction may publish it."""
    declarations=validate_declarations(atomic)
    if role not in declarations or not call_id:
        raise ValueError('explicit declared caller argument required')
    declaration=declarations[role]
    if fixed_anchor is not None and fixed_anchor.source in {
            BindingSource.TASK,BindingSource.RUNTIME_PLAN,BindingSource.DATA_FLOW,BindingSource.REPEAT}:
        if not json_values_equal(fixed_anchor.value,value):
            raise ValueError('caller value conflicts with its fixed input source')
    refs=[]
    if declaration['kind']=='caller_boolean':
        if type(value) is not bool:
            raise ValueError('caller_boolean requires an actual boolean')
    else:
        if (not isinstance(value,list) or not declaration['min_items'] <= len(value) <= declaration['max_items']
                or any(not isinstance(item,str) or not item for item in value)
                or len(set(value)) != len(value)):
            raise ValueError('ordered scope must contain bounded unique entity strings')
        for item in value:
            constraint=GroundingConstraint('caller_scope_identity',GroundingConstraintKind.ARGUMENT_CONCRETE,
                argument_mapping={'entity':BindingExpression('skill_input',source_role='entity')})
            evidence=evidence_store.match_constraint(constraint,{'entity':item},revision)
            if not evidence:
                raise ValueError('ordered scope contains an unauthenticated entity')
            refs.extend(e.evidence_id for e in evidence)
    digest=hashlib.sha256(json_value_key(value).encode()).hexdigest()
    refs.append(f'caller_arg:{call_id}:{role}:{digest}')
    parameter=next(p for p in atomic.inputs if p.name==role)
    return RuntimeBinding(role,copy.deepcopy(value),parameter.semantic_type,BindingSource.CALLER_AUTHORIZED,
        BindingStatus.GROUNDED,BindingResolution.SEMANTIC,list(dict.fromkeys(refs)),revision)


def validate_committed(atomic, role, binding, *, evidence_store, revision):
    """Recheck a carried control without minting a new caller authorization."""
    allowed = {BindingSource.TASK, BindingSource.RUNTIME_PLAN, BindingSource.DATA_FLOW,
               BindingSource.REPEAT, BindingSource.CALLER_AUTHORIZED}
    if (binding.source not in allowed or binding.status is not BindingStatus.GROUNDED
            or binding.resolution is not BindingResolution.SEMANTIC):
        raise ValueError('control input is not a committed semantic authorization')
    if binding.source is BindingSource.CALLER_AUTHORIZED and not any(
            ref.startswith('caller_arg:') for ref in binding.evidence_refs):
        raise ValueError('caller authorization provenance missing')
    # Reuse the pure type/scope validator. Its hypothetical binding is never
    # committed or returned, so autonomous entry cannot create authorization.
    authorize(atomic, role, binding.value, call_id='validation-only',
              evidence_store=evidence_store, revision=revision, fixed_anchor=binding)
