"""Authored Tool entry contracts: normalize symbols, never infer a route."""
from typing import Any

from ..core.bindings import BindingExpression, BindingExprKind, GroundingConstraint
from ..core.contracts import SemanticPredicate
from ..core.serialization import to_primitive


def normalize_entry_contract(value: Any, input_roles) -> dict:
    if not isinstance(value, dict) or set(value) != {'conditions', 'grounding_constraints'}:
        raise ValueError('R10.2 requires an explicit entry_contract with conditions and grounding_constraints')
    if not all(isinstance(value[key], list) for key in value):
        raise ValueError('entry_contract fields must be arrays')
    roles = set(input_roles)

    def check(raw):
        if isinstance(raw, str) and raw.startswith('$'):
            expression = BindingExpression(BindingExprKind.SKILL_INPUT, source_role=raw[1:])
        elif isinstance(raw, (dict, BindingExpression)):
            if isinstance(raw, dict) and set(raw) - {'kind', 'source_role', 'source_step', 'constant', 'transform_id'}:
                raise ValueError('unknown entry binding expression field')
            expression = BindingExpression.from_dict(raw)
        else:
            return  # Portable constants are checked by the static validator.
        if expression.kind is BindingExprKind.CONSTANT:
            if expression.source_role or expression.source_step or expression.transform_id:
                raise ValueError('entry constant cannot carry source/transform fields')
            return
        if expression.kind is not BindingExprKind.SKILL_INPUT or expression.source_role not in roles:
            raise ValueError('entry_contract may reference only declared inputs or stable constants')
        if expression.source_step or expression.transform_id or expression.constant is not None:
            raise ValueError('entry input reference cannot carry step/transform/constant fields')

    predicates = [p if isinstance(p, SemanticPredicate) else SemanticPredicate(**p)
                  for p in value['conditions']]
    # Check raw expression fields before dataclass parsing can discard extras.
    for raw_constraint in value['grounding_constraints']:
        mapping = (raw_constraint.argument_mapping if isinstance(raw_constraint, GroundingConstraint)
                   else raw_constraint.get('argument_mapping', {}))
        for raw in mapping.values():
            check(raw)
    constraints = [c if isinstance(c, GroundingConstraint) else GroundingConstraint(**c)
                   for c in value['grounding_constraints']]
    for predicate in predicates:
        for raw in predicate.args.values():
            check(raw)
    for constraint in constraints:
        for raw in constraint.argument_mapping.values():
            check(raw)
    return to_primitive({'conditions': predicates, 'grounding_constraints': constraints})


def parameter_schema(parameters) -> dict:
    types = {'str': 'string', 'string': 'string', 'entity': 'string', 'object': 'string',
             'int': 'integer', 'integer': 'integer', 'float': 'number', 'number': 'number',
             'bool': 'boolean', 'boolean': 'boolean', 'list': 'array', 'array': 'array',
             'dict': 'object', 'object_map': 'object'}
    return {'type': 'object', 'properties': {
        p.name: {'type': types.get(p.semantic_type.casefold(), 'string')} for p in parameters},
        'required': [p.name for p in parameters if p.required], 'additionalProperties': False}


def check_tool_entry(tool, arguments, harness, evidence_store, revision):
    """Pure admission for this actual Tool invocation, including serial steps."""
    from ..agents.protocol import validate_schema_instance, SchemaValidationError
    from ..core.results import ValidationResult
    try:
        entry = normalize_entry_contract(tool.interface.get('entry_contract'),
                                         tool.signature.get('properties', {}))
        validate_schema_instance(arguments, tool.signature)
    except (KeyError, TypeError, ValueError, SchemaValidationError) as exc:
        return ValidationResult.fail('tool', 'tool_entry_contract_invalid', str(exc))
    # JSON persistence leaves typed references as dictionaries. Reconstruct the
    # declared expressions before passing them to the Harness; otherwise they
    # are interpreted as literal values instead of references to these inputs.
    conditions = [SemanticPredicate(**{**p, 'args': {
        role: BindingExpression.from_dict(value) if isinstance(value, dict) else value
        for role, value in p['args'].items()}}) for p in entry['conditions']]
    if conditions:
        report = harness.validator_channel().validate_atomic_effect({
            'effects': conditions, 'bindings': dict(arguments), 'output_candidates': {}})
        if not report.passed:
            return ValidationResult.fail('tool', 'tool_entry_conditions_unsatisfied',
                                         '; '.join(report.messages))
    for constraint in entry['grounding_constraints']:
        if not evidence_store.match_constraint(constraint, arguments, revision):
            return ValidationResult.fail('tool', 'tool_entry_constraint_unsatisfied', constraint['constraint_id'])
    return ValidationResult.ok('tool', entry_contract=True)
