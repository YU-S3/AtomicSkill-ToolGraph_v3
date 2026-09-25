"""Public Tool-local references and bounded collections; no graph bindings.

Only object keys are projected. Never attribute access, JSONPath, indexing,
expressions, implicit conversion or access to a different execution scope.
"""
from collections.abc import Mapping
import re

FIELD_PATH_SCHEMA = {'type':'array', 'minItems':1, 'maxItems':4,
    'items':{'type':'string', 'pattern':r'^[A-Za-z_][A-Za-z0-9_]*$'}}
BOUNDED_COUNT_SCHEMA = {'type':'object', 'required':['source','field'],
    'additionalProperties':False,
    'properties':{'source':{'const':'tool_input'}, 'field':{'type':'string','minLength':1}}}
TOOL_VALUE_REFERENCE_HELP = (
    'Tool IR only: skill_input/local_variable references may have field_path, '
    'a list of 1..4 named object fields. No indices, wildcards, expressions or '
    'JSONPath; absent fields or non-object intermediates fail closed. '
    'FOR_EACH source=bounded_count has count={source:tool_input,field:<input>}; '
    'the input_schema must declare an integer with minimum>=0 and finite maximum '
    '<=max_iterations. Array input collections require maxItems<=max_iterations. '
    'Counts and collections are rejected above their bounds, never clamped. '
    'Declare input_schema with exactly the original input roles/types/required '
    'set; only narrow validation, never change the Atomic boundary. '
    'Worst-case ACTION bound is sum(sequence), max(IF branches), and '
    'max_iterations*body_bound(FOR_EACH), and must fit max_actions.')


def field_path(reference):
    if 'field_path' not in reference:
        return ()
    path = reference['field_path']
    kind = reference.get('kind')
    if kind not in {'skill_input','local_variable'} or not isinstance(path,list) or not 1 <= len(path) <= 4:
        raise ValueError('tool_ir_value_reference_invalid: illegal field_path')
    if reference.get('source_step') or reference.get('transform_id') or reference.get('constant') is not None:
        raise ValueError('tool_ir_value_reference_invalid: projection cannot carry graph or transform authority')
    if any(not isinstance(key,str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',key) is None for key in path):
        raise ValueError('tool_ir_value_reference_invalid: only named object fields are allowed')
    return tuple(path)


def resolve_tool_value_reference(reference, state):
    path = field_path(reference)
    kind = reference.get('kind', reference.get('source'))
    role = reference.get('source_role', reference.get('field',''))
    if kind not in {'skill_input','tool_input','local_variable'}:
        raise ValueError('tool_ir_value_reference_invalid: unsupported source')
    scope = state.local if kind == 'local_variable' else state.bindings
    value = scope.get(role)
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError('tool_ir_value_reference_invalid: missing field or non-object intermediate')
        value = value[key]
    return value


def bounded_count_reference(source):
    count = source.get('count')
    if set(source) != {'source','count'} or not isinstance(count,Mapping) or set(count) != {'source','field'}:
        raise ValueError('tool_ir_bounded_count_invalid: expected only source and count')
    if count['source'] != 'tool_input' or not isinstance(count['field'],str) or not count['field']:
        raise ValueError('tool_ir_bounded_count_invalid: count must name a Tool input')
    return count


def worst_case_actions(program):
    # Caller first uses the shared bounded recursive tree validator.
    total = 0
    for node in program:
        op = node['op']
        if op == 'ACTION': total += 1
        elif op == 'IF': total += max(worst_case_actions(node.get('then_branch',[])), worst_case_actions(node.get('else_branch',[])))
        elif op == 'FOR_EACH':
            n = node.get('max_iterations')
            if type(n) is not int or n <= 0:
                raise ValueError('tool_ir_for_each_unbounded: positive static limit required')
            total += n * worst_case_actions(node.get('body',[]))
    return total


def uses_value_extensions(value):
    if isinstance(value,Mapping):
        return 'field_path' in value or value.get('source') == 'bounded_count' or any(uses_value_extensions(v) for v in value.values())
    return isinstance(value,(list,tuple)) and any(uses_value_extensions(v) for v in value)


def tool_input_schema(parameters, declared=None):
    """Narrow the physical Tool signature without changing the Atomic roles."""
    from .entry_contract import parameter_schema
    import copy
    base = parameter_schema(parameters)
    if declared is None: return base
    if not isinstance(declared,Mapping) or declared.get('type') != 'object' or declared.get('additionalProperties') is not False:
        raise ValueError('tool_ir_input_schema_invalid: closed object signature required')
    if set(declared.get('properties',{})) != set(base['properties']) or set(declared.get('required',[])) != set(base['required']):
        raise ValueError('tool_ir_input_schema_invalid: Atomic input roles/required set must not change')
    for role, shape in base['properties'].items():
        if not isinstance(declared['properties'][role],Mapping) or declared['properties'][role].get('type') != shape['type']:
            raise ValueError('tool_ir_input_schema_invalid: Atomic input type must not change')
    return copy.deepcopy(dict(declared))


def validate_value_contract(program, signature, max_actions, *, strict=False):
    """One lexical schema traversal, shared by publication and invocation.

    Legacy programs lacking new references retain their original schema. New
    explicit input schemas opt into finite collection bounds. Existing declared
    maxItems constraints are always honored, even without that opt-in.
    """
    if strict and (type(max_actions) is not int or max_actions <= 0):
        raise ValueError('tool_ir_bounded_max_actions: positive integer required')
    inputs = signature.get('properties',{})
    def reference(ref, locals_):
        path = field_path(ref)
        if not path: return
        kind = ref.get('kind', ref.get('source'))
        role = ref.get('source_role', ref.get('field',''))
        schema = (locals_ if kind == 'local_variable' else inputs).get(role)
        for key in path:
            if not isinstance(schema,Mapping) or schema.get('type') != 'object' or key not in schema.get('properties',{}) or key not in schema.get('required',[]):
                raise ValueError('tool_ir_value_reference_invalid: projected field must be declared and required in its scope')
            schema = schema['properties'][key]
    def scan(value, locals_):
        if isinstance(value,Mapping):
            if 'field_path' in value: reference(value,locals_)
            for child in value.values(): scan(child,locals_)
        elif isinstance(value,list):
            for child in value: scan(child,locals_)
    def visit(nodes, locals_):
        for node in nodes:
            op = node['op']
            for key,value in node.items():
                if key not in {'body','then_branch','else_branch'}: scan(value,locals_)
            if op == 'IF':
                visit(node.get('then_branch',[]),dict(locals_)); visit(node.get('else_branch',[]),dict(locals_))
            if op != 'FOR_EACH': continue
            source, limit = node.get('collection_source',{}), node.get('max_iterations')
            if strict and (type(limit) is not int or limit <= 0 or limit > max_actions):
                raise ValueError('tool_ir_for_each_unbounded: max_iterations must fit max_actions')
            item = {}
            kind = source.get('source')
            if kind == 'bounded_count':
                count = bounded_count_reference(source)
                schema = inputs.get(count['field'],{})
                low, high = schema.get('minimum'), schema.get('maximum')
                if schema.get('type') != 'integer' or type(low) is not int or low < 0 or type(high) is not int or high < low or type(limit) is not int or high > limit:
                    raise ValueError('tool_ir_bounded_count_invalid: input integer needs 0<=minimum<=maximum<=max_iterations')
                if count['field'] not in signature.get('required',[]):
                    raise ValueError('tool_ir_bounded_count_invalid: count input must be required')
                item = {'type':'integer'}
            elif kind in {'tool_input','local_variable'}:
                schema = (inputs if kind == 'tool_input' else locals_).get(source.get('field'),{})
                high = schema.get('maxItems')
                if strict or high is not None:
                    if schema.get('type') != 'array' or type(high) is not int or high < 0 or type(limit) is not int or high > limit:
                        raise ValueError('tool_ir_collection_bound_invalid: array requires maxItems<=max_iterations')
                item = schema.get('items',{})
            visit(node.get('body',[]),{**locals_,node.get('iteration_variable',''):item})
    visit(program,{})
    bound = worst_case_actions(program)
    if strict and bound > max_actions:
        raise ValueError('tool_ir_worst_case_actions_exceeded: program static action bound exceeds max_actions')
    return bound
