"""Single public registry for Tool collection schema, help and interpreters."""
from copy import deepcopy
from .value_reference import BOUNDED_COUNT_SCHEMA

TOOL_IR_COLLECTION_SOURCES = {
    "tool_input": {
        "required_fields": [
            "field"
        ],
        "optional_fields": [],
        "static_constraints": "Declared array input; v2 maxItems <= max_iterations.",
        "runtime_semantics": "Enumerate caller input list at loop entry.",
        "empty_result_semantics": "Zero iterations; no global absence claim.",
        "refresh_semantics": "Entry snapshot only."
    },
    "local_variable": {
        "required_fields": [
            "field"
        ],
        "optional_fields": [],
        "static_constraints": "List in current lexical scope; v2 finite maxItems.",
        "runtime_semantics": "Enumerate current local list, never another occurrence.",
        "empty_result_semantics": "Zero iterations.",
        "refresh_semantics": "Entry snapshot only."
    },
    "action_catalog": {
        "required_fields": [],
        "optional_fields": [
            "field",
            "where",
            "project",
            "distinct",
            "refresh_each_iteration"
        ],
        "static_constraints": "Public primitive roles only; exact filters accept scalar literals or {source:tool_input|local_variable,field:role} in current scope.",
        "runtime_semantics": "Read current public admissible catalog; candidates are not effects.",
        "empty_result_semantics": "Filtered/projected empty entry aborts with tool_ir_selector_no_match; use an IF match guard for optional lookup.",
        "refresh_semantics": "Optional refresh_each_iteration selects first unseen value; empty after progress ends loop."
    },
    "semantic_evidence": {
        "required_fields": [],
        "optional_fields": [
            "field",
            "where",
            "project",
            "distinct"
        ],
        "static_constraints": "Public predicate roles only.",
        "runtime_semantics": "Read current semantic facts, with joint role filtering.",
        "empty_result_semantics": "Filtered/projected empty entry fails tool_ir_selector_no_match; guard optional lookup.",
        "refresh_semantics": "Entry snapshot only."
    },
    "binding_evidence": {
        "required_fields": [],
        "optional_fields": [
            "field",
            "where",
            "project",
            "distinct"
        ],
        "static_constraints": "Current occurrence authorized evidence only.",
        "runtime_semantics": "Read current binding evidence.",
        "empty_result_semantics": "Zero iterations.",
        "refresh_semantics": "Entry snapshot only."
    },
    "local_deterministic": {
        "required_fields": [
            "values"
        ],
        "optional_fields": [
            "project",
            "distinct"
        ],
        "static_constraints": "Nonempty bounded portable literal list; no episode identifiers.",
        "runtime_semantics": "Enumerate declared deterministic values.",
        "empty_result_semantics": "Empty literal source statically invalid.",
        "refresh_semantics": "Entry snapshot only."
    },
    "bounded_count": {
        "required_fields": [
            "count"
        ],
        "optional_fields": [],
        "static_constraints": "Required integer input: 0 <= minimum <= maximum <= max_iterations <= max_actions.",
        "runtime_semantics": "count={source:tool_input,field:role}; range(count), no clamp.",
        "empty_result_semantics": "Count 0 gives zero iterations.",
        "refresh_semantics": "Entry snapshot only."
    }
}

def collection_source_names():
    return tuple(TOOL_IR_COLLECTION_SOURCES)

def public_collection_sources(condition_sources, action_fields):
    result = [dict(source=name, public=True, **deepcopy(spec)) for name,spec in TOOL_IR_COLLECTION_SOURCES.items()]
    details = (
        {
            "source": "action_catalog",
            "description": (
                "Enumerate values projected from the current public admissible "
                "primitive-action catalog at FOR_EACH entry. This exposes "
                "candidates only; it does not select an action or prove an effect."
                " A filtered/projected FOR_EACH with zero matches aborts the entire "
                "Tool with tool_ir_selector_no_match. Optional candidates need an "
                "IF condition.match guard before enumeration; false skips that "
                "lookup and permits subsequent work."
            ),
            "entry_fields": list(action_fields),
            "where": {
                "action_type": (
                    "optional exact action_type from primitive_actions"
                ),
                "argument_role": (
                    "required with semantic_compatible_with and names an exact "
                    "argument role from that primitive action"
                ),
                "semantic_compatible_with": {
                    "source": list(condition_sources),
                    "field": (
                        "required field or declared role in the selected source"
                    ),
                    "semantic_type": "optional public semantic type",
                },
            },
            "direct_argument_filter_encoding": (
                "Put each optional exact portable primitive argument value directly "
                "at where.<argument_role>. Do not wrap argument filters in another "
                "object. Every role is checked against the selected action_type's "
                "public primitive signature."
            ),
            "projection": {
                "field": (
                    "a top-level action_catalog entry field, used instead of project"
                ),
                "project": {
                    "kind": ["field", "argument"],
                    "field": "required when kind=field",
                    "role": (
                        "required when kind=argument and names a primitive "
                        "argument role"
                    ),
                },
            },
            "distinct": "optional boolean",
            "refresh_each_iteration": (
                "Optional boolean, FOR_EACH action_catalog only. False (default) "
                "uses the entry snapshot. True queries the current revision before "
                "each iteration and takes the first unseen projected value; vanished "
                "values are skipped and new values are visible. Empty after progress "
                "ends the loop; strict empty entry still rejects. Bounds are unchanged."
            ),
        },
    )
    result[2].update(details[0])
    result[2]['direct_argument_filter_encoding'] += (
        ' A filter may also read an exact current Tool input/local value with '
        '{source:tool_input|local_variable,field:role}; this is equality, not semantic matching.')
    result[-1]['count'] = deepcopy(BOUNDED_COUNT_SCHEMA)
    return result

def collection_source_schema(condition_sources):
    NONEMPTY_STRING_SCHEMA = {"type":"string","minLength":1}
    schema = {
        "type": "object",
        "required": ["source"],
        "allOf": [{"anyOf": [
            {"properties": {"source": {"not": {"const": "bounded_count"}}}},
            {"required": ["count"], "additionalProperties": False,
             "properties": {"source": {"const":"bounded_count"}, "count": BOUNDED_COUNT_SCHEMA}}
        ]}],
        "anyOf": [
            {"not": {"required": ["refresh_each_iteration"]}},
            {"properties": {"source": {"const": "action_catalog"}}},
        ],
        "properties": {
            "refresh_each_iteration": {"type": "boolean", "description": "Only for action_catalog FOR_EACH: re-query current catalog before each iteration, selecting the first unseen value; default false preserves the entry snapshot."},
            "source": {
                "type": "string",
                "enum": list(collection_source_names()),
            },
            "field": NONEMPTY_STRING_SCHEMA,
            "count": BOUNDED_COUNT_SCHEMA,
            "values": {"type": "array", "minItems": 1},
            "where": {
                "type": "object",
                "description": (
                    "Filter action_catalog with optional where.action_type and "
                    "direct where.<primitive_argument_role>=<exact portable value> "
                    "entries. There is no primitive_argument_filters wrapper. "
                    "Static validation closes every direct role against the public "
                    "Harness primitive signature."
                ),
                "properties": {
                    "action_type": {"type": "string"},
                    "predicate": {"type": "string"},
                    "argument_role": {"type": "string"},
                    "semantic_compatible_with": {
                        "type": "object",
                        "required": ["source", "field"],
                        "properties": {
                            "source": {
                                "type": "string",
                                "enum": list(condition_sources),
                            },
                            "field": NONEMPTY_STRING_SCHEMA,
                            "semantic_type": {"type": "string"},
                        },
                    },
                },
            },
            "project": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["field", "argument"]},
                    "field": NONEMPTY_STRING_SCHEMA,
                    "role": NONEMPTY_STRING_SCHEMA,
                },
            },
            "distinct": {"type": "boolean"},
        },
        "description": (
            "FOR_EACH selector over an existing Tool IR source. For source="
            "action_catalog or semantic_evidence, a filtered/projected selector "
            "with ZERO matching entries aborts the ENTIRE Tool with "
            "tool_ir_selector_no_match; it does not mean zero harmless iterations. "
            "For optional action candidates, first guard the lookup using IF with "
            "the matching action_catalog condition.match, then enumerate only in "
            "that true branch. Query false permits continued execution. For source="
            "action_catalog, each current public entry has action_id, revision, "
            "action_type, and arguments. Filter with where.action_type and optional "
            "exact primitive-argument values directly as where.<argument_role>; "
            "do not use a primitive_argument_filters wrapper. When "
            "semantic_compatible_with is used, "
            "where.argument_role names the action argument and the nested "
            "source/field names the comparison authority. Project a top-level entry "
            "field with field or use project.kind=argument plus project.role to put "
            "that primitive argument value into the loop variable. distinct is an "
            "optional boolean. The catalog supplies candidates, not effect evidence."
        ),
    }
    for name, contract in TOOL_IR_COLLECTION_SOURCES.items():
        if contract['required_fields']:
            schema['allOf'].append({'anyOf':[
                {'properties':{'source':{'not':{'const':name}}}},
                {'required':contract['required_fields']} ]})
    return schema
