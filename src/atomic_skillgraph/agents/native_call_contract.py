"""Request-local descriptions and diagnostics, never execution authority."""
from dataclasses import asdict, dataclass
import hashlib
import json

NATIVE_CALL_CONTRACT_VERSION = 'r103.native-call-contract.v1'
IMPLEMENTATION_HELP = (
    "Call this implementation with only the Atomic input properties declared in this "
    "function's parameters, directly at the top level. Do not attach control or output "
    "submission fields belonging to environment_action or validate_current_atomic. "
    "Outputs are produced by the implementation and certified by the existing validators. "
    "Properties absent from required may already be grounded; omission is allowed only "
    "as specified by this call's actual schema. ")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class NativeCallContractView:
    tool_name: str
    call_kind: str
    scope: str
    input_schema_json: str
    result_owner: str
    version: str = NATIVE_CALL_CONTRACT_VERSION

    @classmethod
    def from_tool(cls, tool):
        return cls(tool.name, tool.call_kind, tool.scope, canonical(tool.input_schema), tool.result_owner)

    @property
    def schema_sha256(self):
        return hashlib.sha256(self.input_schema_json.encode()).hexdigest()

    def to_dict(self):
        schema = json.loads(self.input_schema_json)
        composed = any(k in schema for k in ('anyOf', 'oneOf', 'allOf'))
        return {**asdict(self), 'schema_sha256': self.schema_sha256,
                'allowed_root_properties': None if composed else sorted(schema.get('properties', {})),
                'required_root_properties': None if composed else sorted(schema.get('required', []))}


def schema_diagnostics(value, schema, failure, *, path='$', limit=64):
    """Inspect only after strict rejection; retain branch identity, never choose one."""
    from .protocol import SchemaValidationError, validate_schema_instance
    issues = []
    omitted = 0

    def visit(item, spec, where, branch=None):
        nonlocal omitted
        try:
            validate_schema_instance(item, spec, path=where)
        except SchemaValidationError as exc:
            if len(issues) >= limit:
                omitted += 1
                return
            row = {'object_path': where, 'first_failure': str(exc), 'failure_path': exc.path,
                   'constraint': exc.constraint, 'actual': exc.actual}
            if branch is not None:
                row['branch'] = branch
            if isinstance(item, dict):
                properties = spec.get('properties', {})
                required = spec.get('required', [])
                # These are this schema's own fields, never a union of branches.
                row.update(allowed_properties=sorted(properties), required_properties=sorted(required),
                    actual_properties=sorted(item), additional_properties=spec.get('additionalProperties', True),
                    unexpected_properties=sorted(set(item)-set(properties)) if spec.get('additionalProperties') is False else [],
                    missing_required_properties=sorted(set(required)-set(item)))
            issues.append(row)
            for key in ('anyOf', 'oneOf', 'allOf'):
                for index, child in enumerate(spec.get(key, [])):
                    visit(item, child, where, f'{branch + "/" if branch else ""}{key}[{index}]')
            if isinstance(item, dict):
                for key, child in item.items():
                    child_spec = spec.get('properties', {}).get(key, spec.get('additionalProperties'))
                    if isinstance(child_spec, dict):
                        visit(child, child_spec, f'{where}.{key}', branch)
            elif isinstance(item, list) and isinstance(spec.get('items'), dict):
                for index, child in enumerate(item):
                    visit(child, spec['items'], f'{where}[{index}]', branch)

    try:
        visit(value, schema, path)
        return {'first_failure': str(failure), 'issues': issues, 'truncated': omitted > 0,
                'omitted_issue_count': omitted}
    except Exception as exc:
        return {'first_failure': str(failure), 'diagnostic_build_error': type(exc).__name__}


def rejection_diagnostic(tool, call, failure):
    view = NativeCallContractView.from_tool(tool)
    return {'version': view.version, 'tool_name': tool.name, 'call_kind': view.call_kind,
            'scope': view.scope, 'schema_sha256': view.schema_sha256,
            'call_id': call.call_id, 'error_code': 'runtime_agent_schema_error', 'executed': False,
            'repair_scope': 'native_call_schema_only',
            **schema_diagnostics(call.arguments, tool.input_schema, failure)}
