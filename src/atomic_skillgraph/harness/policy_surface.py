"""Request-local policy projection from the production harness action schema."""
from dataclasses import dataclass
from ..agents.protocol import NativeToolSpec
from ..core.refs import content_hash

@dataclass(frozen=True)
class BenchmarkPolicySurface:
    revision: int
    catalog: dict
    tool: NativeToolSpec
    schema_hash: str

def exact_tuple_surface(harness):
    from .scienceworld_actions import action_schema, compact
    items = harness.action_catalog()
    schema = action_schema(items)
    tool = NativeToolSpec('environment_action',
        'Execute one exact current catalog action using action_type and arguments. '
        'Only these two fields are allowed. Do not submit action_id, intent, support_call_id, '
        'candidate_bindings, candidate_outputs, output_mapping, or implementation inputs. '
        'At node scope this is a preparation action; explicitly validate_current_atomic to submit completion.',
        schema, call_kind='environment_action', scope='task', result_owner='environment_and_validator')
    return BenchmarkPolicySurface(harness._revision, compact(items), tool, content_hash(schema))
