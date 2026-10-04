"""Explicit paid provider capability probe; never substitutes another model."""
from .protocol import NativeToolSpec, validate_schema_instance


def run_provider_capability_probe(provider):
    schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok'], 'additionalProperties': False}
    turn = provider.complete([{'role': 'user', 'content': 'Return one capability_probe call with ok=true.'}],
                             tools=[NativeToolSpec('capability_probe', 'Check native tool support', schema)])
    if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != 'capability_probe':
        raise ValueError('Provider does not meet native tool capability')
    validate_schema_instance(turn.tool_calls[0].arguments, schema)
    return {'supported': bool(turn.tool_calls[0].arguments['ok']), 'usage': {
        name: getattr(turn, name) for name in ['prompt_tokens','completion_tokens','total_tokens','reasoning_tokens']}}
