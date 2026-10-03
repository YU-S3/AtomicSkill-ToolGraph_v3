"""Ephemeral obligations derived from validated P1, never registered assets."""
from types import SimpleNamespace
from ..core.serialization import to_primitive


def contracts_for(expansion, search, harness_profile):
    result = {}
    for instance in search.missing_instances:
        requirement = instance.requirement
        ref = 'gap://' + instance.instance_id
        inputs = {p.name: p for p in requirement.expected_inputs}
        # An explicitly identical input/output role transfers that same value;
        # it never discovers a fresh identity or upgrades grounding on its own.
        identities = {p.name: {'kind': 'input_identity', 'input_role': p.name}
            for p in requirement.expected_outputs
            if p.name in inputs and inputs[p.name].semantic_type == p.semantic_type}
        result[ref] = {'ref': ref, 'instance_id': instance.instance_id,
            'canonical_intent': requirement.intent,
            'atomic_contract': {'inputs': to_primitive(requirement.expected_inputs),
                'outputs': to_primitive(requirement.expected_outputs),
                'preconditions': to_primitive(requirement.precondition_hints),
                'effects': to_primitive(requirement.desired_effects),
                'validator_spec': {'validator_id': 'harness_atomic_effect', 'identity_strict': True,
                                   'output_derivations': identities}},
            'seeded_guideline': {}, 'harness_profile': harness_profile,
            'origin': 'dynamic_gap'}
    return result


def runtime_record(value):
    return SimpleNamespace(provisional_ref=value['ref'], **{k: v for k, v in value.items() if k != 'ref'})
