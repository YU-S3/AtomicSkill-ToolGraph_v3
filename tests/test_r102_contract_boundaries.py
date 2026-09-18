"""Pure contract/identity boundaries, not benchmark-specific policy fixtures."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents.skill_guidance import normalize_guideline, guidance_view
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind, GroundingConstraint, GroundingConstraintKind
from atomic_skillgraph.core.contracts import ParameterSpec
from atomic_skillgraph.harness.protocol import HarnessActionSpec
from atomic_skillgraph.runtime.evidence_store import GroundingEvidenceStore
from atomic_skillgraph.tooling.entry_contract import normalize_entry_contract, parameter_schema
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments.fakes import FakeHarness, fake_task
from test_batch_evolution import _system_config


@pytest.mark.parametrize('expression', [
    {'kind': 'skill_input', 'source_role': 'future'},
    {'kind': 'skill_input', 'source_role': 'given', 'source_step': 'later'},
    {'kind': 'skill_input', 'source_role': 'given', 'transform_id': 'hidden'},
    {'kind': 'skill_input', 'source_role': 'given', 'expression': 'select_first'},
    {'kind': 'constant', 'constant': 'portable', 'source_role': 'given'},
    {'kind': 'tool_output', 'source_role': 'given', 'source_step': 'later'},
])
def test_entry_rejects_future_or_malformed_references(expression):
    entry = {'conditions': [], 'grounding_constraints': [{
        'constraint_id': 'authored', 'kind': 'argument_concrete',
        'argument_mapping': {'value': expression}}]}
    with pytest.raises(ValueError):
        normalize_entry_contract(entry, ['given'])


def test_P02_formal_numbered_role_is_not_episode_literal():
    guide = {'steps': ['Inspect $item_1 and verify the declared effect.'], 'notes': []}
    assert normalize_guideline(guide, formal_roles=['item_1']) == guide
    with pytest.raises(ValueError):
        normalize_guideline(guide, formal_roles=['other'])
    with pytest.raises(ValueError):
        normalize_guideline({'steps': ['Go to cabinet_7.']}, formal_roles=['item_1'])
    atomic = SimpleNamespace(inputs=[ParameterSpec('item_1', 'entity')], outputs=[], guideline=guide)
    assert guidance_view(atomic) == {'soft_reference': True, 'guidance_absent': False, **guide}


def test_D01_D02_catalog_identity_survives_but_existence_and_affordance_do_not():
    store = GroundingEvidenceStore()
    store.replace_action_catalog([HarnessActionSpec('a', 0, 'TAKE', {'object': 'observed_1'}, 'take observed 1', 'take observed 1', {})], 0)
    expr = BindingExpression(BindingExprKind.SKILL_INPUT, source_role='chosen')
    concrete = GroundingConstraint('identity', GroundingConstraintKind.ARGUMENT_CONCRETE,
        argument_mapping={'object': expr})
    store.replace_action_catalog([], 1)
    assert store.match_constraint(concrete, {'chosen': 'observed_1'}, 1)
    assert not store.match_constraint(concrete, {'chosen': 'unknown_1'}, 1)
    assert not store.match_constraint(concrete, {}, 1)
    assert not store.match_constraint(replace(concrete, kind=GroundingConstraintKind.ARGUMENT_EXISTS), {'chosen': 'observed_1'}, 1)
    assert not store.match_constraint(replace(concrete, kind=GroundingConstraintKind.HARNESS_AFFORDANCE, action_type='TAKE'), {'chosen': 'observed_1'}, 1)


def test_R02_old_configuration_is_readable_but_cannot_execute(tmp_path):
    config = _system_config(tmp_path)
    config['repair_revision'] = 'R10.1'
    with AtomicSkillGraphSystem(config, harness=FakeHarness()) as system:
        digest = system.knowledge_digest()
        with pytest.raises(ValueError, match='protocol migration'):
            system.run_task(fake_task('old', 'apple_1'))
        assert digest == system.knowledge_digest()
        assert not system.usage.events


def test_V04_parameter_schema_preserves_types_and_optionality():
    schema = parameter_schema([ParameterSpec('count', 'integer'), ParameterSpec('ready', 'boolean', required=False)])
    assert schema == {'type': 'object', 'properties': {'count': {'type': 'integer'}, 'ready': {'type': 'boolean'}},
                      'required': ['count'], 'additionalProperties': False}


def test_semantic_argument_presence_does_not_satisfy_authored_evidence_requirement():
    from atomic_skillgraph.tooling.entry_contract import check_tool_entry
    tool = SimpleNamespace(signature=parameter_schema([ParameterSpec('term', 'entity')]),
        interface={'entry_contract': {'conditions': [], 'grounding_constraints': []}})
    evidence = GroundingEvidenceStore()
    assert check_tool_entry(tool, {'term': 'category'}, None, evidence, 0).passed
    assert not check_tool_entry(tool, {}, None, evidence, 0).passed
    tool.interface['entry_contract']['grounding_constraints'] = [{
        'constraint_id': 'author_requires_evidence', 'kind': 'argument_exists',
        'required_resolution': 'semantic',
        'argument_mapping': {'value': {'kind': 'skill_input', 'source_role': 'term'}}}]
    from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
    result = check_tool_entry(tool, {'term': 'category'}, object.__new__(AlfWorldAdapter), evidence, 0)
    assert not result.passed
    assert result.failure_codes == ['tool_entry_constraint_unsatisfied']


@pytest.mark.parametrize('kind,verifier,allowed', [
    ('argument_exists','',True),('argument_concrete','',True),('harness_affordance','',True),
    ('current_context','agent.at_location',False),('custom_adapter','invented',False),
    ('argument_exists','invented',False),
])
def test_formal_constraint_schema_static_and_runtime_agree(tmp_path,kind,verifier,allowed):
    from atomic_skillgraph.agents.protocol import validate_schema_instance, SchemaValidationError
    from atomic_skillgraph.agents.structured_submission import TOOL_PROPOSAL_SCHEMA
    from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
    from atomic_skillgraph.tooling.entry_contract import check_tool_entry
    from atomic_skillgraph.tooling.proposal import tool_proposal_from_dict
    from test_r1021_i import source_case, author
    from types import SimpleNamespace
    system,ctx,provider,c,n=source_case(tmp_path)
    try:
        provider.choose=lambda r,_:author(r,create=True)
        built,_=system._build_tool_for_occurrence(c,system._canonical_atomic_for_occurrence(c),n,ctx.trace_builder.trace,source_task=ctx.task)
        constraint={'constraint_id':'authored','kind':kind,'verifier_id':verifier,
            'action_type':'GO_TO','argument_mapping':{'destination':{'kind':'skill_input','source_role':'container'}}}
        schema=TOOL_PROPOSAL_SCHEMA['properties']['entry_contract']['properties']['grounding_constraints']['items']
        if allowed: validate_schema_instance(constraint,schema)
        else:
            with pytest.raises(SchemaValidationError): validate_schema_instance(constraint,schema)
        harness=object.__new__(AlfWorldAdapter)
        assert harness.supports_constraint(kind,verifier) is allowed
        raw=author(provider.requests[-1],create=True)[1]
        raw['entry_contract']['grounding_constraints']=[constraint]
        system.harness.supports_constraint=harness.supports_constraint
        static=system.tool_static_validator.validate_proposal(tool_proposal_from_dict(raw),built.atomic,system.harness)
        assert ('tool_entry_constraint_unsupported' not in static.failure_codes) is allowed
        store=GroundingEvidenceStore()
        store.replace_action_catalog([HarnessActionSpec('go',0,'GO_TO',{'destination':'cabinet_1'},'go','go',{})],0)
        tool=copy.deepcopy(built.tool);tool.interface['entry_contract']=raw['entry_contract']
        assert check_tool_entry(tool,{'container':'cabinet_1'},harness,store,0).passed is allowed
    finally: system.close()


def test_L02_entry_contract_changes_executable_identity_and_failure_cache(tmp_path):
    from test_r102_execution import _opened, _prepare
    from atomic_skillgraph.evolution.aligner import _tool_signature
    from atomic_skillgraph.runtime.invocation_transaction import execution_cache_key
    system, ctx, occurrence, invocations, _ = _opened(tmp_path)
    try:
        before = invocations[0]
        changed = copy.deepcopy(before)
        changed.tools[0].interface['entry_contract']['grounding_constraints'] = [{
            'constraint_id': 'given', 'kind': 'argument_concrete',
            'argument_mapping': {'object': {'kind': 'skill_input', 'source_role': 'object'}}}]
        assert before.tools[0].artifact == changed.tools[0].artifact
        assert _tool_signature(before.tools[0]) != _tool_signature(changed.tools[0])
        args = _prepare(system, ctx, occurrence, before).normalized_arguments
        assert execution_cache_key(before, args, occurrence, ctx) != execution_cache_key(changed, args, occurrence, ctx)
    finally:
        system.close()


@pytest.mark.parametrize('representation', ['structured', 'dollar', 'constant'])
def test_entry_json_roundtrip_checks_authored_given_values_against_actual_facts(representation):
    import json
    from atomic_skillgraph.harness.alfworld import AlfWorldValidatorChannel
    from atomic_skillgraph.tooling.entry_contract import check_tool_entry
    validator = AlfWorldValidatorChannel()
    harness = SimpleNamespace(validator_channel=lambda: validator)
    given = {'item': 'parcel_1', 'place': 'surface_1'}
    def reference(role):
        if representation == 'structured':
            return {'kind': 'skill_input', 'source_role': role}
        if representation == 'constant':
            return {'kind': 'constant', 'constant': given[role]}
        return '$' + role
    # Exercise the persisted JSON form emitted by ToolBuilder, not only $roles.
    entry = normalize_entry_contract({'conditions': [
        {'predicate': 'agent.holds', 'args': {'object': reference('item')}},
        {'predicate': 'agent.at_location', 'args': {'location': reference('place')}}],
        'grounding_constraints': []}, given)
    tool = SimpleNamespace(signature=parameter_schema([ParameterSpec(r, 'entity') for r in given]),
        interface={'entry_contract': json.loads(json.dumps(entry))})
    evidence = GroundingEvidenceStore()
    assert not check_tool_entry(tool, given, harness, evidence, 0).passed
    for revision, (action, arguments) in enumerate([
        ('TAKE', {'object': 'parcel_1', 'source': 'box_1'}),
        ('GO_TO', {'destination': 'surface_1'})], 1):
        validator.record(HarnessActionSpec(str(revision), revision - 1, action, arguments, '', '', {}),
                         accepted=True, revision=revision, done=False, won=False)
    facts_before = copy.deepcopy(validator._facts)
    assert check_tool_entry(tool, given, harness, evidence, 2).passed
    assert validator._facts == facts_before and validator.revision == 2
    if representation != 'constant':
        assert not check_tool_entry(tool, {**given, 'item': 'parcel_2'}, harness, evidence, 2).passed
    validator.record(HarnessActionSpec('3', 2, 'GO_TO', {'destination': 'elsewhere_1'}, '', '', {}),
                     accepted=True, revision=3, done=False, won=False)
    assert not check_tool_entry(tool, given, harness, evidence, 3).passed
