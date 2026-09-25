import copy
from dataclasses import replace
from types import SimpleNamespace
import pytest

from atomic_skillgraph.harness import scienceworld_actions as actions
from atomic_skillgraph.harness.scienceworld import ScienceWorldValidatorChannel
from atomic_skillgraph.harness.scienceworld_public import fact, normalize_listing
from atomic_skillgraph.harness.protocol import HarnessActionSpec
from atomic_skillgraph.harness.public_discovery import project_discovery
from atomic_skillgraph.agents.protocol import validate_schema_instance
from atomic_skillgraph.runtime.support_call_surface import build_surface

def test_public_local_observation_survives_corroborating_catalog_and_ambiguity_cannot_certify_absence():
    def item(location):
        return HarnessActionSpec(location, 1, 'TAKE', {'object': 'light_2', 'source': location}, '', '', {})
    def project(text, catalog):
        return project_discovery(observation=text, action_signature=None, accepted=True, revision=1,
                                 catalog=catalog, episode_id='test')
    frame = project('On the surface 1, you see a light 2.', [item('surface_1'), item('surface_2')])
    assert {(r.entity,r.location) for r in frame.records} == {('light_2','surface_1')}
    assert not frame.unresolved_relations
    conflict = project('On the surface 1, you see a light 2.', [item('surface_2')])
    assert not conflict.records and conflict.unresolved_relations
    assert conflict.inspected_scopes[0].status == 'ambiguous'

def test_support_identity_deduplicates_proofs_not_routes(tmp_path):
    from test_r10_runtime import setup
    from test_oldfirst_support import candidate
    system, ctx, occurrence, _, _ = setup(tmp_path, lambda *a: pytest.fail('no API'))
    try:
        atomic = system.skills.get_atomic(occurrence.node_ref)
        preview = {'status':'proven','output_mapping':{'entity':'object'},'input_mapping':{'query':'object'},
                   'anchor_inputs':['object'],'mapping_evidence':['proof1']}
        row = replace(candidate(), atomic_ref=str(atomic.ref), mapping_previews=(preview,
                      {**preview,'mapping_evidence':['proof2']}))
        routes = {row.atomic_ref: [SimpleNamespace(implementation=SimpleNamespace(ref='skill://route@1.0.0'))]}
        kwargs = dict(skills=system.skills,routes=routes,consumer=atomic,occurrence=occurrence,ctx=ctx,session_id='s')
        surface = build_surface([row], **kwargs)
        assert len(surface.options) == 1
        assert surface.options[0].public()['proof_alternatives'] == [['proof1'],['proof2']]
        assert 'proof_alternatives' not in surface.public_candidates()[0]
        routes[row.atomic_ref][0].implementation.ref = 'skill://other@1.0.0'
        assert build_surface([row], **kwargs).options[0].support_call_id != surface.options[0].support_call_id
    finally:
        system.close()

def test_exact_tuples_private_ids_stale_and_ambiguous():
    raw = [{'action':'move egg to bowl','template_id':9,'obj_ids':[4,5]},
           {'action':'move cup to table','template_id':9,'obj_ids':[6,7]},
           {'action':'wait1','template_id':18,'obj_ids':[]},
           {'action':'look around','template_id':3,'obj_ids':[]}]
    cat = actions.catalog(raw, 3)
    assert len(cat) == 3 and actions.catalog(list(reversed(raw)), 3) == cat
    assert actions.compact(cat)['MOVE']['tuples'] == [['cup','table'],['egg','bowl']]
    schema = actions.action_schema(cat)
    validate_schema_instance({'action_type':'WAIT1','arguments':{}},schema)
    with pytest.raises(ValueError):
        validate_schema_instance({'action_type':'WAIT1','arguments':{},'support_call_id':'bad'},schema)
    for kwargs in [('MOVE',{'entity':'egg','container':'table'},3),('WAIT1',{},2)]:
        with pytest.raises(ValueError): actions.resolve(cat,*kwargs)
    duplicate = [cat[0],replace(cat[0],action_id='different',raw_action='other')]
    with pytest.raises(ValueError,match='ambiguous'): actions.resolve(duplicate,cat[0].action_type,cat[0].arguments,3)
    assert actions.parse('pour paint in cup into bowl') == ('POUR',{'source':'paint in cup','destination':'bowl'})
    assert actions.parse('disconnect terminal 1') == ('DISCONNECT',{'entity':'terminal 1'})

def test_evidence_output_has_real_witness_not_null():
    validator = ScienceWorldValidatorChannel()
    validator.revision = 3
    effect = {'predicate':'measurement.observed','args':{'subject':'$target','instrument':'$meter','evidence':None}}
    request = {'effects':[effect], 'known_bindings':{'target':'water','meter':'thermometer'}, 'current_revision':3}
    assert not validator.resolve_atomic_effect(request).passed
    validator.facts = [fact('measurement.observed',{'subject':'water','instrument':'thermometer','evidence':'observed:3'},3)]
    assert validator.resolve_atomic_effect(request).passed
    assert not validator.resolve_atomic_effect({**request,'current_revision':2}).passed
    assert not validator.validate_atomic_effect({'effects':[effect],'bindings':{}}).passed
    changed = copy.deepcopy(request)
    changed['known_bindings']['target'] = 'ice'
    assert not validator.resolve_atomic_effect(changed).passed

def test_only_sibling_listing_order_is_normalized():
    assert normalize_listing('Room\n\ta\n\tb\nEnd') == normalize_listing('Room\n\tb\n\ta\nEnd')
    assert normalize_listing('first\nsecond') != normalize_listing('second\nfirst')
    assert normalize_listing('Room\n\ta\n\ta') != normalize_listing('Room\n\ta')


def test_public_listing_heads_and_measurement_cannot_be_inferred_from_a_mention():
    from atomic_skillgraph.harness.scienceworld_public import listed_identity, action_evidence
    assert listed_identity('a thermometer, currently reading a temperature of 10 degrees celsius', {'thermometer'}) == 'thermometer'
    assert listed_identity('a counter. On the counter is: a bowl', {'counter', 'bowl'}) == 'counter'
    assert listed_identity('a counter. On the counter is: a bowl', {'bowl'}) is None
    selected = HarnessActionSpec('measure', 1, 'USE', {'instrument':'thermometer','target':'sample'}, '', '', {})
    yes, _ = action_evidence(selected, 'the thermometer measures a temperature of -2.5 degrees celsius', 2, 'episode')
    no, _ = action_evidence(selected, 'The thermometer cannot measure this object.', 2, 'episode')
    assert any(f['predicate'] == 'measurement.observed' for f in yes)
    assert not any(f['predicate'] == 'measurement.observed' for f in no)


def test_current_evidence_projection_keeps_history_addressable():
    channel = ScienceWorldValidatorChannel()
    channel.facts = [fact('time.progressed', {'evidence':'first'},1), fact('time.progressed', {'evidence':'second'},2)]
    assert [f['args']['evidence'] for f in channel.snapshot()['facts']] == ['second']
    assert channel.validate_atomic_effect({'effects':[{'predicate':'time.progressed','args':{'evidence':'first'}}]}).passed
    assert len(channel.facts) == 2


def test_official_goal_authority_is_validator_only():
    channel = ScienceWorldValidatorChannel()
    goal = fact('scienceworld.matter_goal_satisfied', {'task':'public task'}, 4)
    goal.update(source_kind='official_score', effect_domain='world')
    channel.facts = [goal]
    assert channel.snapshot()['facts'] == []
    assert channel.validate_atomic_effect({'effects':[{'predicate':goal['predicate'],'args':goal['args']}]}).passed


def test_offered_but_rejected_use_and_broken_device_do_not_prove_effects():
    from atomic_skillgraph.harness.scienceworld_public import action_accepted, action_evidence
    assert not action_accepted("I'm not sure how to use those two things together.")
    assert not action_accepted('Ambiguous request: choose one')
    assert action_accepted('The light appears broken, and can\'t be activated or deactivated.')
    action = HarnessActionSpec('on', 1, 'ACTIVATE', {'device':'light'}, '', '', {})
    no, _ = action_evidence(action, 'The light appears broken, and can\'t be activated or deactivated.', 2, 'ep')
    yes, _ = action_evidence(action, 'The light is now activated.', 2, 'ep')
    assert not any(f['predicate'] == 'device.active' for f in no)
    assert any(f['predicate'] == 'device.active' for f in yes)
