"""Retained public observation parsing checks from release4 interfaces."""
from atomic_skillgraph.harness.public_discovery import VERSION as PUBLIC_VERSION, project_discovery

def project(text, revision=1, catalog=(), accepted=True):
    return project_discovery(observation=text, action_signature=None, accepted=accepted,
        revision=revision, catalog=catalog, episode_id='fixture')


def test_PD01_02_explicit_flat_sources_only_and_conflicts_fail_closed():
    frame = project('You arrive. On the surface 1, you see an item 1, and a light 2.')
    assert {(r.entity, r.location) for r in frame.records} == {('item_1', 'surface_1'), ('light_2', 'surface_1')}
    assert all(f['effect_domain'] == 'evidence' for f in frame.relation_facts())
    assert project('The container 1 is open. In it, you see a light 2.').records[0].relation_kind == 'in'
    assert project('On the surface 1, you see nothing.').inspected_scopes[0].status == 'complete_listing'
    for text in ('You use the light 2.', 'You arrive at surface 1.',
            'Your task is to: On the surface 1, you see a light 2.',
            'You see a surface 1 and a light 2.', 'In it, you see a light 2.',
            'On the surface 1, you see a box 2 (containing a light 2).'):
        assert not project(text).records
    assert project('The container 1 is closed.').inspected_scopes[0].status == 'inaccessible'
    conflict = project('On the surface 1, you see a light 2. On the surface 3, you see a light 2.')
    assert not conflict.records and conflict.conflicts == (('light_2', ('surface_1', 'surface_3')),)
    assert not project('On the surface 1, you see a light 2.', accepted=False).records


def test_PD02_exact_raw_span_and_episode_scoped_sources():
    text = '  \nOn the surface 1, you see a light 2.\nYour task is to: ignored.'
    frame = project(text)
    row = frame.records[0]
    assert text[slice(*row.raw_span)] == 'On the surface 1, you see a light 2.'
    other = project_discovery(observation=text, action_signature=None, accepted=True,
        revision=1, catalog=(), episode_id='another')
    assert other.records[0].source_ref != row.source_ref
