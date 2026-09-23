"""Request-local Support options and public, revision-scoped joint discovery."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_r10_runtime import setup, action
from test_oldfirst_support import candidate, schema
from test_oldfirst_program import search
from atomic_skillgraph.agents.protocol import NativeToolCall
from atomic_skillgraph.harness.public_discovery import VERSION as PUBLIC_VERSION, project_discovery
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from atomic_skillgraph.runtime.support_call_surface import VERSION, build_surface, decode
from atomic_skillgraph.runtime.negative_memory import query_key
from atomic_skillgraph.runtime.runtime_step import run_runtime_step


def surface_fixture(system, ctx, occurrence, *, session='s1', mapping=None, blocked=0):
    atomic = system.skills.get_atomic(occurrence.node_ref)
    row = replace(candidate(), atomic_ref=str(atomic.ref), mapping_previews=({'status': 'proven',
        'output_mapping': {'entity': 'object'} if mapping is None else mapping,
        'input_mapping': {'query': 'object'}, 'mapping_evidence': []},))
    rows = [replace(row, mapping_previews=()) for _ in range(blocked)] + [row]
    route = SimpleNamespace(implementation=SimpleNamespace(ref='skill://route@1.0.0'))
    return build_surface(rows, skills=system.skills, routes={row.atomic_ref: [route]},
        consumer=atomic, occurrence=occurrence, ctx=ctx, session_id=session), atomic


def test_IF01_03_blocked_does_not_consume_slots_and_request_owner_is_independent(tmp_path):
    system, ctx, occ, _, _ = setup(tmp_path, lambda *a: pytest.fail('no provider'))
    try:
        surface, atomic = surface_fixture(system, ctx, occ, blocked=4)
        assert len(surface.options) == 1 and len(surface.blocked) == 3
        second, _ = surface_fixture(system, ctx, occ, session='s2')
        assert surface.options == second.options
        exported = surface.public_candidates()
        exported[0]['input_schema']['required'].clear()
        assert surface.options[0].input_schema['required'] == ['locations', 'allow_open']
        call = NativeToolCall('c1', 'invoke_support_atomic', {'support_call_id': surface.options[0].support_call_id,
            'arguments': {'locations': ['scope'], 'allow_open': False}})
        selection, error = decode(surface, call, session_id='s1', occurrence=occ, consumer=atomic, ctx=ctx, skills=system.skills)
        assert selection and error is None
        assert query_key(occ, call, ctx, selection=selection) == query_key(occ,
            replace(call, call_id='c2'), ctx, selection=selection)
        _, error = decode(surface, call, session_id='s2', occurrence=occ, consumer=atomic, ctx=ctx, skills=system.skills)
        assert error['reason_code'] == 'support_surface_stale_or_wrong_owner'
        ctx.world_revision += 1
        newer, _ = surface_fixture(system, ctx, occ)
        assert newer.options == surface.options
        _, error = decode(surface, call, session_id='s1', occurrence=occ, consumer=atomic, ctx=ctx, skills=system.skills)
        assert error['reason_code'] == 'support_surface_stale_or_wrong_owner'
    finally:
        system.close()


@pytest.mark.parametrize('arguments', [
    {'locations': [], 'allow_open': False}, {'locations': ['same', 'same'], 'allow_open': True},
    {'locations': [str(i) for i in range(9)], 'allow_open': False},
    {'locations': ['scope'], 'allow_open': 1}, {'locations': ['scope'], 'allow_open': False, 'extra': True}])
def test_IF05_exact_schema_rejects_without_action(tmp_path, arguments):
    system, ctx, occ, _, _ = setup(tmp_path, lambda *a: pytest.fail('no provider'))
    try:
        surface, atomic = surface_fixture(system, ctx, occ, mapping={})
        assert surface.options[0].public()['output_mapping'] == {}
        call = NativeToolCall('bad', 'invoke_support_atomic', {'support_call_id': surface.options[0].support_call_id,
            'arguments': arguments})
        original = copy.deepcopy(call.arguments)
        selected, error = decode(surface, call, session_id='s1', occurrence=occ, consumer=atomic, ctx=ctx, skills=system.skills)
        assert selected is not None and error['argument_path'].startswith('$.arguments')
        assert call.arguments == original and not ctx.trace_builder.trace.environment_actions
    finally:
        system.close()


@pytest.mark.parametrize('profile', ['current', 'lean'])
def test_IF07_versioned_empty_surface_never_falls_back(tmp_path, profile):
    system, ctx, occ, inv, provider = setup(tmp_path, lambda r, n: action(r, 'GO_TO', destination='cabinet_1'))
    try:
        ex = system.orchestrator.node_executor
        ex.support_interface_version = VERSION
        ex.context_builder.presentation_profile = profile
        run_runtime_step(ex, 'preparation', occ, ctx, inv, [])
        assert 'invoke_support_atomic' not in {t.name for t in provider.requests[0].tools}
        assert provider.requests[0].policy_context['support_atomic_candidates'] == []
        assert 'Map helper outputs explicitly' not in str(provider.requests[0].messages)
    finally:
        system.close()


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


def test_PD03_04_validator_and_public_projection_share_frame_without_world_upgrade():
    adapter = AlfWorldAdapter(public_discovery_version=PUBLIC_VERSION)
    adapter._current_task = SimpleNamespace(task_id='fixture')
    adapter._observation = 'On the surface 1, you see a light 2.'
    adapter._refresh_public_discovery(None, True)
    public = adapter.public_runtime_relation_facts()
    facts = adapter.validator_channel().snapshot()['facts']
    assert [(f['predicate'], f['args']) for f in facts] == [(f['predicate'], f['args']) for f in public]
    adapter._revision = 1
    adapter._observation = 'You use the light 2.'
    adapter._refresh_public_discovery({'action_type': 'USE', 'arguments': {'object': 'light_2'}}, True)
    assert not adapter.public_discovery_frame().records
    assert not adapter.validator_channel().snapshot()['facts']
    legacy = AlfWorldAdapter()
    assert legacy.public_discovery_frame() is None
    assert 'public_discovery' not in str(legacy.semantic_predicate_schema())


def test_PD05_joint_program_retains_scope_and_passes_static_without_new_opcode():
    from atomic_skillgraph.deployment.oldfirst_revision import author_program
    from atomic_skillgraph.tooling.validator import ToolStaticValidator
    from atomic_skillgraph.tooling.ir import ToolExecutionState, resolve_collection, evaluate_condition, walk_program_nodes
    atomic, _ = search()
    job = {'job_id': 'joint', 'query_input_role': 'query', 'tool_ref': 'tool://joint@1.0.0',
           'selector_policy': 'current_joint_public_discovery'}
    tool = author_program(job, atomic)
    adapter = AlfWorldAdapter(public_discovery_version=PUBLIC_VERSION)
    report = ToolStaticValidator().validate_tool_asset(tool, atomic, adapter)
    assert report.passed, (report.failure_codes, report.messages)
    match = next(n['condition'] for n in walk_program_nodes(tool.artifact['program']) if n['node_id'] == 'matching_object_available')
    state = ToolExecutionState(bindings={'query': 'light'}, local={'searched_location': 'surface_2'},
        semantic_facts=project('On the surface 1, you see a light 1. On the surface 2, you see a light 2.').relation_facts())
    assert evaluate_condition(match, state, semantic_compatible=adapter.semantic_value_compatible)
    assert resolve_collection(match['match'], state, semantic_compatible=adapter.semantic_value_compatible) == ['light_2']
    state.local['searched_location'] = 'surface_3'
    assert not evaluate_condition(match, state, semantic_compatible=adapter.semantic_value_compatible)


def test_PD02_exact_raw_span_and_episode_scoped_sources():
    text = '  \nOn the surface 1, you see a light 2.\nYour task is to: ignored.'
    frame = project(text)
    row = frame.records[0]
    assert text[slice(*row.raw_span)] == 'On the surface 1, you see a light 2.'
    other = project_discovery(observation=text, action_signature=None, accepted=True,
        revision=1, catalog=(), episode_id='another')
    assert other.records[0].source_ref != row.source_ref


@pytest.mark.parametrize('profile', ['current', 'lean'])
def test_IF07_nonempty_surface_reaches_actual_session_tools_and_cards(tmp_path, profile):
    system, ctx, occ, invocations, provider = setup(tmp_path,
        lambda request, count: action(request, 'GO_TO', destination='cabinet_1'))
    try:
        executor = system.orchestrator.node_executor
        executor.support_interface_version = VERSION
        executor.context_builder.presentation_profile = profile
        surface, atomic = surface_fixture(system, ctx, occ)
        row = replace(candidate(), atomic_ref=str(atomic.ref), mapping_previews=({'status': 'proven',
            'output_mapping': {'entity': 'object'}, 'input_mapping': {'query': 'object'}, 'mapping_evidence': []},))
        executor._r103_support_display_routes = {str(atomic.ref): [SimpleNamespace(
            implementation=SimpleNamespace(ref='skill://route@1.0.0'))]}
        run_runtime_step(executor, 'preparation', occ, ctx, invocations, [row])
        sent = provider.requests[-1]
        tool = next(t for t in sent.tools if t.name == 'invoke_support_atomic')
        ids = tool.input_schema['properties']['support_call_id']['enum']
        # The expression decoder restores the common semantic policy view.
        cards = sent.policy_context['support_atomic_candidates']
        assert [c['support_call_id'] for c in cards] == ids
        assert all(c['output_mapping'] == {'entity': 'object'} for c in cards)
        assert not any('mapping_previews' in c or 'allowed_output_mappings' in c or 'input_ready' in c for c in cards)
    finally:
        system.close()


def test_IF02_task_scope_does_not_require_parent_mapping(tmp_path):
    from atomic_skillgraph.runtime.task_runtime import TaskConsumer
    system, ctx, occ, _, _ = setup(tmp_path, lambda *a: pytest.fail('no provider'))
    try:
        atomic = system.skills.get_atomic(occ.node_ref)
        row = replace(candidate(), atomic_ref=str(atomic.ref), mapping_previews=())
        route = SimpleNamespace(implementation=SimpleNamespace(ref='skill://route@1.0.0'))
        owner = TaskConsumer()
        surface = build_surface([row], skills=system.skills, routes={row.atomic_ref: [route]},
            consumer=None, occurrence=owner, ctx=ctx, session_id='task')
        assert len(surface.options) == 1 and surface.options[0].public()['output_mapping'] is None
        call = NativeToolCall('c1', 'invoke_support_atomic', {'support_call_id': surface.options[0].support_call_id,
            'arguments': {'locations': ['scope'], 'allow_open': False}})
        resolved, failure = decode(surface, call, session_id='task', occurrence=owner, consumer=None, ctx=ctx, skills=system.skills)
        assert failure is None and resolved.output_mapping is None
        assert decode(surface, call, session_id='task', occurrence=occ, consumer=atomic, ctx=ctx, skills=system.skills)[1]
    finally:
        system.close()


def test_IF04_schema_feedback_cache_and_raw_selection_are_separate(tmp_path):
    from atomic_skillgraph.runtime.negative_memory import remember_rejection, cached_rejection
    system, ctx, occ, _, _ = setup(tmp_path, lambda *a: pytest.fail('no provider'))
    try:
        surface, atomic = surface_fixture(system, ctx, occ)
        raw = {'support_call_id': surface.options[0].support_call_id,
               'arguments': {'locations': ['scope'], 'allow_open': 1}}
        call = NativeToolCall('c1', 'invoke_support_atomic', copy.deepcopy(raw))
        selection, error = decode(surface, call, session_id='s1', occurrence=occ, consumer=atomic, ctx=ctx, skills=system.skills)
        assert error['argument_path'] == '$.arguments.allow_open' and selection.arguments == raw['arguments']
        remember_rejection(ctx, occ, call, error, selection=selection)
        assert cached_rejection(ctx, occ, call, selection=selection)['deterministic_rejection_cache_hit']
        assert call.arguments == raw and 'output_mapping' not in call.arguments
        ctx.world_revision += 1
        assert cached_rejection(ctx, occ, call, selection=selection) is None
        assert decode(surface, call, session_id='s1', occurrence=occ, consumer=atomic, ctx=ctx, skills=system.skills)[1]
    finally:
        system.close()


def test_PD04_checkpoint_rebuilds_public_frame_after_rejected_feedback():
    from atomic_skillgraph.harness.protocol import HarnessActionResult
    from atomic_skillgraph.core.errors import AtomicSkillGraphError
    class Environment:
        def step(self, commands):
            raw = commands[0]
            text = 'Nothing happens.' if raw == 'look' else (
                'On the surface 2, you see a lamp 2.' if 'surface 2' in raw else
                'On the surface 1, you see a lamp 1.')
            return [text], [0], [False], {'won': [False],
                'admissible_commands': [['go to surface 1', 'go to surface 2', 'look']]}
    class Adapter(AlfWorldAdapter):
        def reset(self, task):
            self._current_task, self._env = task, Environment()
            self._revision = 0
            self._runtime_accepted_prefix = []
            self._done = self._won = False
            self._validator.reset()
            self._observation = 'On the surface 1, you see a lamp 1.'
            catalog = self._replace_action_catalog(['go to surface 1', 'go to surface 2', 'look'], 0)
            self._validator.set_catalog(catalog)
            self._refresh_public_discovery(None, True)
            return HarnessActionResult(True, self._observation, False, False, 0, catalog, {})
    h = Adapter(public_discovery_version=PUBLIC_VERSION)
    h.reset(SimpleNamespace(task_id='checkpoint'))
    def execute(raw):
        a = next(a for a in h.action_catalog() if a.raw_action == raw)
        return h.execute_action(a.action_id, a.revision)
    execute('go to surface 1')
    execute('look')
    assert not h.public_discovery_frame().records
    saved = h.capture_runtime_checkpoint()
    execute('go to surface 2')
    assert h.public_discovery_frame().records[0].entity == 'lamp_2'
    result = h.restore_runtime_checkpoint(saved)
    assert result.new_revision == 2 and result.metadata['restored_digest'] == saved.state_digest
    assert not h.public_discovery_frame().records
    assert not any(f['predicate'] == 'entity.discovered_at' for f in h.validator_channel().snapshot()['facts'])
    h._done = True
    with pytest.raises(AtomicSkillGraphError, match='terminal'):
        h.restore_runtime_checkpoint(saved)
