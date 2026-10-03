"""Mechanism.v2 boundaries: no model, no benchmark answer recipes."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.mechanism_profile import VERSION, resolve, bind
from atomic_skillgraph.agents.decision_frame import project_decision_frame, expand_frame
from atomic_skillgraph.agents.protocol import validate_schema_instance, SchemaValidationError
from atomic_skillgraph.evolution.shared_contract import instantiate, add_program_controls, schema
from atomic_skillgraph.evolution.extraction_view import CONTRACT_FIELDS, source_reference_view
from atomic_skillgraph.evolution.preparation_realizer import uncovered_intervals, learning_view
from atomic_skillgraph.governance.ledger import EvidenceEvent, EvidenceEventType
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from atomic_skillgraph.harness.public_discovery import VERSION as DISCOVERY
from atomic_skillgraph.runtime.support_call_surface import VERSION as SUPPORT
from atomic_skillgraph.planner.requirement_agent import REQUIREMENT_SCHEMA, requirement_bundle_from_dict
from atomic_skillgraph.planner.semantic_vocabulary import requirement_schema, check_bundle
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.runtime.loop_guard import ActionLoopGuard
from atomic_skillgraph.runtime.node_executor import _loop_progress


def profile_config():
    return {'mechanism_profile': VERSION, 'repair_revision': 'R10.3',
        'runtime': {'support_interface_version': SUPPORT},
        'harness': {'adapter': 'alfworld_v3', 'public_discovery_version': DISCOVERY}}


def test_profile_actual_bind_and_legacy_reader_isolation(tmp_path):
    cfg = profile_config()
    for section, field in [('runtime', 'support_interface_version'), ('harness', 'public_discovery_version')]:
        wrong = copy.deepcopy(cfg)
        del wrong[section][field]
        with pytest.raises(ValueError, match='requires explicit'):
            resolve(wrong)
    with StateDatabase(tmp_path / 'state.sqlite3', r103=True) as db:
        system = SimpleNamespace(config=cfg, readonly=False, database=db)
        profile = bind(system)
        assert profile['support_interface_version'] == SUPPORT
        system.readonly = True
        assert bind(system) == profile
        system.config = {}
        with pytest.raises(ValueError, match='reader'):
            bind(system)


@pytest.mark.parametrize('scope', ['task', 'node'])
def test_frame_preserves_goal_repeat_identity_catalog_feedback(scope):
    fact = {'predicate': 'agent.holds', 'args': {'object': 'creditcard_1'}, 'constraints': ['same identity']}
    raw = {'task_goal': 'Put TWO distinct creditcards on the same dresser, then keep them there.',
        'current_state_snapshot': {'current_atomic': {'summary': 'Deliver one card', 'effects': [fact]},
            'confirmed': [fact] * 3, 'downstream_obligations': {'distinct': ['creditcard_2']}},
        'task_runtime_frame': {'task_contract': {'effects': [fact]}, 'last_step': {'diagnostic': 'task feedback'}},
        'current_observation': 'Unknown qualifier remains verbatim.',
        'execution_frame': {'repeat': {'distinct': ['creditcard_2']}, 'last_step': {'diagnostic': 'node feedback'}},
        'source_proof': {'expanded': ['audit only'] * 500},
        'current_action_catalog': [{'action_id': 'a1', 'action_type': 'HEAT', 'arguments': {'object': 'apple_1'}}]}
    before = copy.deepcopy(raw)
    result, audit = project_decision_frame(raw, scope=scope)
    frame = expand_frame(result)
    assert raw == before and audit['after_body_utf8_bytes'] < audit['before_body_utf8_bytes']
    assert frame['goal']['task_qualifiers'] == raw['task_goal']
    assert frame['state']['confirmed'] == raw['current_state_snapshot']['confirmed']
    assert frame['goal']['downstream'] == {'distinct': ['creditcard_2']}
    assert frame['execution']['repeat'] == raw['execution_frame']['repeat']
    assert frame['feedback']['last_call'] == {'diagnostic': 'node feedback'}
    assert frame['task_runtime']['last_step'] == {'diagnostic': 'task feedback'}
    assert result['current_action_catalog'] == raw['current_action_catalog']


def test_p1_vocab_native_and_python_identical_no_invented_hold_predicate():
    harness = AlfWorldAdapter(public_discovery_version=DISCOVERY)
    native = requirement_schema(REQUIREMENT_SCHEMA, harness.semantic_predicate_schema())
    requirement = {'requirement_id': 'held', 'intent': 'hold the specified object',
        'desired_effects': [{'predicate': 'agent.holds', 'args': {'object': '$item'}, 'effect_domain': 'world'}],
        'expected_inputs': [], 'expected_outputs': [], 'precondition_hints': [],
        'semantic_variants': [], 'required': True, 'rationale': 'formal held goal'}
    body = {'requirements': [requirement], 'repeat_blocks': []}
    validate_schema_instance(body, native)
    check_bundle(requirement_bundle_from_dict(body), native)
    for predicate, args in [('object.in_hand', {'object': '$item'}), ('agent.holds', {'wrong_role': '$item'})]:
        wrong = copy.deepcopy(body)
        wrong['requirements'][0]['desired_effects'][0].update(predicate=predicate, args=args)
        with pytest.raises(SchemaValidationError):
            validate_schema_instance(wrong, native)
    for value in (42, False, {'arbitrary': 'JSON'}, {'kind': 'skill_input'},
                  {'kind': 'constant', 'constant': 42}):
        wrong = copy.deepcopy(body)
        wrong['requirements'][0]['desired_effects'][0]['args']['object'] = value
        with pytest.raises(SchemaValidationError):
            validate_schema_instance(wrong, native)
        with pytest.raises(SchemaValidationError):
            check_bundle(requirement_bundle_from_dict(wrong), native)
    bound = copy.deepcopy(body)
    bound['requirements'][0]['desired_effects'][0]['args']['object'] = {'kind': 'skill_input', 'source_role': 'item'}
    check_bundle(requirement_bundle_from_dict(bound), native)


def shared_example():
    from test_r1021_boundaries import typed_preparation_example
    from test_r1021_final import raw_proposal
    proposal, normalized = typed_preparation_example()
    raw = raw_proposal(proposal)
    shared = {'group_id': 'pair', 'rationale': 'shared exact contract',
        'candidate_contract': {k: v for k, v in raw.items() if k in CONTRACT_FIELDS},
        'source_instances': [{**{k: v for k, v in raw.items() if k not in CONTRACT_FIELDS},
            'source_slot': slot, 'source_id': slot} for slot in ('current', 'history')]}
    return shared, normalized


def test_shared_contract_one_generation_cannot_override_source_namespace():
    submitted, normalized = shared_example()
    validate_schema_instance([submitted], schema([{'group_id': 'pair'}]))
    rows = instantiate(submitted, {'current': 'current', 'history': 'history'})
    assert rows[0]['proposal']['effects'] == rows[1]['proposal']['effects']
    submitted['source_instances'][1]['source_id'] = 'current'
    with pytest.raises(ValueError, match='identity'):
        instantiate(submitted, {'current': 'current', 'history': 'history'})


def test_alias_ordinary_reuse_sidecar_namespace_transport_and_independent_candidates():
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.agents import UsageLedger
    from atomic_skillgraph.evolution.extractor_session import ExtractorSession
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    submitted, normalized = shared_example()
    _, aliases = source_reference_view(normalized)
    original = submitted['source_instances'][0]['input_provenance_refs']
    reverse = {v: k for k, v in aliases.items()}
    for value in original.values():
        value['authority_ref'] = reverse[value['authority_ref']]
    submitted['source_instances'][1]['input_provenance_refs'] = copy.deepcopy(original)
    ordinary = instantiate({**submitted, 'source_instances': [
        {**r, 'source_id': r['source_slot']} for r in submitted['source_instances']]},
        {'current': 'current', 'history': 'history'})[0]['proposal']
    body = {'occurrences': [ordinary], 'generalizations': [submitted], 'reuse_existing': []}
    provider = ScriptedAgentProvider([FakeReply.structured(body)])
    session = ReplayAgentSession(provider, system_prompt='E1', usage_ledger=UsageLedger(), usage_bucket='extractor_e1')
    batch = ExtractorSession(session, mechanism_profile={'version': VERSION}).propose_batch(normalized,
        generalization_groups=[{'group_id': 'pair', 'history': normalized}])
    assert len(batch.occurrences) == 1
    assert batch.occurrences[0].input_provenance_refs == {
        k: {**v, 'authority_ref': aliases[v['authority_ref']]} for k, v in original.items()}
    assert batch.generalizations[0]['_transport_error'] == 'extractor_history_reference_namespace_invalid'
    assert not batch.transport_rejections
    assert len(provider.requests) == 1


def test_preparation_intervals_exclude_covered_and_terminal_not_last_primitive_only():
    actions = [{'event_index': i, 'event_id': 'a' + str(i), 'accepted': True, 'done': i == 6}
        for i in range(7)]
    source = {'actions': actions, 'covered_program_event_indices': [3, 4]}
    assert [[a['event_index'] for a in span] for span in uncovered_intervals(source)] == [[0, 1, 2]]
    view = learning_view(source)
    assert view['actions'][3]['learning_responsibility'].startswith('existing successful Program')
    assert source['actions'][3] == actions[3] and 'learning_responsibility' not in actions[3]


def test_covered_program_events_identical_after_immutable_trace_readback(tmp_path):
    from atomic_skillgraph.evolution.preparation_realizer import covered_program_events
    from atomic_skillgraph.traces.schema import TraceRecord, TaskRecord, RuntimeSpan, ToolExecutionRecord
    from atomic_skillgraph.traces.store import TraceStore
    from atomic_skillgraph.core.serialization import to_primitive
    trace = TraceRecord('trace_span_roundtrip', 3, TaskRecord('generic_task', 'fixture', 'generic goal', '', 'sig'),
        {}, {}, {}, 0.0)
    trace.environment_actions = [{} for _ in range(4)]
    trace.runtime_spans = [RuntimeSpan('ok', 'program', 'node', 0, 3, None, True),
                           RuntimeSpan('failed', 'program', 'other', 3, 4, None, True)]
    trace.tool_executions = [ToolExecutionRecord('attempt', 'node', 'tool://generic@1',
        {'started': True, 'completed': True, 'atomic_effect_passed': True, 'failure_code': ''}, 'ok'),
        ToolExecutionRecord('failed_attempt', 'other', 'tool://other@1',
        {'started': True, 'completed': False, 'atomic_effect_passed': False, 'failure_code': 'failed'}, 'failed')]
    trace.metadata['runtime_rollbacks'] = [{'discarded_action_start': 1, 'discarded_action_end': 2}]
    before = to_primitive(trace)
    assert covered_program_events(trace) == [0, 2]
    store = TraceStore(tmp_path)
    store.save_atomic(trace)
    loaded = store.load(trace.trace_id)
    assert isinstance(loaded.runtime_spans[0], dict)
    assert isinstance(loaded.tool_executions[0], dict)
    assert covered_program_events(loaded) == covered_program_events(before) == [0, 2]
    assert to_primitive(trace) == before == store.load_payload(trace.trace_id)


def test_new_control_scope_public_entry_not_backfilled_historical_authority():
    from test_r1021_boundaries import typed_preparation_example, typed_atomicizer
    from atomic_skillgraph.system import AtomicSkillGraphSystem
    proposal, normalized = typed_preparation_example()
    occurrence, = typed_atomicizer().validate_and_canonicalize([proposal], normalized)
    # Construction is the same contract materializer; no source input mutation.
    system = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    system.atomicizer = typed_atomicizer()
    atomic = system._canonical_atomic_for_occurrence(occurrence)
    entry = normalized['actions'][0]['before_revision']
    normalized['boundary_authorities'].setdefault('locals', []).append({'kind': 'public_catalog', 'available_revision': entry,
        'value': 'surface_1', 'resolution': 'concrete'})
    declaration = {'parameter': {'name': 'scopes', 'semantic_type': 'list', 'required': True,
        'required_resolution': 'semantic', 'runtime_resolvable': True},
        'authorization': {'kind': 'ordered_entity_scope', 'element_semantic_type': 'entity',
            'min_items': 1, 'max_items': 4, 'unique_items': True}}
    old = copy.deepcopy(occurrence.input_bindings)
    candidate, call = add_program_controls(atomic, occurrence, [declaration], {'scopes': ['surface_1']}, normalized)
    assert occurrence.input_bindings == old and 'scopes' not in occurrence.input_provenance_refs
    assert call.input_bindings['scopes'] == ['surface_1']
    assert candidate.validator_spec['input_authorization']['scopes']['max_items'] == 4
    with pytest.raises(ValueError, match='public pre-entry'):
        add_program_controls(atomic, occurrence, [declaration], {'scopes': ['hidden_place_1']}, normalized)


def test_progress_clock_no_credit_and_novel_discovery_unblocks_loop():
    guard = ActionLoopGuard()
    ctx = SimpleNamespace(exploration_policy_view=lambda: {'found': 'cloth_1', 'revision': 1})
    before = _loop_progress(ctx, {'completed': False, 'revision': 1})
    ctx.exploration_policy_view = lambda: {'found': 'cloth_1', 'revision': 5}
    assert before == _loop_progress(ctx, {'completed': False, 'revision': 5})
    for _ in range(2):
        assert not guard.inspect(action_type='X', arguments={}, observation='same', catalog=[], progress=before).blocked
    assert guard.inspect(action_type='X', arguments={}, observation='same', catalog=[], progress=before).blocked
    assert not guard.inspect(action_type='X', arguments={}, observation='same', catalog=[],
        progress={'found': 'cloth_2'}).blocked


def test_cloth_public_identity_without_take_does_not_become_affordance():
    adapter = AlfWorldAdapter(public_discovery_version=DISCOVERY)
    adapter._current_task = SimpleNamespace(task_id='generic_public_identity')
    adapter._observation = 'The cabinet 4 is open. In it, you see a cloth 1, and a spraybottle 1.'
    adapter._refresh_public_discovery(None, True)
    frame = adapter.public_discovery_frame()
    assert any(r.entity == 'cloth_1' for r in frame.records)
    assert not any(a.action_type == 'TAKE' for a in adapter.action_catalog())
    assert 'HEAT' in {a['action_type'] for a in adapter.primitive_action_schema() if a.get('public_semantics')}


def test_qualification_rejects_same_source_and_no_online_credit(tmp_path):
    from test_r103_generalization import sources
    from atomic_skillgraph.deployment.qualification import verify
    from atomic_skillgraph.core.refs import content_hash
    from atomic_skillgraph.core.serialization import atomic_create_json
    from atomic_skillgraph.governance.projections import LifecycleProjection
    # Malformed certificates never enter the ledger even with recomputed hashes.
    with StateDatabase(tmp_path / 'state.sqlite3', r103=True) as db:
        bind(SimpleNamespace(config=profile_config(), readonly=False, database=db))
        certificate = {'artifact_hashes': {}, 'version': 'skillcompiler.train-qualification.v2', 'cases': []}
        digest = content_hash(certificate)
        atomic_create_json(tmp_path / 'artifacts/deployment_qualification' / (digest + '.json'), certificate)
        event = EvidenceEvent.create(task_id='source', trace_id='qualification', occurrence_id='q',
            attempt_id='q', sequence_no=0, artifact_ref='skill://candidate@1.0.0', artifact_kind='atomic',
            event=EvidenceEventType.DEPLOYMENT_QUALIFIED, metadata={'version': 'skillcompiler.train-qualification.v2',
                'certificate_hash': digest, 'certificate': certificate})
        with pytest.raises(ValueError):
            verify(event, db)
        assert not db.rows('SELECT * FROM evidence_events')


def test_p1_one_bounded_vocabulary_repair_before_retrieval():
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.agents import UsageLedger
    from atomic_skillgraph.planner.requirement_agent import RequirementAgent
    from atomic_skillgraph.core.contracts import TaskContract, SemanticPredicate
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    good = {'requirements': [{'requirement_id': 'held', 'intent': 'hold one object',
        'desired_effects': [{'predicate': 'agent.holds', 'args': {'object': '$object'}, 'effect_domain': 'world'}],
        'expected_inputs': [], 'expected_outputs': [], 'precondition_hints': [],
        'semantic_variants': [], 'required': True, 'rationale': 'required effect'}], 'repeat_blocks': []}
    bad = copy.deepcopy(good)
    bad['requirements'][0]['desired_effects'][0]['predicate'] = 'object.in_hand'
    from atomic_skillgraph.agents.protocol import AgentTurn, NativeToolCall
    class RawProvider:
        def __init__(self):
            self.requests = []
        def snapshot(self):
            return {'provider': 'raw_schema_fixture'}
        def complete(self, messages, *, tools):
            self.requests.append(messages)
            number = len(self.requests)
            return AgentTurn('', [NativeToolCall(f'vocab_{number}', tools[0].name, bad if number == 1 else good)],
                'tool_calls', 1, 1, 2, 0, 0)
    provider = RawProvider()
    session = ReplayAgentSession(provider, system_prompt='P1', usage_ledger=UsageLedger(), usage_bucket='planner_p1')
    agent = RequirementAgent(session, predicate_specs=AlfWorldAdapter().semantic_predicate_schema())
    bundle = agent.propose(SimpleNamespace(goal='hold one object'),
        TaskContract([SemanticPredicate('agent.holds', {'object': '$object'})]), 'public state', 'alfworld_v3')
    assert bundle.requirements[0].desired_effects[0].predicate == 'agent.holds'
    assert len(provider.requests) == 2
    assert session.snapshot()['protocol_repairs_used'] == 1


def test_two_train_production_qualifications_frozen_filter_no_online_credit(tmp_path, monkeypatch):
    from atomic_skillgraph.system import AtomicSkillGraphSystem
    from atomic_skillgraph.core.bindings import BindingExpression
    from atomic_skillgraph.core.contracts import SemanticPredicate
    from atomic_skillgraph.core.serialization import to_primitive
    from atomic_skillgraph.evolution.identity_matching import raw_hash
    from atomic_skillgraph.evolution.aligner import _tool_signature
    from atomic_skillgraph.evolution.replay_certificates import ReplayCertificates
    from atomic_skillgraph.deployment.qualification import qualify_routes, qualified
    from atomic_skillgraph.deployment.freeze_consistency import converge
    from atomic_skillgraph.knowledge.source_snapshots import publish
    from test_r103_generalization import sources
    from test_r10_runtime import CheckpointHarness, route
    from experiments.r10_world_checks import install_fixture
    bank, trace, context = sources(tmp_path, monkeypatch)
    try:
        bank.config.update(profile_config())
        bank.mechanism_profile = bind(bank)
        n = context['normalized']
        p = context['batch'].generalizations[0]['source_proposals'][0]['proposal']
        from atomic_skillgraph.evolution.extractor_session import parse_occurrence_payload
        proposal = parse_occurrence_payload(p)
        occurrence, = bank.atomicizer.validate_and_canonicalize([proposal], n)
        atomic = bank._canonical_atomic_for_occurrence(occurrence)
        reference = bank.learning_source_store.stage(trace, n, occurrence, atomic, bank.harness.profile_name, proposal=proposal)
        bank.traces.save_atomic(trace)
        parent = publish(bank.data_dir, bank.traces.load_payload(trace.trace_id))
        bank.ledger.append_transaction([], companion_write=lambda c: bank.learning_source_store.commit(c, reference, parent))
        rows = bank.database.rows('SELECT * FROM learning_source_index ORDER BY sample_key')
        assert len(rows) == 2
        expr = BindingExpression('skill_input', source_role='target')
        bank.harness.reset(route.fixture_task(bank.harness.case))
        atomic, impl = install_fixture(bank, 'qualified_open', [],
            [SemanticPredicate('container.open', {'container': expr})], [('GO_TO', 'destination'), ('OPEN', 'object')], source_target='cabinet_1')
        tool = bank.tools.get(impl.tool_bindings[0].tool_ref)
        with bank.database.transaction() as connection:
            connection.execute("UPDATE artifact_index SET status='candidate' WHERE artifact_ref IN (?,?,?)",
                (str(atomic.ref), str(impl.ref), str(tool.ref)))
        class IndependentHarness(CheckpointHarness):
            def reset(self, task):
                # Two different public scenes, chosen by each immutable fixture
                # identity. This is fixture construction, never production policy.
                number = 1 if task.task_id.endswith('1') else 2
                self.case = replace(self.case, locations=(f'surface_{number}', 'cabinet_1'))
                return super().reset(task)
        bank.qualification_harness_factory = lambda: IndependentHarness(route.RouteCase('qualification'))
        certificates = ReplayCertificates(bank.ledger)
        for row in rows:
            task_record = bank.traces.load_payload(row['source_trace_id'])['task']
            case = {'trace_id': row['source_trace_id'], 'source_task': task_record,
                'bindings': {'target': 'cabinet_1'}, 'prefix': []}
            task = bank._replay_source_authority().resolve(case)
            result = bank._replay_tool_candidate_result(task, tool, case, requested_task_id=task.task_id)
            assert result.passed, result
            event = certificates.certificate(_tool_signature(tool), case, result, artifact_ref=str(tool.ref),
                trace_id=row['source_trace_id'], task_id=task.task_id, tool=tool, semantic_profile=bank.harness.profile_name)
            bank.ledger.append_transaction([event])
        result = qualify_routes(bank)
        assert result and result[0]['status'] == 'qualified', result
        from atomic_skillgraph.deployment.qualification import verify
        for row in bank.database.rows("SELECT * FROM evidence_events WHERE event_type=?", (EvidenceEventType.DEPLOYMENT_QUALIFIED.value,)):
            verify(EvidenceEvent.from_row(row), bank.database)
        assert qualified(bank.database, str(impl.ref))
        assert [i.ref for i in bank.skills.implementations_for(atomic.ref, mode='frozen')] == [impl.ref]
        assert tool.ref in bank.tools.list_refs(mode='frozen')
        stats = bank.projection.stats(str(tool.ref), 'tool')
        assert stats.execution_support.get('direct_success_tasks', []) == []
        report = converge(bank)
        digest = bank.knowledge_digest()
        assert converge(bank)['ledger_watermark'] == report['ledger_watermark']
        assert bank.knowledge_digest() == digest
        assert report['projection_watermark'] == bank.ledger.max_rowid()
        from atomic_skillgraph.core.contracts import CompositeSkill, CompositeOccurrence, TaskContract
        from atomic_skillgraph.core.refs import SkillRef
        from atomic_skillgraph.deployment.qualification import qualify_composites, composite_closure
        graph = CompositeSkill(SkillRef('qualified_graph', '1.0.0'), 'open specified container',
            [CompositeOccurrence('open', 'open', atomic.ref, {'target': BindingExpression('skill_input', source_role='target')})],
            ['open'], [], [], TaskContract([SemanticPredicate('container.open', {'container': '$target'})]),
            {}, {}, {}, {}, 'candidate')
        bank.skills.register_composite(graph)
        closure = composite_closure(bank.database, str(graph.ref), bank.harness.profile_name)
        assert closure['passed'], closure
        result = qualify_composites(bank)
        assert result and result[0]['status'] == 'qualified', result
        bank.projection.consume_new_events()
        assert qualified(bank.database, str(graph.ref))
        assert graph.ref in [g.ref for g in bank.skills.composites(mode='frozen')]
        assert bank.projection.stats(str(graph.ref), 'composite').deployment_qualification
        from atomic_skillgraph.deployment.train_bank_compiler import qualification_inventory
        inventory = qualification_inventory(bank)
        assert {c['kind'] for c in inventory['certificates']} == {'production_program', 'composition_closure'}
        assert all(c['online_direct_success_credit'] is False for c in inventory['certificates'])
    finally:
        bank.close()


def test_mixed_gap_actual_validated_output_returns_to_existing_graph(tmp_path):
    from test_r10_runtime import setup, action
    from experiments.r10_world_checks import install_fixture
    from atomic_skillgraph.core.bindings import BindingExpression
    from atomic_skillgraph.core.contracts import (SemanticPredicate, ColdStartPlanProposal,
        ColdStartPlanStep, ProposedEdge, ParameterSpec)
    from atomic_skillgraph.core.serialization import to_primitive
    def choose(request, number):
        if number == 1:
            return action(request, 'GO_TO', destination='cabinet_1')
        if number == 2:
            return 'validate_current_atomic', {'candidate_bindings': {'target': 'cabinet_1'},
                                              'candidate_outputs': {'target': 'cabinet_1'}}
        pytest.fail('Verified successors should execute without another Runtime decision: ' + str(request.policy_context))
    system, ctx, parent, _, provider = setup(tmp_path, choose)
    try:
        system.harness.case = replace(system.harness.case, terminal_action='TAKE')
        role = BindingExpression('skill_input', source_role='target')
        at = SemanticPredicate('agent.at_location', {'location': role})
        opened = SemanticPredicate('container.open', {'container': role})
        atomic, impl = install_fixture(system, 'known_open', [at], [opened], [('OPEN', 'object')], source_target='cabinet_1')
        parameter = ParameterSpec('target', 'entity', runtime_resolvable=True, required_resolution='concrete')
        gap = {'ref': 'gap://nav', 'canonical_intent': 'reach the specified public destination',
            'atomic_contract': {'inputs': [to_primitive(parameter)], 'outputs': [to_primitive(parameter)],
                'preconditions': [], 'effects': [to_primitive(at)], 'validator_spec': {
                    'output_derivations': {'target': {'kind': 'input_identity', 'input_role': 'target'}}}},
            'seeded_guideline': {}, 'harness_profile': system.harness.profile_name, 'origin': 'dynamic_gap'}
        steps = [ColdStartPlanStep('gap', ['nav'], 'dynamic_gap', gap['ref'], 'seeded_only',
                    {'target': BindingExpression('constant', constant='cabinet_1')}, {}),
                 ColdStartPlanStep('open', ['open'], 'verified', str(atomic.ref), 'direct_or_seeded',
                    {'target': BindingExpression('data_flow', source_step='gap', source_role='target')}, {}),
                 ColdStartPlanStep('take', ['take'], 'verified', str(parent.node_ref), 'direct_or_seeded',
                    {'object': BindingExpression('constant', constant='egg_1'),
                     'source': BindingExpression('constant', constant='cabinet_1')}, {})]
        proposal = ColdStartPlanProposal('mixed', steps, ['gap', 'open', 'take'],
            [ProposedEdge('output', 'data_flow', 'gap', 'open', 'target', 'target')], [],
            {'nav': ['gap'], 'open': ['open'], 'take': ['take']}, [])
        plan = copy.deepcopy(ctx.plan)
        plan.source = 'mixed_execution'
        plan.occurrences = []
        plan.control_sequence = []
        plan.cold_start_plan = proposal
        plan.cold_start_scaffold = {'executable_step_ids': proposal.control_sequence}
        plan.planner_audit['dynamic_gaps'] = {gap['ref']: gap}
        plan.planner_audit['complete_graph_coverage'] = False
        system.planner.build_plan = lambda *args, **kwargs: plan
        trace = system.orchestrator.run_task(ctx.task)
        assert trace.benchmark_success
        assert [r.action_type for r in trace.environment_actions] == ['GO_TO', 'OPEN', 'TAKE']
        assert len(provider.requests) == 2
        assert [r.candidate_source for r in trace.cold_start_steps] == ['dynamic_gap', 'verified', 'verified']
        assert all(r.local_effect_passed for r in trace.cold_start_steps)
        assert all(r.result['started'] for r in trace.tool_executions)
        assert not trace.runtime_plan['planner_audit']['complete_graph_coverage']
    finally:
        system.close()
