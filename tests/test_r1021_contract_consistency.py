"""One authored contract through learning, admission and actual invocation."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.contracts import ParameterSpec
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.typed_boundary import validate_input_specs
from atomic_skillgraph.evolution.typed_repairs import _atomic, _implementation
from atomic_skillgraph.evolution.contract_canonicalizer import AtomicContractCanonicalizer
from atomic_skillgraph.tooling.proposal import validate_output_semantic_constraints


@pytest.mark.parametrize('resolution,constraints,required,valid', [
    ('semantic', {}, True, True),
    ('concrete', {'result': {'compatible_with_input': 'item'}}, True, True),
    ('relation_verified', {'result': {'compatible_with_input': 'item'}}, True, True),
    ('semantic', {'result': {'compatible_with_input': 'item'}}, True, False),
    ('concrete', {'result': {'compatible_with_input': 'item'}}, False, False),
    ('concrete', {'result': {'compatible_with_input': 'unknown'}}, True, False),
    ('semantic', None, True, False), ('semantic', [], True, False),
])
def test_C1_identical_contract_at_e1_canonical_repair_registry(tmp_path, resolution, constraints, required, valid):
    from test_typed_repair_session import _atomic as fixture_atomic
    from test_atomic_contract_canonicalizer import _runtime
    a = fixture_atomic()
    a.inputs[0].required = required
    a.outputs[0].required_resolution = resolution
    a.validator_spec['output_semantic_constraints'] = constraints
    raw = to_primitive(a)
    proposal = SimpleNamespace(input_specs=a.inputs, output_specs=a.outputs,
        input_roles={'item': 'cup'}, output_roles={'result': 'cup'}, output_semantic_constraints=constraints)
    authority = {'item': {'available_revision': 0, 'semantic_type': 'entity', 'resolution': 'semantic'}}
    db, skills, _, _ = _runtime(tmp_path)
    try:
        calls = [lambda: validate_input_specs(proposal, authority, {'before_revision': 0}),
            lambda: validate_output_semantic_constraints(a.inputs, a.outputs, constraints),
            lambda: _atomic(raw), lambda: AtomicContractCanonicalizer().canonicalize(a),
            lambda: skills.register_atomic(a)]
        for call in calls:
            if valid:
                call()
            else:
                with pytest.raises(ValueError): call()
        assert to_primitive(a) == raw
        if valid:
            assert to_primitive(skills.get_atomic(a.ref)) == raw
        else:
            assert not skills.list_refs('atomic')
    finally: db.close()


@pytest.mark.parametrize('provided,target_required', [(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('scope', ['registered', 'runtime_trial'])
def test_C3_optional_preflight_and_runner(tmp_path, provided, target_required, scope):
    from test_r102_execution import _opened
    from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
    system, ctx, occurrence, invocations, _ = _opened(tmp_path)
    try:
        compiled = copy.deepcopy(invocations[0])
        compiled.atomic.inputs.append(ParameterSpec('extra', 'entity', required=False))
        tool = compiled.tools[0]
        tool.signature['properties']['extra'] = {'type': 'string'}
        if target_required: tool.signature.setdefault('required', []).append('extra')
        compiled.implementation.tool_bindings[0].parameter_mapping['extra'] = BindingExpression(
            BindingExprKind.SKILL_INPUT, source_role='extra')
        compiled.spec = system.invocation_compiler.compile(compiled.atomic, compiled.implementation,
            compiled.tools, ctx.binding_store.snapshot_for_node(occurrence))
        args = {'object': 'egg_1', 'source': 'cabinet_1'}
        if provided:
            args['extra'] = 'cabinet_1'
        pre = system.invocation_compiler.preflight(compiled, call_name=compiled.spec.name,
            call_id='optional', arguments=args, occurrence=occurrence, binding_store=ctx.binding_store,
            evidence_store=ctx.evidence_store, revision=ctx.world_revision)
        if target_required and not provided:
            assert not pre.passed and pre.failure_code == 'implementation_mapping_error'
            assert 'extra' in pre.message
            return
        assert pre.passed, pre
        assert ('extra' in pre.normalized_arguments) == provided
        runner = system.orchestrator.node_executor.implementation_runner
        recorded = []
        original = runner.tool_runner.run
        def observed(tool, arguments, *args, **kwargs):
            recorded.append(copy.deepcopy(arguments))
            return original(tool, arguments, *args, **kwargs)
        runner.tool_runner.run = observed
        result = execute_invocation(runner,
            compiled, pre, occurrence, ctx, agent_prepared=True, execution_scope=scope)
        assert result.completed and result.atomic_effect_passed, result
        actual = recorded[0]
        assert ('extra' in actual) == provided
        if provided: assert actual['extra'] == args['extra']
    finally: system.close()


@pytest.mark.parametrize('value', [0, False, '', [], None])
def test_C3_falsey_values_are_not_missing(value):
    from atomic_skillgraph.runtime.invocation_compiler import _tool_arguments
    atomic = SimpleNamespace(inputs=[ParameterSpec('x','entity',required=False)])
    tool = SimpleNamespace(signature={'properties':{'arg':{}}, 'required':[]})
    mapping = {'arg':BindingExpression(BindingExprKind.SKILL_INPUT,source_role='x')}
    for first in [False,True]:
        assert _tool_arguments(mapping, {'x':value}, {}, atomic, tool, first=first) == {'arg':value}
        assert _tool_arguments(mapping, {}, {}, atomic, tool, first=first) == {}
    for kind in [BindingExprKind.TOOL_OUTPUT, BindingExprKind.DATA_FLOW]:
        mapping['arg'] = BindingExpression(kind,source_role='x',source_step='previous')
        with pytest.raises((KeyError, ValueError)): _tool_arguments(mapping, {}, {}, atomic, tool)


@pytest.mark.parametrize('edit,path', [
    (lambda a:a.pop('ref'), 'ref'),
    (lambda a:a.update(ref=None), 'ref'),
    (lambda a:a.update(summary={}), 'summary'),
    (lambda a:a.update(inputs=None), 'inputs'),
    (lambda a:a['inputs'][0].update(required='false'), 'required'),
    (lambda a:a['effects'][0].update(args={'x':{'kind':'skill_output','source_role':'result'}}), 'args.x'),
    (lambda a:a['effects'][0].update(args=None), 'args'),
])
def test_C5_untrusted_replacement_is_content_error(edit,path):
    from test_typed_repair_session import _atomic as fixture_atomic, _review
    from atomic_skillgraph.evolution.typed_repair_session import TypedRepairProposalSession, TypedRepairDecision
    from atomic_skillgraph.evolution.typed_repairs import TypedRepairEngine
    raw = to_primitive(fixture_atomic()); edit(raw)
    review = _review()
    decision = TypedRepairDecision(review.review_id,'propose','revise_atomic_contract',(raw,),'fixture')
    payload = {'decisions':[{**to_primitive(decision),'replacements':[raw]}]}
    calls = [lambda:_atomic(raw), lambda:TypedRepairProposalSession.parse(payload,[review]),
        lambda:TypedRepairProposalSession.build_proposals([decision],[review]),
        lambda:TypedRepairEngine.build_proposal(decision.operation,review.target_refs,[raw],review.evidence)]
    for call in calls:
        with pytest.raises(ValueError,match=path): call()


@pytest.mark.parametrize('internal_fault', [False, True])
def test_C6_real_run_task_maintenance_and_runner_commit(tmp_path, monkeypatch, internal_fault):
    """Fixture selection only; run_task, validators, ledger and runner stay real."""
    import json
    from experiments import run_v3_train as train
    from experiments import self_tooling_targeted as route
    from fixtures.r921_self_tooling_cases import fixture_config
    from test_r10_runtime import CheckpointHarness, StepProvider, action
    from atomic_skillgraph.system import AtomicSkillGraphSystem
    from atomic_skillgraph.evolution.typed_repair_session import TypedRepairReview
    from atomic_skillgraph.evolution.typed_repairs import RepairEvidence

    case = route.RouteCase('maintenance_fixture', terminal_action='TAKE')
    harness = CheckpointHarness(case)
    task = route.fixture_task(case)
    harness.split = 'train'
    harness.load_balanced_tasks = lambda *_: [task]
    output = tmp_path/'run'
    config = fixture_config(output)
    config['trace_data_dir'] = str(output)
    config['experiment'].update(phase='train', name='run', output_dir=str(output),
        max_task_attempts=2, task_manifest_path=str(output/'task_manifest.json'))
    config_path = tmp_path/'fixture.json'
    config_path.write_text(json.dumps(config))
    holder = {}
    def choose(request, _):
        if any(t.name == 'submit_typed_repair' for t in request.tools):
            raw = to_primitive(holder['atomic']); raw.pop('ref')
            return 'submit_typed_repair', {'decisions':[{'review_id':'fixture_review',
                'decision':'propose','operation':'revise_atomic_contract','replacements':[raw],'rationale':'fixture'}]}
        catalog = route.newest_catalog(request) or request.policy_context.get('current_action_catalog', {})
        if isinstance(catalog, dict): catalog = catalog.get('actions', [])
        for kind in ['TAKE', 'OPEN', 'GO_TO']:
            available = [a for a in catalog if a['action_type']==kind and
                (kind!='GO_TO' or a['arguments'].get('destination')=='cabinet_1')]
            if available:
                return 'environment_action', {'action_id':available[0]['action_id']}
        pytest.fail('fixture has no expected public action')
    class ContentProvider(StepProvider):
        def complete(self, messages, *, tools):
            if any(t.name == 'submit_typed_repair' for t in tools):
                from experiments.fakes import FakeProviderRequest
                from atomic_skillgraph.agents.protocol import AgentTurn, NativeToolCall
                request = FakeProviderRequest(tuple(copy.deepcopy(messages)), tuple(tools))
                self.requests.append(request)
                name, args = choose(request, len(self.requests))
                return AgentTurn('', [NativeToolCall(f'bad_{len(self.requests)}',name,args)],
                    'tool_calls',10,10,20,0,1.0,{'provider':'fixture','usage_status':'reported'})
            return super().complete(messages,tools=tools)
    provider = ContentProvider(choose)
    def factory(cfg, **kwargs):
        system = AtomicSkillGraphSystem(cfg, harness=harness, provider=provider, **kwargs)
        holder['system'] = system
        original_preflight = system.preflight
        system.preflight = lambda **kw: original_preflight(require_api_key=False, initialize_harness=False, require_empty_bank=kw['require_empty_bank'])
        def plan(task, *args, **_):
            from atomic_skillgraph.core.results import RuntimeLinearPlan
            atomic = route.parent_atomic(case)
            holder['atomic'] = atomic
            return RuntimeLinearPlan(task.task_id, 'full_dynamic', None, [], [], [], [], harness.task_contract(task), {'fixture_controlled':True})
        system.planner.build_plan = plan
        system.maintenance_interval = 1
        system.extraction_policy.decide = lambda trace: SimpleNamespace(should_extract=False,reasons=['diagnostic_no_e1'])
        system.evolution_maintenance.build_batch_reviews = lambda *_: []
        def reviews(**_):
            if internal_fault: raise KeyError('internal_index_corrupt')
            system.skills.register_atomic(holder['atomic'])
            return [TypedRepairReview('fixture_review','atomic',(str(holder['atomic'].ref),),
                ('revise_atomic_contract',), {}, (RepairEvidence('ev','source','trace_source','cluster',{}),))]
        system.evolution_maintenance.build_typed_reviews = reviews
        system.evolution_maintenance.build_composite_sequence_reviews = lambda **_: []
        return system
    monkeypatch.setattr(train, 'AtomicSkillGraphSystem', factory)
    monkeypatch.setattr(train, '_selection', lambda _: ([task.task_type],1,1))
    monkeypatch.setattr(train, '_train_protocol', lambda _: ('fixture',42,None,None))
    monkeypatch.setattr(train, '_validate_formal_config', lambda *_:None)
    monkeypatch.setattr(train, '_r7_train_reference_manifest', lambda _:None)
    monkeypatch.setattr(train, 'ensure_provider_capability', lambda *_,**__: {'passed':True})
    # No experiment identity discovery is needed for this temporary fixture.
    monkeypatch.setattr(train, 'hash_code', lambda _: 'fixture_code')
    class EndOfTaskFixture(Exception): pass
    def finish(*_, **__): raise EndOfTaskFixture()
    monkeypatch.setattr(train, '_run_final_batch_maintenance', finish)
    with pytest.raises(KeyError if internal_fault else EndOfTaskFixture):
        train.run(config_path)
    import sqlite3
    db = sqlite3.connect(holder['system'].database.path)
    row = db.execute('select state,result_json from run_tasks').fetchone()
    db.close()
    if internal_fault:
        assert row[0] == 'infrastructure_failed'
        assert (output/'.task_checkpoint').exists()
    else:
        assert row[0] == 'completed', row
        assert json.loads(row[1])['benchmark_success']
        assert not (output/'.task_checkpoint').exists()
    traces = [json.loads(p.read_text()) for p in (output/'traces').glob('*.json')
              if not p.name.endswith(('.owner.json','.claim.json'))]
    maintenance = next(t for t in traces if t.get('metadata',{}).get('trace_kind')=='maintenance')
    assert maintenance['ended_at'] >= maintenance['started_at']
    if internal_fault:
        assert maintenance['infrastructure_failure']
    else:
        assert not maintenance['infrastructure_failure']
        assert maintenance['metadata']['typed_proposal_count'] == 0
        assert 'ref' in maintenance['metadata']['semantic_proposal_error']
        assert maintenance['llm_usage'] and maintenance['agent_sessions']
        assert 'replacements' in json.dumps(maintenance['agent_sessions'])
