"""S01-S12: production interpreter, transactions and request projections."""
import copy
from types import SimpleNamespace
import pytest
from test_oldfirst_program import search
from test_r10_runtime import setup, action
from atomic_skillgraph.runtime.checkpoint import capture, restore
from atomic_skillgraph.runtime.runtime_step import run_runtime_step, completed_step_ids_for_policy
from atomic_skillgraph.runtime.invocation_transaction import cache_lookup
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.core.results import NodeExecutionStatus
from atomic_skillgraph.core.serialization import to_primitive


def run_search(system, ctx, occurrence, *, query='absent_kind', scope=None, allow=True, tool=None):
    _, default = search()
    tool = tool or default
    saved = capture(ctx, occurrence.occurrence_id)
    try:
        result = system.orchestrator.node_executor.implementation_runner.tool_runner.run(tool,
            {'query': query, 'locations': scope or list(ctx.harness.case.locations), 'allow_open': allow},
            ctx, occurrence_id=occurrence.occurrence_id)
        return result
    finally:
        restore(ctx, saved, 'fixture_search_failure')


def test_S01_S02_S06_history_survives_three_rollbacks_cache_and_native_step(tmp_path):
    system, ctx, occ, inv, provider = setup(tmp_path, lambda r, n: action(r, 'GO_TO', destination='cabinet_1'))
    try:
        digest = ctx.harness._runtime_state_digest()
        original_bindings = copy.deepcopy(vars(ctx.binding_store))
        ids = []
        for index in range(3):
            result = run_search(system, ctx, occ, scope=[ctx.harness.case.locations[index % len(ctx.harness.case.locations)]])
            assert not result.completed
            ids.append(list(ctx.search_history.attempts)[-1])
            assert ctx.harness._runtime_state_digest() == digest
            assert ctx.binding_store._bindings == original_bindings['_bindings']
        before = len(ctx.trace_builder.trace.environment_actions)
        ctx.rejected_runtime_candidates['test'] = {'failure_code': 'tool_ir_execution_error',
            'failure_layer': 'tool', 'search_observation_ids': [ids[0]]}
        cache_lookup(ctx, 'test', occ, 'cache_call')
        assert len(ctx.trace_builder.trace.environment_actions) == before
        assert len(ctx.search_history.attempts) == 3
        run_runtime_step(system.orchestrator.node_executor, 'preparation', occ, ctx, inv, [])
        view = provider.requests[-1].policy_context['exploration_memory']['search_history']
        assert view['groups'][0]['rolled_back']
        assert set(view['groups'][0]['checked_no_match']) == set(ctx.harness.case.locations[:3])
        assert all(a.world_disposition == 'rolled_back' for a in ctx.search_history.attempts.values())
        run_runtime_step(system.orchestrator.node_executor, 'preparation', occ, ctx, inv, [])
        assert provider.requests[-1].policy_context['exploration_memory']['search_history'] == view
        from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
        from atomic_skillgraph.runtime.budget import RuntimeBudget
        fresh = TaskRuntimeContext.create(ctx.task, ctx.plan, ctx.harness, system.orchestrator.create_trace_builder(ctx.task), RuntimeBudget())
        assert fresh.search_history.policy_view()['groups'] == []
    finally:
        system.close()


def test_S03_S04_method_is_typed_and_unreached_scope_is_not_no_match(tmp_path):
    system, ctx, occ, _, _ = setup(tmp_path, lambda *a: pytest.fail('unexpected provider'))
    try:
        for query, allow in [('absent', True), ('absent', False), ('another', True)]:
            run_search(system, ctx, occ, query=query, allow=allow)
        assert len(ctx.search_history.policy_view()['groups']) == 3
        run_search(system, ctx, occ, scope=['unreachable'])
        a = list(ctx.search_history.attempts.values())[-1]
        assert all(c.outcome != 'no_matching_candidate' for c in a.checks)
        assert not a.output_authorized and not a.current_truth_authorized
        ctx.budget.global_action_budget = ctx.budget.used_global_actions + 1
        with pytest.raises(BudgetExhausted):
            run_search(system, ctx, occ)
        a = list(ctx.search_history.attempts.values())[-1]
        assert a.execution_outcome == 'interrupted' and a.world_disposition == 'rolled_back'
        assert any(c.outcome in {'interrupted', 'unchecked'} for c in a.checks)
    finally:
        system.close()


def test_S05_matched_selector_with_invalid_output_never_becomes_absence(tmp_path):
    system, ctx, occ, _, _ = setup(tmp_path, lambda *a: pytest.fail('unexpected provider'))
    try:
        _, tool = search()
        tool.interface['output_schema']['required'].append('missing_output')
        tool.interface['output_schema']['properties']['missing_output'] = {'type': 'string'}
        result = run_search(system, ctx, occ, query='egg', tool=tool)
        assert not result.completed
        a = list(ctx.search_history.attempts.values())[-1]
        assert any(c.outcome == 'matched_candidate' for c in a.checks)
        assert not a.output_authorized
    finally:
        system.close()


def test_S07_observer_does_not_change_actions_validation_outputs_or_cost(tmp_path):
    outcomes = []
    for enabled in (False, True):
        system, ctx, occ, _, provider = setup(tmp_path/str(enabled), lambda *a: pytest.fail('unexpected provider'))
        try:
            if not enabled:
                ctx.search_history = None
            result = run_search(system, ctx, occ)
            outcomes.append((to_primitive(result),
                [(a.action_type, a.arguments, a.accepted) for a in ctx.trace_builder.trace.environment_actions],
                ctx.world_revision, ctx.budget.used_global_actions, list(system.usage.events)))
        finally:
            system.close()
    assert outcomes[0] == outcomes[1]


def test_S08_completed_status_is_not_program_credit():
    records = [SimpleNamespace(step_id=status.value, status=status) for status in NodeExecutionStatus]
    result = completed_step_ids_for_policy(records + records)
    assert 'agent_completed_before_invocation' in result
    assert set(result) == {'already_satisfied', 'direct_autonomous_success', 'direct_agent_prepared_success',
                           'agent_completed_before_invocation', 'seeded_success'}


def test_S09_budget_call_finalizer_is_once_and_preserves_exception(tmp_path):
    system, ctx, occ, inv, provider = setup(tmp_path, lambda r, n: action(r, 'GO_TO', destination='cabinet_1'))
    try:
        ctx.budget.global_action_budget = 0
        with pytest.raises(BudgetExhausted):
            run_runtime_step(system.orchestrator.node_executor, 'preparation', occ, ctx, inv, [])
        calls = ctx.trace_builder.trace.native_tool_calls
        assert len(calls) == 1 and calls[0].preflight_result['interrupted_by_budget']
        assert 'accepted' not in calls[0].preflight_result
        assert len(provider.requests) == 1 and len(system.usage.events) == 1
    finally:
        system.close()


def test_S10_S11_common_summary_exact_preview_dedup_and_codec_roundtrip():
    from atomic_skillgraph.agents.runtime_policy_projection import project_support_presentation
    from atomic_skillgraph.agents.runtime_expression_codec import project_lean, restore_lean
    preview = {'status': 'proven', 'output_mapping': {'object':'object'}, 'relevant_revision':1, 'mapping_evidence':['e1']}
    other = {**preview, 'mapping_evidence':['e2']}
    raw = {'support_atomic_candidates': [{'atomic_ref':'skill://a@1.0.0', 'input_schema': {'maxItems':8},
        'mapping_previews':[preview, copy.deepcopy(preview), other]}], 'exploration_memory': {'search_history': {'groups':[{'query':True}]}}}
    before = copy.deepcopy(raw)
    view, audit = project_support_presentation(raw, {'skill://a@1.0.0':'Original purpose'})
    assert raw == before and audit['removed_exact_previews'] == 1
    assert view['support_atomic_candidates'][0]['mapping_previews'] == [preview,other]
    lean, codec = project_lean(view)
    assert restore_lean(lean,codec) == view
    assert lean['support_atomic_candidates'][0]['summary'] == 'Original purpose'


def test_S12_all_attempt_costs_are_unique_and_unknown_is_not_zero():
    from experiments.compiler_metrics import task_attempt_costs
    def trace(e, request, reasoning):
        return {'task':{'task_id':'one'}, 'llm_usage':[{'event_id':e,'prompt_tokens':3,'completion_tokens':2,'reasoning_tokens':reasoning}],
                'provider_requests':[{'request_id':request,'usage_status':'reported'}]}
    failed, final = trace('a','ra',None), trace('b','rb',1)
    costs = task_attempt_costs([failed,final,failed])
    assert costs == {'one':{'prompt_tokens':6,'completion_tokens':4,'reasoning_tokens':None,'total_tokens':10,'call_count':2}}
    final['provider_requests'][0]['usage_status']='unknown'
    assert task_attempt_costs([failed,final])['one']['total_tokens'] is None
    conflict = copy.deepcopy(failed);conflict['llm_usage'][0]['prompt_tokens']=99
    with pytest.raises(ValueError, match='conflicting usage'):
        task_attempt_costs([failed,conflict])


def test_S06_S09_real_invocation_cache_and_budget_audit(tmp_path):
    from atomic_skillgraph.deployment.oldfirst_revision import author_implementation
    from atomic_skillgraph.core.status import SkillStatus, ToolStatus
    from atomic_skillgraph.core.results import RuntimeOccurrence
    system, ctx, parent, _, provider = setup(tmp_path, lambda *a: pytest.fail('not configured'))
    try:
        atomic, tool = search()
        impl = author_implementation({'job_id':'history_test','implementation_ref':'skill://scope_impl@1.0.0'}, atomic, tool)
        atomic.status = impl.status = SkillStatus.ACTIVE
        tool.status = ToolStatus.ACTIVE
        impl.compatibility = {'harness_profiles':['fake_v3']}
        system.skills.register_atomic(atomic)
        system.tools.register(tool)
        system.skills.register_implementation(impl)
        from atomic_skillgraph.core.bindings import BindingExpression
        occ = RuntimeOccurrence('search','search',atomic.ref,[],
            {'query':BindingExpression('skill_input',source_role='object')},[impl.ref],atomic.effects)
        ctx.begin_occurrence(occ);ctx.budget.begin_node(occ.occurrence_id)
        ctx.binding_store.resolve_occurrence_specs(occ,ctx.world_revision)
        inv = system.invocation_compiler.compile_candidates(occ,ctx.binding_store,task_id=ctx.task_id)
        assert len(inv)==1
        provider.choose = lambda r,n: (inv[0].spec.name, {'query':'egg',
            'locations':['countertop_1','countertop_2'], 'allow_open':True})
        ex = system.orchestrator.node_executor
        run_runtime_step(ex,'preparation',occ,ctx,inv,[])
        assert len(ctx.search_history.attempts)==1, str(ctx.trace_builder.trace.native_tool_calls[-1].preflight_result)
        before = len(ctx.trace_builder.trace.environment_actions)
        run_runtime_step(ex,'preparation',occ,ctx,inv,[])
        assert len(ctx.search_history.attempts)==1 and len(ctx.trace_builder.trace.environment_actions)==before
        assert ctx.search_history.cache_hits[-1]['observation_ids']==list(ctx.search_history.attempts)
        assert ctx.runtime_step_feedback[occ.occurrence_id]['cached_rejection']
        provider.choose = lambda r,n: (inv[0].spec.name, {'query':'egg',
            'locations':list(ctx.harness.case.locations), 'allow_open':True})
        ctx.budget.global_action_budget = ctx.budget.used_global_actions+1
        with pytest.raises(BudgetExhausted):
            run_runtime_step(ex,'preparation',occ,ctx,inv,[])
        interrupted=ctx.trace_builder.trace.native_tool_calls[-1]
        assert interrupted.preflight_result['interrupted_by_budget']
        assert len(interrupted.preflight_result['attempt_refs']['tool_executions'])==1
        assert len(interrupted.preflight_result['attempt_refs']['implementation_invocations'])==1
        assert len(ctx.trace_builder.trace.native_tool_calls)==3 and len(provider.requests)==3
        assert list(ctx.search_history.attempts.values())[-1].authorizing_call_id==interrupted.call_id
    finally:
        system.close()


def test_S02_S10_dynamic_consumes_same_history(tmp_path):
    from atomic_skillgraph.runtime.task_runtime import run_dynamic
    from dataclasses import replace
    system,ctx,occ,_,provider=setup(tmp_path,lambda r,n:('environment_action', {
        'action_id':action(r,'GO_TO',destination='cabinet_1')[1]['action_id']}))
    try:
        run_search(system,ctx,occ)
        history=ctx.search_history
        system.harness.case=replace(system.harness.case,terminal_action='GO_TO')
        run_dynamic(system.orchestrator.node_executor,ctx,rescue=True)
        assert ctx.search_history is history
        assert provider.requests[0].policy_context['exploration_memory']['search_history']['groups']
    finally:
        system.close()


@pytest.mark.parametrize('profile', ['current', 'lean'])
@pytest.mark.parametrize('mode', ['node', 'dynamic'])
def test_S10_actual_context_has_original_summary_and_history(profile, mode):
    import json
    from atomic_skillgraph.agents.context_builder import ContextBuilder
    from atomic_skillgraph.agents.runtime_expression_codec import restore_lean
    builder = ContextBuilder(); builder.presentation_profile = profile
    ref = 'skill://search@1.0.0'
    summary = 'Find a matching public candidate within the supplied scope.'
    lookup = builder.selected_support_summaries(SimpleNamespace(get_atomic=lambda r: SimpleNamespace(summary=summary)),
        [SimpleNamespace(atomic_ref=ref)])
    assert lookup == {ref: summary}
    candidate = {'atomic_ref': ref, 'input_schema': {'type':'object', 'required':['scope'],
        'properties': {'scope': {'type':'array', 'maxItems':8, 'uniqueItems':True}}}}
    history = {'version':'r103.search-history.v1', 'groups':[{'query':'object', 'checked_no_match':['location']}]}
    kwargs = dict(task_goal='Find the object', observation='At a location', action_catalog=[],
        relevant_action_history=[], remaining_budget={}, exploration_memory={'search_history':history},
        support_summary_lookup=lookup, projection_audit={})
    if mode == 'node':
        rendered = builder.runtime_node(**kwargs, atomic_contract={'summary':'Goal','inputs':[],'outputs':[],
            'preconditions':[], 'effects':[], 'guideline':{'steps':[]}}, implementation_invocations=[],
            support_atomic_candidates=[candidate])
    else:
        rendered = builder.dynamic_task(**kwargs, task_runtime_frame={'capability_candidates':[candidate]})
    view = json.loads(rendered[rendered.index('\n{')+1:])
    if profile == 'lean':
        view = restore_lean(view, kwargs['projection_audit']['release_expression'])
    candidates = view['support_atomic_candidates'] if mode == 'node' else view['task_runtime_frame']['capability_candidates']
    assert candidates[0]['summary'] == summary
    assert candidates[0]['input_schema'] == candidate['input_schema']
    assert view['exploration_memory']['search_history'] == history


def test_S12_S14_report_final_task_all_attempts_and_finite_gate(tmp_path):
    import json, sqlite3
    from atomic_skillgraph.core.serialization import atomic_write_json as write
    from experiments.protocol import AttemptTraceLedger, SCHEMA_VERSION, hash_config
    from experiments.release_report import write_release_report
    from experiments.released_dev_checks import check_episode
    from atomic_skillgraph.deployment.release_protocol import ReleaseError
    config = {'bank_release':{'expected_bank_digest':'bank'}, 'deployment':{'presentation_profile':'current'}}
    manifest = dict(run_id='fixture', config_hash=hash_config(config), code_commit='code', knowledge_digest='bank')
    write(tmp_path/'run_manifest.json',manifest)
    write(tmp_path/'config.json',{**config, '_config_path':'loader-only'})
    write(tmp_path/'execution_code_manifest.json',{'code_hash':'code'})
    entry = dict(task_id='task',task_signature='signature')
    write(tmp_path/'task_manifest.json',{'tasks':[entry]})
    write(tmp_path/'progress.json',{'state':'completed','completed':1})
    ledger = AttemptTraceLedger(tmp_path/'attempt_history',tmp_path/'traces')
    traces=[]
    for i in (1,2):
        attempt=ledger.begin(run_id='fixture',**entry,attempt_kind='task',sequence=i)
        trace={'schema_version':SCHEMA_VERSION,'trace_id':f'trace_{i}', 'task':entry,
            'metadata':{'runtime_search_history':{'attempts':[],'cache_hits':[],'projections':[]}},
            'benchmark_success':i==2,'environment_actions':[],
            'llm_usage':[{'event_id':f'event{i}','prompt_tokens':3,'completion_tokens':2,'reasoning_tokens':None}],
            'provider_requests':[{'request_id':f'request{i}','session_id':f'session{i}','usage_status':'reported'}]}
        write(tmp_path/'traces'/f'trace_{i}.json',trace)
        ledger.capture(attempt,reason='fixture')
        traces.append(trace)
    with sqlite3.connect(tmp_path/'run_state.sqlite3') as db:
        db.execute('CREATE TABLE run_tasks(task_id TEXT, trace_id TEXT, state TEXT)')
        db.execute("INSERT INTO run_tasks VALUES ('task','trace_2','completed')")
    report=write_release_report(tmp_path,config,resource_traces=traces+traces[:1],digest_after='bank')
    assert report['tasks']==report['successes']==1 and report['total_recorded_tokens']==10
    result=check_episode(tmp_path,code_hash='code',bank_digest='bank',entry=entry,profile='current')
    assert result['costs']['total_tokens']==10
    for field in ('code_hash','bank_digest','profile'):
        args=dict(code_hash='code',bank_digest='bank',entry=entry,profile='current');args[field]='wrong'
        with pytest.raises(ReleaseError,match='identity/completion'):
            check_episode(tmp_path,**args)
    write(tmp_path/'progress.json',{'state':'running','completed':0})
    with pytest.raises(ReleaseError,match='identity/completion'):
        check_episode(tmp_path,code_hash='code',bank_digest='bank',entry=entry,profile='current')
