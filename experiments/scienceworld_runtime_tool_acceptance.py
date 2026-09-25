"""Isolated F2 transport fixture on real ScienceWorld JVMs.

Only Agent submissions and the full-dynamic plan are controlled. No official
score, R0/R1 decision, credit, lifecycle status or source execution is fabricated.
The live Runtime finishes each public task after the controlled Tool boundary.
This acceptance bank is never an input to the formal learned campaign.
"""
import argparse
import copy
import json
from pathlib import Path

from atomic_skillgraph.core.results import RuntimeLinearPlan,RuntimeOccurrence
from atomic_skillgraph.core.bindings import BindingExpression
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from .benchmark_protocol import BenchmarkExperimentProtocol
from .scienceworld_config import make_config
from .protocol import ManifestStore, RunManifest, TaskManifest, ensure_task_manifest, hash_code, hash_config
from .fakes import FakeProviderRequest, FakeReply


class BoundaryProvider:
    def __init__(self, delegate, builder=False):
        self.delegate, self.builder = delegate, builder
        self.phase = 0
        self.reuse = False
        self.requests = []
        self.builder_counter = lambda: 0
        self.reuse_builder_before = None
        self.reuse_verified = False

    def snapshot(self):
        return {'provider':'scienceworld_f2_controlled_boundary','fixture_generated':True,
                'live_delegate':self.delegate.snapshot()}

    def set_request_context(self, **kwargs):
        self.delegate.set_request_context(**kwargs)

    @property
    def request_record_count(self):
        return self.delegate.request_record_count

    def request_records_since(self, offset):
        return self.delegate.request_records_since(offset)

    def complete(self,messages,*,tools):
        request=FakeProviderRequest(tuple(copy.deepcopy(messages)),tuple(tools))
        self.requests.append({'messages':copy.deepcopy(messages),'tools':[t.to_openai() for t in tools]})
        names={t.name for t in tools}
        if self.builder:
            atomic=request.policy_context['canonical_atomic']
            if ({p['name'] for p in atomic['inputs']}!={'wait_steps'} or
                {e['predicate'] for e in atomic['effects']}!={'time.progressed'}):
                # Only the designated transport trial is controlled. Unrelated
                # Evolution requests keep their ordinary production provider.
                return self.delegate.complete(messages,tools=tools)
            effects=atomic['effects']
            arguments={'proposal_version':'2','decision':'create','summary':'Three bounded public time steps',
                'atomic_ref':request.policy_context['atomic_ref'],'inputs':atomic['inputs'],'outputs':atomic['outputs'],
                'entry_contract':{'conditions':[],'grounding_constraints':[]},'max_actions':3,
                'input_schema':{'type':'object','required':['wait_steps'],'additionalProperties':False,
                    'properties':{'wait_steps':{'type':'integer','minimum':1,'maximum':3}}},
                'program':[{'op':'FOR_EACH','node_id':'steps','iteration_variable':'step',
                    'collection_source':{'source':'bounded_count','count':{'source':'tool_input','field':'wait_steps'}},
                    'max_iterations':3,'body':[{'op':'ACTION','node_id':'wait','action_type':'WAIT1','argument_mapping':{},
                            'expected_effects':[{'predicate':'time.progressed','args':{'evidence':None},'effect_domain':'evidence'}]}]},
                           {'op':'RETURN','node_id':'return_evidence','output_sources':{
                               'evidence_ref':{'source':'semantic_evidence','where':{'predicate':'time.progressed'},
                                   'project':{'kind':'argument','role':'evidence'},'distinct':True}}}],
                'final_effects':effects,'path_expectations':[],'evidence_outputs':[],
                'rationale':'Fixture-controlled bounded procedure; evidence comes only from real accepted WAIT1 actions.'}
            name='create_tool'
        elif self.phase==0 and not self.reuse:
            name='request_runtime_automation'
            arguments={'reason':'F2 transport acceptance of a bounded public procedure','intended_capability':'Advance three time steps with witnessed evidence'}
            self.phase=1
        elif self.phase==1 and not self.reuse:
            interface=request.policy_context['runtime_automation_interface']
            name='propose_runtime_automation_atomic'
            arguments={'draft_id':'three_public_steps','intent':'advance_three_public_steps',
                'inputs':[{'name':'wait_steps','semantic_type':'integer','required':True,'runtime_resolvable':False,'required_resolution':'semantic'}],
                'outputs':[{'name':'evidence_ref','semantic_type':'evidence_ref','required':True,'runtime_resolvable':False,'required_resolution':'semantic'}],
                'preconditions':[],'effects':[{'predicate':'time.progressed','args':{'evidence':'$evidence_ref'},'effect_domain':'evidence'}],
                'input_binding_specs':{'wait_steps':{'kind':'constant','value':3}},'source_occurrence_id':interface['source_occurrence_id'],
                'rationale':'Three public time steps produce a fresh harness witness.'}
            self.phase=2
        elif self.phase==0 and self.reuse:
            spec=next(t for t in tools if t.name.startswith('invoke_impl_'))
            name=spec.name
            role=next(iter(spec.input_schema['properties']))
            arguments={role:3}
            self.reuse_builder_before=self.builder_counter()
            self.phase=2
        else:
            if self.reuse:
                # The fourth episode is a finite route-reuse acceptance, not
                # a score test. Stop through the ordinary Agent boundary once
                # the stored route has actually completed its three actions.
                assert self.reuse_verified,'Persistent route did not complete'
                reply=FakeReply.tool('report_runtime_status',{'status':'give_up','detail':'Finite persistent-route acceptance complete; no official success claimed.'},prompt_tokens=0,completion_tokens=0,reasoning_tokens=0)
                return reply.materialize(call_id=f'finite_stop_{len(self.requests)}',tools=tools,request=request)
            if self.phase==2 and not self.reuse:
                feedback=request.policy_context.get('task_runtime_frame',{}).get('last_step',{})
                if feedback.get('tool')=='propose_runtime_automation_atomic':
                    assert feedback.get('r1_passed'),feedback
            return self.delegate.complete(messages,tools=tools)
        reply=FakeReply.tool(name,arguments,prompt_tokens=0,completion_tokens=0,reasoning_tokens=0)
        turn=reply.materialize(call_id=f'controlled_{self.builder}_{len(self.requests)}',tools=tools,request=request)
        turn.provider_metadata={'provider':'scienceworld_f2_fixture','fixture_generated':True,'external_tokens':0}
        return turn


def run(root, resume=False):
    root=Path(root).resolve()
    path=make_config(root)
    config=load_config(path)
    config['experiment']['task_manifest_path']=str(root/'acceptance_task_manifest.json')
    config['extraction']['extract_full_dynamic_success']=False
    protocol=BenchmarkExperimentProtocol.scienceworld(config['harness']['manifest'])
    # Fixed transport fixtures; these are Train-only, not selected from Dev/Test.
    entries=[r for r in protocol.manifest['tasks'] if r['task_type']=='2-1'][:4]
    assert len(entries)==4
    harness=ScienceWorldAdapter()
    with AtomicSkillGraphSystem(config,harness=harness) as system:
        runtime=BoundaryProvider(system._provider('runtime_dynamic'))
        builder=BoundaryProvider(system._provider('tool_builder'),builder=True)
        runtime.builder_counter=lambda:len(builder.requests)
        system._provider_override={'runtime':runtime,'tool_builder':builder,'default':system._provider('extractor')}
        system.orchestrator.planner.build_plan=lambda task,harness,**kw:RuntimeLinearPlan.full_dynamic(
            task.task_id,harness.task_contract(task),reason='controlled_F2_transport_fixture')
        initial=system.knowledge_digest()
        tasks=[TaskManifest(i,r['task_id'],r['task_signature'],initial,'scienceworld','train',json.dumps(r)) for i,r in enumerate(entries)]
        manifest=RunManifest.create(run_id=config['experiment']['name'],phase='train',config_hash=hash_config(config),
            code_commit=hash_code(Path(__file__).resolve().parents[1]),knowledge_digest=initial,tasks=tasks,
            metadata={'experiment_kind':'engineering_acceptance','not_a_formal_result':True})
        store=ManifestStore(root,system.database)
        if resume:
            manifest=store.load(config['experiment']['name'])
        else:
            store.persist_before_run(manifest)
        ensure_task_manifest(config['experiment']['task_manifest_path'],manifest)
        completed=[]
        action_counts=[]
        tool_windows=json.loads((root/'tool_windows.json').read_text()) if resume and (root/'tool_windows.json').exists() else []
        runner=system.orchestrator.node_executor.implementation_runner.tool_runner
        original_run=runner.run
        def counted_run(*args,**kwargs):
            before=(len(runtime.requests),len(builder.requests),runtime.request_record_count,builder.request_record_count)
            result=original_run(*args,**kwargs)
            after=(len(runtime.requests),len(builder.requests),runtime.request_record_count,builder.request_record_count)
            assert before==after,'Provider called inside Tool'
            if runtime.reuse:
                assert runtime.reuse_builder_before==len(builder.requests),'Persistent route called ToolBuilder'
                assert result.started and result.completed and result.executed_action_count==3
                runtime.reuse_verified=True
            tool_windows.append({'before':before,'after':after,'provider_delta':0,'result':to_primitive(result)})
            atomic_write_json(root/'tool_windows.json',tool_windows)
            return result
        runner.run=counted_run
        execute=harness.execute_action
        def recorded_action(*args,**kwargs):
            action_counts.append({'runtime_requests':len(runtime.requests),'builder_requests':len(builder.requests)})
            return execute(*args,**kwargs)
        harness.execute_action=recorded_action
        for ordinal,row in enumerate(entries):
            stored=system.database.execute('SELECT state,trace_id FROM run_tasks WHERE run_id=? AND task_id=?',
                (manifest.run_id,row['task_id'])).fetchone()
            if resume and stored['state']=='completed':
                saved=system.traces.load_payload(stored['trace_id'])
                assert not saved['infrastructure_failure']
                completed.append({'task_id':row['task_id'],'trace_id':stored['trace_id'],'official_score':saved['official_score'],'resumed_existing':True})
                continue
            runtime.phase=0
            runtime.reuse=ordinal==3
            task=protocol.task(row,harness)
            if runtime.reuse:
                # Committed observations are admitted only by the ordinary
                # maintenance transaction (including replay/credit/Trace),
                # never by fixture-written lifecycle state.
                system.run_maintenance(triggering_task_id=completed[-1]['task_id'],
                    milestone='f2_before_persistent_reuse', finalize_pending=True)
                routes=[i for i in system.skills.implementations(mode='frozen')
                    if {e.predicate for e in system.skills.get_atomic(i.abstract_ref).effects}=={'time.progressed'}]
                assert len(routes)==1,'Expected one actually promoted Active persistent route'
                route=routes[0];atomic=system.skills.get_atomic(route.abstract_ref)
                # A caller-provided semantic count is a Plan anchor, not an
                # observed world fact. The stored route uses production R0/R1.
                occurrence=RuntimeOccurrence('reuse','reuse',atomic.ref,[],
                    {atomic.inputs[0].name:BindingExpression('constant',constant=3)},[route.ref],atomic.effects)
                system.orchestrator.planner.build_plan=lambda task,harness,**kw:RuntimeLinearPlan(
                    task.task_id,'new_workflow','',[occurrence],['reuse'],[],[],harness.task_contract(task),
                    {'selected_by':'controlled_persistent_reuse_fixture','task_contract_covered':False})
            sequence=store.mark_task_running(manifest.run_id,task.task_id,max_attempts=3)
            before_builder=len(builder.requests)
            try:
                trace=system.run_task(task,attempt_id=f'F2_attempt_{ordinal}_{sequence}')
            finally:
                atomic_write_json(root/'controlled_requests.json',{'runtime':runtime.requests,'builder':builder.requests})
            atomic_write_json(root/f'episode{ordinal}.json',to_primitive(trace))
            store.mark_task_completed(manifest.run_id,task.task_id,trace_id=trace.trace_id,result={'knowledge_digest_after':system.knowledge_digest()})
            assert not trace.infrastructure_failure
            if ordinal<3:
                trials=trace.metadata.get('runtime_tool_trials',{})
                assert len(trials)==1,to_primitive(trace)
                trial=next(iter(trials.values()))
                assert trial['r1']['admission_eligible'],trial
                tools=trace.tool_executions
                assert any(t.result['executed_step_count']==3 and t.result['completed'] for t in tools),to_primitive(tools)
            else:
                assert runtime.reuse_verified,'Missing persistent reuse boundary proof'
                assert any(t.result['started'] and t.result['completed'] for t in trace.tool_executions)
            completed.append({'task_id':task.task_id,'official_score':trace.official_score,
                'trace_id':trace.trace_id,'support_sources':len(system.runtime_support_store.committed()),
                'builder_calls':len(builder.requests)-before_builder})
            atomic_write_json(root/'progress.json',completed)
        observations=system.runtime_support_store.committed()
        assert len(observations)>=2
        assert tool_windows and all(w['provider_delta']==0 for w in tool_windows)
        atomic_write_json(root/'result.json',{'passed':True,'not_a_formal_result':True,'episodes':completed,
            'observations':observations,'action_request_counters':action_counts,
            'persistent_reuse_without_builder':True})
        atomic_write_json(root/'controlled_requests.json',{'runtime':runtime.requests,'builder':builder.requests})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--resume',action='store_true')
    run(**vars(p.parse_args()))
