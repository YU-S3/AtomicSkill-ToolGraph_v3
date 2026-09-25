"""Real-JVM qualification and isolated publication of the authored reference.

The fixture inputs below are not installed in the bank or runtime. Only Train
worlds and their current public catalogs are inspected. Failed/uncovered assets
remain Draft; publication never manufactures DIRECT_SUCCESS or training credit.
"""
import argparse
import copy
import json
from pathlib import Path

from atomic_skillgraph.core.results import RuntimeLinearPlan,RuntimeOccurrence
from atomic_skillgraph.core.serialization import atomic_write_json,to_primitive
from atomic_skillgraph.deployment.scienceworld_reference import reference_atomics
from atomic_skillgraph.deployment.scienceworld_workflows import reference_workflows
from atomic_skillgraph.deployment.train_bank_compiler import clone_bank,bank_digest
from atomic_skillgraph.deployment.release_protocol import DDL,sha,verify_deployments
from atomic_skillgraph.deployment.workflow_entry_closure import audit_workflow
from atomic_skillgraph.evolution.identity_matching import raw_hash
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.tool_runner import ToolRunner
from atomic_skillgraph.validation.tool_validator import ToolValidator
from atomic_skillgraph.validation.atomic_validator import AtomicValidator
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from atomic_skillgraph.traces.schema import EnvironmentActionRecord
from atomic_skillgraph.system import AtomicSkillGraphSystem,load_config
from .scienceworld_config import make_config
from .scienceworld_manifest import load,task_from_entry
from .fakes import FakeProviderRequest,FakeReply


def public_action(ctx,kind,arguments):
    candidates=[a for a in ctx.action_catalog if a.action_type==kind and a.arguments==arguments]
    if len(candidates)!=1: raise ValueError(f'Fixture action not uniquely public: {kind} {arguments}')
    a=candidates[0];ctx.budget.consume_action()
    r=ctx.harness.execute_action(a.action_id,a.revision)
    event=EnvironmentActionRecord(a.action_id,a.revision,kind,arguments,r.accepted,r.observation,r.done,r.won,r.new_revision,'fixture_setup')
    ctx.trace_builder.trace.environment_actions.append(event)
    ctx.update_after_action(r,{**to_primitive(event),'occurrence_id':'','origin':'qualification_fixture'})
    if not r.accepted:raise ValueError(r.observation)


def fixture_inputs(number,ctx):
    public_action(ctx,'TELEPORT',{'destination':'workshop' if number in (10,11,28) else 'kitchen'})
    room='workshop' if number in (10,11,28) else 'kitchen'
    rows={1:{'destination':'kitchen'},2:{'container':'freezer'},3:{'entity':'thermometer'},
        4:{'entity':'thermometer','container':'counter'},5:{'entity':'thermometer'},6:{'entity':'thermometer'},
        8:{'device':'stove'},9:{'device':'stove'},12:{'instrument':'thermometer','target':'counter'},
        13:{'source':'glass cup','destination':'bowl'},14:{'container':'bowl'},15:{},
        16:{'thermometer':'thermometer','target':'counter'},17:{'location':'kitchen'},
        18:{'query':'thermometer','locations':['hallway','bathroom','kitchen']},
        19:{'query':'sodium chloride','location':'kitchen','containers':['glass jar']},20:{'wait_steps':3},
        24:{'query':'thermometer','locations':['kitchen']},
        25:{'query':'sodium chloride','location':'kitchen','containers':['glass jar']},
        26:{'query':'thermometer','locations':['kitchen']},
        29:{'source':'glass cup','destination':'bowl'},30:{'wait_steps':3,'thermometer':'thermometer','target':'counter'}}
    if number==9:public_action(ctx,'ACTIVATE',{'device':'stove'})
    if number in (10,11,28):
        pairs=sorted([a.arguments for a in ctx.action_catalog if a.action_type=='CONNECT'],key=lambda a:json.dumps(a,sort_keys=True))
        if not pairs:raise ValueError('No public CONNECT tuple in fixture')
        pair=pairs[0]
        if number==11:
            public_action(ctx,'CONNECT',pair);return {'entity':pair['left']}
        if number==28:return {**pair,'device':'switch'}
        return pair
    if number in (7,27):
        rooms=sorted({a.arguments['destination'] for a in ctx.action_catalog if a.action_type=='TELEPORT'})
        for room in rooms:
            public_action(ctx,'TELEPORT',{'destination':room})
            candidates=sorted({a.arguments['entity'] for a in ctx.action_catalog if a.action_type=='READ'})
            if candidates:
                return {'entity':candidates[0]} if number==7 else {'query':candidates[0],'locations':[room]}
        raise ValueError('No public READ fixture in this Train world')
    if number in (21,22,23):
        queries=['query'] if number==21 else ['query_'+c for c in 'abc'[:number-20]]
        return {**{q:'thermometer' for q in queries},'locations':['kitchen'],'wait_steps':1}
    return rows[number]


class EntryProvider:
    def __init__(self,arguments):self.arguments=arguments;self.calls=0
    def snapshot(self):return {'provider':'reference_entry_fixture','fixture_generated':True}
    def complete(self,messages,*,tools):
        self.calls+=1
        request=FakeProviderRequest(tuple(messages),tuple(tools))
        if self.calls!=1:raise RuntimeError('Post-bootstrap Agent request: '+json.dumps(request.policy_context.get('execution_frame',{})))
        offered=[t.name for t in tools if t.name.startswith('invoke_impl_')]
        if not offered:raise RuntimeError(json.dumps({'offered':[t.name for t in tools],'context':request.policy_context}))
        name=offered[0]
        return FakeReply.tool(name,self.arguments,prompt_tokens=0,completion_tokens=0,reasoning_tokens=0).materialize(
            call_id='reference_bootstrap',tools=tools,request=request)


def run(root, resume=False):
    root=Path(root).resolve()
    cfg=load_config(make_config(root))
    cfg['data_dir']=str(root/'bank');cfg['trace_data_dir']=str(root/'audit')
    cfg['experiment']['task_manifest_path']=None
    if (root/'bank/state.sqlite3').exists() and not resume:raise FileExistsError('Use --resume for a qualified prefix')
    manifest=load(Path(cfg['harness']['manifest']))
    rows=reference_atomics();graphs=reference_workflows()
    audits=json.loads((root/'audit/real_jvm_replays.json').read_text()) if resume and (root/'audit/real_jvm_replays.json').exists() else []
    qualified=set()
    with AtomicSkillGraphSystem(cfg,harness=ScienceWorldAdapter(),provider=EntryProvider({})) as system:
        for atomic,impl,tool in rows:
            static=ToolStaticValidator().validate_tool_asset(tool,atomic,system.harness)
            if not static.passed:raise ValueError(to_primitive(static))
            system.skills.register_atomic(atomic);system.tools.register(tool);system.skills.register_implementation(impl)
        for graph in graphs:system.skills.register_composite(graph)
        qualified={r['artifact_ref'] for r in system.database.rows("SELECT artifact_ref FROM artifact_index WHERE status='active'")}
        for number,(atomic,impl,tool) in enumerate(rows,1):
            if len([a for a in audits if a['id']==f'A{number:02d}'])==2:
                continue
            category='3-1' if number in (10,11,28) else '2-1'
            entries=[e for e in manifest['tasks'] if e['task_type']==category][:2]
            passes=[]
            for entry in entries:
                task=task_from_entry(entry,'')
                plan=RuntimeLinearPlan.full_dynamic(task.task_id,system.harness.task_contract(task),reason='authored_qualification')
                ctx=system.orchestrator._create_context(task,plan)
                check={'id':f'A{number:02d}','task_id':task.task_id,'task_signature':entry['task_signature'],'passed':False}
                try:
                    arguments=fixture_inputs(number,ctx)
                    result=ToolRunner(ToolValidator()).run(tool,arguments,ctx,occurrence_id=check['id'])
                    occurrence=RuntimeOccurrence(check['id'],check['id'],atomic.ref,[],{},[impl.ref],atomic.effects)
                    r1=AtomicValidator().validate_execution_result(atomic,occurrence,arguments,result.output_candidates,
                        system.harness.validator_channel(),current_revision=ctx.world_revision,
                        semantic_compatible=system.harness.semantic_value_compatible)
                    check.update(arguments=arguments,result=to_primitive(result),r1=to_primitive(r1),
                        passed=bool(result.completed and r1.passed),provider_request_delta=0)
                except (ValueError,KeyError,StopIteration) as exc:
                    check['fixture_failure']=str(exc)
                tracepath=root/'audit/replays'/f'A{number:02d}_{entry["variation_idx"]}.json'
                atomic_write_json(tracepath,to_primitive(ctx.trace_builder.trace))
                check.update(trace_path=str(tracepath),trace_sha256=sha(tracepath))
                audits.append(check);passes.append(check['passed'])
                atomic_write_json(root/'audit/real_jvm_replays.json',audits)
            if all(passes):
                for asset in (atomic,impl,tool):
                    system.database.execute("UPDATE artifact_index SET status='active' WHERE artifact_ref=?",(str(asset.ref),))
                    qualified.add(str(asset.ref))
                system.database.connection.commit()
            print(json.dumps({'asset':f'A{number:02d}','independent_replays':passes}),flush=True)
        # G03 exercises the actual Composite executor and explicit DataFlow.
        # Other fragments remain in the inventory until their own two-source
        # qualification; none is granted official task-contract coverage.
        from atomic_skillgraph.planner.compiler import PlanCompiler
        from atomic_skillgraph.runtime.composite_executor import VerifiedCompositeExecutor
        graph=graphs[2];workflow_checks=[]
        closure=audit_workflow(system.skills,graph,system.harness)
        for entry in [e for e in manifest['tasks'] if e['task_type']=='2-1'][:2]:
            provider=EntryProvider({'query_a':'counter','query_b':'thermometer','locations':['hallway','bathroom','kitchen']})
            system._provider_override=provider
            task=task_from_entry(entry,'')
            system._current_task_id=task.task_id
            system._current_task_usage_start=len(system.usage.events)
            plan=PlanCompiler(system.skills).from_composite(task,system.harness.task_contract(task),graph,mode='frozen',audit={
                'selected_by':'explicit_partial_workflow_integration','task_contract_covered':False})
            # The caller's search intent is a semantic plan anchor, not a
            # fabricated discovered entity. The Agent still submits the entry
            # call; its identity outputs must traverse normal DataFlow.
            from atomic_skillgraph.core.bindings import BindingExpression
            plan.occurrences[0].binding_specs={k:BindingExpression('constant',constant=v)
                for k,v in provider.arguments.items() if k.startswith('query')}
            ctx=system.orchestrator._create_context(task,plan);ctx.runtime_config=cfg['runtime']
            results=[]
            for step in plan.control_sequence:
                occurrence=plan.occurrence(step);atomic=system.skills.get_atomic(occurrence.node_ref)
                ctx.binding_store.apply_data_flow(plan,step,ctx.validated_outputs,revision=ctx.world_revision)
                ctx.binding_store.resolve_occurrence_specs(occurrence,ctx.world_revision,input_specs=atomic.inputs)
                ctx.begin_occurrence(occurrence)
                result=VerifiedCompositeExecutor(system.orchestrator.node_executor).run_occurrence(occurrence,ctx)
                results.append(to_primitive(result))
                if not result.atomic_effect_passed:break
                ctx.binding_store.publish_validated_outputs(occurrence,result.validated_outputs,result.atomic_witness_refs,
                    ctx.world_revision,certified_bindings=result.validated_output_bindings)
                ctx.validated_outputs[occurrence.occurrence_id]=result.validated_outputs
            passed=(len(results)==len(plan.occurrences) and all(r['atomic_effect_passed'] for r in results)
                and len(ctx.trace_builder.trace.environment_actions)>=3 and provider.calls==1 and closure['post_bootstrap_program_closed'])
            tracepath=root/'audit/replays'/f'G03_{entry["variation_idx"]}.json'
            atomic_write_json(tracepath,to_primitive(ctx.trace_builder.trace))
            workflow_checks.append({'passed':passed,'task_id':task.task_id,'task_signature':entry['task_signature'],
                'results':results,'bootstrap_requests':provider.calls,'post_bootstrap_requests':max(0,provider.calls-1),
                'selected_by':'explicit_partial_workflow_integration','trace_path':str(tracepath),'trace_sha256':sha(tracepath)})
        if not all(r['passed'] for r in workflow_checks):raise RuntimeError('Authored SW-WF01 failed')
        system.database.execute("UPDATE artifact_index SET status='active' WHERE artifact_ref=?",(str(graph.ref),))
        qualified.add(str(graph.ref));system.database.connection.commit()
        atomic_write_json(root/'audit/zero_llm_workflows.json',workflow_checks)
        closures=[audit_workflow(system.skills,g,system.harness) for g in graphs]
        atomic_write_json(root/'audit/workflow_entry_closure.json',closures)
        system.database.execute(DDL)
        for r in system.database.rows('SELECT * FROM artifact_index ORDER BY artifact_ref'):
            ref=r['artifact_ref'];digest=raw_hash(system.artifacts.get_payload(ref))
            receipt={'passed':True,'experiment_kind':'authored_reference','artifact_ref':ref,'source_payload_hash':digest,
                'effective_status':r['status'],'execution_credit_delta':0,'historical_direct_success_claimed':False,
                'qualification':workflow_checks if ref==str(graph.ref) else [a for a in audits if a['id']==system.artifacts.get_payload(ref).get('metadata',{}).get('reference_inventory_id')]}
            relative=Path('publication_checks')/(raw_hash(ref)+'.json');atomic_write_json(system.data_dir/relative,receipt)
            system.database.execute('INSERT INTO release_deployments VALUES(?,?,?,?,?,?,?,?,?)',
                (ref,r['artifact_kind'],r['status'],'authored_revision',ref,digest,digest,str(relative),sha(system.data_dir/relative)))
        system.database.connection.commit();verify_deployments(system.database,system.data_dir)
        inventory=[{'ref':r['artifact_ref'],'kind':r['artifact_kind'],'status':r['status']} for r in system.database.rows('SELECT * FROM artifact_index ORDER BY artifact_ref')]
        atomic_write_json(root/'audit/reference_inventory.json',inventory)
    frozen=root/'frozen/data_v3';clone_bank(root/'bank',frozen);digest=bank_digest(frozen)
    atomic_write_json(frozen/'freeze_manifest.json',{'schema_version':3,'knowledge_digest':digest,'experiment_kind':'authored_reference'})
    atomic_write_json(root/'frozen/digest.json',{'digest':digest})
    atomic_write_json(root/'audit/release_checks.json',{'go':True,'experiment_kind':'authored_reference','learned_train_asset_source':False,
        'frozen':str(frozen),'digest':digest,'registered_assets':len(inventory),'qualified_active_assets':len(qualified),
        'unqualified_assets_remain_draft':True,'partial_workflow_not_p0':True,'sw_wf01_passed':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--resume',action='store_true')
    run(**vars(p.parse_args()))
