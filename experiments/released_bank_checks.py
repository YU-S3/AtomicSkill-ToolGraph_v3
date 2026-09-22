"""Bounded real-world acceptance of published programs, not an evaluation policy.

Only this test fixes entry selection. Every value comes from public catalog;
all binding, preflight, Tool execution and output checks are production code.
"""
import copy
from pathlib import Path
from atomic_skillgraph.core.bindings import BindingExpression
from atomic_skillgraph.core.edges import GraphEdge
from atomic_skillgraph.core.results import RuntimeOccurrence,RuntimeLinearPlan
from atomic_skillgraph.core.serialization import atomic_write_json,to_primitive
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.runtime.invocation_transaction import execute_invocation
from atomic_skillgraph.traces.compiler_observer import initialize,finalize,node_window
from atomic_skillgraph.system import AtomicSkillGraphSystem
from .run_v3_r103_validation import resolve_tasks


class NoModel:
    def complete(self,messages,*,tools=None):
        raise AssertionError('program/automatic entry unexpectedly requested a model')
    def snapshot(self):return {'provider':'no_model_acceptance','fixture_generated':True}


def run(config,entry,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(config);config['trace_data_dir']=str(output)
    with AtomicSkillGraphSystem(config,readonly=True,provider=NoModel()) as system:
        before=system.knowledge_digest()
        task=resolve_tasks(system,[entry])[0]
        atomics={a.metadata.get('catalog_id'):a for a in system.skills.atomics(mode='frozen')
                 if a.metadata.get('release_revision')}
        impls={i.metadata.get('catalog_id'):i for i in system.skills.implementations(mode='frozen')
               if i.metadata.get('release_revision')}
        occurrences=[]
        for step,code in [('inspect','C01'),('acquire','C03'),('return','C05')]:
            a=atomics[code];i=impls[code]
            occurrences.append(RuntimeOccurrence(step,step,a.ref,[],{},[i.ref],a.effects))
        inspect,acquire,deliver=occurrences
        edges=[GraphEdge('inspect_source','data_flow','inspect','acquire','location','source'),
               GraphEdge('inspect_permission','data_flow','inspect','acquire','allow_open','allow_open'),
               GraphEdge('take_object','data_flow','acquire','return','object','object'),
               GraphEdge('return_permission','data_flow','acquire','return','allow_open','allow_open'),
               GraphEdge('return_destination','data_flow','inspect','return','location','destination')]
        for edge in edges:
            next(o for o in occurrences if o.step_id==edge.target_step).binding_specs[edge.target_role]=BindingExpression(
                'data_flow',source_step=edge.source_step,source_role=edge.source_role)
        plan=RuntimeLinearPlan(task.task_id,'atomic_composition',None,occurrences,
            [o.step_id for o in occurrences],edges,[],system.harness.task_contract(task),{})
        builder=system.orchestrator.create_trace_builder(task)
        builder.trace.metadata['experiment_kind']='controlled_release_automation_acceptance'
        initialize(builder,persistent_refs=[str(t.ref) for t in system.tools.tools()],request_snapshot=lambda:[])
        ctx=TaskRuntimeContext.create(task,plan,system.harness,builder,RuntimeBudget(global_action_budget=100))
        ctx.runtime_config=config['runtime']
        executor=system.orchestrator.node_executor

        def begin(occ):
            ctx.begin_occurrence(occ);ctx.budget.begin_node(occ.occurrence_id)
            ctx.binding_store.apply_data_flow(plan,occ.step_id,revision=ctx.world_revision)
            ctx.binding_store.resolve_occurrence_specs(occ,ctx.world_revision,
                input_specs=system.skills.get_atomic(occ.node_ref).inputs)
            return system.invocation_compiler.compile_candidates(occ,ctx.binding_store,task_id=ctx.task_id)

        def publish(occ,result):
            if not (result.completed and result.atomic_effect_passed and not result.failure_code):
                builder.trace.metadata['acceptance_failure']=to_primitive(result)
                system.traces.save_atomic(builder.finish())
                raise AssertionError(to_primitive(result))
            ctx.binding_store.publish_validated_outputs(occ,result.validated_outputs,result.atomic_witness_refs,
                ctx.world_revision,certified_bindings=result.validated_output_bindings)
            ctx.validated_outputs[occ.occurrence_id]=result.validated_outputs

        def explicit(occ,args):
            candidates=begin(occ)
            if len(candidates)!=1:raise AssertionError('controlled published route not unique')
            compiled=candidates[0]
            preflight=system.invocation_compiler.preflight(compiled,call_name=compiled.spec.name,
                call_id='controlled_'+occ.step_id,arguments=args,occurrence=occ,binding_store=ctx.binding_store,
                evidence_store=ctx.evidence_store,revision=ctx.world_revision)
            if not preflight.passed:raise AssertionError(to_primitive(preflight))
            result=execute_invocation(executor.implementation_runner,compiled,preflight,occ,ctx,
                                      agent_prepared=True,authorizing_native_call_id='controlled_'+occ.step_id)
            publish(occ,result)
            return result

        scope=list(dict.fromkeys(a.arguments['destination'] for a in ctx.action_catalog if a.action_type=='GO_TO'))[:8]
        found=None
        for destination in scope:
            explicit(inspect,{'location':destination,'allow_open':True})
            found=next((a for a in ctx.action_catalog if a.action_type=='TAKE'),None)
            if found:break
        if found is None:raise AssertionError('bounded public scope exposed no TAKE; no hidden-state fallback')
        explicit(acquire,{'object':found.arguments['object'],'source':found.arguments['source'],'allow_open':True})
        candidates=begin(deliver);ctx.graph_bootstrap_completed=True
        with node_window(deliver,ctx):
            result=executor.try_autonomous(deliver,candidates,ctx)
        if result is None:raise AssertionError('explicit dataflow successor was not ready')
        publish(deliver,result)
        trace=builder.finish();finalize(trace)
        system.traces.save_atomic(trace)
        links=trace.metadata['compiler_observability']['program_invocation_links']
        flows=trace.metadata['compiler_observability']['dataflow_consumptions']
        multi=[p for p in links if p['complete_success'] and len(p['canonical_action_indices'] or [])>=2 and p['provider_request_refs']==[]]
        automatic=[p for p in links if p['origin']=='graph_entry_auto' and p['complete_success'] and p['provider_request_refs']==[]]
        consumed=[c for c in flows if c['consumer_occurrence_id']==deliver.occurrence_id and c['producer_invocation_id']]
        passed=bool(multi and automatic and consumed and before==system.knowledge_digest())
        report={'passed':passed,'entry_selection':'fixed public-only boundary acceptance, not natural planner',
            'trace_id':trace.trace_id,'multi_action_intervals':multi,'automatic_intervals':automatic,
            'dataflow_consumptions':consumed,'bank_digest_before':before,'bank_digest_after':system.knowledge_digest()}
        atomic_write_json(output/'acceptance.json',report)
        if not passed:raise AssertionError(report)
        return report


if __name__=='__main__':
    import argparse,json
    from .run_v3_r103_validation import declared_entries
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    config=json.loads((args.prepared_root/'prepare_manifest.json').read_text())['config']
    config['data_dir']=str(args.prepared_root/'work/data_v3')
    config['bank_release']={'source_seed':42}
    config['harness']['split']='train'
    entry=next(e for e in declared_entries() if int(e['task_id'].split('_')[2])==2)
    print(json.dumps(run(config,entry,args.output)),flush=True)
