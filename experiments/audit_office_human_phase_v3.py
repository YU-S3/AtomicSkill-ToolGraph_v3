"""Read-only replay of served policies, arithmetic ABIs and final evidence windows."""
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from atomic_skillgraph.empirical import office_human_policy as p
from atomic_skillgraph.empirical.contracts import PublicTask, validate_schema_instance
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.budget_governor import input_token_bound

ROOT=Path(__file__).parents[1]
PACKAGE=ROOT/'resources/office_human_frontier_v3'

def read(path):return json.loads(Path(path).read_text())

def main(archive,output):
    archive=Path(archive)
    spec=importlib.util.spec_from_file_location('office_phase_view',PACKAGE/'integration/frontier.py')
    frontier=importlib.util.module_from_spec(spec);spec.loader.exec_module(frontier)
    bundle=read(PACKAGE/'bank/bank.json')
    bank=SimpleNamespace(readonly=True,environment=bundle['program'][0]['environment'],
        all=lambda kind:bundle[kind],get=lambda key:next((a for kind in frontier.KINDS for a in bundle[kind] if a['id']==key),None))
    view=frontier.HumanBankView(bank,bundle)
    policies=[];finishes=[];abis=[]
    for trace_path in sorted((archive/'test/traces').glob('*.json')):
        tr=read(trace_path);task=PublicTask(**tr['task']);view.bind_task(task)
        state=read(archive/'test/checkpoints'/task.task_id/'1/state.json')
        end=TaskContext()
        for k,v in state['executor_state']['context'].items():setattr(end,k,deepcopy(v))
        audits=read(archive/'test/requests'/f'{task.task_id}_attempt1.json')['requests']
        for q in audits:
            if q['purpose']!='runtime':continue
            material=json.loads(q['messages'][1]['content'])
            known=set()
            def refs(v):
                if isinstance(v,dict):
                    if isinstance(v.get('result_id'),str):known.add(v['result_id'])
                    for child in v.values():refs(child)
                elif isinstance(v,list):
                    for child in v:refs(child)
            refs(material)
            ctx=TaskContext()
            for k in ['scope','results','result_origins','sources','window_sources']:
                setattr(ctx,k,deepcopy(getattr(end,k)))
            ctx.memory={k:deepcopy(v) for k,v in end.memory.items() if v['result_id'] in known}
            ctx.results={k:v for k,v in ctx.results.items() if k in known}
            remaining=material.get('recovery',{}).get('remaining_calls',48)
            observed=p.progress(ctx,{},remaining)
            selected=view.selected(task,observed_progress=observed)
            policies.append({'task_id':task.task_id,'request_id':q['id'],
                'old_count':len(material.get('human_frontier_skill_ids',[])),
                'new_count':len(selected),'new_phase':observed['phase'],
                'replay_scope':'only results referenced in this saved request; no synthesized candidate'})
        q=next((q for q in reversed(audits) if q['purpose']=='finish_only'),None)
        if q:
            material=json.loads(q['messages'][1]['content']);prompt=q['messages'][0]['content']
            payload=deepcopy(q['http_attempts'][0]['final_payload_audit']['serialized_parameters'])
            def wire(value,cap):
                return {**payload,'messages':[{'role':'system','content':prompt},
                    {'role':'user','content':json.dumps(value,ensure_ascii=False,allow_nan=False)}],'tools':[]}
            rebuilt=p.finish_evidence(end,material,prompt,max_bytes=16384,output_cap=2048,
                payload_builder=wire,state={},policy=view.policy(task,'finalize'))
            assert input_token_bound(wire(rebuilt['materials'],2048))==rebuilt['serialized_input_bytes']<=16384
            assert all(end.resolve_public(w['source_ref']) is not None for w in rebuilt['materials']['finish_evidence'])
            finishes.append({'task_id':task.task_id,'request_id':q['id'],
                'old_omitted':len(tr['execution']['finish_evidence']['omitted_refs']),
                'new_evidence':rebuilt})
        for a in tr['execution']['attempts']:
            new=next(x for x in bundle['program'] if Path(x['source_file']).name=='series_metrics.py')
            error=None
            try:validate_schema_instance(a['arguments'],new['input_schema'])
            except ValueError as exc:error=str(exc)
            abis.append({'task_id':task.task_id,'attempt_id':a['id'],'old_program_id':a['program_id'],
                'original_arguments':a['arguments'],'new_program_id':new['id'],
                'old_arguments_accepted_by_new_abi':error is None,'validation_detail':error,
                'historical_arguments_rewritten':False})
    assert len(policies)==816 and len(finishes)==84 and len(abis)==11
    assert max(x['new_count'] for x in policies)<=2
    result={'schema':'office.human.phase.v3.offline-replay','paid_http':0,
        'historical_score_used_for_routing':False,'runtime_requests':len(policies),
        'old_skill_count':dict(Counter(x['old_count'] for x in policies)),
        'new_skill_count':dict(Counter(x['new_count'] for x in policies)),
        'finish_requests':len(finishes),'program_abis':len(abis),
        'max_finish_bytes':max(x['new_evidence']['serialized_input_bytes'] for x in finishes),
        'policies':policies,'finish_replays':finishes,'abi_audit':abis,
        'limitation':'Offline visibility and integrity check only; no accuracy claim and no inferred ready candidates.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['policies','finish_replays','abi_audit']},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();main(args.archive,args.output)
