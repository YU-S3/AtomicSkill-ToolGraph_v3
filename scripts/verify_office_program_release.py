"""Zero-model replay of saved Office operations against production Docker/Bank.

This derives controlled engineering fixtures, never naturally learned assets.
All paths are explicit; output must be new and cannot be a formal Bank.
"""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from atomic_skillgraph.empirical import LEARNING_MATERIAL_VERSION, BINDING_POLICY_VERSION
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.local_validation import canonical_binding
from atomic_skillgraph.empirical.model_view import expand_material
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments.run_empirical import code_identity, write_json
from atomic_skillgraph.harness.simple_protocol import Broker
from skillcompiler_bench_contracts.office import grep, select_paths, _cached_scan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_fixture(source, names):
    """Parameterize explicit top-level literals; retain the recorded calculation."""
    tree=ast.parse(source);values={};refs={};types={}
    for index,node in enumerate(tree.body):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
        name=node.targets[0].id
        if name not in names:continue
        value=ast.literal_eval(node.value);values[name]=value
        refs[name]={'kind':'source_literal','path':['body',index,'value']}
        types[name]=({'type':'array','items':{'type':'string' if isinstance(value[0],str) else 'number'}}
                     if isinstance(value,list) else {'type':'number'})
        node.value=ast.parse('inputs['+repr(name)+']',mode='eval').body
    assert set(values)==set(names)
    last=tree.body[-1]
    assert isinstance(last,ast.Expr) and isinstance(last.value,ast.Call) and last.value.func.id=='print'
    expression=ast.unparse(last.value.args[0])
    tree.body[-1]=ast.parse("return {'status':'ok','outputs':{'computed':json.loads("+expression+")}}").body[0]
    function=ast.parse('def run(ctx, inputs):\n    pass').body[0];function.body=tree.body
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))),values,refs,object_schema(types,list(types))


def replay(root, config, task, source, parameters, *, original_reference=None):
    root.mkdir()
    config=deepcopy(config);config['data_dir']=str(root/'bank');config['experiment']['output_dir']=str(root)
    config['learning']['material_version']=LEARNING_MATERIAL_VERSION
    config['learning']['binding_policy']=BINDING_POLICY_VERSION
    config.pop('budget',None)  # Fixture responses cannot enter a real cost ledger.
    os.environ[config['llm']['api_key_env']]='offline-intercepted-fixture'
    system=EmpiricalSystem(config)
    write_json(root/'config.json',system.config)
    frozen=None
    try:
        system.adapter.reset(task)
        broker=Broker(system.adapter,64)
        started=time.perf_counter();result=broker.call('execute_python',{'source':source});elapsed=time.perf_counter()-started
        write_json(root/'source_replay.json',{'task':asdict(task),'source':source,'result':result,'tools':broker.events,'seconds':elapsed})
        assert result['accepted'] and 'computed' in result['data'] and not broker.unknown
        assert all(e['state']=='finished' for e in broker.events)
        if original_reference is not None:assert result['data']['computed']==original_reference
        trace={'task':asdict(task),'tools':broker.events,'execution':{},'score':None}
        experience=system.learner._experience(task,trace);evidence=experience['local_evidence'][0]
        compiled,values,refs,input_schema=compile_fixture(source,parameters)
        spec={'goal':'Replay the recorded local operation','execution_intent':'program_requested','result_role':'intermediate',
              'input_schema':input_schema,'output_schema':object_schema({'computed':{'type':'object'}},['computed'])}
        proposed={'case_id':task.physical_key,'inputs':values,'input_refs':refs,'local_evidence_ref':evidence['id'],
                  'reference_fields':{'computed':['data','computed']}}
        canonical_binding(proposed,experience,spec)
        proposal={'decision':'propose_skill_and_program_spec','skill':spec,
                  'realization_request':{'skill_id':'$new','action':'build','case_bindings':[proposed]}}
        pending=iter([('submit_learning',proposal),('submit_program',{'source':compiled})]);sent=[]
        import requests
        original_post=requests.post
        def fixture_post(url,**kwargs):
            name,value=next(pending);sent.append(deepcopy(kwargs['json']))
            body={'model':'offline-fixture','choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':'',
                'tool_calls':[{'id':'fixture','type':'function','function':{'name':name,'arguments':json.dumps(value)}}]}}],
                'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}
            return SimpleNamespace(ok=True,status_code=200,headers={},json=lambda:body)
        requests.post=fixture_post
        try:learned=system.learner.learn(task,trace)
        finally:
            requests.post=original_post
            write_json(root/'intercepted_wire.json',{'actual_http':0,'fixture_responses':True,'payloads':sent,'requests':system.requests})
        assert not learned['errors'] and len(sent)==2,learned
        program=system.bank.all('program')[0]
        binding=system.bank.jobs()[0]['case_bindings'][0]
        trial=system.bank.attempts(program['id'])[0]
        assert trial['outcome']=='positive',trial['validation']
        program=system.bank.get(program['id']);assert system.bank.program_eligible(program)
        system.bank.freeze(root/'frozen_bank');frozen=Bank(root/'frozen_bank',readonly=True)
        before=frozen.digest()
        # Same physical source verifies replay only. Cross-physical consumption is
        # independently covered by test_office_program_release, never asserted here.
        def consume(stage,prompt,material,*args,**kwargs):
            data=expand_material(material)['bindings']['inputs']['computed']
            assert data==result['data']['computed']
            return {'action':'finish','answer':json.dumps(data,ensure_ascii=False),'used_inputs':['computed']}
        execution=Executor(frozen,consume,system.worker,system.planner,frozen=True).run(task,system.adapter,Broker(system.adapter,64),
            {'nodes':[{'id':'operation','execution_mode':'program','program_id':program['id'],
                       'args':{k:{'literal':v} for k,v in values.items()}},
                      {'id':'consume','execution_mode':'dynamic','goal':'consume the acquired object',
                       'args':{'computed':{'from':'operation','field':'computed'}}}]})
        assert execution['attempts'][0]['outputs_consumed'] and frozen.digest()==before
        write_json(root/'experience.json',experience);write_json(root/'binding.json',binding)
        write_json(root/'trial.json',trial);write_json(root/'frozen_consumption.json',execution)
        return {'passed':True,'fixture_only':True,'natural_learning_claim':False,'model_http':0,
                'program_id':program['id'],'source_seconds':elapsed,'source_native_calls':len(broker.events),
                'frozen_before':before,'frozen_after':frozen.digest()}
    finally:
        if frozen:frozen.close()
        system.close()


def main():
    p=argparse.ArgumentParser()
    for name in ('parent','compute-trace','corpus','original-corpus','output','commit'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.output);assert not root.exists()
    code=code_identity();assert code['git_sha']==args.commit and not code['tracked_dirty']
    # Any accidental provider request fails before network transmission.
    import requests
    def forbidden(*args,**kwargs):raise AssertionError('Zero-model acceptance forbids HTTP')
    requests.sessions.Session.request=forbidden
    root.mkdir(parents=True);write_json(root/'identity.json',{'code':code,'model_http':0,'fixture_only':True})
    source_files=select_paths(args.original_corpus);mirror_files=select_paths(args.corpus)
    original={name:sha(path) for name,path in source_files};mirror={name:sha(path) for name,path in mirror_files}
    assert len(mirror)==697 and original==mirror
    assert all(path.stat().st_mode & 0o222==0 for _,path in mirror_files)
    write_json(root/'corpus.json',{'original':args.original_corpus,'mirror':args.corpus,'count':len(mirror),
        'bytes':sum(path.stat().st_size for _,path in mirror_files),'file_sha256':mirror})
    parent=Path(args.parent)
    request=json.loads((parent/'train/requests/officeqa:UID0221_attempt1.json').read_text())
    sources=[call['arguments']['arguments']['source'] for row in request['requests']
             for call in row.get('response',{}).get('tool_calls',[]) if call['arguments'].get('name')=='execute_python']
    source=sources[-1]
    patterns=next(ast.literal_eval(n.value) for n in ast.parse(source).body
                  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='pats')
    comparisons=[];_cached_scan.cache_clear()
    for pattern in [*patterns,'NO_MATCH_office_release_7d8b2a']:
        row={'pattern':pattern};results=[]
        for label,corpus,cached in [('original_uncached',args.original_corpus,False),
                                  ('mirror_uncached',args.corpus,False),('mirror_cold',args.corpus,True),
                                  ('mirror_cached',args.corpus,True)]:
            start=time.perf_counter();result=grep(corpus,pattern,use_cache=cached);row[label]=time.perf_counter()-start
            results.append(result)
        assert all(value==results[0] for value in results)
        row.update(result_sha256=hashlib.sha256(json.dumps(results[0],sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                   hit_count=len(results[0][0]),truncated=results[0][1]['truncated'],equal=True)
        comparisons.append(row);write_json(root/'search_equivalence.json',comparisons)
    config=json.loads((parent/'train/config.json').read_text());config['harness']['corpus_root']=args.corpus
    task=PublicTask(**next(t for t in json.loads((parent/'train/execution_manifest.json').read_text())['tasks']
                         if t['task_id']=='officeqa:UID0221'))
    search=replay(root/'uid0221',config,task,source,['pats'])
    recorded=json.loads(Path(args.compute_trace).read_text())
    compute_event=next(e for e in recorded['tools'] if 'np.linalg.solve' in e.get('arguments',{}).get('source','')
                       and e.get('result',{}).get('accepted'))
    compute=replay(root/'computation',config,PublicTask(**recorded['task']),compute_event['arguments']['source'],
                   ['y1971','y1972','y1973','actual_jul73'],original_reference=compute_event['result']['data']['computed'])
    write_json(root/'summary.json',{'passed':True,'code':code,'model_http':0,'source_request_sha256':sha(parent/'train/requests/officeqa:UID0221_attempt1.json'),
        'compute_trace':args.compute_trace,'compute_trace_sha256':sha(args.compute_trace),
        'diagnostic_derivation':recorded.get('diagnostic_derivation'),'search':search,'computation':compute,
        'search_equivalent':True,'corpus_files':len(mirror),'natural_learning_claim':False})


if __name__=='__main__':main()
