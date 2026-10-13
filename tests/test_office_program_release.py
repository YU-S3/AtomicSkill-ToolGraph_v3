"""Controlled declarations, real Docker/source qualification/Frozen consumption; zero LLM."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask, object_schema
from atomic_skillgraph.empirical.executor import Executor
from atomic_skillgraph.empirical.local_validation import canonical_binding, source_literals, validate_on_source
from atomic_skillgraph.empirical.model_view import operation_directory, expand_material
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from skillcompiler_bench_contracts.office import grep, _cached_scan
from test_cf2_contracts import office, http, response
from test_empirical import worker


CHAIN = r'''import json, re
pattern = '源值'
factor = 3
hits = ctx.call('grep', {'pattern':pattern})['data']
page = ctx.call('read', {'path':hits[0]['path'], 'offset':hits[0]['offset']})['data']
value = float(re.search(r'=(\d+)', page['text']).group(1))
print(json.dumps({'path':hits[0]['path'], 'offset':hits[0]['offset'], 'result':value*factor}))
'''
COMPILED = r'''def run(ctx, inputs):
    import re
    search = ctx.call('grep', {'pattern':inputs['pattern']})
    if not search.get('accepted'): return {'status':'execution_error','outputs':{}}
    if not search['data']: return {'status':'not_found','outputs':{}}
    hit = search['data'][0]
    page = ctx.call('read', {'path':hit['path'], 'offset':hit['offset']})
    if not page.get('accepted'): return {'status':'execution_error','outputs':{}}
    value = float(re.search(r'=(\d+)', page['data']['text']).group(1))
    return {'status':'ok','outputs':{'path':hit['path'],'offset':hit['offset'],'result':value*inputs['factor']}}
'''


def prepare(tmp_path, worker, *, chain):
    s=office(tmp_path);s.worker=worker;s.adapter.worker=worker
    worker.settings['wall_timeout_seconds']=20
    (s.adapter.corpus/'a.txt').write_bytes('前言\r\n源值=7\r\n'.encode())
    (s.adapter.corpus/'b.txt').write_text('标题\nUnicode 次值=11\n',encoding='utf-8')
    s.adapter_factory=lambda:OfficeAdapter({'office':{'answer':'private'}},s.config)
    broker=Broker(s.adapter,24)
    args={'source':CHAIN} if chain else {'pattern':'源值'}
    assert broker.call('execute_python' if chain else 'grep',args)['accepted']
    trace={'tools':broker.events,'execution':{'prediction':'wrong'},'score':{'hard':False}}
    experience=s.learner._experience(s.adapter.task,trace)
    evidence=experience['local_evidence'][0]
    ins={'pattern':{'type':'string'},**({'factor':{'type':'number'}} if chain else {})}
    outs=({'path':{'type':'string'},'offset':{'type':'integer'},'result':{'type':'number'}} if chain
          else {'hits':{'type':'array','items':{'type':'object'}}})
    values={'pattern':'源值',**({'factor':3} if chain else {})}
    literals=source_literals(CHAIN)
    refs=({k:next({a:r[a] for a in ('kind','path')} for r in literals if r['value']==v)
           for k,v in values.items()} if chain else {'pattern':{'kind':'operation_json','path':['pattern']}})
    fields=({k:['data','computed',k] for k in outs} if chain else {'hits':['data']})
    binding={'case_id':s.adapter.task.physical_key,'local_evidence_ref':evidence['id'],
             'inputs':values,'input_refs':refs,'reference_fields':fields}
    spec={'goal':'Retrieve bounded local evidence'+(' and compute a scaled value' if chain else ''),
          'execution_intent':'program_requested','result_role':'intermediate',
          'input_schema':object_schema(ins,list(ins)), 'output_schema':object_schema(outs,list(outs))}
    source=COMPILED if chain else "def run(ctx, inputs):\n    result=ctx.call('grep',{'pattern':inputs['pattern']})\n    return {'status':'ok','outputs':{'hits':result['data']}}"
    proposal={'decision':'propose_skill_and_program_spec','skill':spec,
              'realization_request':{'skill_id':'$new','action':'build','case_bindings':[binding]}}
    return s,trace,experience,evidence,proposal,source


@pytest.mark.parametrize('chain',[False,True])
def test_source_to_frozen_different_physical_task(tmp_path,monkeypatch,worker,chain):
    s,trace,experience,evidence,proposal,source=prepare(tmp_path,worker,chain=chain)
    sent=http(monkeypatch,[response([proposal],name='submit_learning'),response([{'source':source}],name='submit_program')])
    frozen=None
    try:
        learned=s.learner.learn(s.adapter.task,trace)
        assert not learned['errors'] and len(sent)==2
        program=s.bank.all('program')[0]
        assert s.bank.program_eligible(program) and program['result_role']=='intermediate'
        assert program['local_validation']['source_physical_key']==s.adapter.task.physical_key
        locked=s.bank.jobs()[0]['case_bindings'][0]
        trial=s.bank.attempts(program['id'])[0]
        assert trial['outcome']=='positive' and trial['basis']=='local_check'
        if chain:
            assert [e['name'] for e in evidence['executed_calls']]==['grep','read']
            hit=trial['tools'][0]['result']['data'][0]
            assert trial['tools'][1]['arguments']=={'path':hit['path'],'offset':hit['offset']}
        bad=deepcopy(locked);bad['local_evidence_ref']='local:invented'
        with pytest.raises(ValueError,match='existing host'):
            validate_on_source(s,program,bad,s.adapter.task,experience,'forged-source')
        wrong=s.bank.put('program',{**{k:v for k,v in program.items() if k not in {'id','state','local_validation'}},
            'source':source.replace("'result':value*inputs['factor']","'result':-1") if chain else source.replace("result['data']","[]")})
        failed=validate_on_source(s,wrong,locked,s.adapter.task,experience,'wrong-output')
        assert failed['outcome']=='execution_failure' and not s.bank.program_eligible(s.bank.get(wrong['id']))
        snapshot=tmp_path/'frozen';s.bank.freeze(snapshot);frozen=Bank(snapshot,readonly=True)
        before=frozen.digest()
        task=PublicTask('different','different-physical','retrieve 次值 and compute',split='val')
        adapter=OfficeAdapter({'different':{'answer':'44'}},s.config);adapter.reset(task)
        values={'pattern':'次值',**({'factor':4} if chain else {})}
        def consume(stage,prompt,material,*args,**kwargs):
            inputs=expand_material(material)['bindings']['inputs']
            assert inputs['result']==44 if chain else inputs['hits'][0]['path']=='b.txt'
            answer=str(inputs['result']) if chain else inputs['hits'][0]['text']
            return {'action':'finish','answer':answer,'used_inputs':['result' if chain else 'hits']}
        fields={'result':{'from':'program','field':'result'}} if chain else {'hits':{'from':'program','field':'hits'}}
        broker=Broker(adapter,24)
        execution=Executor(frozen,consume,worker,s.planner,frozen=True).run(task,adapter,broker,
            {'nodes':[{'id':'program','execution_mode':'program','program_id':program['id'],
                       'args':{k:{'literal':v} for k,v in values.items()}},
                      {'id':'consumer','execution_mode':'dynamic','goal':'consume acquired result','args':fields}]})
        assert execution['attempts'][0]['outputs_consumed'] and execution['attempts'][0]['status']=='ok'
        assert broker.events[0]['result']['data'][0]['path']=='b.txt'
        if chain:
            assert broker.events[1]['arguments']=={'path':'b.txt','offset':3}
            assert execution['attempts'][0]['calls']==2
        assert frozen.digest()==before
        invalid_broker=Broker(adapter,24)
        with pytest.raises(ValueError):worker.execute(program,{'pattern':13},invalid_broker)
        assert not invalid_broker.events
        empty=worker.execute(program,{**values,'pattern':'absent-token'},Broker(adapter,24))
        assert empty['status']==('not_found' if chain else 'ok')
        if not chain:assert empty['outputs']['hits']==[]
        adapter.close()
    finally:
        if frozen:frozen.close()
        s.close()


def test_one_feedback_includes_declaration_provenance_and_output_errors(tmp_path,worker):
    s,trace,experience,evidence,proposal,_=prepare(tmp_path,worker,chain=True)
    try:
        proposal['skill']['result_role']='final_answer'
        b=proposal['realization_request']['case_bindings'][0]
        b['inputs']['factor']=999
        b['reference_fields']['result']=['invented']
        with pytest.raises(ValueError) as failure:
            s.learner.validate_learning_proposal(proposal,{s.adapter.task.physical_key:(s.adapter.task,experience)})
        codes={e['code'] for group in failure.value.feedback for e in group['errors']}
        assert {'program_answer_schema_invalid','unbound_input','invalid_output_reference','no_answer_evidence'}<=codes
        assert failure.value.repair_material['proposal']['skill']['result_role']=='final_answer'
        assert not s.bank.all('program')
        # A real typed answer field is eligible; renaming acquired text is not.
        spec=deepcopy(proposal['skill']);spec['output_schema']=object_schema({'answer':{'type':'string'}},['answer'])
        b['inputs']['factor']=3;b['reference_fields']={'answer':['data','computed','answer']}
        copied=deepcopy(experience);copied['local_evidence'][0]['reference']['data']['computed']['answer']='21'
        canonical_binding(b,copied,spec)
        b['reference_fields']['answer']=['data','computed','path']
        with pytest.raises(ValueError,match='no_answer_evidence'):canonical_binding(b,experience,spec)
        assert operation_directory(evidence,s.adapter.task.physical_key)['supported_roles']==['intermediate']
    finally:s.close()


def test_cache_preserves_scope_offsets_limits_and_invalidates(tmp_path):
    (tmp_path/'a.txt').write_bytes(('标题\r\n'+('甲 target\r\n'*45)).encode())
    (tmp_path/'b.txt').write_text('乙 target\n',encoding='utf-8')
    _cached_scan.cache_clear()
    for paths in (None,[],['b.txt'],['b.txt','a.txt','a.txt']):
        args=() if paths is None else (paths,)
        raw=grep(tmp_path,'target',*args,use_cache=False)
        assert grep(tmp_path,'target',*args)==raw
        assert grep(tmp_path,'target',*args)==raw
    hits,audit=grep(tmp_path,'target')
    assert len(hits)==40 and audit['truncated'] and hits[0]['offset']==3
    hits[0]['text']='poison'
    assert grep(tmp_path,'target')[0][0]['text']=='甲 target'
    (tmp_path/'b.txt').write_text('丙 changed\n',encoding='utf-8')
    assert grep(tmp_path,'target',['b.txt'])[0]==[]
    with pytest.raises(ValueError):grep(tmp_path,'target',['../outside.txt'])
    outside=tmp_path.parent/'outside.txt';outside.write_text('target')
    (tmp_path/'link.txt').symlink_to(outside)
    with pytest.raises(ValueError):grep(tmp_path,'target')
    assert _cached_scan.cache_info().hits>=5


def test_program_cannot_mount_corpus_or_expand_tools(tmp_path,worker):
    s,_,_,_,proposal,_=prepare(tmp_path,worker,chain=True)
    try:
        program={**proposal['skill'],'id':'permission-fixture','entry':'run','allowed_tools':['read'],
            'environment':s.config['program_environment'],
            'input_schema':object_schema(),'output_schema':object_schema({'visible':{'type':'boolean'},'escaped':{'type':'boolean'}},['visible','escaped']),
            'source':"def run(ctx, inputs):\n    from pathlib import Path\n    r=ctx.call('read',{'path':'../outside.txt','offset':0})\n    return {'status':'ok','outputs':{'visible':Path("+repr(str(s.adapter.corpus))+ ").exists(),'escaped':r['accepted']}}"}
        result=worker.execute(program,{},Broker(s.adapter,24))
        assert result['status']=='ok' and result['outputs']=={'visible':False,'escaped':False}
        program['source']="def run(ctx,inputs):\n    ctx.call('grep',{'pattern':'target'})\n    return {'status':'ok','outputs':{}}"
        broker=Broker(s.adapter,24)
        result=worker.execute(program,{},broker)
        assert result['status']!='ok' and not any(e.get('backend_invoked') for e in broker.events)
    finally:s.close()


def test_formal_gate_accepts_only_production_preview_shortening():
    from atomic_skillgraph.experiments.audit_officeqa_prefix import verify_directory
    from atomic_skillgraph.empirical.model_view import shrink_learning_preview
    card={'local_evidence_ref':'local:source','source_excerpt':'x'*800,'source_truncated':False,
          'output_references':[{'kind':'json','path':['data','text'],'type':'str','content_hint':'observed'}]}
    body={'experience':{'local_evidence':[deepcopy(card)]}}
    for _ in range(3):
        assert shrink_learning_preview(body)
        verify_directory(body,{card['local_evidence_ref']:card})
    for key,value in [('source_excerpt','invented'),('output_references',[{'kind':'json','path':['gold'],'type':'str'}])]:
        bad=deepcopy(body);bad['experience']['local_evidence'][0][key]=value
        with pytest.raises(AssertionError):verify_directory(bad,{card['local_evidence_ref']:card})


@pytest.mark.parametrize('section',['legal_operations','experience','completed_train_cases'])
def test_formal_gate_compares_source_literals_after_json_serialization(section):
    from atomic_skillgraph.experiments.audit_officeqa_prefix import verify_directory
    sources=source_literals("pairs = [('报告.txt', 120), ('second.txt', 240)]")
    literal=next(r for r in sources if r['path']==['body',0,'value'])
    assert isinstance(literal['value'][0],tuple)
    card={'local_evidence_ref':'local:source','input_sources':[literal]}
    wire=json.loads(json.dumps(card,ensure_ascii=False))
    def material(row):
        if section=='legal_operations':return {section:[row]}
        if section=='experience':return {section:{'local_evidence':[row]}}
        return {section:[{'local_evidence':[row]}]}
    verify_directory(material(wire),{card['local_evidence_ref']:card})
    assert isinstance(literal['value'][0],tuple)  # Auditing must not mutate saved evidence.
    for key,value in [('value',[['报告.txt',121],['second.txt',240]]),
                      ('path',['body',1,'value']),('kind','public_json')]:
        forged=deepcopy(wire);forged['input_sources'][0][key]=value
        with pytest.raises(AssertionError,match='Invented source binding'):
            verify_directory(material(forged),{card['local_evidence_ref']:card})
