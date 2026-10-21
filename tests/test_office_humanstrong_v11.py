"""Actual source-cell replay, direct Docker submission and honest billing ancestry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from atomic_skillgraph.empirical.contracts import PublicTask,program_digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import response,http
from test_office_human_phase_v3 import load
import test_office_humanstrong_v5 as fixtures

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v11'
PARENT=Path('/home/yangchengyu/officeqa_humanstrong_v10_train120_test102_pairs_seed42_20261021/train120')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**k:pytest.fail('Real model HTTP forbidden'))


def test_assets_and_specific_routing():
    bank=json.loads((ROOT/'bank/bank.json').read_text());bridge=load(ROOT/'integration/frontier.py','v11_cards')
    assert [len(bank[k]) for k in ('skill','program','workflow')]==[16,8,0]
    for p in bank['program']:
        assert p['id']=='program_'+program_digest(p)
        assert p['result_role']=='final_answer' and p['allowed_tools']==['read_result']
        assert p['source']==(ROOT/p['source_file']).read_text()
    inputs={'source_pages':[{'text':'| A | A |\n| --- | --- |\n'}]}
    assert [s['title'] for s in bridge.matching('Fit a linear trend with OLS',inputs)]==['regression_coordinates','table_groups']
    assert [s['title'] for s in bridge.matching('Fit degree 2 polynomial regression')]==['polynomial_fit']
    assert not bridge.matching('Not inflation-adjusted nominal yield')
    for p in list(ROOT.rglob('*.py'))+[ROOT/'principles.json']:
        assert 'UID0' not in p.read_text()


class Context:
    def __init__(self,values):self.values=values;self.calls=[]
    def call(self,name,args):
        self.calls.append((name,args));data=self.values[args['result_id']]
        return {'accepted':True,'data':data,'next_offset':None,'offset':0,'end':len(str(data)),'size':len(str(data))}


def scalar(rid):return {'source_ref':{'result_id':rid,'path':['data','value']}}


def display(style='number',digits=2):return {'style':style,'decimals':digits,'scale':'as_is'}


def test_source_cells_keep_blanks_groups_and_reject_ambiguity():
    m=load(ROOT/'programs/source_values.py','v11_cells')
    header='| Date | Value | | Date | Value |'
    first='| Feb | 2 | | Feb | 9 |';second='| Feb | 3 | | Feb | 8 |'
    text=header+'\n| --- | --- | --- | --- | --- |\n'+first+'\n'+second+'\n'
    ctx=Context({'s':text});args={**scalar('s'),'header':header,'row':'Feb','column':4,'row_column':3}
    with pytest.raises(ValueError,match='absent/repeated'):m.read_operand(ctx,args,{},[])
    args['row_text']=first;proof=[]
    assert m.read_operand(ctx,args,{},proof)=='9'
    assert proof[0]['quote']==first
    args['column']=2
    with pytest.raises(ValueError,match='not a plain numeric'):m.read_operand(ctx,args,{},[])
    args['row_text']=first.replace('9','999')
    with pytest.raises(ValueError):m.read_operand(ctx,args,{},[])


def test_computation_contracts_and_wrong_conventions_remain_distinct():
    m=load(ROOT/'programs/final_calculation.py','v11_calc')
    ctx=Context({'a':'100','b':'121'})
    args={'submit_answer':True,'family':'growth','operation':'cagr','unit':'percent','base':scalar('a'),
        'end':scalar('b'),'periods':2,'output_percent':True,'format':display('percent',1)}
    assert m.run(ctx,args)['outputs']['answer']=='10.0%'
    args['format']['scale']='fraction_to_percent'
    assert m.run(ctx,args)['status']=='needs_input'
    args['format']=display();args['submit_answer']=False
    assert m.run(ctx,args)['status']=='needs_input'
    args={'submit_answer':True,'family':'statistics','unit':'widgets','statistic':'std_population',
        'requested_periods':['a','b'],'records':[{'period':p,'source':scalar(p)} for p in ['a','b']],
        'format':display()}
    assert m.run(ctx,args)['outputs']['answer']=='10.50'
    args['statistic']='std_sample';assert m.run(ctx,args)['outputs']['answer']=='14.85'
    args['statistic']='correlation'
    assert m.run(ctx,args)['status']=='needs_input'  # Cannot invent the other series.
    args['statistic']='mean';args['records'].pop()
    assert m.run(ctx,args)['status']=='needs_input'


def saved(uid):
    path=next((PARENT/'episodes').glob('*_'+uid+'/manual/summary.json'))
    return path,json.loads(path.read_text())['cases'][0]


@pytest.mark.parametrize('uid,op,expected',[('UID0060','cagr','13.009%'),('UID0166','deviation','0.05'),
    ('UID0062','difference','6244'),('UID0138','statistics','1.600'),('UID0098','statistics','14.166')])
def test_real_saved_source_to_docker_to_final_without_rewrite(tmp_path,monkeypatch,uid,op,expected):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    cfg,bridge,bundle=fixtures.setup.__wrapped__(tmp_path)
    path,old=saved(uid);old_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    task=PublicTask(**old['task']);program=next(p for p in bundle['program'] if p['operation']==op);seen=[]
    def post(*a,**kw):
        seen.append(kw['json']);assert len(seen)==1,'No second LLM decision after complete final Program'
        material=json.loads(kw['json']['messages'][1]['content'])
        refs=material['provided_source_page_refs']
        if uid=='UID0060':
            page=task.inputs['source_pages'][0]['text']
            header=next(line for line in page.splitlines() if line.startswith('| Fiscal year or month |'))
            cell=lambda row:{'source_ref':refs[0],'header':header,'row':row,'column':5}
            args={'submit_answer':True,'family':'growth','operation':'cagr','unit':'percent per year',
                'base':cell('1947'),'end':cell('1950'),'periods':3,'output_percent':True,'format':display('percent',3)}
        elif uid=='UID0166':
            lines=task.inputs['source_pages'][2]['text'].splitlines()
            header=next(line for line in lines if line.startswith('| Date |'))
            selectors=[]
            for year,target in [('1943','Feb.'),('1944','Feb.'),('1945','1945-Jan.')]:
                start=next(i for i,line in enumerate(lines) if line.startswith('|') and len(line.strip('|').split('|'))>8 and line.strip('|').split('|')[8].strip()==year+'-Jan.')
                row=next(line for line in lines[start:] if line.startswith('|') and len(line.strip('|').split('|'))>8 and line.strip('|').split('|')[8].strip()==target)
                selectors.append({'source_ref':refs[2],'header':header,'row':target,'row_column':8,'row_text':row,'column':11})
            args={'submit_answer':True,'family':'statistics','statistic':'mean_absolute_deviation','unit':'percentage points',
                'requested_periods':['1943-02','1944-02','1945-01'],'records':[{'period':p,'source':s} for p,s in zip(['1943-02','1944-02','1945-01'],selectors)],
                'benchmark':'2.60','format':display()}
        elif uid=='UID0062':
            cells=[]
            for index,year in [(0,'1940'),(1,'1947')]:
                header=next(line for line in task.inputs['source_pages'][index]['text'].splitlines()
                    if line.startswith('| Fiscal year or month |') and 'Department of the Army' in line)
                labels=[s.strip() for s in header.strip('|').split('|')]
                column=next(i for i,s in enumerate(labels) if s.startswith('Department of the Army'))
                cells.append({'source_ref':refs[index],'header':header,'row':year,'column':column})
            args={'submit_answer':True,'family':'difference','operation':'difference','unit':'millions of dollars',
                'from_source':cells[0],'to_source':cells[1],'absolute':False,'format':display(digits=0)}
        elif uid=='UID0138':
            selectors=[]
            for index,column in [(0,1),(1,3),(2,4)]:
                lines=task.inputs['source_pages'][index]['text'].splitlines()
                n=next(i for i,line in enumerate(lines) if line.startswith('| Percent of ') and 'pay deducted' in line)
                row=lines[n].strip('|').split('|')[0].strip();start=n
                while start and lines[start-1].startswith('|'):start-=1
                selectors.append({'source_ref':refs[index],'header':lines[start],'row':row,'column':column})
            periods=['1942-02','1943-02','1944-02']
            args={'submit_answer':True,'family':'statistics','statistic':'median_absolute_deviation_mean','unit':'percent',
                'requested_periods':periods,'records':[{'period':p,'source':s} for p,s in zip(periods,selectors)],'format':display(digits=3)}
        else:
            records=[]
            for year in range(2016,2025):
                index=0 if year<=2020 else 1
                lines=task.inputs['source_pages'][index]['text'].splitlines()
                header=next(line for line in lines if line.startswith('| Means of financing') and 'continued' in line) if index==0 else next(
                    line for line in lines if line.startswith('| Fiscal year or month | Borrowing'))
                records.append({'period':str(year),'source':{'source_ref':refs[index],'header':header,'row':str(year),'column':4 if index==0 else 3}})
            args={'submit_answer':True,'family':'statistics','statistic':'log_trimmed_mean','unit':'millions of dollars',
                'requested_periods':[str(y) for y in range(2016,2025)],'records':records,'trim_per_tail':'0.2','format':display(digits=3)}
        assert program['id'] in {p['id'] for p in material['calls']['programs']}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([{'action':'call_program','name':program['id'],'arguments':args}]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key');monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    bank=Path(cfg['data_dir'])/'bank.sqlite3';before=hashlib.sha256(bank.read_bytes()).hexdigest()
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':expected}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['execution']['prediction']==expected and trace['execution']['reason']=='plan_submitted',trace['execution']
    assert trace['score']['hard'] and trace['execution']['attempts'][0]['submission_by_program']
    assert before==hashlib.sha256(bank.read_bytes()).hexdigest() and old_hash==hashlib.sha256(path.read_bytes()).hexdigest()


def test_polynomial_and_grouped_rolling_are_not_native_rewrites():
    m=load(ROOT/'programs/final_calculation.py','v11_transforms')
    ctx=Context({str(i):str(i*i) for i in range(5)})
    args={'submit_answer':True,'family':'polynomial','operation':'absolute_error','degree':2,'unit':'widgets',
        'requested_periods':['0','1','2','3'],'records':[{'period':str(i),'x':i,'source':scalar(str(i))} for i in range(4)],
        'prediction_x':4,'actual_source':scalar('4'),'format':display()}
    assert m.run(ctx,args)['outputs']['answer']=='0.00'
    args['prediction_x']=3;assert m.run(ctx,args)['status']=='needs_input'
    args={'submit_answer':True,'family':'statistics','statistic':'rolling_mean_range','unit':'widgets',
        'requested_periods':['0','1','2','3'],'records':[{'period':str(i),'source':scalar(str(i)),'group':'a' if i<2 else 'b'} for i in range(4)],
        'window':2,'format':display()}
    result=m.run(ctx,args)
    assert result['outputs']['answer']=='6.00' and result['outputs']['calculation']['window_count']==2


def test_paused_billing_and_frozen_prefix(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT.parents[1]/'experiments'))
    import run_office_humanstrong_v11 as runner
    runner.bind_runner();common=runner.common
    before=common.sha(PARENT/'budget.json')
    plan,ledger,locks=runner.settled_parent(PARENT)
    assert sum(map(common.charge,ledger['attempts'].values()))==169104311
    tasks=common.read(Path(runner.RESOURCES)/'train.json')['tasks']
    selected,receipt=runner.select_prefix(tasks)
    assert len(selected)==62 and selected==tasks[:62] and not receipt['prior_scores_used_for_selection']
    assert common.sha(PARENT/'budget.json')==before
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    root=tmp_path/'run';runner.prepare(root)
    prepared=common.verify(root)
    assert prepared['new_token_cap']==26855424 and prepared['inherited_used_or_reserved']==169104311
    assert prepared['authorized_cumulative_ceiling']==262000000 and len(prepared['episodes'])==124
    monkeypatch.setattr(common,'load_env',lambda _:None)
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in prepared['episodes']])
    assert common.execute(root,'unused')==0
    tasks={t['task_id']:t for t in selected}
    for episode,wire in zip(prepared['episodes'],seen):
        body=json.loads(wire['messages'][1]['content'])
        assert body['task']['inputs']==tasks[episode['task_id']]['inputs']
        if episode['arm']=='off':
            assert not body['calls']['programs'] and 'human_frontier_skill_ids' not in body
        assert common.read(Path(episode['path'])/'freeze_record.json')['unchanged']
    assert len(seen)==124 and common.sha(PARENT/'budget.json')==before
    assert common.status(root)['new_http']==124


@pytest.mark.parametrize('arm',['manual','off'])
def test_all_public_tasks_fit_initial_request(tmp_path,monkeypatch,arm):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    cfg,bridge,bundle=fixtures.setup.__wrapped__(tmp_path)
    rows=[t for split in ('train','test') for t in json.loads((fixtures.RESOURCES/(split+'.json')).read_text())['tasks']]
    records={row['task_id']:{'answer':'PRIVATE_OFFLINE_FIXTURE'} for row in rows}
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in rows])
    with bridge.bridge():
        for row in rows:
            system=EmpiricalSystem(cfg,harness=OfficeAdapter(records,cfg),readonly=True,
                bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm))
            try:trace=system.run_task(PublicTask(**row),learn=False)
            finally:system.close()
            assert trace['execution']['reason']=='agent_submitted',(row['task_id'],trace['execution'])
    assert len(seen)==222
    for row,wire in zip(rows,seen):
        body=json.loads(wire['messages'][1]['content'])
        assert body['task']['inputs']==row['inputs'] and wire.get('tools')
        assert 'PRIVATE_OFFLINE_FIXTURE' not in json.dumps(wire)
        if arm=='off':assert not body['calls']['programs'] and 'human_frontier_skill_ids' not in body


@pytest.mark.parametrize('arm',['manual','off'])
def test_actual_compute_helper_visibility_and_direct_submission(tmp_path,monkeypatch,arm):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    cfg,bridge,bundle=fixtures.setup.__wrapped__(tmp_path)
    task=fixtures.v2.fixture_task(bridge);seen=[]
    task=PublicTask(**{**task.__dict__,'goal':'Return the acquired quantity in units, to two decimals.',
        'inputs':{**task.inputs,'source_pages':[]}})
    def post(*a,**kw):
        seen.append(kw['json']);body=json.loads(kw['json']['messages'][1]['content'])
        if len(seen)==1:
            assert not body['calls']['programs']
            action={'action':'call_tool','name':'execute_python','arguments':{'source':'import json\nprint(json.dumps({"answer_value":0.05}))'}}
        elif arm=='manual':
            assert len(seen)==2,'Final Program must not trigger an extra model turn'
            program=next(p for p in bundle['program'] if p['operation']=='final_scalar')
            assert [p['id'] for p in body['calls']['programs']]==[program['id']]
            rid=next(r['result_id'] for r in body['memory']['recent_results'] if r.get('data',{}).get('computed'))
            action={'action':'call_program','name':program['id'],'arguments':{'submit_answer':True,
                'value_ref':{'result_id':rid,'path':['data','computed','answer_value']},'format':display()}}
        else:
            assert not body['calls']['programs'] and 'human_frontier_skill_ids' not in body
            assert 'final_scalar' not in kw['json']['messages'][0]['content']
            action={'action':'finish','answer':'0.05'}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key');monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'0.05'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,arm=arm))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'] and len(seen)==2
    assert trace['execution']['reason']==('plan_submitted' if arm=='manual' else 'agent_submitted')
