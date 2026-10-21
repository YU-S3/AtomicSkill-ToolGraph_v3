"""Build small scoped cards; reuse unchanged growth and normalization programs."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from atomic_skillgraph.empirical.contracts import digest, program_digest, validate_program

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'office.humanstrong.scoped.v10.20261021'


def statistics_schema():
    number = {'type':['number','string']}
    props = {
        'statistic':{'enum':['mean','geometric_mean','variance_sample','variance_population',
            'std_sample','std_population','correlation','ols_slope','ols_intercept',
            'forecast_minus_actual','actual_minus_forecast']},
        'requested_periods':{'type':'array','minItems':1,'maxItems':500,'uniqueItems':True,
            'items':{'type':'string'}, 'description':'Exact requested fitting/measurement periods, in order; exclude forecast dates.'},
        'records':{'type':'array','minItems':1,'maxItems':500,'items':{'type':'object',
            'properties':{'period':{'type':'string'},'value':number,'x':number},
            'required':['period','value'],'additionalProperties':False},
            'description':'One record per observed period. value=response; x=predictor for paired/OLS statistics, in explicitly requested coordinates. No unit conversion is performed.'},
        'decimals':{'type':'integer','minimum':0,'maximum':18},
        'unit':{'type':'string','minLength':1},
        'forecast_x':number, 'actual':number}
    return {'type':'object','properties':props,'required':['statistic','requested_periods','records','decimals','unit'],
            'additionalProperties':False}


def build():
    parent = json.loads((ROOT.parent/'office_humanstrong_v9/bank/bank.json').read_text())
    v8 = json.loads((ROOT.parent/'office_humanstrong_v8/bank/bank.json').read_text())
    bank = {**deepcopy(parent),'method_version':VERSION,'parent_bank_digest':digest(parent),
        'skill':[],'program':[],'implementation':[],'workflow':[],
        'source_basis':'Complete existing Train120 trace review; prior V8/V9 development evidence disclosed. No task IDs or answers in assets.',
        'status':'development_candidate_not_accepted','measured_performance':None}
    for spec in json.loads((ROOT/'principles.json').read_text()):
        skill = deepcopy(parent['skill'][0]); skill.pop('id')
        skill.update({k:spec[k] for k in ('title','goal','guidance')},method_version=VERSION)
        skill['id']='skill_'+digest(skill);bank['skill'].append(skill)
        if 'operation' not in spec: continue
        op = spec['operation']
        source = parent if op == 'symmetric_difference' else v8
        if op == 'scoped_statistics':
            program = deepcopy(parent['program'][0])
            program.update(source_file='programs/scoped_statistics.py',
                source=(ROOT/'programs/scoped_statistics.py').read_text(), input_schema=statistics_schema(),
                output_schema={'type':'object','properties':{
                    'value':{'type':'string'},'answer':{'type':'string'},'formula':{'type':'string'},
                    'selected_periods':{'type':'array','items':{'type':'string'}},
                    'excluded_periods':{'type':'array','items':{'type':'string'}},
                    'count':{'type':'integer'},'unit':{'type':'string'},'proof_scope':{'type':'string'}},
                    'required':['value','answer','formula','selected_periods','excluded_periods','count','unit','proof_scope'],
                    'additionalProperties':False})
        else: program = deepcopy(next(p for p in source['program'] if p['operation'] == op))
        program.update(method_version=VERSION,operation=op,title=spec['goal']+'; local computation only')
        program['id']='program_'+program_digest(program);validate_program(program);bank['program'].append(program)
        link = dict(kind='implementation',skill_id=skill['id'],program_id=program['id'],goal=spec['goal'],
            origin='human_authored',permission='author_trusted',method_version=VERSION)
        link['id']='implementation_'+digest(link);bank['implementation'].append(link)
    destination=ROOT/'bank';destination.mkdir(exist_ok=True)
    (destination/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    with sqlite3.connect(destination/'bank.sqlite3') as db:
        db.execute('DROP TABLE IF EXISTS assets')
        db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
        for kind in ('skill','program','implementation','workflow'):
            db.executemany('INSERT INTO assets VALUES (?,?,?)',[(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
    print(json.dumps({'digest':digest(bank),'counts':{k:len(bank[k]) for k in ('skill','program','implementation','workflow')},
        'guidance_characters':sum(len(s['guidance']) for s in bank['skill'])}))


if __name__=='__main__':build()
