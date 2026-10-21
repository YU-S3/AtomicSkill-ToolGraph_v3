"""Freeze source-bound final calculators and short actionable cards."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from atomic_skillgraph.empirical.contracts import digest, program_digest, object_schema, validate_program

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'office.humanstrong.source-final.v11.20261021'


def schemas():
    text = {'type':'string','minLength':1}
    number = {'type':['number','string']}
    ref = object_schema({'result_id':text,'path':{'type':'array','minItems':1,'items':{'type':['string','integer']}}}, ['result_id','path'])
    source = {'oneOf':[
        object_schema({'source_ref':ref}, ['source_ref']),
        object_schema({'source_ref':ref,'offset':{'type':'integer','minimum':0},'header':text,'row':text,
            'row_column':{'type':'integer','minimum':0},'row_text':text,
            'column':{'type':'integer','minimum':1}}, ['source_ref','header','row','column'])],
        'description':'Acquired scalar ref, or exact Markdown cell. Header is one exact unique header line. row is exact label at row_column (default 0); for repeated month labels supply full exact row_text. column counts every cell including blanks from 0.'}
    display = object_schema({'style':{'enum':['number','percent','unit_suffix']},
        'decimals':{'type':'integer','minimum':0,'maximum':18},'scale':{'const':'as_is'},'suffix':text}, ['style','decimals','scale'])
    common = {'submit_answer':{'const':True},'format':display,'unit':text}
    growth = {**common,'family':{'const':'growth'},'base':source,'end':source,'periods':number}
    record = object_schema({'period':text,'source':source,'x':number,'x_source':source,'group':text}, ['period','source'])
    stats = {**common,'family':{'const':'statistics'},
        'requested_periods':{'type':'array','minItems':1,'uniqueItems':True,'items':text},
        'records':{'type':'array','minItems':1,'items':record},'forecast_x':number,'actual_source':source,
        'trim_per_tail':number,'window':{'type':'integer','minimum':1}}
    result = {}
    for name in ('cagr','annual_log_growth'):
        props={**growth,'operation':{'const':name}}
        props['output_percent']={'type':'boolean'}
        result[name]=object_schema(props,list(props))
    props={**common,'family':{'const':'midpoint'},'operation':{'const':'symmetric_difference'},'first':source,'second':source}
    result['symmetric_difference']=object_schema(props,list(props))
    props={**common,'family':{'const':'difference'},'operation':{'const':'difference'},
        'from_source':source,'to_source':source,'absolute':{'type':'boolean'}}
    result['difference']=object_schema(props,list(props))
    props={**common,'family':{'const':'polynomial'},'operation':{'enum':['predict','absolute_error']},
        'degree':{'type':'integer','minimum':1,'maximum':5},'prediction_x':number,'actual_source':source,
        'requested_periods':stats['requested_periods'],'records':stats['records']}
    result['polynomial']=object_schema(props,[k for k in props if k!='actual_source'])
    props={**stats,'statistic':{'enum':['mean','geometric_mean','variance_sample','variance_population',
        'std_sample','std_population','correlation','ols_slope','ols_intercept','forecast_minus_actual','actual_minus_forecast',
        'median_absolute_deviation_mean','log_trimmed_mean','theil_t','rolling_mean_range']}}
    result['statistics']=object_schema(props,list(common)+['family','statistic','requested_periods','records'])
    props={**stats,'statistic':{'const':'mean_absolute_deviation'},'benchmark':number}
    result['deviation']=object_schema(props,list(common)+['family','statistic','requested_periods','records','benchmark'])
    return result


def build():
    parent=json.loads((ROOT.parent/'office_humanstrong_v10/bank/bank.json').read_text())
    old=json.loads((ROOT.parent/'office_humanstrong_v5/bank/bank.json').read_text())
    bank={**deepcopy(parent),'method_version':VERSION,'parent_bank_digest':digest(parent),
        'skill':[],'program':[],'implementation':[],'workflow':[],
        'status':'development_candidate_not_accepted','measured_performance':None,
        'source_basis':'Paused V10 Train62 public traces and declared development history; no task IDs or gold in assets.'}
    parts=[(ROOT/'programs/source_values.py').read_text(),
        (ROOT.parent/'office_humanstrong_v2/programs/series_metrics.py').read_text().replace('def run(', 'def calculate_growth(',1),
        (ROOT.parent/'office_humanstrong_v10/programs/scoped_statistics.py').read_text().replace('def run(', 'def calculate_statistics(',1),
        (ROOT.parent/'office_humanstrong_v3/programs/polynomial_forecast.py').read_text().replace('def run(', 'def calculate_polynomial(',1),
        (ROOT.parent/'office_humanstrong_v5/programs/answer_format.py').read_text(),
        (ROOT/'programs/final_calculation_entry.py').read_text()]
    code='\n\n'.join(parts)
    (ROOT/'programs/final_calculation.py').write_text(code)
    final_template=next(p for p in old['program'] if p['source_file'].endswith('/computed_scalar.py'))
    for op,schema in schemas().items():
        p=deepcopy(final_template)
        p.update(method_version=VERSION,title=op+': acquired cells/results -> calculation -> final answer; no LLM rewrite',
            operation=op,source_file='programs/final_calculation.py',source=code,input_schema=schema,
            entry_constraints={'selector':'final_calculation','description':'Explicit full-task source selection and named arithmetic','proof_scope':'literal_source_and_computation'},
            output_schema=object_schema({'answer':{'type':'string'},'calculation':{'type':'object'},
                'evidence':{'type':'array'},'format':{'type':'object'},'proof_scope':{'type':'string'}},
                ['answer','calculation','evidence','format','proof_scope']))
        p['id']='program_'+program_digest(p);validate_program(p);bank['program'].append(p)
    p=deepcopy(final_template)
    p.update(method_version=VERSION,operation='final_scalar',source_file='../office_humanstrong_v5/'+p['source_file'])
    p['id']='program_'+program_digest(p);validate_program(p);bank['program'].append(p)
    for spec in json.loads((ROOT/'principles.json').read_text()):
        skill=deepcopy(parent['skill'][0]);skill.pop('id')
        skill.update({k:spec[k] for k in ('title','goal','guidance')},method_version=VERSION)
        skill['id']='skill_'+digest(skill);bank['skill'].append(skill)
        if 'operation' in spec:
            p=next(p for p in bank['program'] if p['operation']==spec['operation'])
            link=dict(kind='implementation',skill_id=skill['id'],program_id=p['id'],goal=spec['goal'],
                origin='human_authored',permission='author_trusted',method_version=VERSION)
            link['id']='implementation_'+digest(link);bank['implementation'].append(link)
    dest=ROOT/'bank';dest.mkdir(exist_ok=True)
    (dest/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    with NamedTemporaryFile(dir=dest,suffix='.tmp',delete=False) as f:temporary=Path(f.name)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
            for kind in ('skill','program','implementation','workflow'):
                db.executemany('INSERT INTO assets VALUES (?,?,?)',[(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
        temporary.replace(dest/'bank.sqlite3')
    finally:temporary.unlink(missing_ok=True)
    print(json.dumps({'digest':digest(bank),'counts':{k:len(bank[k]) for k in ('skill','program','implementation','workflow')}}))


if __name__=='__main__':build()
