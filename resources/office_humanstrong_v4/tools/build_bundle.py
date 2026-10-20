"""Compose existing pure Programs with one bounded source-reading entry point."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'src'))
from atomic_skillgraph.empirical.contracts import digest,program_digest,object_schema

VERSION='office.humanstrong.programmatic.v4.20261020'


def build():
    parent=json.loads((ROOT.parent/'office_humanstrong_v3/bank/bank.json').read_text())
    bank=deepcopy(parent)
    bank.update(method_version=VERSION,parent_bank_digest=digest(parent),measured_performance=None,
        status='authored_reference_requires_test_observation',
        source_basis='V3 development trace: preserve actual source refs through extraction, computation and final submission. No target answers or task IDs.')
    replacements=json.loads((ROOT/'guidance_changes.json').read_text())
    remap={}
    for skill in bank['skill']:
        if skill['title'] not in replacements:continue
        old=skill.pop('id')
        skill.update(guidance=replacements[skill['title']],method_version=VERSION)
        skill['id']='skill_'+digest(skill);remap[old]=skill['id']
    for p in bank['program']:
        p['source_file']=str((ROOT.parent/'office_humanstrong_v3'/p['source_file']).resolve().relative_to(ROOT.parent.resolve()))
        p['source_file']='../'+p['source_file']
    metrics=next(p for p in parent['program'] if p['source_file'].endswith('/series_metrics.py'))
    rules={}
    for branch in metrics['input_schema']['oneOf']:
        operation=branch['properties']['operation']
        for name in operation.get('enum',[operation.get('const')]):
            rules[name]={'required':branch['required'],'allowed':branch['propertyNames']['enum']}
    source=['METRIC_RULES = '+repr(rules)]
    for path,name in [('office_humanstrong_v2/programs/table_extract.py','extract_table'),
                      ('office_humanstrong_v2/programs/series_metrics.py','calculate_metrics'),
                      ('office_humanstrong_v3/programs/polynomial_forecast.py','calculate_polynomial')]:
        code=(ROOT.parent/path).read_text()
        source.append(code.replace('def run(',f'def {name}(',1))
    source.append((ROOT/'programs/table_calculate_entry.py').read_text())
    code='\n\n'.join(source)
    (ROOT/'programs/table_calculate.py').write_text(code)
    text={'type':'string','minLength':1}
    ref=object_schema({'result_id':text,'path':{'type':'array','items':{'type':['string','integer']}}},
                      ['result_id','path'])
    table=object_schema({'key':text,'source_ref':ref,
        'rows':{'type':'array','items':text,'minItems':1,'maxItems':240,'uniqueItems':True},
        'column':text,'unit':text,'header_hint':{'type':'string'},
        'offset':{'type':'integer','minimum':0},
        'length':{'type':'integer','minimum':1,'maximum':24000}},
        ['key','source_ref','rows','column','unit'])
    props=metrics['input_schema']['properties']
    allowed={k:deepcopy(props[k]) for k in ('operation','decimals','output_percent','absolute','periods')}
    allowed['operation']['enum']+=['predict','absolute_error','absolute_relative_change']
    allowed.update(method={'enum':['series_metrics','polynomial_forecast']},
        degree={'type':'integer','minimum':1,'maximum':5},
        first_x={'type':'number'},x_step={'type':'number'},prediction_x={'type':'number'})
    options=object_schema(allowed,['method','operation','decimals'])
    keys=['from_value','to_value','base_value','new_value','numerator','denominator',
          'values','other_values','observations','actual']
    bindings={'type':'object','propertyNames':{'enum':keys},'minProperties':1,
        'additionalProperties':{'type':'array','items':text,'minItems':1,'maxItems':8,'uniqueItems':True}}
    schema=object_schema({'submit_answer':{'const':True},
        'tables':{'type':'array','items':table,'minItems':1,'maxItems':8},
        'bindings':bindings,'calculation':options,
        'answer_unit':{'type':'string','minLength':1,'description':'Result unit label; no unit conversion is performed.'}},
        ['submit_answer','tables','bindings','calculation','answer_unit'])
    output=object_schema({'answer':{'type':'string'},'calculation':{'type':'object'},
        'evidence':{'type':'array','items':{'type':'object'}},'proof_scope':{'type':'string'}},
        ['answer','calculation','evidence','proof_scope'])
    program={k:deepcopy(v) for k,v in parent['program'][0].items()
        if k not in ('id','title','source','source_file','input_schema','output_schema','allowed_tools')}
    program.update(title='table_calculate: read real table refs, extract literal rows, compute and submit the complete scalar answer',
        source_file='programs/table_calculate.py',source=code,input_schema=schema,output_schema=output,
        allowed_tools=['read_result'],result_role='final_answer',method_version=VERSION)
    program['id']='program_'+program_digest(program);bank['program'].append(program)
    def replace(value):
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        if isinstance(value,list):return [replace(v) for v in value]
        return remap.get(value,value) if isinstance(value,str) else value
    for kind in ('implementation','workflow'):
        bank[kind]=[replace(a) for a in bank[kind]]
        for asset in bank[kind]:
            asset.pop('id');asset['method_version']=VERSION;asset['id']=kind+'_'+digest(asset)
    skill=next(s for s in bank['skill'] if s['title']=='OA-Table')
    link={'kind':'implementation','skill_id':skill['id'],'program_id':program['id'],
        'goal':'Source-bound extraction and named calculation with explicit final submission',
        'origin':'human_authored','permission':'author_trusted','method_version':VERSION}
    link['id']='implementation_'+digest(link);bank['implementation'].append(link)
    dest=ROOT/'bank';dest.mkdir(exist_ok=True)
    (dest/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
    for kind in ('skill','program','implementation','workflow'):
        db.executemany('INSERT INTO assets VALUES (?,?,?)',
            [(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
    db.commit()
    with sqlite3.connect(dest/'bank.sqlite3') as target:db.backup(target)
    db.close()
    print(json.dumps({'version':VERSION,'digest':digest(bank),'counts':{k:len(bank[k]) for k in ('skill','program','implementation','workflow')}}))


if __name__=='__main__':build()
