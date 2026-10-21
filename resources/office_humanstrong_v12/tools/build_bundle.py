"""Build a frozen superset; profiles change enabled relationships, not hidden task routing."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from atomic_skillgraph.empirical.contracts import digest,program_digest,object_schema,validate_program

ROOT=Path(__file__).resolve().parents[1]
VERSION='office.humanstrong.atomic-directions.v12.20261022'


def build():
    previous=ROOT.parent/'office_humanstrong_v11'
    parent=json.loads((previous/'bank/bank.json').read_text())
    bank={**deepcopy(parent),'parent_bank_digest':digest(parent),'method_version':VERSION,
        'skill':[],'program':[],'implementation':[],'workflow':[],
        'source_basis':'Observed V11 development traces; explicit ablation of rules, source calculators and typed submission.',
        'profiles':{'off':'No active Bank assets','rules':'Short rules; native tools only',
                    'selective':'Same rules plus matching source calculators; no generic formatting call',
                    'guarded':'Selective plus explicit typed final submission after native computation'}}
    for old in parent['program']:
        if old['operation']=='final_scalar':continue
        p=deepcopy(old);p.update(method_version=VERSION,source_file='../office_humanstrong_v11/'+p['source_file'])
        p['id']='program_'+program_digest(p);bank['program'].append(p)
    template=deepcopy(parent['program'][0])
    text={'type':'string','minLength':1}
    ref=object_schema({'result_id':text,'path':{'type':'array','items':{'type':['string','integer']}}},['result_id','path'])
    fmt=object_schema({'output_scale':{'enum':['quantity','fraction','percent','percentage_points']},
        'style':{'enum':['number','percent','unit_suffix']},'decimals':{'type':'integer','minimum':0,'maximum':18},'suffix':text},
        ['output_scale','style','decimals'])
    typed=deepcopy(template);typed.update(title='Typed final quantity: explicit scale conversion, then rounding and submission',
        operation='typed_scalar',method_version=VERSION,source_file='programs/typed_scalar.py',
        entry_constraints={'selector':'typed_quantity','description':'Acquired value, scale and unit; explicit final display',
                           'proof_scope':'producer_declared_quantity_and_conversion'},
        source=(ROOT/'programs/typed_scalar.py').read_text(),
        input_schema=object_schema({'submit_answer':{'const':True},'quantity_ref':ref,'format':fmt},['submit_answer','quantity_ref','format']),
        output_schema=object_schema({'answer':text,'quantity':{'type':'object'},'format':{'type':'object'},
            'evidence':{'type':'array'},'proof_scope':text},['answer','quantity','format','evidence','proof_scope']))
    typed['id']='program_'+program_digest(typed);bank['program'].append(typed)
    share=deepcopy(template);source=deepcopy(template['input_schema']['properties']['base'])
    props={'submit_answer':{'const':True},'first':source,'second':source,'total':source,
        'absolute':{'type':'boolean'},'output_scale':{'enum':['fraction','percent']},
        'format':deepcopy(template['input_schema']['properties']['format'])}
    code='\n\n'.join([(previous/'programs/source_values.py').read_text(),
        (ROOT.parent/'office_humanstrong_v5/programs/answer_format.py').read_text(),
        (ROOT/'programs/share_difference_entry.py').read_text()])
    (ROOT/'programs/share_difference.py').write_text(code)
    share.update(title='Difference of source-bound shares of one total; explicit fraction or percent answer',
        operation='share_difference',method_version=VERSION,source_file='programs/share_difference.py',source=code,
        input_schema=object_schema(props,list(props)),
        output_schema=object_schema({'answer':text,'quantity':{'type':'object'},'formula':text,'output_scale':text,
            'evidence':{'type':'array'},'proof_scope':text},['answer','quantity','formula','output_scale','evidence','proof_scope']))
    share['id']='program_'+program_digest(share);bank['program'].append(share)
    for p in bank['program']:validate_program(p)
    for spec in json.loads((ROOT/'principles.json').read_text()):
        skill=deepcopy(parent['skill'][0]);skill.pop('id')
        skill.update({k:spec[k] for k in ('title','goal','guidance')},method_version=VERSION)
        skill['id']='skill_'+digest(skill);bank['skill'].append(skill)
        if spec.get('operation'):
            program=next(p for p in bank['program'] if p['operation']==spec['operation'])
            link={'kind':'implementation','skill_id':skill['id'],'program_id':program['id'],'goal':spec['goal'],
                'origin':'human_authored','permission':'author_trusted','method_version':VERSION}
            link['id']='implementation_'+digest(link);bank['implementation'].append(link)
    dest=ROOT/'bank';dest.mkdir(exist_ok=True)
    (dest/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    with NamedTemporaryFile(dir=dest,suffix='.tmp',delete=False) as f:temp=Path(f.name)
    try:
        with sqlite3.connect(temp) as db:
            db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
            for kind in ('skill','program','implementation','workflow'):
                db.executemany('INSERT INTO assets VALUES (?,?,?)',[(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
        temp.replace(dest/'bank.sqlite3')
    finally:temp.unlink(missing_ok=True)
    print(json.dumps({'digest':digest(bank),'counts':{k:len(bank[k]) for k in ('skill','program','implementation','workflow')}}))


if __name__=='__main__':build()
