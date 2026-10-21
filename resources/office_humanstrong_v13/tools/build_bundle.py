"""Preserve V12 short rules and replace the broad tool menu with one optional finalizer."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from atomic_skillgraph.empirical.contracts import digest,program_digest,object_schema,validate_program

ROOT=Path(__file__).resolve().parents[1]
VERSION='office.humanstrong.final-display.v13.20261022'


def build():
    previous=ROOT.parent/'office_humanstrong_v12'
    parent=json.loads((previous/'bank/bank.json').read_text())
    bank={**deepcopy(parent),'parent_bank_digest':digest(parent),'method_version':VERSION,
        'skill':[],'program':[],'implementation':[],'workflow':[],
        'source_basis':'V12 development observations: correct values lost final precision; bulky source selectors caused retries.',
        'profiles':{'off':'No active Bank assets','rules':'Frozen V12 short-rule control, native tools only',
                    'precise':'Same short rules plus final numeric-display atom, including finish-only',
                    'display':'Precise plus one scalar finalizer after an acquired numeric result exists'}}
    for old in parent['skill']:
        if old['title']=='typed_submission':continue
        skill=deepcopy(old);skill.pop('id');skill['method_version']=VERSION
        skill['id']='skill_'+digest(skill);bank['skill'].append(skill)
    skill=deepcopy(bank['skill'][0]);skill.pop('id');skill.update(json.loads((ROOT/'precision.json').read_text()))
    skill['id']='skill_'+digest(skill);bank['skill'].append(skill)
    program=deepcopy(next(p for p in parent['program'] if p['operation']=='typed_scalar'))
    text={'type':'string','minLength':1}
    props={'submit_answer':{'const':True},'value_ref':program['input_schema']['properties']['quantity_ref'],
        'source_scale':{'enum':['quantity','fraction','percent','percentage_points']},'unit':text,
        'format':program['input_schema']['properties']['format']}
    source=(previous/'programs/typed_scalar.py').read_text().split('\ndef run(ctx,inputs):')[0]+'\n'+(ROOT/'programs/scalar_display_entry.py').read_text()
    (ROOT/'programs/scalar_display.py').write_text(source)
    program.update(title='Submit an acquired final scalar with explicit scale and exact decimal display',operation='scalar_display',
        method_version=VERSION,source_file='programs/scalar_display.py',source=source,
        input_schema=object_schema(props,list(props)),
        output_schema=object_schema({'answer':text,'value':text,'format':{'type':'object'},'source_scale':text,'unit':text,
                                    'evidence':{'type':'array'},'proof_scope':text},
                                   ['answer','value','format','source_scale','unit','evidence','proof_scope']),
        entry_constraints={'selector':'acquired_numeric_leaf','description':'Caller explicitly declares source and output scales',
                           'proof_scope':'numeric_conversion_and_display_only'})
    program['id']='program_'+program_digest(program);validate_program(program);bank['program'].append(program)
    link={'kind':'implementation','skill_id':skill['id'],'program_id':program['id'],'goal':skill['goal'],
          'origin':'human_authored','permission':'author_trusted','method_version':VERSION}
    link['id']='implementation_'+digest(link);bank['implementation'].append(link)
    dest=ROOT/'bank';dest.mkdir(exist_ok=True)
    (dest/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    with NamedTemporaryFile(dir=dest,suffix='.tmp',delete=False) as file:temp=Path(file.name)
    try:
        with sqlite3.connect(temp) as db:
            db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
            for kind in ('skill','program','implementation','workflow'):
                db.executemany('INSERT INTO assets VALUES (?,?,?)',[(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
        temp.replace(dest/'bank.sqlite3')
    finally:temp.unlink(missing_ok=True)
    print(json.dumps({'digest':digest(bank),'counts':{k:len(bank[k]) for k in ('skill','program','implementation','workflow')}}))


if __name__=='__main__':build()
