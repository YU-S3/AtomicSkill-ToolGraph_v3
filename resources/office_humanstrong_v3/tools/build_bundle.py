"""Build a small, versioned human Bank revision over the frozen V2 assets."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]/'src'))
from atomic_skillgraph.empirical.contracts import digest, program_digest, object_schema

VERSION = 'office.humanstrong.programmatic.v3.20261020'


def build():
    parent = json.loads((ROOT.parent/'office_humanstrong_v2/bank/bank.json').read_text())
    bank = deepcopy(parent)
    bank.update(method_version=VERSION, parent_bank_digest=digest(parent), measured_performance=None,
                status='authored_revision_requires_paired_observation',
                source_basis='V2 development failures: incomplete evidence and missing polynomial calculation; no gold or task-specific constants in assets.')
    changes = json.loads((ROOT/'guidance_changes.json').read_text())
    remap = {}
    for s in bank['skill']:
        if s['title'] not in changes: continue
        old = s.pop('id')
        s['guidance'] = changes[s['title']]
        s['guidance'] = s['guidance'].replace('All five Programs', 'All six Programs')
        s['method_version'] = VERSION
        s['id'] = 'skill_'+digest(s); remap[old] = s['id']
    for p in bank['program']:
        p['source_file'] = '../office_humanstrong_v2/'+p['source_file']
    ref = object_schema({'result_id':{'type':'string'}, 'path':{'type':'array','items':{'type':['string','integer']}}}, ['result_id','path'])
    numeric = {'type':['number','string']}
    text = {'type':'string','minLength':1}
    observation = object_schema({'period':text,'x':numeric,'value':numeric,'unit':text,'source_ref':ref},
                                ['period','x','value','unit','source_ref'])
    properties = {'operation':{'enum':['predict','absolute_error']}, 'degree':{'type':'integer','minimum':1,'maximum':5},
        'required_periods':{'type':'array','items':text,'minItems':2,'maxItems':240,'uniqueItems':True},
        'observations':{'type':'array','items':observation,'minItems':2,'maxItems':240},
        'prediction_x':numeric, 'unit':text, 'decimals':{'type':'integer','minimum':0,'maximum':18},
        'actual':object_schema({'value':numeric,'unit':text,'source_ref':ref}, ['value','unit','source_ref'])}
    schema = object_schema(properties, [k for k in properties if k!='actual'])
    fields = {k:{'type':'string'} for k in ['unit','center','scale','coefficient_basis','prediction','answer','formula','proof_scope']}
    fields.update(count={'type':'integer'}, degree={'type':'integer'},
                  coefficients={'type':'array','items':{'type':'string'}},
                  actual={'type':['string','null']}, absolute_difference={'type':['string','null']})
    program = {k:deepcopy(v) for k,v in parent['program'][0].items()
               if k not in ('id','title','source','source_file','input_schema','output_schema','allowed_tools')}
    program.update(title='Polynomial OLS forecast or absolute prediction error; explicit period coverage and source operands',
        source_file='programs/polynomial_forecast.py', source=(ROOT/'programs/polynomial_forecast.py').read_text(),
        input_schema=schema, output_schema=object_schema(fields,list(fields)), allowed_tools=[],
        method_version=VERSION, result_role='intermediate')
    program['id']='program_'+program_digest(program);bank['program'].append(program)
    def replace(value):
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        if isinstance(value,list):return [replace(v) for v in value]
        return remap.get(value,value) if isinstance(value,str) else value
    for kind in ('implementation','workflow'):
        bank[kind]=[replace(a) for a in bank[kind]]
        for asset in bank[kind]:
            asset.pop('id');asset['method_version']=VERSION;asset['id']=kind+'_'+digest(asset)
    skill=next(s for s in bank['skill'] if s['title']=='OA-Advanced')
    link={'kind':'implementation','skill_id':skill['id'],'program_id':program['id'],
          'goal':'Execute explicitly specified polynomial OLS and prediction error',
          'origin':'human_authored','permission':'author_trusted','method_version':VERSION}
    link['id']='implementation_'+digest(link);bank['implementation'].append(link)
    dest=ROOT/'bank';dest.mkdir(exist_ok=True)
    (dest/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
    for kind in ('skill','program','implementation','workflow'):
        db.executemany('INSERT INTO assets VALUES (?,?,?)',[(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
    db.commit()
    with sqlite3.connect(dest/'bank.sqlite3') as target:db.backup(target)
    db.close()
    print(json.dumps({'version':VERSION,'digest':digest(bank),'counts':{k:len(bank[k]) for k in ('skill','program','implementation','workflow')}}))


if __name__=='__main__':build()
