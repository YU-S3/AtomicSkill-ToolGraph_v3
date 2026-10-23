"""Build native scoped guidance and linked, explicitly typed calculations."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from atomic_skillgraph.empirical.contracts import digest, program_digest, validate_program

ROOT=Path(__file__).resolve().parents[1]
VERSION='office.humanstrong.operation-bound.v15.20261024'


def build():
    parent=json.loads((ROOT.parent/'office_humanstrong_v14/bank/bank.json').read_text())
    bank={**deepcopy(parent),'method_version':VERSION,'parent_bank_digest':digest(parent),
        'skill':[],'program':[],'implementation':[],'workflow':[],
        'status':'development_candidate_not_accepted','measured_performance':None,
        'source_basis':'Public formulas and table semantics checked against V14 failures and historical/Trace2Skill successful processes. No task IDs, gold values or answer lookup. Development-exposed human reference, not learned or blind Test evidence.'}
    cards=json.loads((ROOT/'cards.json').read_text()); replaced={c['title'] for c in cards}
    specs=[s for s in parent['skill'] if s['title'] not in replaced]
    for s in specs:
        s=deepcopy(s);s.pop('id');s['method_version']=VERSION;s['id']='skill_'+digest(s);bank['skill'].append(s)
    for card in cards:
        s={**deepcopy(parent['skill'][0]),**{k:v for k,v in card.items() if k in ('title','goal','guidance')}}
        s.pop('id');s['method_version']=VERSION
        s['execution_intent']='program_requested' if card.get('program') else 'guidance_only'
        s['applicability']={k:card[k] for k in ('when','source_pattern') if k in card}
        s['id']='skill_'+digest(s);bank['skill'].append(s)
    # Preserve the existing scalar implementation; no scorer-dependent formatting.
    scalar=deepcopy(parent['program'][0]);scalar['method_version']=VERSION
    scalar['id']='program_'+program_digest(scalar);bank['program'].append(scalar)
    common={'submit_answer':{'type':'boolean','enum':[True]},'values':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':240},
        'unit':{'type':'string','minLength':1},'decimals':{'type':'array','items':{'type':'integer','minimum':0,'maximum':12},'minItems':1,'maxItems':3},
        'fixed_decimals':{'type':'boolean'},'operand_refs':{'type':'array','items':{'type':'object'}}}
    schema_fields={
        'midpoint_submit':{'absolute':{'type':'boolean'},'output_percent':{'type':'boolean'},'percent_suffix':{'type':'boolean'}},
        'growth_submit':{'intervals':{'type':'number','exclusiveMinimum':0}},
        'paired_statistics':{'other_values':deepcopy(common['values']),
            'periods':{'type':'array','items':{'type':'string'},'minItems':2,'maxItems':240},
            'expected_periods':{'type':'array','items':{'type':'string'},'minItems':2,'maxItems':240},
            'statistic':{'type':'string','enum':['share_and_correlation','absolute_log_growth_cv_difference']}},
        'baseline_z':{'spread':{'type':'string','enum':['sample','population']},'target':{'type':'number'},
            'threshold':{'type':'number','exclusiveMinimum':0},'two_sided':{'type':'boolean'}}}
    for card in cards:
        if not card.get('program'):continue
        op=card['program'];props={'operation':{'type':'string','enum':[op]},**deepcopy(common),**schema_fields[op]}
        if op in ('midpoint_submit','growth_submit'):props['values']['maxItems']=2
        p={**deepcopy(scalar),'operation':op,'title':card['goal']+'; explicit operands, validated formula, direct final submission',
            'source_file':'programs/verified_math.py','source':(ROOT/'programs/verified_math.py').read_text(),
            'allowed_tools':[],'entry_constraints':{'description':'Caller binds actual source operands and explicit units; this calculator validates arithmetic, not source semantics.'},
            'input_schema':{'type':'object','properties':props,'required':[k for k in props if k not in ('operand_refs','fixed_decimals','percent_suffix')],'additionalProperties':False},
            'output_schema':{'type':'object','properties':{'answer':{'type':'string','minLength':1},'value':{},'calculation':{'type':'object'},'evidence':{'type':'array'},'proof_scope':{'type':'string'}},
                             'required':['answer','value','calculation','evidence','proof_scope'],'additionalProperties':False}}
        p['id']='program_'+program_digest(p);validate_program(p);bank['program'].append(p)
    links={'scalar_display':'answer_only',**{c['program']:c['title'] for c in cards if c.get('program')}}
    for p in bank['program']:
        s=next(s for s in bank['skill'] if s['title']==links[p['operation']])
        link={'kind':'implementation','skill_id':s['id'],'program_id':p['id'],'goal':s['goal'],
              'origin':'human_authored','permission':'author_trusted','method_version':VERSION}
        link['id']='implementation_'+digest(link);bank['implementation'].append(link)
    folder=ROOT/'bank';folder.mkdir(exist_ok=True)
    (folder/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    path=folder/'bank.sqlite3';path.unlink(missing_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
        for kind in ('skill','program','implementation','workflow'):
            db.executemany('INSERT INTO assets VALUES (?,?,?)',[(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
    print(json.dumps({'counts':{k:len(bank[k]) for k in ('skill','program','implementation','workflow')},'hash':digest(bank)}))


if __name__=='__main__':build()
