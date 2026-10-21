"""Prune unhelpful coverage; isolate the observed normalization/scale confusion."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from atomic_skillgraph.empirical.contracts import digest, program_digest, validate_program

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'office.humanstrong.sparse.v9.20261021'


def build():
    parent = json.loads((ROOT.parent/'office_humanstrong_v8/bank/bank.json').read_text())
    spec = json.loads((ROOT/'principles.json').read_text())[0]
    skill = deepcopy(next(s for s in parent['skill'] if s['title']=='midpoint_difference'))
    skill.pop('id'); skill.update({k:spec[k] for k in ('title','goal','guidance')},method_version=VERSION)
    skill['id']='skill_'+digest(skill)
    program=deepcopy(next(p for p in parent['program'] if p['operation']=='symmetric_difference'))
    program.update(method_version=VERSION,title='Absolute midpoint-normalized ratio; decimal output, no percent scaling')
    program['input_schema']['properties']['output_percent']={'const':False}
    program['id']='program_'+program_digest(program); validate_program(program)
    link=dict(kind='implementation',skill_id=skill['id'],program_id=program['id'],goal=spec['goal'],
              origin='human_authored',permission='author_trusted',method_version=VERSION)
    link['id']='implementation_'+digest(link)
    bank={**deepcopy(parent),'method_version':VERSION,'parent_bank_digest':digest(parent),
          'skill':[skill],'program':[program],'implementation':[link],'workflow':[],
          'source_basis':'V8 development observation: retain no tied family as proven; test a narrower scale contract.',
          'status':'development_candidate_not_accepted','measured_performance':None}
    destination=ROOT/'bank';destination.mkdir(exist_ok=True)
    (destination/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    with sqlite3.connect(destination/'bank.sqlite3') as db:
        db.execute('DROP TABLE IF EXISTS assets')
        db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
        for kind in ('skill','program','implementation','workflow'):
            db.executemany('INSERT INTO assets VALUES (?,?,?)',[(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
    print(json.dumps({'digest':digest(bank),'guidance_characters':len(skill['guidance'])}))


if __name__=='__main__':build()
