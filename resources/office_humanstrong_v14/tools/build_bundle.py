"""Build author-trusted native assets without task IDs, scores or answer lookups."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from atomic_skillgraph.empirical.contracts import digest,program_digest,validate_program

ROOT=Path(__file__).resolve().parents[1]
VERSION='office.humanstrong.submission-facts.v14.20261023'


def build():
    parent=json.loads((ROOT.parent/'office_humanstrong_v13/bank/bank.json').read_text())
    bank={**deepcopy(parent),'method_version':VERSION,'parent_bank_digest':digest(parent),
          'skill':[],'program':[],'implementation':[],'workflow':[],
          'profiles':{'manual':'Short operation/fact AtomicSkills plus optional scalar submission','off':'No Bank assets'},
          'source_basis':'Public evidence and observed V13 execution failures; human-authored development-exposed reference, not learned Ours.'}
    specs=[s for s in parent['skill'] if s['title'] not in ('final_precision',)]
    # Preserve parent rules, then add scoped cards as the same native asset kind.
    cards=json.loads((ROOT/'cards.json').read_text())
    for old in specs:
        s=deepcopy(old);s.pop('id');s['method_version']=VERSION
        if s['title']=='compound_growth':s['guidance']=s['guidance'].replace('and display %','and follow the requested final display')
        if s['title']=='revision_basis':s['guidance']='Honor an explicit as-of release. Otherwise compare already supplied same-definition historical revisions before binding the value. Once all required periods and headers are verified, compute; do not expand to every bulletin merely because years repeat.'
        s['id']='skill_'+digest(s);bank['skill'].append(s)
    template=deepcopy(bank['skill'][0])
    for card in cards:
        s={**deepcopy(template),**{k:v for k,v in card.items() if k not in ('when','source_pattern')}}
        s.pop('id');s['applicability']={k:card[k] for k in ('when','source_pattern') if k in card}
        if s['title']=='answer_only':s['execution_intent']='program_requested'
        s['id']='skill_'+digest(s);bank['skill'].append(s)
    p=deepcopy(parent['program'][0]);p['method_version']=VERSION
    p['title']='Submit an acquired final scalar with explicit scale and requested display'
    source=(ROOT/p['source_file']).read_text()
    p['input_schema']['properties']['format']['properties']['fixed_decimals']={
        'type':'boolean','description':'True only when an exact fixed-width decimal format is requested; default false removes redundant fractional zeros after rounding.'}
    p['source']=source;p['id']='program_'+program_digest(p);validate_program(p);bank['program'].append(p)
    skill=next(s for s in bank['skill'] if s['title']=='answer_only')
    link={'kind':'implementation','skill_id':skill['id'],'program_id':p['id'],'goal':skill['goal'],
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
