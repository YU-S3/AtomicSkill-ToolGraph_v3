"""Reproducibly build a separately versioned human reference; no model requests."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'src'))
from atomic_skillgraph.empirical.contracts import digest, program_digest

VERSION = 'office.humanstrong.programmatic.v2.20261020'


def main():
    old = json.loads((ROOT.parent/'office_humanstrong_v1/bank/bank.json').read_text())
    bank = deepcopy(old)
    bank.update(method_version=VERSION, input_protocol='office.external-minimax.public-pages.v1',
        measured_performance=None, status='offline_validated_only_no_accuracy_claim',
        source_basis='Human revision of V1 using external three-seed Train-derived skill procedures and audited execution failures; development exposed, not blind Test.',
        default_workflow_id=None, workflow_runtime='stored_reference_only_native_dynamic_runtime')
    sections = re.split(r'^## (OA-[^\n]+)\n', (ROOT/'guidance.md').read_text(), flags=re.M)
    cards = dict(zip(sections[1::2], sections[2::2])); remap = {}
    bank['skill'] = []
    by_title = {s['title']: s for s in old['skill']}
    for title, guidance in cards.items():
        asset = deepcopy(by_title.get(title, old['skill'][0]))
        asset.pop('id'); asset.update(title=title, goal=title + ': ' + guidance.strip().split('. ')[0],
            guidance=guidance.strip(), method_version=VERSION)
        asset['id'] = 'skill_' + digest(asset)
        if title in by_title: remap[by_title[title]['id']] = asset['id']
        bank['skill'].append(asset)
    for p in bank['program']:
        previous = p['id']; p['source'] = (ROOT/p['source_file']).read_text();p['method_version'] = VERSION
        if p['source_file'].endswith('series_metrics.py'):
            generic=p['input_schema']['oneOf'][-1]
            for op,count in [('symmetric_difference',2),('share_difference',4)]:
                schema=deepcopy(generic);props=schema['properties'];props.pop('other_values')
                props['operation']={'const':op};props['values'].update(minItems=count,maxItems=count)
                schema['required'].append('output_percent')
                if op=='share_difference':props['absolute']={'type':'boolean'};schema['required'].append('absolute')
                p['input_schema']['oneOf'].append(schema)
            # Factor repeated common schemas without changing any accepted input.
            branches=p['input_schema']['oneOf'];properties={}
            for branch in branches:properties.update(branch['properties'])
            properties['operation']={'enum':list(dict.fromkeys(op for b in branches for op in
                ([b['properties']['operation']['const']] if 'const' in b['properties']['operation'] else b['properties']['operation']['enum'])))}
            properties['values']={'type':'array','items':{'type':['number','string']},'minItems':1,'maxItems':5000}
            p['input_schema']={'type':'object','properties':properties,'additionalProperties':False,
                'oneOf':[{'properties':{k:v for k,v in b['properties'].items() if k in ('operation','values')},
                    'required':b['required'],'propertyNames':{'enum':list(b['properties'])}} for b in branches]}
            p['title']='Exact scalar/series formulas; explicit change, symmetric difference, share difference and units'
        p['id'] = 'program_' + program_digest(p); remap[previous] = p['id']
    def replace(value):
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        if isinstance(value,list):return [replace(v) for v in value]
        return remap.get(value,value) if isinstance(value,str) else value
    for kind in ('implementation','workflow'):
        bank[kind] = [replace(a) for a in bank[kind]]
        for a in bank[kind]:
            a.pop('id');a['method_version']=VERSION;a['id']=kind+'_'+digest(a)
    dest=ROOT/'bank';dest.mkdir(exist_ok=True)
    (dest/'bank.json').write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    # Distribution SQLite is a transparent snapshot, not a learned/promotion database.
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
    for kind in ('skill','program','implementation','workflow'):
        db.executemany('INSERT INTO assets VALUES (?,?,?)',[(a['id'],kind,json.dumps(a,ensure_ascii=False)) for a in bank[kind]])
    db.commit()
    with sqlite3.connect(dest/'bank.sqlite3') as out:db.backup(out)
    db.close()
    print(json.dumps({'version':VERSION,'bank_digest':digest(bank),'counts':{k:len(bank[k]) for k in ('skill','program','implementation','workflow')}}))


if __name__=='__main__':main()
