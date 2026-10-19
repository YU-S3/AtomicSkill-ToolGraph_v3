"""Zero-HTTP artifact and cross-reference preflight; runs without the project installed."""
from pathlib import Path
import ast,hashlib,json,sqlite3,sys
ROOT=Path(__file__).resolve().parents[1]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def p_digest(p):
    keys=['source','entry','input_schema','output_schema','allowed_tools','environment']
    keys += [k for k in ('result_role','entry_constraints') if k in p]
    return digest({k:sorted(set(p[k])) if k=='allowed_tools' else p[k] for k in keys})
def main():
    import jsonschema
    b=json.loads((ROOT/'bank/bank.json').read_text())
    counts={k:len(b[k]) for k in ('skill','program','implementation','workflow')}
    assert counts=={'skill':11,'program':5,'implementation':5,'workflow':4},counts
    ids={a['id'] for k in counts for a in b[k]}
    assert len(ids)==sum(counts.values())
    for p in b['program']:
        assert p['id']=='program_'+p_digest(p)
        assert 'def run(ctx,inputs)' in p['source'] or 'def run(ctx, inputs)' in p['source']
        compile(p['source'],p['source_file'],'exec')
        assert p['source']==(ROOT/p['source_file']).read_text()
        jsonschema.Draft202012Validator.check_schema(p['input_schema'])
        jsonschema.Draft202012Validator.check_schema(p['output_schema'])
    for x in b['implementation']:assert x['skill_id'] in ids and x['program_id'] in ids
    by={p['id']:p for p in b['program']}
    for w in b['workflow']:
        past=set()
        for n in w['nodes']:
            assert set(n.get('after',[]))<=past
            if 'program_id' in n:assert n['program_id'] in by
            for ref in n.get('args',{}).values():
                if set(ref)=={'from','field'}:assert ref['from'] in past
                else:assert set(ref) in ({'task'},{'literal'},{'unresolved'})
            past.add(n['id'])
        for ref in w['outputs'].values():assert ref['from'] in past
    conn=sqlite3.connect(f'file:{(ROOT/"bank/bank.sqlite3").absolute()}?mode=ro',uri=True)
    count=conn.execute('select count(*) from assets').fetchone()[0];assert count==sum(counts.values())
    conn.close()
    print(json.dumps({'static_validation':'passed','counts':counts,'bank_sha256':digest(b),'model_http':0},indent=2))
if __name__=='__main__':main()
