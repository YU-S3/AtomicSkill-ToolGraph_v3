"""Import authored assets through this repository's real Bank.put, not SQLite edits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def prepare(repo,config_path,destination):
    repo=Path(repo).resolve(); sys.path.insert(0,str(repo/'src'))
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import digest, program_digest
    config=json.loads(Path(config_path).read_text())
    env=config['program_environment']; seed=config['experiment'].get('seed',42)
    source=json.loads((ROOT/'bank/bank.json').read_text(encoding='utf-8'))
    output=Path(destination).resolve()
    if output.exists():raise ValueError('Fresh destination required; never overwrite another Bank')
    output.mkdir(parents=True)
    bank=Bank(output,seed=seed,environment=env)
    remap={};assets=deepcopy(source)
    try:
        for s in assets['skill']:
            old=s.pop('id')
            s.update(bank.put('skill',s))
            remap[old]=s['id']
        for program in assets['program']:
            old=program['id']
            program['environment']=deepcopy(env)
            program['source']=(ROOT/program['source_file']).read_text(encoding='utf-8')
            program['id']='program_'+program_digest(program)
            record=bank.put('program',program)
            remap[old]=record['id']
            program.update(record)
        for impl in assets['implementation']:
            impl.pop('id',None)
            impl['skill_id']=remap[impl['skill_id']]
            impl['program_id']=remap[impl['program_id']]
            actual=bank.put('implementation',impl)
            impl.update(actual)
        for flow in assets['workflow']:
            flow.pop('id',None)
            for node in flow['nodes']:
                if node.get('program_id'):node['program_id']=remap[node['program_id']]
            record=bank.put('workflow',flow)
            flow.update(record)
        counts={k:len(bank.all(k)) for k in ('skill','program','implementation','workflow')}
        if counts!={'skill':10,'program':3,'implementation':3,'workflow':3}:
            raise ValueError('Unexpected imported asset counts '+str(counts))
        manifest={'method_version':source['method_version'],'origin':'human_authored_only_not_learned',
                  'bank_digest':bank.digest(),'environment':env,'seed':seed,
                  'counts':counts,'program_status':'candidate_manual_view_can_call_explicitly',
                  'source_bundle_hash':digest(source),'asset_id_mapping':remap}
        (output/'manual_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        (output/'imported_bank.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'imported':str(output),'counts':counts,'bank_digest':manifest['bank_digest']},ensure_ascii=False))
    finally: bank.close()
    # Intentional: DO NOT Bank.freeze(output/frozen_bank): it drops unqualified human programs.
    # The independent readonly HumanDocVQAView provides explicit manual execution scope.

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',required=True,help='Actual clean AtomicSkill-ToolGraph_v3 checkout')
    ap.add_argument('--config',required=True,help='Actual resolved full DocVQA eval JSON (not YAML template)')
    ap.add_argument('--output',required=True,help='Fresh output directory for native Bank')
    a=ap.parse_args();prepare(a.repo_root,a.config,a.output)
