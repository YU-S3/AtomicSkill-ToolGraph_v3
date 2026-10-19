"""Zero-HTTP native import for the isolated authored OfficeQA reference Bank.

This intentionally does NOT call Bank.freeze(): a normal learned-program freeze
would erase unqualified manual candidates. The HumanBankView is the explicit
manual permission path; it does not change Ours' qualification policy.
"""
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--repo-root',required=True)
    a.add_argument('--config',required=True,help='Fully resolved evaluation config JSON, not an unresolved YAML')
    a.add_argument('--output',required=True,help='Fresh Bank directory')
    args=a.parse_args()
    repo=Path(args.repo_root).resolve()
    sys.path.insert(0,str(repo/'src'))
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import digest
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('office_strong_frontier',ROOT/'integration/frontier.py')
    frontier=module_from_spec(spec);spec.loader.exec_module(frontier)
    cfg=json.loads(Path(args.config).read_text(encoding='utf8'))
    if cfg.get('experiment',{}).get('benchmark')!='officeqa' or cfg['harness']['adapter']!='officeqa':
        raise ValueError('OfficeQA config required')
    bundle=json.loads((ROOT/'bank/bank.json').read_text(encoding='utf8'))
    dest=Path(args.output).resolve()
    if dest.exists():raise ValueError('Provide a non-existent independent directory; do not overwrite experiments')
    actual=frontier.native_import(bundle,dest,cfg['program_environment'],cfg.get('experiment',{}).get('officeqa_method_version'),cfg['experiment'].get('seed',42))
    bank=Bank(dest,readonly=True,seed=cfg['experiment'].get('seed',42),environment=cfg['program_environment'])
    try:
        counts={k:len(bank.all(k)) for k in frontier.KINDS}
        assert counts=={'skill':12,'program':5,'implementation':5,'workflow':4},counts
        for imp in bank.all('implementation'):
            assert bank.get(imp['skill_id']) and bank.get(imp['program_id'])
        for wf in bank.all('workflow'):
            for n in wf['nodes']:
                if 'program_id' in n:assert bank.get(n['program_id'])
                if 'skill_id' in n:assert bank.get(n['skill_id'])
        view=frontier.HumanBankView(bank,actual,arm='manual')
        print(json.dumps({'import':'passed','counts':counts,'method_version':frontier.VERSION,
            'bank_digest':bank.digest(),'human_permissions':'view_only','model_http':0},indent=2))
    finally:bank.close()
if __name__=='__main__':main()
