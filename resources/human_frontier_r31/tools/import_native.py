"""Import programmatic human Bank with Bank.put; never falsify local qualification."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys,json,hashlib

ROOT=Path(__file__).resolve().parents[1]

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--repo-root',required=True)
    pa.add_argument('--output',required=True)
    args=pa.parse_args()
    dest=Path(args.output).resolve()
    if dest.exists():raise ValueError('Use a NEW directory; old runs/assets are immutable')
    sys.path.insert(0,str(Path(args.repo_root).resolve()/'src'))
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import program_digest,validate_workflow
    src=json.loads((ROOT/'bank.json').read_text())
    bank=Bank(dest,seed=42,environment=src['program_environment'])
    try:
        for skill in src['skill']:bank.put('skill',deepcopy(skill))
        for prog in src['program']:
            recorded=bank.put('program',deepcopy(prog))
            if recorded['id']!=prog['id'] or recorded['id']!='program_'+program_digest(prog):
                raise ValueError('Program digest changed from authored source')
        for impl in src['implementation']:bank.put('implementation',deepcopy(impl))
        for wf in src['workflow']:
            validate_workflow(wf,bank=bank)
            bank.put('workflow',deepcopy(wf))
        with bank.db:
            bank.db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',
                            ('manual_authority',src['method_version']))
        actual={k:len(bank.all(k)) for k in ('skill','program','implementation','workflow')}
        if actual!=src['counts']:raise ValueError('Native asset counts differ')
        sha=hashlib.sha256((dest/'bank.sqlite3').read_bytes()).hexdigest()
        receipt={'status':'manual_imported','source_bank_sha256':hashlib.sha256((ROOT/'bank.json').read_bytes()).hexdigest(),
                 'sqlite_sha256':sha,'native_bank_digest':bank.digest(),'asset_counts':actual,
                 'programs_have_actual_worker_qualification':False,
                 'runtime_use':'Use isolated author_trusted R3.1 native ProgramWorker path; DO NOT call Bank.freeze which deletes candidate Programs'}
        (dest/'manual_import_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(receipt,ensure_ascii=False,indent=2))
    finally:bank.close()
if __name__=='__main__':main()
