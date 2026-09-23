"""Verified authored release source feeding the shared production Frozen loop."""
from __future__ import annotations
import argparse
import copy
import json
import time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

from atomic_skillgraph.system import AtomicSkillGraphSystem,load_config
from atomic_skillgraph.deployment.bank_release import load_release_source
from atomic_skillgraph.deployment.release_protocol import ReleaseError
from atomic_skillgraph.knowledge.r103_protocol import validate_config
from .protocol import (AttemptTraceLedger,TaskManifest,hash_code,hash_config,task_signature,
    ALFWORLD_FORMAL_TASK_TYPES,capture_execution_manifest)
from .reference_manifest import load_formal_reference_manifest,select_reference_tasks,validate_reference_disjoint
from .run_v3_frozen_eval import run_frozen_tasks

ROOT=Path(__file__).resolve().parents[1]

def run(config_path,*,resume=False,task_entries=None):
    started=time.monotonic();started_at=datetime.now(timezone.utc)
    config_path=Path(config_path).resolve();config=load_config(config_path)
    errors=validate_config(config)
    if errors:raise ReleaseError('; '.join(errors))
    experiment=config['experiment']
    if experiment.get('experiment_kind')!='bank_release_eval' or experiment.get('phase')!='test':
        raise ReleaseError('released runner requires its explicit experiment kind and phase')
    source=load_release_source(config['bank_release']['release_manifest'],config)
    output=Path(experiment['output_dir']);run_id=experiment['name']
    if output.exists() and not resume:raise FileExistsError(output)
    if resume and (output/'run_state.sqlite3').is_file():
        import sqlite3
        with sqlite3.connect(f'file:{output / "run_state.sqlite3"}?mode=ro',uri=True) as db:
            row=db.execute('SELECT state FROM run_manifests WHERE run_id=?',(run_id,)).fetchone()
            if row and row[0]=='completed':raise ReleaseError('completed evaluation cannot be resumed or overwritten')
    output.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'stage':'verified_release','seed':config['bank_release']['source_seed'],
        'output':str(output),'profile':config['deployment']['presentation_profile']}),flush=True)
    from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
    probe_hash=hash_config({'llm':config['llm']})
    probe_root=Path(config.get('bank_release',{}).get('provider_probe_dir',output/'provider_probe'))
    capability=ensure_provider_capability(config,output_dir=probe_root,config_hash=probe_hash,
        code_hash=hash_code(ROOT),run_if_missing=not resume)
    capture_execution_manifest(ROOT,output,hash_config(config_path),capability,capability_config_hash=probe_hash)
    from atomic_skillgraph.core.serialization import atomic_write_json
    atomic_write_json(output/'config.json',config)
    atomic_write_json(output/'provider_probe_reference.json',{'path':str(probe_root),'manifest':capability})
    ledger=AttemptTraceLedger(output/'attempt_history',output/'traces')
    if resume:ledger.recover_pending(run_id=run_id)
    if ledger.unresolved(run_id=run_id):raise ReleaseError('unresolved paid attempt; archive/account before recovery')
    with AtomicSkillGraphSystem(config,readonly=True) as system:
        preflight=system.preflight(require_api_key=True,initialize_harness=True)
        if not preflight.get('passed'):raise ReleaseError(f'release preflight failed: {preflight}')
        train=load_formal_reference_manifest(ROOT/'data/baseline_manifests/train_120.json',manifest_id='train_120')
        test=load_formal_reference_manifest(ROOT/'data/baseline_manifests/test_ood_full_134.json',manifest_id='test_ood_full_134')
        disjoint=validate_reference_disjoint(train,test)
        if task_entries is None:tasks=select_reference_tasks(system.harness,test)
        else:
            # A dev entry is the full reference identity, never an env index.
            tasks=list(task_entries(system.harness))
            train=test=None;disjoint=None
        if task_entries is None and len(tasks)!=134:raise ReleaseError('Test134 count mismatch')
        digest=system.knowledge_digest()
        if digest!=source.manifest['knowledge_digest']:raise ReleaseError('System and release knowledge digest disagree')
        items=tuple(TaskManifest(i,t.task_id,task_signature(t),f'frozen:{digest}',t.benchmark,str(system.harness.split),
            json.dumps({'task_type':t.task_type,'game_file':t.context.get('game_file','')},sort_keys=True)) for i,t in enumerate(tasks))
        original=source.manifest['source_metadata']['provenance']
        source_run=SimpleNamespace(run_id=original['source_run_id'],code_commit=original['source_code_commit'],metadata={})
        result=run_frozen_tasks(system=system,config=config,config_path=config_path,tasks=tasks,task_items=items,
            output_dir=output,resume=resume,run_id=run_id,phase='test',digest_before=digest,
            max_task_attempts=experiment['max_task_attempts'],attempt_ledger=ledger,train_manifest=source_run,
            invocation_started_at=started_at,invocation_started_monotonic=started,expected_total=len(tasks),
            labels=list(ALFWORLD_FORMAL_TASK_TYPES),per_type=0,experiment_seed=config['bank_release']['source_seed'],
            protocol='r103_frozen134',source_train_replay=False,reference_train_manifest=train,
            reference_test_manifest=test,counts={label:sum(t.task_type==label for t in tasks) for label in ALFWORLD_FORMAL_TASK_TYPES},
            reference_disjoint_audit=disjoint)
    return result

def dev(release_root,seed,suite,profiles):
    """Finite, predeclared six-task paired run; no retries for ordinary failure."""
    if suite == 'oldfirst-coverage6':
        if profiles != ['current']:
            raise ReleaseError('Release4 coverage uses the fixed current profile')
        from .release4_coverage import run_coverage
        run_coverage(release_root, seeds=(seed,))
        return 0
    from atomic_skillgraph.core.serialization import atomic_write_json
    from .run_v3_r103_validation import declared_entries,resolve_tasks
    from .release_report import compare_dev
    if seed!=42 or suite not in {'frozen-dev6', 'oldfirst-dev3'} or profiles!=['current','lean']:
        raise ReleaseError('paired dev requires a declared seed42 suite with current and lean')
    root=Path(release_root).resolve();prepared=root/'seed42'
    base=json.loads((prepared/'prepare_manifest.json').read_text())['config']
    frozen=prepared/'frozen';manifest=json.loads((frozen/'release_manifest.json').read_text())
    selected={int(e['task_id'].split('_')[2]):e for e in declared_entries()}
    entries=[selected[n] for n in ((36,2,34) if suite == 'oldfirst-dev3' else (2,34,36,40,1,7))]
    if suite == 'oldfirst-dev3' and manifest['protocol_version'] != 'r103.oldfirst-release.v1':
        raise ReleaseError('oldfirst dev requires its own published source')
    atomic_write_json(root/'dev/declared_tasks.json',entries)
    for i,entry in enumerate(entries):
        for profile in (profiles if i%2==0 else list(reversed(profiles))):
            output=root/'dev'/profile/entry['task_id']
            config=copy.deepcopy(base);config.pop('_config_path',None)
            config['data_dir']=str(frozen);config['trace_data_dir']=str(output)
            config['harness']['split']='train'
            config['experiment'].update(name=output.name,experiment_kind='bank_release_eval',phase='test',runtime_mode='frozen',
                freeze_skills=True,allow_long_term_knowledge_writes=False,output_dir=str(output),task_manifest_path=str(output/'task_manifest.json'))
            config['bank_release']={'protocol_version':manifest['protocol_version'],'source_seed':seed,
                'release_manifest':str(frozen/'release_manifest.json'),'expected_bank_digest':manifest['knowledge_digest']}
            config['bank_release']['provider_probe_dir']=str(root/'dev/provider_probe')
            config['deployment']={'presentation_profile':profile,'presentation_version':'r103.frozen-expression.v2',
                'input_authorization_version':'r103.caller-input.v1','programs':True,'automatic_entry':True,'capture_final_request':True,'repeat_index':1}
            path=root/'dev/configs'/f'{profile}_{entry["task_id"]}.json'
            atomic_write_json(path,config)
            if i==0 and profile==profiles[0]:
                if suite == 'oldfirst-dev3':
                    from .oldfirst_bank_checks import run as check_automation
                else:
                    from .released_bank_checks import run as check_automation
                load_release_source(frozen/'release_manifest.json',config)
                check_automation(config,selected[2],root/'dev/automation')
            run(path,task_entries=lambda h,e=entry:resolve_tasks(SimpleNamespace(harness=h),[e]))
            if suite == 'oldfirst-dev3':
                from .released_dev_checks import check_episode
                check_episode(output, code_hash=manifest['code_hash'], bank_digest=manifest['knowledge_digest'],
                              entry=entry, profile=profile)
    compare_dev(root/'dev')
    if suite == 'oldfirst-dev3':
        from .released_dev_checks import verify_dev
        verify_dev(root, write=True)
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('--config',required=True,type=Path);p.add_argument('--resume',action='store_true')
    p=sub.add_parser('aggregate');p.add_argument('--plan',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('dev');p.add_argument('--release-root',required=True,type=Path);p.add_argument('--seed',required=True,type=int,choices=(42,43,44))
    p.add_argument('--suite',required=True,choices=('frozen-dev6','oldfirst-dev3','oldfirst-coverage6'));p.add_argument('--profiles',nargs='+',required=True,choices=('current','lean'))
    args=parser.parse_args(argv)
    if args.command=='run':return run(args.config,resume=args.resume)
    if args.command=='dev':return dev(args.release_root,args.seed,args.suite,args.profiles)
    from .release_report import aggregate
    aggregate(args.plan,args.output);return 0

if __name__=='__main__':raise SystemExit(main())
