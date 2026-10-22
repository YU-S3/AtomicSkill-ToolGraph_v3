"""Read-only paired evaluation; explicit source, selection and shared metered budget."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'sha', 'config', 'selection', 'output', 'campaign', 'env-file'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--bank', help='Existing immutable control Bank; otherwise import this source')
    parser.add_argument('--crop-mode', choices=['guidance', 'grounded'], default='grounded')
    parser.add_argument('--single-node', action='store_true', help='One manual Skill; text-only response with no tools')
    parser.add_argument('--single-node-crop', action='store_true', help='One manual Skill with only optional image inspection')
    parser.add_argument('--guidance-role', choices=['user', 'system'], default='user',
                        help='Single-node ablation: where the complete manual Skill is delivered')
    parser.add_argument('--evidence-interface', choices=['legacy', 'scoped', 'compact', 'typed'], default='legacy',
                        help='Single-node crop submission: raw result IDs or supplied image handles')
    parser.add_argument('--skill-revision', choices=['v6', 'v7', 'v8', 'v9', 'v10', 'v11'], default='v6',
                        help='Manual single-node Skill to import into a fresh Bank')
    parser.add_argument('--principles', default='', help='Comma-separated v8/v10/v11 rule IDs; empty selects only the core')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.single_node = args.single_node or args.single_node_crop
    if not args.single_node and args.guidance_role != 'user':
        parser.error('--guidance-role applies only to single-node evaluation')
    if args.evidence_interface not in ('legacy', 'typed') and not args.single_node_crop:
        parser.error('--evidence-interface requires --single-node-crop')
    if args.skill_revision != 'v6' and (not args.single_node or args.bank):
        parser.error('--skill-revision selects a fresh single-node Bank; omit --bank')
    if args.principles and (args.skill_revision not in ('v8', 'v10', 'v11') or args.bank):
        parser.error('--principles requires a fresh v8/v10/v11 Bank')
    if args.skill_revision in ('v9', 'v10', 'v11') and args.evidence_interface != 'typed':
        parser.error('v9/v10/v11 requires --evidence-interface typed')
    if args.evidence_interface == 'typed' and not args.single_node:
        parser.error('Typed submission requires single-node evaluation')
    repo, root, campaign = map(lambda x: Path(x).resolve(), (args.repo,args.output,args.campaign))
    sys.path.insert(0,str(repo/'src'))
    sys.path.insert(0,str(repo/'experiments/docvqa_human_bank/integration'))
    from human_view import HumanDocVQAView, METHOD, image_result_bridge
    if args.single_node:
        from single_node import SingleNodeDocVQAView as HumanDocVQAView, METHOD, single_node_bridge as image_result_bridge
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import PublicTask, digest
    from atomic_skillgraph.experiments.canonical_manifest import verify, sha256
    from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity
    from atomic_skillgraph.experiments.run_empirical import code_identity, load_env, run, write_json
    code=code_identity()
    if code['git_sha']!=args.sha or code['tracked_dirty']:raise ValueError('Expected clean pinned source')
    selection=json.loads(Path(args.selection).read_text())
    split=selection['split'];ids=selection['task_ids']
    if split not in ('train','val','test') or len(ids)!=len(set(ids)):raise ValueError('Invalid selection')
    authority=repo/'data/main_experiment_v1';verify(authority)
    canonical=json.loads((authority/'docvqa'/(split+'.json')).read_text())['tasks']
    config=deepcopy(json.loads(Path(args.config).read_text()))
    datasets=Path(config['harness']['evaluator_records']).parent.parent
    material=json.loads((datasets/'materialization.json').read_text())
    if material['authority_sha256']!=sha256(authority/'manifest.json'):raise ValueError('Materialization authority differs')
    data_path=datasets/'docvqa'/(split+'.json')
    gold_path=datasets/'docvqa'/('evaluator_records_'+split+'.json')
    for path in (data_path,gold_path):
        if sha256(path)!=material['files_sha256'][path.relative_to(datasets).as_posix()]:raise ValueError('Materialized identity differs')
    rows=json.loads(data_path.read_text())['tasks']
    if [x['task_id'] for x in rows]!=[x['task_id'] for x in canonical]:raise ValueError('Canonical members/order differ')
    lookup={x['task_id']:x for x in rows}
    if not set(ids)<=lookup.keys():raise ValueError('Selection outside split')
    tasks=[PublicTask(**lookup[k]) for k in ids]
    for task in tasks:
        for image in task.inputs['images']:
            if sha256(image)!=material['external_files_sha256'][image]:raise ValueError('Original image changed')
    root.mkdir(parents=True,exist_ok=True)
    # A single campaign lock serializes the production governor across processes.
    with (campaign/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        limits=json.loads((campaign/'budget.json').read_text())['limits']
        config['data_dir']=str(root/'bank')
        config['harness']['evaluator_records']=str(gold_path)
        config['manifest']=str(authority/'docvqa'/(split+'.json'))
        config['experiment'].update(output_dir=str(root/split),runtime_mode='frozen')
        config['experiment']['docvqa_human_reference']={'method':METHOD,'crop_mode':None if args.single_node else args.crop_mode,
            'development_only':split=='train','workflow_execution':False,
            'single_node':args.single_node,
            'programs':'optional_image_crop' if args.single_node_crop else 'off' if args.single_node else 'auto'}
        if args.single_node:
            config['experiment']['docvqa_human_reference']['guidance_role'] = args.guidance_role
            config['experiment']['docvqa_human_reference']['evidence_interface'] = args.evidence_interface
            config['experiment']['docvqa_human_reference']['skill_revision'] = args.skill_revision
        config['budget'].update(limits,train_task_tokens=96000,eval_task_tokens=96000)
        config['budget'].pop('train_solve_tokens',None);config['budget'].pop('train_learning_tokens',None)
        config['program_worker']['record_execution_receipt']=True
        write_json(root/'import_config.json',config)
        if not (root/'bank').exists():
            if args.bank:shutil.copytree(args.bank,root/'bank')
            else:
                sys.path.insert(0,str(repo/'experiments/docvqa_human_bank/tools'))
                if args.single_node:
                    from single_node import prepare
                else:
                    from import_native import prepare
                prepare(repo,root/'import_config.json',root/'bank',**({'crop':args.single_node_crop,
                    'skill_revision':args.skill_revision,
                    'principle_ids':args.principles.split(',') if args.principles else []} if args.single_node else {}))
        manual=json.loads((root/'bank/manual_manifest.json').read_text())
        if manual['method_version']!=METHOD:raise ValueError('Bank/view version differs')
        if args.single_node:
            config['experiment']['docvqa_human_reference']['skill_revision'] = manual.get('skill_revision', 'v6')
            if (manual.get('skill_revision') in ('v9', 'v10', 'v11')) != (args.evidence_interface == 'typed'):
                raise ValueError('Typed submission and matching Bank must be selected together')
            if manual.get('typed_submission'):
                config['experiment']['docvqa_human_reference']['programs'] = 'typed_final_answer_and_optional_crop'
            if manual.get('skill_revision') in ('v8', 'v10', 'v11'):
                config['experiment']['docvqa_human_reference']['principle_ids'] = manual['principle_ids']
        if args.single_node and manual['program_status']!=('optional_image_crop' if args.single_node_crop else 'disabled'):
            raise ValueError('Requested crop mode differs from the frozen Bank')
        factory=lambda bank:HumanDocVQAView(bank,manual,**({} if args.bank else {'crop_mode':args.crop_mode}))
        bank=Bank(root/'bank',readonly=True,seed=42,environment=config['program_environment'])
        try:view_hash=factory(bank).view_hash
        finally:bank.close()
        identity={'method':'human_authored_reference_not_learned_ours','source':code,
            'method_version':METHOD,'bank_digest':manual['bank_digest'],'view_hash':view_hash,
            'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'selection':selection,'selection_hash':digest(selection),'canonical_split_hash':sha256(authority/'docvqa'/(split+'.json')),
            'campaign_budget':str(campaign/'budget.json'),'run_seed':42,'generation_seed_status':'unsupported'}
        plan={'identity':identity,'config':config}
        if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text())!=plan:raise ValueError('Frozen plan changed')
        write_json(root/'plan.json',plan)
        if args.check:print(json.dumps({'ready':True,'tasks':len(tasks),'method':METHOD,'http':0}));return
        load_env(args.env_file)
        log=FormalLog(root,identity,config,resume=args.resume)
        governor=BudgetGovernor(campaign/'budget.json',**limits)
        before=tree_identity(root/'bank')['sha256']
        log.freeze(root/'bank',newly_created=not (root/'final_frozen_manifest.json').exists())
        try:
            with image_result_bridge(**({'guidance_role': args.guidance_role,
                'evidence_interface': args.evidence_interface} if args.single_node else {})):
                summary=run(config,tasks,root/split,readonly=True,resume=args.resume,
                    bank_view_factory=factory,formal_log=log,budget_governor=governor,
                    task_metadata={x['task_id']:x['task_metadata'] for x in canonical},freeze_on_complete=False)
            after=tree_identity(root/'bank')['sha256']
            if before!=after:raise RuntimeError('Read-only Bank changed')
            if not summary['complete'] or summary['unknown_billing_attempts']:raise RuntimeError('Incomplete/unmetered run')
            cases=summary['cases']
            result={'complete':True,'tasks':len(cases),'correct':summary['successes'],
                'anls':sum(t['score']['soft'] for t in cases)/len(cases),'tokens':summary['known_total_tokens'],
                'empty':sum(not (t['execution'].get('prediction') or '').strip() for t in cases),
                'programs':summary['program_invocations'],'bank_before':before,'bank_after':after,
                'unknown_billing':summary['unknown_billing_attempts'],
                'http':sum(len(r.get('http_attempts',[])) for t in cases for r in t['requests'])}
            write_json(root/'result.json',result)
            log.finish('completed');print(json.dumps(result),flush=True)
        except BaseException:
            log.finish('stopped');raise


if __name__=='__main__':main()
