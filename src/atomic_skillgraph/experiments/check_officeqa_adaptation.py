"""One bounded canonical prefix through real Train, freeze and readonly Val."""
import argparse
import json
from pathlib import Path
import yaml

from ..empirical.budget_governor import BudgetGovernor
from ..empirical.contracts import PublicTask
from ..empirical.system import validate_config
from .canonical_manifest import ordered_train, sha256, verify
from .formal_log import FormalLog, tree_identity
from .run_empirical import code_identity, load_env, run, write_json
from .run_formal import corpus_identity, model_settings, resolved_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--corpus-root', required=True)
    parser.add_argument('--cost-source', required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root, datasets = Path(args.output), Path(args.datasets)
    if root.exists(): raise ValueError('A check needs a new independent directory')
    code = code_identity()
    if code['tracked_dirty'] or not code['git_sha']: raise ValueError('Pinned clean source required')
    if args.env_file: load_env(args.env_file)
    authority = Path('data/main_experiment_v1'); verify(authority)
    materialized = json.loads((datasets/'materialization.json').read_text())
    if materialized['authority_sha256'] != sha256(authority/'manifest.json'): raise ValueError('Canonical authority mismatch')
    for name, value in materialized['files_sha256'].items():
        if sha256(datasets/name) != value: raise ValueError('Changed materialized input: '+name)
    base = model_settings(yaml.safe_load(Path('configs/default.yaml').read_text()),
                          json.loads(Path('models.lock.json').read_text())['current_test'])
    profile = json.loads(Path('benchmark_profiles.json').read_text())['profiles']['officeqa']
    limits = {'token_limit':1392000, 'request_limit':120, 'finish_reserve':0}
    parent = json.loads(Path('configs/atomic_unified_seed42_budget.json').read_text())['cells']['officeqa']
    prior = json.loads((Path(args.cost_source)/'budget.json').read_text())
    if prior['unknown_billing'] or any(a['status']!='metered' for a in prior['attempts'].values()):
        raise ValueError('Old cost must be sealed before a new check')
    if sum(a['accounted_tokens'] for a in prior['attempts'].values())+limits['token_limit']>parent['token_limit']:
        raise ValueError('Check cannot exceed the existing cumulative authority')
    base['budget'] = {**parent, **limits}
    configurations, selected = {}, {}
    for split, count in [('train',6),('val',2)]:
        rows = [PublicTask(**r) for r in json.loads((datasets/'officeqa'/(split+'.json')).read_text())['tasks']]
        canonical = json.loads((authority/'officeqa'/(split+'.json')).read_text())['tasks']
        if [t.task_id for t in rows] != [t['task_id'] for t in canonical]: raise ValueError('Canonical members/order changed')
        selected[split] = (ordered_train(rows,42) if split=='train' else rows)[:count]
        configurations[split] = validate_config(resolved_config(base,profile,'officeqa',42,split,root,
                                                                 datasets,authority,args.corpus_root))
    identity = {'method':'ours', 'benchmark':'officeqa', 'purpose':'bounded adaptation check',
                'source_commit':code['git_sha'], 'source_sha256':code['source_sha256'], 'run_seed':42,
                'initial_bank':'fresh empty independent Bank', 'limits':limits,
                'authority_sha256':sha256(authority/'manifest.json'),
                'materialization_sha256':sha256(datasets/'materialization.json'),
                'corpus_sha256':corpus_identity(args.corpus_root)['sha256'],
                'selection':{k:[t.task_id for t in tasks] for k,tasks in selected.items()},
                'cost_source':args.cost_source, 'prior_ledger_sha256':sha256(Path(args.cost_source)/'budget.json')}
    root.mkdir(parents=True)
    write_json(root/'selection.json',identity)
    if not args.execute:
        write_json(root/'completion.json',{'status':'prepared_no_model_requests'});return
    governor = BudgetGovernor(root/'budget.json',**limits)
    logger = FormalLog(root,identity,configurations)
    result = {}
    try:
        result['train'] = run(configurations['train'],selected['train'],root/'train',formal_log=logger,budget_governor=governor)
        if not result['train']['complete']:
            logger.finish('stopped')
            write_json(root/'completion.json',{'status':'stopped_at_task_boundary','completed_train_tasks':result['train']['tasks']})
            return
        frozen = root/'train/frozen_bank';before = tree_identity(frozen)
        result['val'] = run(configurations['val'],selected['val'],root/'val',readonly=True,formal_log=logger,budget_governor=governor)
        after = tree_identity(frozen)
        if before != after: raise ValueError('Readonly evaluation changed the Frozen Bank')
        write_json(root/'freeze_verification.json',{'before':before,'after':after,'unchanged':True})
        logger.finish('completed')
        write_json(root/'completion.json',{'status':'completed','runs':{k:{x:v[x] for x in ('tasks','successes','known_total_tokens','knowledge_digest')}
                                                                      for k,v in result.items()},'frozen_hash':before['sha256']})
    except SystemExit as exc:
        status = 'stopped_at_task_boundary' if exc.code == 75 else 'failed'
        logger.finish('stopped' if exc.code == 75 else 'failed')
        write_json(root/'completion.json',{'status':status,'exit_code':exc.code})
        if exc.code != 75: raise
    except Exception as exc:
        logger.finish('failed')
        write_json(root/'completion.json',{'status':'failed','error':str(exc)});raise


if __name__ == '__main__': main()
