"""Read-only replay and separate interpretation export for the closed LM1 batch."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from ..empirical.bank import Bank
from ..empirical.contracts import PublicTask, digest, validate_schema_instance
from ..empirical.choice_guidance import validate_proposal, validate_check, select, checked_asset, STATS_KEY
from ..harness.scorers.livemath import evaluate
from .canonical_manifest import sha256
from .run_empirical import write_json
from .run_livemath_lm1_validation import summarize


def read(path):return json.loads(Path(path).read_text())


def export(parent,destination):
    parent,destination=Path(parent).resolve(),Path(destination).resolve()
    if destination.is_relative_to(parent):raise ValueError('Review must not overwrite the closed batch')
    before={p.relative_to(parent).as_posix():sha256(p) for p in parent.rglob('*') if p.is_file()}
    requests=[];replays=[];reason_counts=Counter();published=0
    for path in sorted((parent/'train/events').glob('*/requests.json')):
        audit=read(path);requests.extend(audit['requests'])
        learning=read(path.parent/'learning.json').get('learning',{})
        published+=bool(learning.get('persisted_skill_id'))
        for q in audit['requests']:
            calls=q['response']['tool_calls'];tool=q['tools'][0]['function'];schema=tool['parameters']
            assert len(calls)==1 and calls[0]['name']==tool['name']
            value=calls[0]['arguments'];material=read_material(q);reason='accepted'
            try:validate_schema_instance(value,schema)
            except ValueError:reason='grounding_schema_invalid' if q['purpose']=='guidance_grounding' else 'proposal_schema_invalid'
            if reason=='accepted':
                try:
                    if q['purpose']=='guidance_grounding':
                        supported=validate_check(value,material['proposal'],material['source'],schema)
                        if not supported:reason='grounding_contradicted' if value['status']=='contradicted' else 'grounding_insufficient'
                    else:validate_proposal(value,material['source'],material.get('related',[]),schema)
                except ValueError:reason='grounding_quote_invalid' if q['purpose']=='guidance_grounding' else 'proposal_scope_invalid'
            reason_counts[reason]+=1
            replays.append({'logical_request_id':q['id'],'purpose':q['purpose'],'repair':q['repair'],
                'schema_sha256':digest(schema),'host_replay':reason,'original_learning_decision':learning.get('decision'),
                'original_errors':learning.get('errors',[])})
            if q is audit['requests'][-1] and learning.get('decision')=='rejected':assert reason!='accepted'
    assert len(requests)==31 and published==9
    bank=Bank(parent/'train/frozen_bank',readonly=True)
    try:
        assets=bank.all('skill');assert len(assets)==9 and all(checked_asset(a) for a in assets)
        config=read(parent/'config.json');policy=config['runtime']['choice_guidance']
        val=read(parent/'val_tasks.json');saved=read(parent/'scope_precheck.json')['tasks'];replayed=[]
        for task,original in zip(val,saved):
            selection=bank.select_guidance(PublicTask(**task),policy)
            # The original precheck stores the unchanged selection payload with its public task ID.
            audit=original.get('selection',original)
            for key in ('audit','injected_ids','injected_chars'):
                if key in audit:assert selection[key]==audit[key]
            replayed.append(selection)
        assert len(val)==17 and not any(x['injected_ids'] for x in replayed)
        train=[PublicTask(**r['task']) if 'task' in r else PublicTask(**r) for r in read(parent/'source_records.json')]
        stats=json.loads(bank.db.execute('SELECT value FROM metadata WHERE key=?',(STATS_KEY,)).fetchone()[0])
        # Frozen host statistics are used unchanged; only the own-source asset is removed.
        train_selection=[select([a for a in assets if a['evidence_source']['physical_key']!=t.physical_key],stats,t,policy) for t in train]
        cross=sum(any(a['evidence_source']['physical_key']!=t.physical_key for a in s['selected']) for t,s in zip(train,train_selection))
        assert len(train)==60 and cross==0
        frozen_digest=bank.digest()
    finally:bank.close()
    original=read(parent/'result.json');new_summary=summarize(parent,original)
    assert new_summary['method_comparison_status']=='not_run_no_exposure' and new_summary['paired_n']==0
    assert new_summary['net_gain'] is None and new_summary['candidate_effort']=='high'
    traces=[read(p) for p in (parent/'val').glob('*/*/trace.json')]
    all_requests=[q for p in parent.glob('**/requests.json') for q in read(p)['requests']]
    physical=[a for q in all_requests for a in q.get('http_attempts',[])]
    tokens=sum(a['raw_usage']['total_tokens'] for a in physical)
    assert len(traces)==34 and len(physical)==65 and tokens==688595
    assert new_summary['arms']['A']['correct']==4 and new_summary['arms']['C']['correct']==2
    # Gold-only strata live in the separate audit, never in Runtime materials.
    private_path=next(Path(p) for p in read(parent/'manifest.json')['input_hashes'] if p.endswith('/resources/private/evaluator_records_val.json'))
    private=read(private_path)
    strata=[]
    for t in traces:
        task=t['task'];record=private[task['task_id']];raw=evaluate(t['execution']['prediction'],record['correct_choice'],record['choices'])
        assert bool(raw['em'])==t['score']['hard']
        pattern=r'\b(?:option|statement|conclusion)s?\b'
        gold=record['correct_choice'];gold_text=gold['text'] if isinstance(gold,dict) else next(c['text'] for c in record['choices'] if c['label']==gold)
        strata.append({'task_id':task['task_id'],'contains_meta_option':any(re.search(pattern,c['text'],re.I) for c in task['inputs']['choices']),
            'gold_is_meta_option':bool(re.search(pattern,gold_text,re.I)),'original_correct':t['score']['hard']})
    after={p.relative_to(parent).as_posix():sha256(p) for p in parent.rglob('*') if p.is_file()}
    assert before==after
    result={'parent_manifest_sha256':sha256(parent/'manifest.json'),'original_files_sha256':before,
        'new_actual_http':0,'new_model_tokens':0,'historical_learning_responses':31,'historical_val_records':34,
        'historical_http':65,'historical_tokens':tokens,'published_assets_unchanged':9,
        'host_replay':replays,'replay_reason_counts':dict(reason_counts),'selection_pairs':153,'val_exposed':0,
        'train_cross_source_exposed':cross,'frozen_digest_before_after':frozen_digest,
        'interpretation_export':new_summary,'score_only_strata':strata,'archive_unchanged':True}
    write_json(destination/'review.json',result)
    return {k:v for k,v in result.items() if k not in {'original_files_sha256','host_replay','score_only_strata','interpretation_export'}}


def read_material(request):
    content=request['messages'][1]['content']
    if not isinstance(content,str):content=content[0]['text']
    return json.loads(content)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(export(a.parent,a.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
