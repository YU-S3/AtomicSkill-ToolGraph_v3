"""Release evaluation statistics over immutable task/attempt records."""
import json
import math
from pathlib import Path
from statistics import mean,median
from atomic_skillgraph.core.serialization import atomic_write_json


def execution_boundary_metrics(trace, compiler):
    from collections import Counter
    plan = trace.get('runtime_plan') or {}
    audit = plan.get('planner_audit') or trace.get('planner_audit') or {}
    selected = next((r for r in audit.get('composite_candidates', [])
                     if r.get('composite_ref') == plan.get('source_composite_ref')), {})
    rejected = Counter()
    for call in trace.get('native_tool_calls', []):
        result = call.get('preflight_result') or {}
        if call.get('call_kind') == 'support_atomic_invocation' and result.get('accepted') is False:
            rejected[result.get('error_code') or result.get('error') or result.get('failure_code') or 'unknown'] += 1
    steps = trace.get('metadata', {}).get('runtime_steps', [])
    return {'graph_started': plan.get('source') == 'stored_composite',
        'program_static_closure': selected.get('program_static_closure'),
        'prepared_runtime_calls': sum(s.get('mode') == 'preparation' for s in steps),
        'support_rejections_by_code': dict(rejected),
        'program_actions': compiler.get('program_policy_actions'),
        'automatic_successors': compiler.get('llm_free_complete_successors'),
        'normal_runtime_decisions': compiler.get('residual_runtime_decisions')}

def _percentile(values,q):
    if not values:return None
    values=sorted(values);index=(len(values)-1)*q;lo=math.floor(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)

def write_release_report(output,config,*,resource_traces,digest_after):
    from .compiler_metrics import trace_metrics
    from atomic_skillgraph.core.serialization import to_primitive
    from atomic_skillgraph.deployment.publication_audit import json_lines
    from atomic_skillgraph.agents.runtime_expression_codec import _hash
    from collections import defaultdict
    output=Path(output);rows=[];usage={};requests={};expressions=[];compiler=[]
    task_usage=defaultdict(set)
    seen=set()
    for raw in resource_traces:
        trace=to_primitive(raw)
        if not trace.get('trace_id') or trace['trace_id'] in seen:continue
        seen.add(trace['trace_id'])
        task=trace.get('task',{}).get('task_id')
        events=trace.get('llm_usage',[])
        for event in events:
            identity=event.get('event_id')
            if not identity:raise ValueError('usage event without identity')
            if identity in usage and usage[identity]!=event:raise ValueError('conflicting usage identity')
            usage[identity]=event
            task_usage[task].add(identity)
        for request in trace.get('provider_requests',[]):
            if request['request_id'] in requests and requests[request['request_id']]!=request:
                raise ValueError('conflicting provider request identity')
            requests[request['request_id']]=request
            final=request.get('final_payload_audit') or {}
            projections=trace.get('metadata',{}).get('runtime_context_projection_audits',[])
            matches=[p for p in projections if p.get('session_id')==request['session_id']
                and p.get('release_expression',{}).get('lean_payload_hash')
                    in final.get('policy_context_sha256',[])]
            expressions.append({'task_id':task,'trace_id':trace['trace_id'],**request,
                'profile':config['deployment']['presentation_profile'],
                'matching_projection_audits':matches,
                'final_lean_payload_matched':bool(matches)})
        rows.append({'task_id':task,'trace_id':trace['trace_id'],'official_success':trace.get('benchmark_success'),
            'infrastructure_failure':trace.get('infrastructure_failure'),'recorded_tokens':sum(e['prompt_tokens']+e['completion_tokens'] for e in events),
            'prompt_tokens':sum(e['prompt_tokens'] for e in events),'completion_tokens':sum(e['completion_tokens'] for e in events),
            'reasoning_tokens':sum(e['reasoning_tokens'] for e in events) if all(e.get('reasoning_tokens') is not None for e in events) else None})
        metrics = trace_metrics(trace) or {'capture_status':'missing'}
        compiler.append({'task_id':task,'trace_id':trace['trace_id'],**metrics,
            **execution_boundary_metrics(trace, metrics)})
    manifest=json.loads((output/'run_manifest.json').read_text())
    # Terminal manifest identities determine scored tasks; failed prefixes stay
    # in the cost ledger but never count as extra evaluation episodes.
    import sqlite3
    with sqlite3.connect(output/'run_state.sqlite3') as db:
        selected={r[0]:r[1] for r in db.execute("SELECT task_id,trace_id FROM run_tasks WHERE state='completed'")}
    scored=[r for r in rows if selected.get(r['task_id'])==r['trace_id']]
    scored_compiler=[r for r in compiler if selected.get(r['task_id'])==r['trace_id']]
    if len(scored)!=len(selected):raise ValueError('completed task missing its scored Trace')
    costs=[sum(usage[key]['prompt_tokens']+usage[key]['completion_tokens'] for key in task_usage[task]) for task in selected]
    summary={'tasks':len(selected),'successes':sum(r['official_success'] is True for r in scored),
        'success_rate':mean(r['official_success'] is True for r in scored) if scored else None,
        'total_recorded_tokens':sum(e['prompt_tokens']+e['completion_tokens'] for e in usage.values()),
        'unknown_usage_requests':sum(r.get('usage_status')!='reported' for r in requests.values()),
        'physical_requests':len(requests),'token_mean':mean(costs) if costs else None,'token_median':median(costs) if costs else None,
        'token_p90':_percentile(costs,.9),'token_p95':_percentile(costs,.95),'rows':scored,
        'all_attempt_rows':rows,'bank_digest':config['bank_release']['expected_bank_digest'],'profile':config['deployment']['presentation_profile'],
        'config_hash':manifest.get('config_hash'),'code_hash':manifest.get('code_commit')}
    atomic_write_json(output/'summary.json',summary);atomic_write_json(output/'all_usage.json',list(usage.values()))
    summary['total_tokens']=None if summary['unknown_usage_requests'] else summary['total_recorded_tokens']
    summary['prompt_tokens']=sum(e['prompt_tokens'] for e in usage.values())
    summary['completion_tokens']=sum(e['completion_tokens'] for e in usage.values())
    summary['reasoning_tokens']=sum(e['reasoning_tokens'] for e in usage.values()) if all(e.get('reasoning_tokens') is not None for e in usage.values()) else None
    summary['programmer_tokens']=sum(e['prompt_tokens']+e['completion_tokens'] for e in usage.values() if e.get('bucket')=='tool_builder_runtime')
    for field in ('messages','tools'):
        values=[r.get('final_payload_audit',{}).get(field+'_utf8_bytes') for r in requests.values()]
        summary['actual_'+field+'_bytes']=sum(values) if all(v is not None for v in values) else None
    atomic_write_json(output/'summary.json',summary)
    json_lines(output/'runtime_expression_requests.jsonl',expressions)
    json_lines(output/'compiler_task_metrics.jsonl',scored_compiler)
    json_lines(output/'compiler_attempt_metrics.jsonl',compiler)
    atomic_write_json(output/'runtime_expression_coverage.json',{'physical_requests':len(requests),
        'captured':sum(bool(r.get('final_payload_audit')) for r in requests.values()),
        'matched_lean_requests':sum(r['final_lean_payload_matched'] for r in expressions)})
    from .compiler_metrics import aggregate as compiler_aggregate
    atomic_write_json(output/'compiler_summary.json',compiler_aggregate(
        [{'task_id':r['task_id'],'compiler_diagnostics':r if r.get('version') else None} for r in scored_compiler],
        resource_summary={k:summary[k] for k in ('total_recorded_tokens','total_tokens','unknown_usage_requests','programmer_tokens')}))
    for request in expressions:
        atomic_write_json(output/'provider_payload_audit'/f"{request['request_id']}.json",request.get('final_payload_audit'))
    if digest_after!=config['bank_release']['expected_bank_digest']:
        raise ValueError('release bank mutated')
    atomic_write_json(output/'bank_digest_audit.json',{'before':manifest['knowledge_digest'],'after':digest_after,'unchanged':True})
    return summary

def aggregate(plan_path,output):
    from .protocol import hash_config
    plan=json.loads(Path(plan_path).read_text());results=[]
    if {(r['seed'],r['rep']) for r in plan['runs']}!={(42,1),(42,2),(42,3),(43,1),(44,1)} or len(plan['runs'])!=5:
        raise ValueError('evaluation matrix mismatch')
    code_hashes=set();bank_hashes={}
    for run in plan['runs']:
        config=json.loads(Path(run['config']).read_text())
        if hash_config(config)!=run['config_hash'] or config['deployment']['presentation_profile']!=plan['profile']:
            raise ValueError('evaluation plan config/profile mismatch')
        path=Path(run['output'])/'summary.json'
        result=json.loads(path.read_text()) if path.exists() else None
        if result is not None:
            if (result['config_hash']!=run['config_hash'] or result['profile']!=plan['profile']
                or result['bank_digest']!=config['bank_release']['expected_bank_digest']):
                raise ValueError('evaluation result identity mismatch')
            code_hashes.add(result['code_hash'])
            prior=bank_hashes.setdefault(run['seed'],result['bank_digest'])
            if prior!=result['bank_digest']:raise ValueError('repetitions used different banks')
        results.append({**run,'result':result})
    if len(code_hashes)>1:raise ValueError('mixed evaluator code versions')
    repeats=[r for r in results if r['seed']==42]
    from .run_v3_released_frozen import ROOT
    reference=json.loads((ROOT/'data/baseline_manifests/test_ood_full_134.json').read_text())
    per_task=[]
    for task in reference['tasks']:
        outcomes=[]
        for rep in repeats:
            matches=[r for r in (rep['result'] or {}).get('rows',[]) if r['task_id']==task['task_id']]
            outcomes.append(matches[0]['official_success'] if len(matches)==1 else None)
        count=sum(v is True for v in outcomes);missing=len(outcomes)!=3 or any(v is None for v in outcomes)
        per_task.append({'task_id':task['task_id'],'outcomes':outcomes,'success_count':count,'missing':missing,
            'pass1':None if missing else count/3,'pass2':None if missing else math.comb(count,2)/3,
            'pass3':None if missing else int(count==3)})
    complete=not any(r['missing'] for r in per_task)
    def finished(run):
        result=run.get('result') or {}
        return result.get('tasks') == len(reference['tasks']) and {
            row['task_id'] for row in result.get('rows',[])} == {t['task_id'] for t in reference['tasks']}
    payload={'runs':results,'main_three_seed':[r for r in results if r['rep']==1],
        'fixed_bank_seed42':{k:mean(r[k] for r in per_task) if complete else None for k in ('pass1','pass2','pass3')},
        'paired_tasks':per_task,'complete':all(finished(r) for r in results),
        'main_three_seed_complete':all(finished(r) for r in results if r['rep']==1),
        'seed42_repeats_complete':complete}
    output=Path(output);output.mkdir(parents=True,exist_ok=True);atomic_write_json(output/'summary.json',payload)
    return payload

def compare_dev(root):
    root=Path(root);entries=json.loads((root/'declared_tasks.json').read_text());pairs=[]
    for entry in entries:
        current=json.loads((root/'current'/entry['task_id']/'summary.json').read_text())
        lean=json.loads((root/'lean'/entry['task_id']/'summary.json').read_text())
        pairs.append({'task_id':entry['task_id'],'current':current,'lean':lean,
            'new_failure':current['successes']==1 and lean['successes']==0,
            'rescued_failure':current['successes']==0 and lean['successes']==1,
            'recorded_token_delta':lean['total_recorded_tokens']-current['total_recorded_tokens']})
    result={'pairs':pairs,'current_successes':sum(p['current']['successes'] for p in pairs),
        'lean_successes':sum(p['lean']['successes'] for p in pairs),
        'new_failures':sum(p['new_failure'] for p in pairs),'rescued_failures':sum(p['rescued_failure'] for p in pairs)}
    atomic_write_json(root/'comparison/summary.json',result);return result
