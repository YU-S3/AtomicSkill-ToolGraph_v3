"""Offline compiler observations; no model calls or bank writes."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median

from atomic_skillgraph.core.serialization import to_primitive, atomic_write_json
from atomic_skillgraph.traces.canonical import canonical_action_indices
from atomic_skillgraph.traces.compiler_observer import VERSION


def ratio(n, d):
    return n / d if d else None


def trace_metrics(trace):
    t = to_primitive(trace)
    if not isinstance(t, dict):
        t = to_primitive(vars(trace))
    obs = t.get('metadata', {}).get('compiler_observability')
    if not obs:
        return None
    links, entries = obs['program_invocation_links'], obs['node_entry_windows']
    canonical = set(canonical_action_indices(t))
    union = lambda group, key: set().union(*(set(p.get(key) or []) for p in group))
    program = union(links, 'canonical_action_indices') & canonical
    eligible = [e for e in entries if not (e['bootstrap'] or e['already_satisfied'] or e['terminal_skipped'])]
    started = {p['implementation_attempt_id'] for p in links if p['started']}
    automatic = {p['implementation_attempt_id'] for p in links if p['started'] and p['origin'] == 'graph_entry_auto'}
    complete = [e for e in eligible if e['outcome'] == 'automatic_complete_success'
                and all(p['complete_success'] for p in links if p['implementation_attempt_id'] == e['selected_invocation_attempt_id'])]
    automatic_entries = [e for e in eligible if e['selected_invocation_attempt_id'] in automatic]
    free = [e for e in complete if e.get('provider_request_refs') == [] and e['capture_complete']]
    non_source = [p for p in links if p.get('source_task_member') is False
                  and p['origin'] in {'graph_entry_auto','agent_selected_registered'}
                  and p.get('evidence_origin') == 'learned' and p['asset_origin'] == 'persistent']
    source_known = any(p.get('source_task_member') is not None for p in links)
    consumed_producers = {c['producer_invocation_id'] for c in obs['dataflow_consumptions'] if c['consumed']}
    non_source_metrics = (dict(started=len({p['implementation_attempt_id'] for p in non_source if p['started']}),
        completed=len({p['implementation_attempt_id'] for p in non_source if p['complete_success']}),
        consumed=len({p['implementation_attempt_id'] for p in non_source if p['implementation_attempt_id'] in consumed_producers}))
        if source_known else None)
    runtime_requests, repairs = [], []
    runtime_sessions = [s for s in t.get('agent_sessions', []) if s.get('session_type', '').startswith('runtime')]
    observed_sessions = True
    for session in runtime_sessions:
        audits = session.get('snapshot', {}).get('runtime_request_context_audits')
        if audits is None:
            observed_sessions = False
            continue
        for a in audits:
            (repairs if a['repair_in_progress'] else runtime_requests).append((session['session_id'], a['request_sequence']))
    buckets = Counter(u.get('bucket', 'unknown') for u in t.get('llm_usage', []))
    compilation = obs['compilation']
    rollback = t.get('metadata', {}).get('runtime_rollbacks', [])
    return dict(version=VERSION, capture_status=obs['capture_status'], missing_reasons=obs['missing_reasons'],
        selected_route=compilation.get('selected_route', 'unknown'), accepted_graph=compilation.get('accepted_graph'),
        structural_reuse_kind=compilation.get('structural_reuse_kind', 'unknown'),
        canonical_policy_actions=len(canonical), all_policy_actions=len(t.get('environment_actions', [])),
        program_canonical_actions=len(program), program_policy_actions=len(union(links, 'policy_action_indices')),
        program_persistent_actions=len(union([p for p in links if p['asset_origin']=='persistent'], 'canonical_action_indices') & canonical),
        program_task_local_actions=len(union([p for p in links if p['asset_origin']=='task_local'], 'canonical_action_indices') & canonical),
        rollback_policy_actions=len(t.get('environment_actions', []))-len(canonical),
        physical_restore_replay_actions=sum(r.get('restore_replay_action_count', 0) for r in rollback),
        program_started=bool(started), program_started_attempts=len(started),
        program_complete=any(p['complete_success'] for p in links),
        dataflow_consumed=any(c['consumed'] for c in obs['dataflow_consumptions']),
        eligible_successor_entries=len(eligible), autonomous_started_entries=len(automatic_entries),
        autonomous_complete_entries=len(complete),
        autonomous_terminal_effect_entries=sum(e['outcome']=='automatic_terminal_effect' for e in eligible),
        llm_free_complete_successors=len(free),
        llm_free_complete_successors_by_origin=dict(Counter(next((p.get('evidence_origin','unknown') for p in links
            if p['implementation_attempt_id']==e['selected_invocation_attempt_id']),'unknown') for e in free)),
        entry_outcomes=dict(Counter(e['outcome'] for e in entries)),
        residual_runtime_decisions=len(set(runtime_requests)) if observed_sessions else None,
        runtime_protocol_repairs=len(set(repairs)) if observed_sessions else None,
        provider_http_attempts=len(t.get('provider_requests', [])),
        unknown_usage_attempts=sum(r.get('usage_status') != 'reported' for r in t.get('provider_requests', [])),
        logical_calls_by_bucket=dict(buckets),
        non_source_deployment=non_source_metrics,
        non_source_missing_reason=None if source_known else 'no_verified_admission_source_membership_at_deployment',
        evidence_kind=t.get('metadata', {}).get('experiment_kind', 'unknown'))


def distribution(values):
    values = sorted(v for v in values if v is not None)
    def quantile(p):
        x = (len(values)-1)*p
        i = int(x)
        return values[i] + (values[min(i+1,len(values)-1)]-values[i])*(x-i)
    return dict(known=len(values), mean=mean(values) if values else None,
        median=median(values) if values else None, p90=quantile(.9) if values else None,
        p95=quantile(.95) if values else None)


def aggregate(rows, *, resource_summary=None):
    rows = list(rows)
    known = [r['compiler_diagnostics'] for r in rows if r.get('compiler_diagnostics') is not None]
    total = len(rows)
    summed = lambda name: sum(k[name] for k in known)
    compiled = sum(k['accepted_graph'] is True for k in known)
    result = dict(version=VERSION, task_count=total, observed_tasks=len(known), unknown_tasks=total-len(known),
        graph_compilation_coverage=ratio(compiled,total) if known else None,
        accepted_graph_tasks=compiled if known else None,
        route_distribution=dict(Counter(k['selected_route'] for k in known)),
        program_task_coverage=ratio(summed('program_started'),total) if known else None,
        program_completed_task_coverage=ratio(summed('program_complete'),total) if known else None,
        dataflow_consumed_task_coverage=ratio(summed('dataflow_consumed'),total) if known else None,
        program_executed_action_share=ratio(summed('program_canonical_actions'),summed('canonical_policy_actions')),
        autonomous_successor_coverage=ratio(summed('autonomous_started_entries'),summed('eligible_successor_entries')),
        autonomous_complete_success_rate=ratio(summed('autonomous_complete_entries'),summed('autonomous_started_entries')),
        non_source_deployment=({name:sum(k['non_source_deployment'][name] for k in known if k['non_source_deployment'] is not None)
            for name in ('started','completed','consumed')} if any(k['non_source_deployment'] is not None for k in known) else None),
        residual_runtime_decisions=distribution(k['residual_runtime_decisions'] for k in known),
        cost_distributions={name: distribution(r.get(name) for r in rows) for name in
            ('total_tokens','prompt_tokens','completion_tokens','reasoning_tokens','call_count')},
        # Existing report resource rows include auxiliary/failed attempts once.
        resource_accounting=resource_summary,
        caveat='Known observed numerators use all task denominators; inspect unknown_tasks. No causal token-saving or energy claim.')
    for key in ('canonical_policy_actions','all_policy_actions','program_canonical_actions','program_policy_actions',
                'program_persistent_actions','program_task_local_actions','rollback_policy_actions',
                'physical_restore_replay_actions','eligible_successor_entries','autonomous_started_entries',
                'autonomous_complete_entries','autonomous_terminal_effect_entries','llm_free_complete_successors',
                'provider_http_attempts','unknown_usage_attempts'):
        result[key] = summed(key) if known else None
    return result


DICTIONARY = '''# Compiler observation metrics

Version: r103.compiler-observation.v1. Read-only derivation; old missing capture is null, never zero.

- Graph coverage: one final accepted P0/P2 graph per task / all tasks (failures included). P2 novelty is unknown.
- Program action share: union of Tool-span action event indices, intersected with final canonical policy indices / all canonical policy actions. Multiple Tools/helpers never multiply an event. Rollback policy work and physical restoration are separate.
- Program task coverage: real started attempts / all tasks. Complete and consumed coverage are separate.
- Automatic denominator: reached non-bootstrap, non-satisfied, non-terminal-skip entries. Complete requires started/completed, valid outputs/effects and all Tools complete, no terminal interruption or failure. Zero denominator is null.
- LLM-free complete: complete automatic entry plus actual before/after request snapshots proving an empty interval. Missing provider boundaries are unknown.
- Residual decisions: distinct Runtime session/request sequence pairs excluding protocol repairs. HTTP attempts and all usage buckets remain separate. Observer does not infer logical requests from session count.
- Dataflow: compiled successor actually reads the unchanged typed DATA_FLOW binding. Publication, value coincidence and overwritten/re-certified bindings receive no inferred credit.
- Non-source deployment: original immutable admission cases with typed completed replay results supply benchmark/task signatures; legacy, fixture or missing provenance stays null. Only actual registered online starts count, never offline replay or later support-credit counts.
- Costs: original report is accounting authority, including failed/auxiliary attempts and online ToolBuilder. Linked invocations never add usage. Reasoning is a subset of completion. Unknown usage is not filled with zero. Per-task distributions include unsuccessful tasks; resource totals include auxiliary attempts separately.
- Public initial hash is not a complete hidden-state digest. Existing unreliable/unavailable full digest remains null. Fixture evidence must not be presented as learned deployment. No causal token/energy saving is inferred from action coverage.
'''


def write_reports(rows, output_dir, *, resource_summary=None):
    rows = list(rows)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tasks = [{k: r.get(k) for k in ('task_id','trace_id','benchmark_success','official_won','compiler_diagnostics')}
             for r in rows]
    (root/'compiler_task_metrics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in tasks), encoding='utf-8')
    atomic_write_json(root/'compiler_summary.json', aggregate(rows, resource_summary=resource_summary))
    (root/'compiler_metrics_dictionary.md').write_text(DICTIONARY, encoding='utf-8')


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    # This thin reader need not reinterpret old learning-stage counters.
    # Use the original accounting authority, never recompute fees from links.
    from experiments.report import _usage_report, _sum_usage
    rows, usages = [], []
    for f in sorted(a.traces.glob('trace_*.json')):
        trace = json.loads(f.read_text(encoding='utf-8'))
        meta = trace.get('metadata', {})
        usage = _usage_report(trace, meta)
        usages.append(usage['episode_total'])
        if meta.get('trace_kind') == 'maintenance' or trace['task'].get('task_type') == 'maintenance':
            continue
        rows.append(dict(task_id=trace['task']['task_id'],trace_id=trace['trace_id'],
            benchmark_success=trace.get('benchmark_success'),compiler_diagnostics=trace_metrics(trace),
            **usage['episode_total']))
    write_reports(rows,a.output,resource_summary={**_sum_usage(usages),
        'scope':'provided immutable Trace collection; formal task/attempt selection remains runner authority'})


if __name__ == '__main__':
    main()
