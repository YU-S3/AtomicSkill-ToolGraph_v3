"""ScienceWorld reporting: continuous official score and exact recorded usage."""
from collections import defaultdict
import json
from pathlib import Path
import math


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def usage_summary(events):
    """Exact recorded sums. Reasoning is already included in completion."""
    return {'physical_requests': len(events),
        'logical_requests': len({e['logical_call_id'] for e in events}),
        'unknown_usage_requests': sum(e.get('prompt_tokens') is None or e.get('completion_tokens') is None for e in events),
        'recorded_prompt_tokens': sum(e.get('prompt_tokens') or 0 for e in events),
        'recorded_completion_tokens': sum(e.get('completion_tokens') or 0 for e in events),
        'recorded_reasoning_tokens': sum(e.get('reasoning_tokens') or 0 for e in events),
        'unknown_reasoning_requests': sum(e.get('reasoning_tokens') is None for e in events),
        'recorded_total_tokens': sum((e.get('prompt_tokens') or 0) + (e.get('completion_tokens') or 0) for e in events),
        'retries': sum(e.get('attempt', 1) > 1 for e in events),
        'failed_requests': sum(e.get('status') == 'failed' for e in events),
        'recorded_latency_ms': sum(e.get('latency_ms') or 0 for e in events)}

def summarize(rows, output, *, operation_prefix=None):
    def mean(values): return sum(values) / len(values) if values else None
    groups = {key: defaultdict(list) for key in ('task_type', 'macro_type')}
    for row in rows:
        for key in groups:
            groups[key][row[key]].append(row)
    result = {'episodes': len(rows), 'mean_score_micro': mean([r['official_score'] for r in rows]),
        'mean_normalized_score_micro': mean([r['normalized_score'] for r in rows]),
        'success100_micro': mean([int(r['perfect_success']) for r in rows]),
        'actions': sum(r['environment_actions'] for r in rows),
        'wall_time_ms': sum(r['wall_time_ms'] for r in rows)}
    for name, key in [('30_task_type', 'task_type'), ('10_category', 'macro_type')]:
        result[f'macro_{name}_score'] = mean([mean([r['official_score'] for r in values]) for values in groups[key].values()])
        result[f'macro_{name}_success100'] = mean([mean([int(r['perfect_success']) for r in values]) for values in groups[key].values()])
        result[f'macro_{name}_observed_groups'] = len(groups[key])
        result[f'macro_{name}_expected_groups'] = 30 if key == 'task_type' else 10
    events = [json.loads(line) for path in Path(output).rglob('provider_calls.jsonl')
              for line in path.read_text().splitlines() if line.strip()]
    if operation_prefix is not None:
        events = [e for e in events if str(e.get('operation','')).startswith(operation_prefix)]
    if len({e['provider_attempt_id'] for e in events}) != len(events):
        raise ValueError('Duplicate physical provider attempt identity')
    result['usage'] = usage_summary(events)
    result['usage_by_role'] = {role: usage_summary([e for e in events if e.get('role') == role])
                              for role in sorted({e.get('role', 'unknown') for e in events})}
    by_task = defaultdict(list)
    for event in events:
        by_task[event.get('task_id')].append(event)
    task_rows = []
    for row in rows:
        task_events = by_task.get(row['task_id'], [])
        task_rows.append({**row, 'usage': usage_summary(task_events),
            'usage_capture_present': bool(task_events),
            # These baselines do not expose A/I/T/G or Support observers.
            'selected_route': None, 'search_scope_signature': None,
            'tool_execution_signature': None})
    result['per_task'] = task_rows
    tokens = [r['usage']['recorded_total_tokens'] for r in task_rows if r['usage_capture_present']]
    actions = [r['environment_actions'] for r in rows]
    total_tokens = result['usage']['recorded_total_tokens']
    total_score = sum(r['official_score'] for r in rows)
    result['efficiency'] = {'tokens_per_task': total_tokens / len(rows) if rows else None,
        'tokens_per_score_point': total_tokens / total_score if total_score > 0 else None,
        'actions_per_task': mean(actions),
        'p50_tokens': percentile(tokens, .5), 'p90_tokens': percentile(tokens, .9),
        'token_quantile_observed_tasks': len(tokens), 'token_quantile_unknown_tasks': len(rows) - len(tokens),
        'p50_actions': percentile(actions, .5), 'p90_actions': percentile(actions, .9),
        'over_100k_token_tasks': sum(t > 100000 for t in tokens),
        'top_10_percent_tasks_token_share': sum(sorted(tokens, reverse=True)[:math.ceil(len(tokens)*.1)]) / sum(tokens) if sum(tokens) else None,
        'unassigned_recorded_tokens': total_tokens - sum(tokens),
        'environment_moves': sum(r['environment_moves'] for r in rows) if all('environment_moves' in r for r in rows) else None}
    failures = list(Path(output).rglob('infrastructure_failure.json')) if operation_prefix is None else None
    result['reliability'] = {'protocol_rejections': sum(r.get('protocol_rejections', 0) for r in rows),
        'infrastructure_failure_receipts': len(failures) if failures is not None else None,
        'unknown_usage_requests': result['usage']['unknown_usage_requests'],
        'provider_retries': result['usage']['retries']}
    return result
