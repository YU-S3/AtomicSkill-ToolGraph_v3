"""Continuous benchmark scoring beside the unchanged execution/cost observers."""
from collections import defaultdict
from atomic_skillgraph.core.serialization import atomic_write_json
from .report import write_reports


def score_summary(traces):
    rows = []
    for trace in traces:
        task = trace['task']
        score = trace.get('official_score')
        if score is None:
            raise ValueError(f"Missing official ScienceWorld score: {task['task_id']}")
        task_type = task['task_type']
        rows.append({'task_id':task['task_id'], 'task_type':task_type, 'macro_type':task_type.split('-')[0],
            'official_score':score, 'normalized_score':score/100, 'perfect_success':score == 100,
            'environment_done':trace.get('environment_done'), 'trace_id':trace['trace_id']})
    def mean(values):
        return sum(values)/len(values) if values else None
    summary = {'episodes':len(rows),'mean_score_micro':mean([r['official_score'] for r in rows]),
        'mean_normalized_score_micro':mean([r['normalized_score'] for r in rows]),
        'success100_micro':mean([r['perfect_success'] for r in rows]), 'per_task':rows}
    for field, target in [('task_type',30),('macro_type',10)]:
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        summary[field+'_groups_observed'] = len(groups)
        summary[field+'_groups_expected'] = target
        summary[field+'_mean_score_macro'] = mean([mean([r['official_score'] for r in g]) for g in groups.values()])
        summary[field+'_success100_macro'] = mean([mean([r['perfect_success'] for r in g]) for g in groups.values()])
    return summary


def write_scienceworld_reports(traces, output, *, auxiliary=()):
    summary = score_summary(traces)
    atomic_write_json(output/'scienceworld_scores.json',summary)
    write_reports(traces,output,title='ScienceWorld execution, learning and recorded usage',auxiliary_usage_traces=auxiliary)
    return summary
