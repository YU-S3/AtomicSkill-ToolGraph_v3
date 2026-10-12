"""Single-seed allocation derived from canonical counts and existing request loops."""
from .canonical_manifest import ADAPTER_NAMES

BENCHMARKS=('searchqa','livemath','officeqa','spreadsheetbench')
TASK_LIMITS={'train_task_tokens':200000,'train_solve_tokens':120000,
             'train_learning_tokens':80000,'eval_task_tokens':96000}


def revised_limits(original, requested, revision=None):
    """A fresh carryforward may change caps only with its explicit revision receipt."""
    if original==requested:
        return dict(original)
    if not revision or revision.get('schema')!='budget-revision.v1' or not revision.get('authorization'):
        raise ValueError('Budget increase requires an explicit authorization receipt')
    if revision.get('old_limits')!=original or revision.get('new_limits')!=requested:
        raise ValueError('Budget revision differs from the source ledger or resolved configuration')
    if set(requested)!=set(original) or requested.get('finish_reserve')!=original.get('finish_reserve'):
        raise ValueError('Budget revision must preserve the existing reserve policy')
    for key in ('token_limit','request_limit'):
        if type(requested.get(key)) is not int or requested[key]<original[key]:
            raise ValueError('Invalid cumulative budget increase: '+key)
    return dict(requested)


def allocation(manifest, profiles, retries):
    cells={}
    for benchmark in BENCHMARKS:
        profile=profiles[ADAPTER_NAMES.get(benchmark,benchmark)]
        counts={split:manifest['benchmarks'][benchmark]['splits'][split]['count'] for split in ('train','val','test')}
        if profile['interaction']=='single_answer':
            # AnswerExecutor: 8 decisions + one bounded finish. Learning:
            # Extractor and grounding/Builder each at most two requests.
            solve=9
        else:
            # Executor loop, one repair per Runtime decision; initial Planner
            # and its single replan each with one repair; one bounded finish.
            solve=2*max(16,4*profile['native_calls']+16)+4+1
        cells[benchmark]={**TASK_LIMITS,
            'token_limit':counts['train']*200000+(counts['val']+counts['test'])*96000,
            'request_limit':(counts['train']*(solve+4)+(counts['val']+counts['test'])*solve)*(retries+1),
            'finish_reserve':0}
    return {'policy':'canonical per-task ceilings; retries included, no automatic borrowing',
        'total_limits':{key:sum(c[key] for c in cells.values()) for key in ('token_limit','request_limit','finish_reserve')},
        'cells':cells}
