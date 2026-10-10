"""Single-seed allocation derived from canonical counts and existing request loops."""
from .canonical_manifest import ADAPTER_NAMES

BENCHMARKS=('searchqa','livemath','officeqa','spreadsheetbench')
TASK_LIMITS={'train_task_tokens':200000,'train_solve_tokens':120000,
             'train_learning_tokens':80000,'eval_task_tokens':96000}


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
