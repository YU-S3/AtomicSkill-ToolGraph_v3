"""Public-question coverage; phase telemetry never hides callable capabilities."""
import re

VERSION = 'office.humanstrong.selection.v2'
CORE = ['OA-Target', 'OA-Locate', 'OA-Table', 'OA-Units', 'OA-Stats', 'OA-Commit', 'OA-Fallback']
SPECIAL = {
    'OA-Periods': r'year|month|quarter|period|series|annual|weekly|daily|between|from',
    'OA-Fiscal': r'fiscal|calendar|fy\b|cy\b|transition quarter',
    'OA-Treasury': r'treasury|debt|securit|tender|bid|matur|refunding|gold|silver|reserve|interest',
    'OA-Advanced': r'forecast|arima|regress|correlat|box.cox|gini|quantile|percentile|z.score|cpi|real (?:value|dollar)',
    'OA-Revision': r'revis|report|bulletin|published|as of|recorded|preliminary|adjusted|current dollar',
}


def select_names(goal, phase='locate'):
    return CORE + [name for name, pattern in SPECIAL.items() if re.search(pattern, goal, re.I)]


def callable_programs(*args, **kwargs):
    return ['scoped_search.py', 'table_extract.py', 'period_ledger.py', 'series_metrics.py', 'treasury_transform.py']


GLOBAL_RULE = '''Use the supplied source pages as primary document evidence. Inspect them before searching; do not repeat retrieval merely to satisfy a phase label. Use candidate files and source hints only to resolve a specific missing or ambiguous fact. All advertised Programs are available now: select one when its exact operation and input contract fit. A Program validates computation, not the meaning of your chosen row or formula. Use existing result references for provenance; do not add evidence_state fields to native actions. Finish promptly once the requested quantity is established; no compulsory verification call.'''
FINISH_ONLY_RULE = '''Return only the requested answer payload, preserving its sign, scale, precision and list order. No tools are available here. Use the supplied exact source spans and computed values; do not substitute a nearby measure, invent a missing operand or change a supported answer solely because the answer channel changed. Omit explanation and unit suffixes unless the question explicitly requires them.'''
