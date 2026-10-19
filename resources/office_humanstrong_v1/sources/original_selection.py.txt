"""Deterministic public-question + actually-observed-phase selection, no LLM router."""
import re

VERSION='office.humanstrong.selection.v1'
PHASES={
    'locate':['OA-Target','OA-Locate'],
    'extract':['OA-Target','OA-Table'],
    'compute':['OA-Table','OA-Stats'],
    'finalize':['OA-Commit']}


def category(goal):
    q=goal.lower()
    # Specific markers, not the near-universal words Treasury, year, total or dollars.
    if re.search(r'box.cox|arima|forecast|autoreg|trend regression',q):return 'OA-Advanced'
    if re.search(r'currency|exchange rate|yen|rupee|fine pounds|in 32nds|quotation|par quote',q):return 'OA-Units'
    if re.search(r'maturity|tenders|accepted bids|issued versus|retired|outstanding|debt held by|government account|new cash|refunding',q):return 'OA-Treasury'
    if re.search(r'(?:from|between|during|over|through|to)\s+(?:the\s+)?(?:\w+\s+)?(?:19|20)\d{2}.{0,36}(?:19|20)\d{2}|every year|each month|month.by.month|time.series',q):return 'OA-Periods'
    if re.search(r'calendar.{0,12}fiscal|fiscal year|fy\s?\d{4}|fiscal month|transition quarter',q):return 'OA-Fiscal'
    if re.search(r'population|standard deviation|sample|regression|correlation|cagr|geometric mean|gini|z.score|slope|percentage change|growth rate',q):return 'OA-Stats'
    return None


def select_names(goal, phase='locate', *, max_cards=3):
    if phase not in PHASES:raise ValueError('Unknown phase')
    cards=list(PHASES[phase])
    specialized=category(goal)
    if specialized and specialized not in cards and phase!='finalize':
        cards.append(specialized)
    return cards[:max_cards]


def callable_programs(goal,phase,table_window_available=False):
    special=category(goal)
    if phase=='locate':return ['scoped_search.py']
    if phase=='extract':
        return (['table_extract.py'] if table_window_available else []) + (['period_ledger.py'] if special in {'OA-Periods','OA-Fiscal'} else [])
    if phase=='compute':
        priority={'OA-Periods':['period_ledger.py','series_metrics.py'],
                  'OA-Fiscal':['period_ledger.py','series_metrics.py'],
                  'OA-Units':['treasury_transform.py','series_metrics.py'],
                  'OA-Advanced':['treasury_transform.py','series_metrics.py']}
        return priority.get(special,['series_metrics.py'])
    return []


GLOBAL_RULE='''Solve the specific requested measure from real source documents. Once a complete sourced answer is available, finish promptly using the current native runtime_step finish action. For numeric answers return a bare numeric payload with required scale, sign and precision; omit a percent/currency/unit suffix unless the exact requested template requires it. Do not replace source evidence with remembered statistics. Never submit a table of intermediate values when the question asks for one statistic.'''
FINISH_ONLY_RULE='''This is the final plain-text answer channel: no tools or runtime_step actions are possible. Return only the best source-grounded final numeric/list/date payload and preserve its correct scale and precision. Do not add explanatory prose or invent missing data. Do not output XML tags.'''
