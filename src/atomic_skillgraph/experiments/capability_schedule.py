"""Public-input source grouping and explicit post-Train unit authority."""
from collections import Counter
from copy import deepcopy
import re

from ..empirical.contracts import digest

VERSION = 'capability-source-schedule.v1'
GROUPING = 'public-goal-token-jaccard.v1'


def schedule(tasks, spec):
    grouping, generation = spec['source_grouping'], spec['generation']
    if (grouping != {'version':GROUPING,'max_source_cases_per_unit':4,
                     'source_coverage_policy':'include_each_of_48_once'} or
        generation['max_learning_units'] != 12 or len(tasks) != 48):
        raise ValueError('An explicit complete 48-source, twelve-unit plan is required')
    public = [{k:t[k] for k in ('task_id','physical_key','goal')} for t in tasks]
    if len({t['task_id'] for t in public}) != 48 or len({t['physical_key'] for t in public}) != 48:
        raise ValueError('Distinct source IDs required')
    words = {t['physical_key']:set(re.findall(r'[a-z]{3,}',t['goal'].casefold())) for t in public}
    frequency = Counter(word for tokens in words.values() for word in tokens)
    words = {key:{word for word in tokens if frequency[word] <= 24} for key,tokens in words.items()}
    algorithm = {'version':GROUPING,'input_fields':['task_id','physical_key','goal'],
        'tokens':'lowercase ASCII words of at least three characters; document frequency <=24',
        'order':'lexicographic task_id anchor, highest Jaccard to anchor, task_id tie-break',
        'interpretation':'public lexical relation only; not a certified shared capability'}
    pending = sorted(public,key=lambda t:t['task_id']);units=[]
    def similarity(a,b):
        left,right = words[a['physical_key']],words[b['physical_key']]
        return len(left & right)/len(left | right) if left | right else 0
    while pending:
        anchor = pending.pop(0)
        related = sorted(pending,key=lambda t:(-similarity(anchor,t),t['task_id']))[:3]
        members = [anchor,*related];keys = {t['physical_key'] for t in members}
        pending = [t for t in pending if t['physical_key'] not in keys]
        links = [{'task_id':t['task_id'],'physical_key':t['physical_key'],
            'jaccard_to_anchor':similarity(anchor,t),'shared_public_terms':sorted(words[anchor['physical_key']] & words[t['physical_key']])}
            for t in related]
        value = {'sources':members,'public_relation':links,'algorithm_hash':digest(algorithm),
                 'relation_status':'lexical_overlap' if all(link['shared_public_terms'] for link in links) else 'weak_public_relation'}
        units.append({'unit_id':digest(value),**value})
    return {'version':VERSION,'algorithm':algorithm,'algorithm_hash':digest(algorithm),
        'public_inputs_hash':digest(sorted(public,key=lambda t:t['task_id'])),'units':units}


def unit_limits(config, spec):
    """Upper bound for induction, proposal, its one repair, and content check."""
    generation = spec['generation'];learning = config['learning'];llm = config['llm']
    caps = [llm['purpose_overrides'][purpose]['max_completion_tokens'] for purpose in
            ('capability_induction','guidance_learning','guidance_grounding')]
    caps[1:1] = [caps[1]] * learning['guidance_repair_limit']
    attempts_per_call = 1+llm['max_retries']
    calculated = {'token_limit':sum(learning['extractor_payload_max_bytes']+cap for cap in caps)*attempts_per_call,
                  'request_limit':len(caps)*attempts_per_call}
    if generation['unit_budget'] != calculated:
        raise ValueError('P2 unit budget must match full Provider wire/output/retry bounds')
    return deepcopy(calculated)


def validate_plan(plan, config):
    declared = schedule(plan['generation_tasks'],plan['settings'])
    if plan['source_schedule'] != declared or plan['candidate_units'] != [
            [t['physical_key'] for t in unit['sources']] for unit in declared['units']]:
        raise ValueError('Frozen public source schedule changed')
    if plan['unit_limits'] != unit_limits(config,plan['settings']):
        raise ValueError('P2 unit authority changed')
    limit=plan['settings']['candidate']['max_candidates_to_selection']
    if limit != plan['settings']['paired_evaluation']['number_of_candidates_that_may_use_selection'] or limit != 2:
        raise ValueError('Separate selection cap must remain two')
    budget=plan['settings']['budget_additional_authority_proposed_not_spent']
    if budget.get('execution_contract')!='best_effort_bounded' or any(
        plan['new_limits'][key]>budget[field] for key,field in
        [('token_limit','max_new_total_tokens'),('request_limit','max_new_physical_http_attempts')]):
        raise ValueError('Versioned P2 cannot enlarge declared authority')


def request_bound(config, spec):
    units=spec['generation']['max_learning_units'];selected=spec['candidate']['max_candidates_to_selection']
    tasks=sum(spec['split']['size'][key] for key in ('selection_screen','selection_confirmation'))
    # The existing AnswerExecutor has eight normal decisions and one possible final fallback.
    return {'generation_physical_http_upper':units*unit_limits(config,spec)['request_limit'],
        'routing_physical_http_upper':selected*tasks*(1+config['llm']['max_retries']),
        'pair_physical_http_upper':selected*tasks*2*9,
        'full_pair_and_routing_minimum_http':selected*tasks*3,
        'at_least_one_induction_per_unit_plus_full_pair_and_routing_http':units+selected*tasks*3,
        'full_coverage_guaranteed_by_80_http':False,
        'val_test_theoretical_reserve_tokens':117*config['budget']['eval_task_tokens']}
