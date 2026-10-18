"""Finite Train-feedback edits of model-induced predicates, without answer-rule literals."""
from copy import deepcopy
import json
from . import math_effect_program_r3 as p
from .contracts import digest

VERSION='livemath.effect-evolution.p7.r3.neighborhood.v1'


def neighborhood(parent,sources,tokenizer_path,limit=6):
    if not p.checked(parent):raise ValueError('Mutation parent must be an actual checked learned candidate')
    rule=parent['candidate']['public_choice_rule']
    if rule is None:return []
    variants=[]
    for field in ('require_all','exclude_any'):
        for i in range(len(rule[field])):
            v=deepcopy(rule);v[field].pop(i)
            if v['require_all']:variants.append((f'delete_{field}_{i}',v))
    for i,pattern in enumerate(rule['exclude_any']):
        variants.append((f'promote_exclusion_{i}',{'require_all':[pattern],
            'exclude_any':[x for j,x in enumerate(rule['exclude_any']) if j!=i]}))
    result=[];seen=set()
    for operation,rule_ in variants[:limit]:
        if digest(rule_) in seen:continue
        seen.add(digest(rule_));v=deepcopy(parent['candidate']);v['public_choice_rule']=rule_
        v.update(category='empirical_public_choice_predicate',rule_basis='benchmark_prior',
            goal='Apply a Train-tested public-choice hypothesis; this is not a mathematical proof.',
            steps=['Check the declared public applicability conditions.',
                'Require the positive text patterns and reject the exclusion patterns: '+json.dumps(rule_,ensure_ascii=False),
                'Select only a unique matching option. Otherwise continue ordinary mathematical reasoning.'],
            stop_condition='No match or multiple matches: abstain. Empirical accuracy never establishes theorem truth.',
            edit_rationale='Automatic one-condition neighborhood of a model-generated predicate; '+operation)
        evidence={**parent['model_evidence'],'origin':'host_train_feedback_neighborhood',
            'natural_model_candidate':False,'operation':operation,'search_version':VERSION,
            'source_response_hash':parent['model_evidence']['response_hash']}
        p.validate(v,sources,tokenizer_path)
        result.append(p.candidate(v,sources,evidence,[parent]))
    return result


def import_text(parent,sources,tokenizer_path):
    """Lossless prior learned text; old promotion decisions are not imported."""
    from .math_effect_contract import R2,checked_candidate,card
    if parent.get('method_version')==R2:
        if not checked_candidate(parent):raise ValueError('Unchecked prior R2 text')
        value={**deepcopy(parent['candidate']),'public_choice_rule':None}
        evidence={**parent['model_evidence'],'origin':'lossless_text_contract_conversion',
            'parent_method':R2,'parent_candidate_hash':digest(parent),'natural_model_candidate':False}
        p.validate(value,sources,tokenizer_path)
        converted=p.candidate(value,sources,evidence,[parent])
        if card(converted['candidate'])!=card(parent['candidate']):raise ValueError('Text conversion changed the model-visible Skill')
        return converted
    if not p.checked(parent) or parent['candidate']['public_choice_rule'] is not None:raise ValueError('Only learned text may enter the text policy pool')
    p.validate(parent['candidate'],sources,tokenizer_path)
    return deepcopy(parent)


def rule_rank(q):
    # All selection uses Train only. Shorter syntax breaks equal empirical evidence.
    return (-q['correct'],q['losses_vs_historical'],q['dispatched']-q['correct'],
        len(q['program']['source']),q['candidate_id'])
