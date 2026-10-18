"""Public syntax routing only: no label, history, model call or theorem-truth rule."""
import json
import re
from .math_effect_contract import card, checked_candidate, token_count, R2


def features(task):
    choices=task.inputs['choices'];text=task.goal+' '+json.dumps(choices,ensure_ascii=False)
    patterns={'quantifier_syntax':r'\\forall|\\exists|for (?:every|all|each)|there exists',
        'equivalence_syntax':r'if and only if|equivalent|\\iff|necessary|sufficient',
        'bound_syntax':r'bound|inequality|\\leq?|\\geq?|asymptotic|converg',
        'scope_syntax':r'local|global|compact|regular|finite|infinite',
        'meta_option_present':r'stronger.*(?:prov|result|statement|conclusion)|none of (?:the|these)|all of (?:the|these)'}
    return {'comparable_conclusions':len(choices)>1,
            **{key:bool(re.search(pattern,text,re.I)) for key,pattern in patterns.items()}}


def select(candidates, task, tokenizer_path, max_tokens=600, *, version=None):
    if version==R2 or candidates and candidates[0]['method_version']==R2:
        return select_r2(candidates,task,tokenizer_path,max_tokens)
    public=features(task);selected=[];skipped=[]
    for c in candidates:  # order is the frozen policy, never response dependent
        if not checked_candidate(c):raise ValueError('Invalid P7 candidate identity')
        v=c['candidate'];a=v['applicability']
        applicable=all(public[k] for k in a['required_public_indicators']) and not any(public[k] for k in a['excluded_indicators'])
        if any(old['candidate']['scope']==v['scope'] for old in selected):applicable=False
        if applicable and token_count([card(x['candidate']) for x in [*selected,c]],tokenizer_path)<=max_tokens:
            selected.append(c)
        else:skipped.append(c['id'])
    return {'selected':selected,'features':public,'skipped':skipped,
            'injection_tokens':token_count([card(c['candidate']) for c in selected],tokenizer_path) if selected else 0,
            'routing_http':0}


def stronger_meta(task):
    matches=[]
    for choice in task.inputs['choices']:
        text=' '.join(choice['text'].split())
        # A relation to the other concrete answers AND truth AND a provable strengthening.
        others=re.search(r'\b(?:other|remaining|above|listed|preceding|following)\b.{0,70}\b(?:options?|statements?|claims?|answers?|conclusions?)\b|\b(?:options?|statements?|claims?|answers?|conclusions?)\b.{0,45}\b(?:above|below)\b',text,re.I)
        truth=re.search(r'\b(?:true|correct|valid|hold|holds)\b',text,re.I)
        strength=re.search(r'\b(?:stronger|strengthen\w*|sharper|improv\w*)\b',text,re.I)
        proof=re.search(r'\b(?:prov\w*|establish\w*|deduc\w*|obtain\w*|demonstrat\w*)\b',text,re.I)
        if others and truth and strength and proof:
            matches.append({'label':choice['label'],'text':choice['text']})
    return {'has_stronger_meta':bool(matches),'matches':matches}


def features_r2(task):
    value=features(task);meta=stronger_meta(task)
    value['meta_option_present']=meta['has_stronger_meta']
    value['has_stronger_meta']=meta['has_stronger_meta']
    return value


def select_r2(candidates,task,tokenizer_path,max_tokens=600):
    public=features_r2(task);selected=[];skipped=[];modes=set()
    for c in candidates:
        if not checked_candidate(c) or c['method_version']!=R2:raise ValueError('R2 requires its own candidate identity')
        a=c['candidate']['applicability'];mode=a['mode']
        ok=(mode not in modes and all(public[k] for k in a['require_all']) and not any(public[k] for k in a['exclude_any']))
        if ok and token_count([card(x['candidate']) for x in [*selected,c]],tokenizer_path)<=max_tokens:
            selected.append(c);modes.add(mode)
        else:skipped.append(c['id'])
    return {'selected':selected,'features':public,'meta_detection':stronger_meta(task),'skipped':skipped,
        'injection_tokens':token_count([card(x['candidate']) for x in selected],tokenizer_path) if selected else 0,'routing_http':0}
