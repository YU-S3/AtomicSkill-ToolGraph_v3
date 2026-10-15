"""Public syntax routing only: no label, history, model call or theorem-truth rule."""
import json
import re
from .math_effect_contract import card, checked_candidate, token_count


def features(task):
    choices=task.inputs['choices'];text=task.goal+' '+json.dumps(choices,ensure_ascii=False)
    patterns={'quantifier_syntax':r'\\forall|\\exists|for (?:every|all|each)|there exists',
        'equivalence_syntax':r'if and only if|equivalent|\\iff|necessary|sufficient',
        'bound_syntax':r'bound|inequality|\\leq?|\\geq?|asymptotic|converg',
        'scope_syntax':r'local|global|compact|regular|finite|infinite',
        'meta_option_present':r'stronger.*(?:prov|result|statement|conclusion)|none of (?:the|these)|all of (?:the|these)'}
    return {'comparable_conclusions':len(choices)>1,
            **{key:bool(re.search(pattern,text,re.I)) for key,pattern in patterns.items()}}


def select(candidates, task, tokenizer_path, max_tokens=600):
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
