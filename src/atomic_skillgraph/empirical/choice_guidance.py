"""Host-bound public choice evidence and frozen, Train-only applicability statistics."""
import hashlib
import json
import math
import re
import unicodedata
from copy import deepcopy

from . import (CHOICE_GUIDANCE_POLICY_VERSION, CHOICE_GUIDANCE_SELECTION_VERSION,
               CHOICE_NORMALIZER_VERSION)
from . import PUBLIC_CHOICE_POLICY_VERSION
from .contracts import digest, validate_schema_instance
from .public_process import VERSION as PROCEDURE_VERSION, REQUIRED, features, intents

QUALIFICATION = 'verified_submission_not_general_proof'
LIMITATION = 'Checked against one verified submission under its stated premises; advisory, not a general proof.'
STATS_KEY = 'choice_guidance_retrieval_stats'
STOP = set('a an the and or of to in on for from by with as at is are be been being this that these those '
           'it its if then such any each all let suppose given find determine which what where when '
           'strongest statement question option options following choose answer true false can may '
           'must only exists exist there have has about under into than not positive integer integers'.split())
LATEX = set('left right big bigl bigr biggl biggr mathbb mathcal mathrm mathbf mathit text operatorname '
            'frac dfrac tfrac sqrt begin end aligned array cases quad qquad hspace vspace '
            'cdot times le leq ge geq neq in subset subseteq forall exists to mapsto '
            'sum prod int lim infinity infty overline underline underbrace overset displaystyle'.split())


def words(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'\\([a-z]+)', lambda m: ' ' if m[1] in LATEX else ' ' + m[1] + ' ', text)
    return {w for w in re.findall(r'[^\W\d_]+', text, re.UNICODE) if len(w) > 1 and w not in STOP}


def valid_choices(task):
    choices = task.inputs.get('choices')
    if not isinstance(choices, list) or len(choices) < 2: return False
    labels = []
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get('label'), str) or not isinstance(choice.get('text'), str): return False
        label = normalize_label(choice['label'])
        if not label or not choice['text'].strip(): return False
        labels.append(label)
    return len(labels) == len(set(labels))


def normalize_label(value):
    return value.strip().upper().rstrip('.):')


def submitted_label(content, choices):
    matches = re.findall(r'<answer>(.*?)</answer>', content, re.I | re.S)
    answer = matches[-1].strip() if matches else next((s.strip() for s in reversed(content.splitlines()) if s.strip()), '')
    labels = {normalize_label(c['label']) for c in choices}
    for candidate in [answer, answer.split()[0] if answer.split() else '']:
        if normalize_label(candidate) in labels: return normalize_label(candidate)
    matched = [normalize_label(c['label']) for c in choices if c['text'].strip().casefold() == answer.casefold()]
    return matched[0] if len(matched) == 1 else None


def build_verified_public_source(task, experience, identity):
    if (experience.get('empty_answer') or experience.get('completion_truncated') or
        experience.get('answer_status') not in (None, 'valid') or not experience.get('submission') or
        (experience.get('score') or {}).get('hard') is not True):
        return {'eligible': False, 'reason': 'insufficient_reusable_evidence'}
    if not valid_choices(task): return {'eligible': False, 'reason': 'source_choice_unresolvable'}
    label = submitted_label(experience['submission'], task.inputs['choices'])
    selected = [c for c in task.inputs['choices'] if normalize_label(c['label']) == label]
    if len(selected) != 1: return {'eligible': False, 'reason': 'source_choice_unresolvable'}
    text = selected[0]['text']
    process=experience.get('public_process_evidence')
    if identity.get('method_version')==PROCEDURE_VERSION:
        if not identity.get('projection_version') or not identity.get('public_materialization_sha256'):
            return {'eligible':False,'reason':'public_projection_identity_missing'}
        if not process or process.get('verification')!='source_compatible' or not process.get('public_steps'):
            return {'eligible':False,'reason':'public_operation_evidence_missing'}
    source={'eligible': True, 'source_task_id': task.task_id, 'physical_key': task.physical_key,
            'projection_version': identity.get('projection_version'), 'submitted_label': label,
            'selected_choice_text': text, 'selected_choice_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'source_goal': task.goal, 'source_public_inputs_hash': digest(task.inputs),
            'outcome_evidence': {'hard': True, 'final_answer_hash': digest(experience['submission'])},
            'qualification': QUALIFICATION, 'source_run_id': identity.get('source_run_id'),
            'source_record_hash': identity.get('source_record_hash') or digest(experience)}
    source.update({k:identity[k] for k in ('method_version','public_materialization_sha256','choice_seed') if identity.get(k) is not None})
    if identity.get('method_version')==PROCEDURE_VERSION:
        source.update(public_process_evidence=deepcopy(process),public_features=sorted(features(task)),
            observed_operation_intents=sorted({step['kind'] for step in process['public_steps']} & intents(task)))
    return source


def proposal_schema(method_version=None):
    from .prompts import CHOICE_PROPOSAL
    schema=deepcopy(CHOICE_PROPOSAL)
    if method_version==PROCEDURE_VERSION:
        schema['properties'].update(knowledge_type={'enum':['procedural_advisory','domain_fact']},
            operation_intent={'enum':list(REQUIRED)},
            prerequisites={'type':'array','items':{'enum':sorted(set().union(*REQUIRED.values()))},'minItems':2,'maxItems':3,'uniqueItems':True},
            source_process_quote={'type':'string','minLength':1,'maxLength':512})
    return schema


def validate_proposal(proposal, source, related, schema):
    # Collect all structural field errors for a single bounded repair; never rewrite fields.
    errors = []
    try: validate_schema_instance(proposal, schema)
    except ValueError as exc: errors.append(str(exc))
    if not isinstance(proposal, dict): raise ValueError('; '.join(errors))
    for field, value in proposal.items():
        if field in schema['properties']:
            try: validate_schema_instance(value, schema['properties'][field])
            except ValueError as exc: errors.append(field + ': ' + str(exc))
    decision = proposal.get('decision')
    procedure=proposal.get('knowledge_type')=='procedural_advisory'
    body = {'goal', 'guidance', 'applicability'} | ({'operation_intent','prerequisites','source_process_quote','knowledge_type'} if procedure else {'scope_terms'})
    if source.get('method_version')==PROCEDURE_VERSION:
        body |= {'knowledge_type'}
    old_id = proposal.get('existing_skill_id')
    if old_id is not None and (not isinstance(old_id, str) or old_id not in {a['id'] for a in related}): errors.append('existing_skill_id was not offered as a checked related asset')
    asset_fields=body|{'knowledge_type','operation_intent','prerequisites','source_process_quote'}
    if decision == 'no_change' and (asset_fields & proposal.keys() or 'existing_skill_id' in proposal): errors.append('no_change must not supply asset fields')
    if decision == 'reuse_existing' and (asset_fields & proposal.keys() or old_id is None): errors.append('reuse_existing requires an offered ID and no new body')
    if decision == 'upsert_guidance':
        if not body <= proposal.keys(): errors.append('upsert_guidance requires goal, guidance, scope_terms, applicability')
        for field in body - {'scope_terms','prerequisites'}:
            if field in proposal and (not isinstance(proposal[field], str) or not proposal[field].strip()): errors.append(field + ' must be nonempty')
        terms = proposal.get('scope_terms', [])
        if isinstance(terms, list) and all(isinstance(t, str) for t in terms):
            normalized = [words(t) for t in terms]
            if any(len(t) != 1 or not t <= words(source['source_goal']) for t in normalized): errors.append('scope_terms must each locate one normalized topic word in source_goal')
            if len(set().union(*normalized)) != len(terms): errors.append('scope_terms must be distinct after normalization')
        if isinstance(proposal.get('guidance'), str) and re.search(r'\b(?i:select|choose|answer|option)\s+[A-Z]\b', proposal['guidance']): errors.append('guidance must not bind instructions to an answer label')
        if procedure:
            kind=proposal.get('operation_intent');needed=proposal.get('prerequisites',[])
            if source.get('method_version')!=PROCEDURE_VERSION or kind not in source.get('observed_operation_intents',[]):
                errors.append('procedural advisory requires a recorded public operation of the same kind')
            if not isinstance(needed,list) or not all(isinstance(x,str) for x in needed) or not isinstance(kind,str) or kind not in REQUIRED or not REQUIRED[kind]<=set(needed)<=set(source.get('public_features',[])):
                errors.append('procedure prerequisites must retain the recorded operation conditions')
            quote=proposal.get('source_process_quote')
            if not isinstance(quote,str) or not quote.strip() or quote not in source.get('public_process_evidence',{}).get('public_process',''):
                errors.append('procedure quote is absent from the actual public process')
            if 'scope_terms' in proposal:errors.append('procedure uses explicit prerequisites, not theorem topic routing')
            text='\n'.join(proposal.get(k,'') for k in ('goal','guidance','applicability') if isinstance(proposal.get(k,''),str))
            if re.search(r'\b(mathematically (proved|proven)|guarantees?|proves? that|always (true|correct))\b',text,re.I):
                errors.append('procedure cannot certify a theorem or guarantee an answer')
            if not errors and len(json.dumps(render({'id':'skill_'+digest(proposal),**proposal}),ensure_ascii=False))+2>512:
                errors.append('complete procedural advisory must fit 512 Unicode characters')
        elif any(k in proposal for k in ('operation_intent','prerequisites','source_process_quote')):
            errors.append('domain facts cannot acquire procedural routing fields')
    if errors: raise ValueError('; '.join(errors))
    return next((a for a in related if a['id'] == old_id), None)


def validate_check(check, proposal, source, schema):
    validate_schema_instance(check, schema)
    policy_version=PUBLIC_CHOICE_POLICY_VERSION if source.get('method_version')==PROCEDURE_VERSION else CHOICE_GUIDANCE_POLICY_VERSION
    if check['policy_version'] != policy_version or check['proposal_hash'] != digest(proposal):
        raise ValueError('grounding identity mismatch')
    proposal_text = '\n'.join(proposal.get(k, '') for k in ('goal', 'guidance', 'applicability'))
    source_text = source['source_goal'] + '\n' + source['selected_choice_text']
    if proposal.get('knowledge_type')=='procedural_advisory':
        source_text=source.get('public_process_evidence',{}).get('public_process','')
    for field, text in [('proposal_quote', proposal_text), ('source_quote', source_text)]:
        if not check[field].strip() or check[field] not in text: raise ValueError('grounding quote not found: ' + field)
    return check['status'] == 'supported'


def checked_asset(asset):
    if asset.get('method_version')=='choice-capability-batch.v1':
        from .capability_batch import checked_asset as checked_capability
        return checked_capability(asset)
    check, source = asset.get('grounding_check', {}), asset.get('evidence_source', {})
    proposal = asset.get('grounded_proposal')
    policy_version=PUBLIC_CHOICE_POLICY_VERSION if source.get('method_version')==PROCEDURE_VERSION else CHOICE_GUIDANCE_POLICY_VERSION
    if (asset.get('guidance_policy_version') != policy_version or
        asset.get('execution_intent') != 'guidance_only' or not proposal or
        check.get('status') != 'supported' or source.get('qualification') != QUALIFICATION or
        check.get('source_hash') != digest(source)): return False
    from .prompts import GUIDANCE_CHECK
    try:
        validate_proposal(proposal, source, [{'id': proposal.get('existing_skill_id')}], proposal_schema(source.get('method_version')))
        validate_check({k: v for k, v in check.items() if k != 'source_hash'}, proposal, source, GUIDANCE_CHECK)
        return asset.get('method_version')==source.get('method_version') and all(asset.get(k) == proposal.get(k) for k in
            ('goal', 'guidance', 'applicability', 'scope_terms','knowledge_type','operation_intent','prerequisites','source_process_quote'))
    except (ValueError, KeyError, TypeError): return False


def retrieval_stats(cases):
    sources = {}
    for row in cases:
        task = row['task']
        sources.setdefault(task['physical_key'], {'goal_hash': digest(task['goal']), 'terms': sorted(words(task['goal']))})
    df = {}
    for row in sources.values():
        for term in row['terms']: df[term] = df.get(term, 0) + 1
    return {'normalizer_version': CHOICE_NORMALIZER_VERSION, 'stats_version': 'choice-guidance.stats.v1',
            'N': len(sources), 'df': df, 'sources': sources, 'source_public_goals_hash': digest(sources)}


def render(asset):
    if asset.get('method_version')=='choice-capability-batch.v1':
        from .capability_batch import render as render_capability
        return render_capability(asset)
    if asset.get('knowledge_type')=='procedural_advisory':
        return {'skill_id':asset['id'],'guidance':asset['guidance'],'applicability':asset['applicability'],
                'operation_intent':asset['operation_intent'],'qualification':'procedural_advisory_not_proof'}
    return {'skill_id': asset['id'], 'goal': asset['goal'], 'guidance': asset['guidance'],
            'applicability': asset['applicability'], 'qualification': asset['evidence_source']['qualification'],
            'source_check_status': asset['grounding_check']['status'], 'limitation': LIMITATION}


def select(assets, stats, task, policy):
    procedural=[a for a in assets if a.get('method_version')=='choice-capability-batch.v1']
    if procedural:
        from .capability_batch import selection
        result=selection(procedural,task,policy)
        others=[a for a in assets if a not in procedural]
        if not others:return result
        baseline=select(others,stats,task,policy)
        for asset in result['selected']:
            size=len(json.dumps(render(asset),ensure_ascii=False))+2
            if len(baseline['selected'])<policy['max_items'] and baseline['injected_chars']+size<=policy['max_total_chars']:
                baseline['selected'].append(asset);baseline['injected_ids'].append(asset['id']);baseline['injected_chars']+=size
        baseline['audit']+=result['audit']
        return baseline
    qualified = [a for a in assets if checked_asset(a)]
    if any(a.get('guidance_policy_version') in {CHOICE_GUIDANCE_POLICY_VERSION,PUBLIC_CHOICE_POLICY_VERSION} for a in assets) and (
                      not stats or stats.get('normalizer_version') != CHOICE_NORMALIZER_VERSION or
                      stats.get('source_public_goals_hash') != digest(stats.get('sources', {}))):
        raise ValueError('Nonempty LM1 Bank lacks a valid frozen retrieval statistics identity')
    stats = stats or retrieval_stats([])
    superseded = {a['parent_skill_id'] for a in qualified if a.get('parent_skill_id')}
    query, audit, ranked = words(task.goal), [], []
    public_intents, public_features=intents(task),features(task)
    for asset in assets:
        terms = set().union(*(words(t) for t in asset.get('scope_terms', [])))
        matched = terms & query
        weight = lambda t: 1 + math.log((1 + stats['N']) / (1 + stats['df'].get(t, 0)))
        coverage = sum(weight(t) for t in matched) / sum(weight(t) for t in terms) if terms else 0
        reason = ('unchecked_asset' if asset not in qualified else 'superseded' if asset['id'] in superseded
                  else 'scope_mismatch' if len(matched) < 2 or coverage < .5 else 'eligible')
        if asset.get('knowledge_type')=='procedural_advisory' and asset in qualified and asset['id'] not in superseded:
            reason=('procedural_route_disabled' if not policy.get('public_process') else
                'operation_intent_mismatch' if asset['operation_intent'] not in public_intents else
                'prerequisite_mismatch' if not set(asset['prerequisites'])<=public_features else 'eligible')
        row = {'asset_id': asset['id'], 'matched_scope_terms': sorted(matched), 'coverage': coverage, 'reason': reason}
        row['route']='procedural_advisory' if asset.get('knowledge_type')=='procedural_advisory' else 'domain_fact'
        if row['route']=='procedural_advisory':row.update(public_operation_intents=sorted(public_intents),public_features=sorted(public_features))
        audit.append(row)
        if reason == 'eligible': ranked.append((asset, row))
    selected, chars = [], 0
    for asset, row in sorted(ranked, key=lambda pair: (-pair[1]['coverage'], pair[0]['id'])):
        size = len(json.dumps(render(asset), ensure_ascii=False))
        procedural=asset.get('knowledge_type')=='procedural_advisory'
        if procedural and (size+2>512 or any(a.get('knowledge_type')=='procedural_advisory' for a in selected)):row['reason']='procedural_bound'
        elif len(selected) >= policy['max_items']: row['reason'] = 'item_limit'
        elif chars + size + 2 > policy['max_total_chars']: row['reason'] = 'character_limit'
        else:
            selected.append(deepcopy(asset)); chars += size + 2; row['reason'] = 'selected'
    return {'selected': selected, 'audit': audit, 'injected_ids': [a['id'] for a in selected],
            'injected_chars': chars, 'selection_version':'choice-guidance.procedure-scope.v2' if policy.get('public_process') else CHOICE_GUIDANCE_SELECTION_VERSION,
            'statistics_hash': digest(stats)}
