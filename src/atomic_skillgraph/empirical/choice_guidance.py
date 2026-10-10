"""Host-bound public choice evidence and frozen, Train-only applicability statistics."""
import hashlib
import json
import math
import re
import unicodedata
from copy import deepcopy

from . import (CHOICE_GUIDANCE_POLICY_VERSION, CHOICE_GUIDANCE_SELECTION_VERSION,
               CHOICE_NORMALIZER_VERSION)
from .contracts import digest, validate_schema_instance

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
    return {'eligible': True, 'source_task_id': task.task_id, 'physical_key': task.physical_key,
            'projection_version': identity.get('projection_version'), 'submitted_label': label,
            'selected_choice_text': text, 'selected_choice_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'source_goal': task.goal, 'source_public_inputs_hash': digest(task.inputs),
            'outcome_evidence': {'hard': True, 'final_answer_hash': digest(experience['submission'])},
            'qualification': QUALIFICATION, 'source_run_id': identity.get('source_run_id'),
            'source_record_hash': identity.get('source_record_hash') or digest(experience)}


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
    body = {'goal', 'guidance', 'scope_terms', 'applicability'}
    old_id = proposal.get('existing_skill_id')
    if old_id is not None and old_id not in {a['id'] for a in related}: errors.append('existing_skill_id was not offered as a checked related asset')
    if decision == 'no_change' and (body & proposal.keys() or 'existing_skill_id' in proposal): errors.append('no_change must not supply asset fields')
    if decision == 'reuse_existing' and (body & proposal.keys() or old_id is None): errors.append('reuse_existing requires an offered ID and no new body')
    if decision == 'upsert_guidance':
        if not body <= proposal.keys(): errors.append('upsert_guidance requires goal, guidance, scope_terms, applicability')
        for field in body - {'scope_terms'}:
            if field in proposal and (not isinstance(proposal[field], str) or not proposal[field].strip()): errors.append(field + ' must be nonempty')
        terms = proposal.get('scope_terms', [])
        if isinstance(terms, list) and all(isinstance(t, str) for t in terms):
            normalized = [words(t) for t in terms]
            if any(len(t) != 1 or not t <= words(source['source_goal']) for t in normalized): errors.append('scope_terms must each locate one normalized topic word in source_goal')
            if len(set().union(*normalized)) != len(terms): errors.append('scope_terms must be distinct after normalization')
        if re.search(r'\b(?i:select|choose|answer|option)\s+[A-Z]\b', proposal.get('guidance', '')): errors.append('guidance must not bind instructions to an answer label')
    if errors: raise ValueError('; '.join(errors))
    return next((a for a in related if a['id'] == old_id), None)


def validate_check(check, proposal, source, schema):
    validate_schema_instance(check, schema)
    if check['policy_version'] != CHOICE_GUIDANCE_POLICY_VERSION or check['proposal_hash'] != digest(proposal):
        raise ValueError('grounding identity mismatch')
    proposal_text = '\n'.join(proposal.get(k, '') for k in ('goal', 'guidance', 'applicability'))
    source_text = source['source_goal'] + '\n' + source['selected_choice_text']
    for field, text in [('proposal_quote', proposal_text), ('source_quote', source_text)]:
        if not check[field].strip() or check[field] not in text: raise ValueError('grounding quote not found: ' + field)
    return check['status'] == 'supported'


def checked_asset(asset):
    check, source = asset.get('grounding_check', {}), asset.get('evidence_source', {})
    proposal = asset.get('grounded_proposal')
    if (asset.get('guidance_policy_version') != CHOICE_GUIDANCE_POLICY_VERSION or
        asset.get('execution_intent') != 'guidance_only' or not proposal or
        check.get('status') != 'supported' or source.get('qualification') != QUALIFICATION or
        check.get('source_hash') != digest(source)): return False
    from .prompts import CHOICE_PROPOSAL, GUIDANCE_CHECK
    try:
        validate_proposal(proposal, source, [{'id': proposal.get('existing_skill_id')}], CHOICE_PROPOSAL)
        validate_check({k: v for k, v in check.items() if k != 'source_hash'}, proposal, source, GUIDANCE_CHECK)
        return all(asset.get(k) == proposal.get(k) for k in ('goal', 'guidance', 'applicability', 'scope_terms'))
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
    return {'skill_id': asset['id'], 'goal': asset['goal'], 'guidance': asset['guidance'],
            'applicability': asset['applicability'], 'qualification': asset['evidence_source']['qualification'],
            'source_check_status': asset['grounding_check']['status'], 'limitation': LIMITATION}


def select(assets, stats, task, policy):
    qualified = [a for a in assets if checked_asset(a)]
    if any(a.get('guidance_policy_version') == CHOICE_GUIDANCE_POLICY_VERSION for a in assets) and (
                      not stats or stats.get('normalizer_version') != CHOICE_NORMALIZER_VERSION or
                      stats.get('source_public_goals_hash') != digest(stats.get('sources', {}))):
        raise ValueError('Nonempty LM1 Bank lacks a valid frozen retrieval statistics identity')
    stats = stats or retrieval_stats([])
    superseded = {a['parent_skill_id'] for a in qualified if a.get('parent_skill_id')}
    query, audit, ranked = words(task.goal), [], []
    for asset in assets:
        terms = set().union(*(words(t) for t in asset.get('scope_terms', [])))
        matched = terms & query
        weight = lambda t: 1 + math.log((1 + stats['N']) / (1 + stats['df'].get(t, 0)))
        coverage = sum(weight(t) for t in matched) / sum(weight(t) for t in terms) if terms else 0
        reason = ('unchecked_asset' if asset not in qualified else 'superseded' if asset['id'] in superseded
                  else 'scope_mismatch' if len(matched) < 2 or coverage < .5 else 'eligible')
        row = {'asset_id': asset['id'], 'matched_scope_terms': sorted(matched), 'coverage': coverage, 'reason': reason}
        audit.append(row)
        if reason == 'eligible': ranked.append((asset, row))
    selected, chars = [], 0
    for asset, row in sorted(ranked, key=lambda pair: (-pair[1]['coverage'], pair[0]['id'])):
        size = len(json.dumps(render(asset), ensure_ascii=False))
        if len(selected) >= policy['max_items']: row['reason'] = 'item_limit'
        elif chars + size + 2 > policy['max_total_chars']: row['reason'] = 'character_limit'
        else:
            selected.append(deepcopy(asset)); chars += size + 2; row['reason'] = 'selected'
    return {'selected': selected, 'audit': audit, 'injected_ids': [a['id'] for a in selected],
            'injected_chars': chars, 'selection_version': CHOICE_GUIDANCE_SELECTION_VERSION,
            'statistics_hash': digest(stats)}
