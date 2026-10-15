"""Small read-only policy view for the real Planner → Executor tool loop."""
from copy import deepcopy
import re

from .bank_view import BankView
from .contracts import digest
from .office_effect_contract import VERSION, RUNTIME, guidance, verify_candidate
from .office_operation_mining import read
from .office_runtime import evidence_view


def applicable(candidate, goal):
    kind = candidate['applicability']
    if kind == 'all': return True
    words = goal.lower()
    if kind == 'time_series_or_comparison':
        return bool(re.search(r'\b(between|each|change|growth|compare|comparison|difference|average|trend|years|months)\b', words))
    if kind == 'calculation':
        return bool(re.search(r'\b(calculate|sum|total|mean|average|ratio|percent|difference|deviation|growth|change)\b', words))
    raise ValueError('Unregistered public applicability condition')


class OfficeEffectView(BankView):
    force_planner = True  # Both arms receive the same planning opportunity, even with no Skill.

    def __init__(self, bank, manifest):
        super().__init__(bank, 'learned_assets_off')
        value = read(manifest) if not isinstance(manifest, dict) else deepcopy(manifest)
        if value['method_version'] != VERSION or value['bank_digest'] != bank.digest():
            raise ValueError('V6 Bank/view identity mismatch')
        if any(bank.all(k) for k in ('skill','program','implementation','workflow')):
            raise ValueError('V6 initial effect experiment requires an empty independent base Bank')
        self.candidate = value.get('candidate') if value['enabled'] else None
        if value.get('candidate'): verify_candidate(value['candidate'])
        self.view_hash = digest(value)

    def all(self, kind):
        if kind != 'skill' or not self.candidate: return []
        c = self.candidate
        return [{'id':c['id'], 'goal':c['goal'], 'guidance':guidance(c),
                 'applicability':c['applicability'], 'qualification':'source_checked_development_intervention'}]

    def retrieve(self, query, limit=8):
        return self.all('skill') if self.candidate and applicable(self.candidate, query) else []

    def planning_cards(self, query): return self.retrieve(query)

    def effect_guidance(self, task): return self.retrieve(task.goal)

    def program_options(self, *args, **kwargs): return []

    def routes(self, *args, **kwargs): return []


def runtime_evidence(context, goal, corpus_identity=None):
    value = evidence_view(context, goal)
    value['version'] = RUNTIME
    value['progress'] = {'search_observed':bool(value['searches']), 'text_acquired':bool(value['acquired_pages']),
        'calculation_observed':bool(value['computed']), 'complete_table':'unverified',
        'all_periods':'unverified', 'operands_and_units':'unverified', 'answer_correctness':'unverified'}
    # Planned metadata, model notes and a successful Python status cannot certify operands.
    seen, duplicates = set(), []
    for row in context.memory.values():
        if row['name'] not in {'grep','grep_page'}: continue
        raw = context.results[row['result_id']]
        if not raw.get('accepted'): continue
        args = row['arguments']; data = raw.get('data')
        key = digest([args.get('pattern'),args.get('paths'),args.get('cursor'),corpus_identity,data])
        if key in seen: duplicates.append(key)
        seen.add(key)
    warned = context.effect_notice_versions
    fresh = [k for k in set(duplicates) if k not in warned or warned[k] == context.effect_owner_version]
    if fresh:
        value['loop_feedback'] = 'Identical search returned no new evidence. Check scope or continuation cursor, or change the query; no absence claim is justified.'
    for key in fresh: warned[key] = context.effect_owner_version
    return value


def validate_runtime(config):
    e = config['experiment']
    if (e.get('office_effect_version') != VERSION or e.get('benchmark') != 'officeqa'
            or e.get('runtime_mode') != 'frozen' or not e.get('office_effect_manifest')
            or e.get('manual_reference_version') or config['runtime']['global_action_budget'] != 48
            or config['budget']['eval_task_tokens'] != 192000):
        raise ValueError('V6 requires isolated readonly Office tool-loop with the frozen paired budget')
