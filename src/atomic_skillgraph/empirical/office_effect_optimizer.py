"""Matched task utility, with negative edits retained only in Learner memory."""
from .contracts import digest


def evaluate_pairs(pairs, max_losses=1):
    if not pairs: raise ValueError('Empty comparison cannot qualify a policy')
    wins = losses = ties = empty_off = empty_on = 0
    cost = {arm: {'tokens': 0, 'http': 0, 'tool_calls': 0} for arm in ('off', 'on')}
    scores = {arm: {'hard':0,'soft':0} for arm in ('off','on')}
    seen = set()
    for row in pairs:
        if row['physical_key'] in seen: raise ValueError('Duplicate paired task')
        seen.add(row['physical_key'])
        off, on = row['off'], row['on']
        if (not row.get('same_protocol') or not row.get('bank_unchanged') or
                any(not x.get('complete') or x.get('unknown_billing') for x in (off, on))):
            raise ValueError('Unfinished/unmatched/unknown-billing pair cannot be selected')
        a, b = bool(off['score']['hard']), bool(on['score']['hard'])
        wins += not a and b; losses += a and not b; ties += a == b
        empty_off += off['empty']; empty_on += on['empty']
        for arm in cost:
            for metric in cost[arm]: cost[arm][metric] += row[arm][metric]
            scores[arm]['hard'] += bool(row[arm]['score']['hard'])
            scores[arm]['soft'] += row[arm]['score'].get('soft',float(row[arm]['score']['hard']))
    accept = wins > losses and losses <= max_losses and empty_on <= empty_off
    return {'schema': 'office.effect-utility.v1', 'pairs': len(pairs), 'wins': wins, 'losses': losses,
        'ties': ties, 'empty_off': empty_off, 'empty_on': empty_on, 'cost': cost, 'accepted': accept,
        'scores':scores, 'max_losses': max_losses, 'reason': 'positive_net_utility' if accept else 'retain_parent_no_positive_safe_gain',
        'pairs_sha256': digest(pairs)}


def choose_version(parent, candidate, receipt):
    return candidate if receipt['accepted'] else parent
