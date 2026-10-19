"""Deterministic completeness check for an *explicitly specified* observation series.

This Program does not discover observations and cannot assert that the user's
requested period list matches the source question. Evidence references are
preserved for downstream use, not independently certified here.
"""
from decimal import Decimal, InvalidOperation


def run(ctx, inputs):
    try:
        required = inputs['required_periods']
        observations = inputs['observations']
        if not isinstance(required, list) or not required or len(required) > 240:
            raise ValueError('required_periods must contain 1..240 explicit period strings')
        if any(not isinstance(p, str) or not p.strip() for p in required) or len(set(required)) != len(required):
            raise ValueError('required_periods must be distinct and nonblank')
        if not isinstance(observations, list) or len(observations) > 500:
            raise ValueError('observations must be a bounded list')
        groups = {}
        for obs in observations:
            if not isinstance(obs, dict) or not {'period','value','unit','source_ref'} <= obs.keys():
                raise ValueError('each observation needs period, value, unit and source_ref')
            p, unit = obs['period'], obs['unit']
            if p not in required or not isinstance(unit, str) or not unit.strip():
                raise ValueError('observation period is not requested or unit is missing')
            v = Decimal(str(obs['value']).replace(',',''))
            if not v.is_finite():
                raise ValueError('observation is nonfinite')
            ref = obs['source_ref']
            if not isinstance(ref, dict) or not isinstance(ref.get('result_id'), str) or not isinstance(ref.get('path'), list):
                raise ValueError('source_ref must reference a real previously obtained result')
            groups.setdefault(p, []).append({'period': p, 'value': str(v), 'unit': unit, 'source_ref': ref})
        aligned, missing, conflicts = [], [], []
        for p in required:
            rows = groups.get(p, [])
            if not rows:
                missing.append(p);continue
            signatures = {(r['value'], r['unit']) for r in rows}
            if len(signatures) != 1:
                conflicts.append(p);continue
            aligned.append(rows[0])
        units = {r['unit'] for r in aligned}
        if len(units) > 1:
            conflicts.extend(p for p in required if p in groups and p not in conflicts)
        conflicts = list(dict.fromkeys(conflicts))
        complete = not missing and not conflicts and len(aligned) == len(required)
        return {'status':'ok','outputs':{
            'complete':complete,'required_periods':required,'observed_periods':[p for p in required if p in groups],
            'missing_periods':missing,'conflicting_periods':conflicts,
            'aligned_observations':aligned if not conflicts else [],
            'matched_count':len(aligned),'expected_count':len(required),
            'proof_scope':'explicit_period_bookkeeping_only',
            'note':'No source text was read here. Period semantics, quoted values, and referenced files must be checked in the actual task.'}}
    except (KeyError,ValueError,TypeError,InvalidOperation) as exc:
        return {'status':'needs_input','outputs':{},'detail':str(exc)}
