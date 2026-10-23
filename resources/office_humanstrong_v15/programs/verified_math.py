"""Small explicit calculations. Input selection remains the caller's responsibility."""
from decimal import Decimal, ROUND_HALF_UP, localcontext
import math
import statistics


def display(value, places, fixed=False):
    if type(places) is not int or not 0 <= places <= 12: raise ValueError('decimals must be 0..12')
    with localcontext() as precision:
        precision.prec = 50
        number = Decimal(str(value))
        if not number.is_finite(): raise ValueError('Finite result required')
        number = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
        if not number: number = abs(number)
        answer = format(number, 'f')
    return answer if fixed or '.' not in answer else answer.rstrip('0').rstrip('.')


def numbers(values):
    if not isinstance(values, list) or not values or len(values) > 240: raise ValueError('Nonempty bounded numeric series required')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values): raise ValueError('Finite numeric operands required')
    return values


def calculate(inputs):
    op = inputs['operation']; a = numbers(inputs['values'])
    details = {'operation': op, 'values': a, 'unit': inputs['unit']}
    if not isinstance(inputs['unit'], str) or not inputs['unit'].strip(): raise ValueError('Source unit required')
    if op in ('midpoint_submit', 'growth_submit'):
        if len(a) != 2 or a[0] + a[1] == 0: raise ValueError('Two ordered endpoints with nonzero midpoint required')
        midpoint = (a[1] - a[0]) / ((a[0] + a[1]) / 2)
        if op == 'midpoint_submit':
            value = abs(midpoint) if inputs['absolute'] else midpoint
            if inputs['output_percent']: value *= 100
            details['formula'] = '(end-start)/((start+end)/2)'
        else:
            if min(a) <= 0 or inputs['intervals'] <= 0: raise ValueError('Positive endpoints and elapsed intervals required')
            factor = math.exp(math.log(a[1] / a[0]) / inputs['intervals'])
            value = [factor - 1, factor, midpoint]
            details['formula'] = '[exp(log(end/start)/intervals)-1, exp(log(end/start)/intervals), (end-start)/midpoint]'
    elif op == 'paired_statistics':
        b = numbers(inputs['other_values']); periods = inputs['periods']; expected = inputs['expected_periods']
        if (len(a) != len(b) or len(a) != len(periods) or periods != expected or len(set(periods)) != len(periods)):
            raise ValueError('Both series must match the same complete, unique requested period list')
        details.update(other_values=b, periods=periods, statistic=inputs['statistic'])
        if inputs['statistic'] == 'share_and_correlation':
            da = [x - statistics.fmean(a) for x in a]; db = [x - statistics.fmean(b) for x in b]
            denominator = math.sqrt(sum(x*x for x in da) * sum(x*x for x in db))
            if not denominator or not sum(b): raise ValueError('Nonzero total and variation required')
            value = [100 * sum(a)/sum(b), sum(x*y for x,y in zip(da,db))/denominator]
            details['formula'] = '[100*sum(part)/sum(total), Pearson(part,total)] on identical periods'
        elif inputs['statistic'] == 'absolute_log_growth_cv_difference':
            def cv(series):
                if len(series) < 3 or min(series) <= 0: raise ValueError('At least three positive levels required')
                growth = [math.log(y/x) for x,y in zip(series,series[1:])]
                mean = statistics.fmean(growth)
                if mean == 0: raise ValueError('Zero mean log growth')
                return statistics.pstdev(growth)/mean
            value = abs(cv(a)-cv(b)); details['formula'] = 'abs(population_std(log_growth_a)/mean_a - population_std(log_growth_b)/mean_b)'
        else: raise ValueError('Unknown paired statistic')
    elif op == 'baseline_z':
        if len(a) < 2: raise ValueError('At least two baseline observations required')
        spread = statistics.stdev(a) if inputs['spread'] == 'sample' else statistics.pstdev(a)
        if not spread: raise ValueError('Nonzero baseline spread required')
        target = numbers([inputs['target']])[0]; z = (target-statistics.fmean(a))/spread
        threshold = numbers([inputs['threshold']])[0]
        if threshold <= 0: raise ValueError('Positive threshold required')
        tested = abs(z) if inputs['two_sided'] else z
        value = [z, 'unusual' if tested > threshold else 'normal']
        details.update(formula='(target-baseline_mean)/baseline_sd', spread=inputs['spread'], target=target, two_sided=inputs['two_sided'], threshold=threshold)
    else: raise ValueError('Unknown operation')
    values = value if isinstance(value, list) else [value]
    places = inputs['decimals']
    if len(places) != sum(type(x) in (int,float) for x in values): raise ValueError('One decimals entry per numeric output field required')
    numeric = iter(places)
    fields = [display(x,next(numeric),inputs.get('fixed_decimals',False)) if type(x) in (int,float) else x for x in values]
    answer = '['+', '.join(fields)+']' if isinstance(value,list) else fields[0]
    if inputs.get('percent_suffix'):
        if op != 'midpoint_submit' or not inputs['output_percent']: raise ValueError('Percent suffix requires percent-scaled scalar output')
        answer += '%'
    return {'answer':answer,'value':value,'calculation':details,
        'evidence':inputs.get('operand_refs',[]),
        'proof_scope':'Validates the declared arithmetic, scale and shared period mask; does not verify source-column semantics.'}


def run(ctx, inputs):
    try:
        if inputs['submit_answer'] is not True: raise ValueError('Explicit final submission required')
        return {'status':'ok','outputs':calculate(inputs)}
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        return {'status':'needs_input','outputs':{},'detail':str(error)}
