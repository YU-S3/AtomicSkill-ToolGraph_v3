"""Small explicit-period statistics; validates scope, not the truth of operands."""
from decimal import Decimal, localcontext, ROUND_HALF_UP


def run(ctx, inputs):
    try:
        wanted, records = inputs['requested_periods'], inputs['records']
        if not wanted or len(wanted) != len(set(wanted)):
            raise ValueError('requested_periods must be nonempty and unique')
        by_period = {}
        for row in records:
            period = row['period']
            if period in by_period: raise ValueError('duplicate observation: '+period)
            by_period[period] = row
        missing = [p for p in wanted if p not in by_period]
        if missing: raise ValueError('missing requested periods: '+', '.join(missing))
        rows = [by_period[p] for p in wanted]
        op, digits = inputs['statistic'], inputs['decimals']
        if type(digits) is not int or not 0 <= digits <= 18: raise ValueError('invalid decimals')
        def number(value):
            if type(value) not in (str, int, float): raise ValueError('numeric operands required')
            d = Decimal(str(value))
            if not d.is_finite(): raise ValueError('nonfinite operand')
            return d
        with localcontext() as context:
            context.prec = 50
            y = [number(r['value']) for r in rows]
            n = Decimal(len(y)); mean = sum(y)/n
            formula = op
            if op == 'mean': value = mean
            elif op == 'geometric_mean':
                if min(y) <= 0: raise ValueError('geometric mean requires positive values')
                value = (sum(v.ln() for v in y)/n).exp()
            elif op in ('variance_sample', 'variance_population', 'std_sample', 'std_population'):
                divisor = n - int(op.endswith('sample'))
                if divisor <= 0: raise ValueError('sample statistic needs at least two observations')
                value = sum((v-mean)**2 for v in y)/divisor
                if op.startswith('std'): value = value.sqrt()
            elif op in ('correlation', 'ols_slope', 'ols_intercept', 'forecast_minus_actual', 'actual_minus_forecast'):
                x = [number(r['x']) for r in rows]
                mx = sum(x)/n
                sxx = sum((v-mx)**2 for v in x)
                if not sxx: raise ValueError('predictor has zero variance')
                sxy = sum((a-mx)*(b-mean) for a,b in zip(x,y))
                slope = sxy/sxx; intercept = mean-slope*mx
                if op == 'correlation':
                    syy = sum((v-mean)**2 for v in y)
                    if not syy: raise ValueError('response has zero variance')
                    value = sxy/(sxx*syy).sqrt()
                elif op == 'ols_slope': value = slope
                elif op == 'ols_intercept': value = intercept
                else:
                    fx = number(inputs['forecast_x'])
                    if fx in x: raise ValueError('forecast predictor must be outside fitting observations')
                    forecast = intercept+slope*fx
                    actual = number(inputs['actual'])
                    value = forecast-actual if op == 'forecast_minus_actual' else actual-forecast
            else: raise ValueError('unsupported statistic; use native Python for other definitions')
            quantized = value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
            if quantized == 0: quantized = abs(quantized)
            return {'status':'ok', 'outputs':{
                'value':str(value), 'answer':format(quantized, '.'+str(digits)+'f'), 'formula':formula,
                'selected_periods':wanted, 'excluded_periods':[p for p in by_period if p not in wanted],
                'count':len(rows), 'unit':inputs['unit'], 'proof_scope':'local_computation_only'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as error:
        return {'status':'needs_input', 'outputs':{}, 'detail':str(error)}
