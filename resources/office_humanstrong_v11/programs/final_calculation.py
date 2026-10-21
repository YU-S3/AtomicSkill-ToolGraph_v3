"""Read selected public cells or acquired scalar results without retyping values."""
import re
from decimal import Decimal


def read_operand(ctx, selector, cache, evidence):
    ref = selector['source_ref']
    key = (ref['result_id'], tuple(ref['path']), selector.get('offset', 0))
    if key not in cache:
        result = ctx.call('read_result', {**ref, 'offset': key[2], 'limit': 12000})
        if not result.get('accepted'):
            raise ValueError('Source read refused: '+str(result.get('error')))
        cache[key] = result
    result = cache[key]
    value = result['data']
    receipt = {'source_ref': ref}
    if 'row' in selector:
        if not isinstance(value, str): raise ValueError('Cell selector needs acquired source text')
        lines = value.splitlines(keepends=True)
        header = selector['header'].strip()
        anchors = [i for i, line in enumerate(lines) if line.strip() == header]
        if len(anchors) != 1: raise ValueError('Use one exact, unique table header line in the source window')
        start = anchors[0]
        matches = []
        for i in range(start+1, len(lines)):
            line = lines[i]
            if not line.lstrip().startswith('|'): break
            if i == len(lines)-1 and result.get('next_offset') is not None and not line.endswith('\n'):
                continue  # Never parse a truncated numeric row.
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            row_column=selector.get('row_column',0)
            if row_column < len(cells) and cells[row_column] == selector['row'].strip() and (
                    'row_text' not in selector or line.strip()==selector['row_text'].strip()):
                matches.append((i, cells))
        if len(matches) != 1:
            raise ValueError('Requested row is absent/repeated; supply the exact full row_text and row_column for its date group')
        index, cells = matches[0]
        column = selector['column']
        if type(column) is not int or not 1 <= column < len(cells):
            raise ValueError('column counts all cells from 0, including blank cells; cell 0 is the row label')
        raw = cells[column]
        token = re.fullmatch(r'\s*(\(?[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)\)?)(%)?(?:\s*[rp])?(?:\s+\d+/)?\s*', raw)
        if not token: raise ValueError('Selected cell is not a plain numeric observation: '+raw)
        value = token[1].replace(',', '')
        if token[2]: receipt['source_scale']='percent_number; no automatic division by 100'
        if value.startswith('(') and value.endswith(')'): value = '-'+value[1:-1]
        receipt.update(header=header, row=selector['row'], column=column, cell=raw,
            quote=lines[index].rstrip('\n'), start=key[2]+sum(map(len, lines[:index])))
    elif result.get('next_offset') is not None:
        raise ValueError('Numeric result reference must be complete')
    if type(value) not in (str, int, float): raise ValueError('Select one numeric result leaf, not a container')
    number = Decimal(str(value))
    if not number.is_finite(): raise ValueError('Nonfinite acquired operand')
    receipt['value'] = str(number)
    evidence.append(receipt)
    return str(number)


"""Named formulas with explicit conventions; computation, not operand provenance."""
import math
from decimal import Decimal, localcontext, ROUND_HALF_UP


def calculate_growth(ctx,inputs):
    try:
        op=inputs['operation'];direction='as_supplied';formula=op
        named={'difference','ratio','percent_change','log_return','cagr','annual_log_growth'}
        if 'output_scale' in inputs:raise ValueError('output_scale is unsupported; specify output_percent explicitly')
        if op in named:
            if 'values' in inputs:raise ValueError('This operation requires named operands, not positional values')
            if op=='difference':
                raw=[inputs['from_value'],inputs['to_value']]
                if type(inputs['absolute']) is not bool:raise ValueError('absolute must be explicit boolean')
                formula='abs(to_value-from_value)' if inputs['absolute'] else 'to_value-from_value'
                direction='absolute' if inputs['absolute'] else 'to_minus_from'
            elif op=='ratio':
                raw=[inputs['denominator'],inputs['numerator']];formula='numerator/denominator';direction='numerator_over_denominator'
            else:
                raw=[inputs['base_value'],inputs['new_value']];direction='base_to_new'
                formula={'percent_change':'(new_value-base_value)/base_value','log_return':'ln(new_value/base_value)',
                         'cagr':'exp(ln(new_value/base_value)/periods)-1',
                         'annual_log_growth':'ln(new_value/base_value)/periods'}[op]
        else:raw=inputs['values']
        if not isinstance(raw,list) or not raw or len(raw)>5000:raise ValueError('1..5000 values required')
        if any(type(v) not in (str,int,float) for v in raw):raise ValueError('numeric strings/numbers required')
        with localcontext() as context:
            context.prec=50
            x=[Decimal(str(v)) for v in raw]
            if any(not v.is_finite() for v in x):raise ValueError('nonfinite operand')
            n=Decimal(len(x));mean=sum(x)/n;convention=op
            if op=='sum':v=sum(x)
            elif op=='mean':v=mean
            elif op=='symmetric_difference':
                if len(x)!=2:raise ValueError('values must be [first,second]')
                v=abs(x[1]-x[0])/((abs(x[0])+abs(x[1]))/2)
                formula='abs(second-first)/((abs(first)+abs(second))/2)';direction='absolute_symmetric'
            elif op=='share_difference':
                if len(x)!=4:raise ValueError('values must be [first_part,first_total,second_part,second_total]')
                if type(inputs.get('absolute')) is not bool:raise ValueError('absolute must be explicit boolean')
                v=x[2]/x[3]-x[0]/x[1]
                formula='second_part/second_total-first_part/first_total';direction='second_share_minus_first_share'
                if inputs['absolute']:v=abs(v);formula='abs('+formula+')';direction='absolute_share_difference'
            elif op=='median':
                a=sorted(x);mid=len(a)//2;v=a[mid] if len(a)%2 else (a[mid-1]+a[mid])/2
            elif op in {'difference','ratio','percent_change','log_return','cagr','annual_log_growth'}:
                if len(x)!=2:raise ValueError('exactly two values in [start,end] order required')
                a,b=x
                if op=='difference':v=abs(b-a) if inputs['absolute'] else b-a
                elif op=='ratio':v=b/a
                elif op=='percent_change':v=(b-a)/a
                else:
                    if a<=0 or b<=0:raise ValueError('growth operands must be positive')
                    ratio=b/a
                    if op=='log_return':v=ratio.ln()
                    else:
                        periods=Decimal(str(inputs.get('periods','0')))
                        if periods<=0:raise ValueError('positive elapsed interval count required, not observation count')
                        v=ratio.ln()/periods if op=='annual_log_growth' else (ratio.ln()/periods).exp()-1
            elif op in {'std_population','std_sample','variance_population','variance_sample'}:
                denominator=n-(1 if op.endswith('sample') else 0)
                if denominator<=0:raise ValueError('sample requires at least two observations')
                v=sum((a-mean)**2 for a in x)/denominator
                if op.startswith('std_'):v=v.sqrt()
            elif op in {'gini_population','gini_sample_adjusted'}:
                if min(x)<0 or sum(x)<=0:raise ValueError('Gini requires nonnegative values and positive sum')
                a=sorted(x);v=sum(Decimal(2*i-len(a)-1)*z for i,z in enumerate(a,1))/(n*sum(a))
                if op=='gini_sample_adjusted':
                    if n<=1:raise ValueError('sample correction requires n>1')
                    v=v*n/(n-1)
            elif op in {'correlation','ols_slope','ols_intercept'}:
                raw_y=inputs.get('other_values',[])
                if len(raw_y)!=len(x) or len(x)<2:raise ValueError('paired x/y arrays of equal length required')
                y=[Decimal(str(z)) for z in raw_y]
                if any(not z.is_finite() for z in y):raise ValueError('nonfinite y')
                my=sum(y)/n;sxx=sum((a-mean)**2 for a in x);sxy=sum((a-mean)*(b-my) for a,b in zip(x,y))
                if sxx==0:raise ValueError('zero x variance')
                slope=sxy/sxx
                if op=='ols_slope':v=slope
                elif op=='ols_intercept':v=my-slope*mean
                else:
                    syy=sum((b-my)**2 for b in y)
                    if syy==0:raise ValueError('zero y variance')
                    v=sxy/(sxx*syy).sqrt()
            elif op=='geometric_growth':
                if any(1+a<=0 for a in x):raise ValueError('growth factors must be positive')
                v=(sum((1+a).ln() for a in x)/n).exp()-1
            else:raise ValueError('unsupported named operation')
            percent=inputs.get('output_percent',False)
            if type(percent) is not bool:raise ValueError('output_percent must be boolean')
            if percent:
                if op not in {'ratio','percent_change','cagr','geometric_growth','symmetric_difference','share_difference'}:raise ValueError('percent output is not supported for this operation')
                v*=100;formula='100*('+formula+')'
            digits=inputs.get('decimals')
            if digits is not None:
                if type(digits) is not int or not 0<=digits<=18:raise ValueError('decimals must be 0..18')
                answer=format(v.quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP),'.'+str(digits)+'f')
            else:answer=format(v,'f').rstrip('0').rstrip('.') if '.' in format(v,'f') else format(v,'f')
            if answer in ('-0',''):answer='0'
            return {'status':'ok','outputs':{'value':str(v),'answer':answer,'formula':formula,'direction':direction,
                'operand_refs':inputs.get('operand_refs',[]),'periods':str(inputs.get('periods','not_applicable')),
                'unit':inputs['unit'],'output_percent':percent,'proof_scope':'local_computation_only',
                'convention':convention,'count':len(x)}}
    except (KeyError,ValueError,TypeError,ArithmeticError) as e:return {'status':'needs_input','outputs':{},'detail':str(e)}


"""Small explicit-period statistics; validates scope, not the truth of operands."""
from decimal import Decimal, localcontext, ROUND_HALF_UP


def calculate_statistics(ctx, inputs):
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


"""Polynomial OLS over explicit public observations; no implicit imputation.

Inputs are public, explicitly aligned observations. This computes a polynomial
OLS forecast with an intercept; it cannot certify source/column semantics.
"""
from decimal import Decimal, localcontext, ROUND_HALF_UP


def calculate_polynomial(ctx, inputs):
    try:
        degree = inputs['degree']
        if type(degree) is not int or not 1 <= degree <= 5:
            raise ValueError('degree must be 1..5')
        rows = inputs['observations']
        expected = inputs['required_periods']
        if not degree < len(expected) <= 240 or len(set(expected)) != len(expected):
            raise ValueError('distinct required periods and at least degree+1 observations required')
        if len(rows) != len(expected) or [r['period'] for r in rows] != expected:
            raise ValueError('observations must cover every required period once, in specified order')
        if any(r['unit'] != inputs['unit'] or not r.get('source_ref') for r in rows):
            raise ValueError('consistent units and source references required')
        operation = inputs['operation']
        if operation not in ('predict', 'absolute_error'):
            raise ValueError('choose predict or absolute_error')
        actual = inputs.get('actual')
        if operation == 'absolute_error' and actual is None:
            raise ValueError('absolute_error requires an independently acquired actual value')
        if actual is not None and (actual['unit'] != inputs['unit'] or not actual.get('source_ref')):
            raise ValueError('actual value needs matching units and a source reference')
        digits = inputs['decimals']
        if type(digits) is not int or not 0 <= digits <= 18:
            raise ValueError('decimals must be 0..18')
        with localcontext() as context:
            context.prec = 60
            x = [Decimal(str(r['x'])) for r in rows]
            y = [Decimal(str(r['value'])) for r in rows]
            target = Decimal(str(inputs['prediction_x']))
            observed = Decimal(str(actual['value'])) if actual is not None else None
            if not all(v.is_finite() for v in x+y+[target]+([] if observed is None else [observed])):
                raise ValueError('finite numeric values required')
            if len(set(x)) < degree+1:
                raise ValueError('rank deficient design')
            center = sum(x)/len(x)
            scale = max(abs(v-center) for v in x)
            z = [(v-center)/scale for v in x]
            design = [[Decimal(1)]+[v**j for j in range(1, degree+1)] for v in z]
            width = degree+1
            matrix = [[sum(r[i]*r[j] for r in design) for j in range(width)]
                      +[sum(r[i]*v for r, v in zip(design, y))] for i in range(width)]
            for i in range(width):
                pivot = max(range(i, width), key=lambda j: abs(matrix[j][i]))
                matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
                divisor = matrix[i][i]
                if divisor == 0:
                    raise ValueError('rank deficient design')
                matrix[i] = [v/divisor for v in matrix[i]]
                for j in range(width):
                    if j != i:
                        factor = matrix[j][i]
                        matrix[j] = [a-factor*b for a, b in zip(matrix[j], matrix[i])]
            coefficients = [r[-1] for r in matrix]
            target_z = (target-center)/scale
            prediction = coefficients[-1]
            for c in reversed(coefficients[:-1]):
                prediction = prediction*target_z+c
            error = abs(prediction-observed) if observed is not None else None
            value = prediction if operation == 'predict' else error
            answer = format(value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), f'.{digits}f')
            return {'status': 'ok', 'outputs': {
                'count': len(rows), 'degree': degree, 'unit': inputs['unit'],
                'center': str(center), 'scale': str(scale),
                'coefficient_basis': 'ascending powers of (x-center)/scale, including intercept',
                'coefficients': [str(v) for v in coefficients],
                'prediction': str(prediction), 'actual': str(observed) if observed is not None else None,
                'absolute_difference': str(error) if error is not None else None, 'answer': answer,
                'formula': 'unweighted polynomial OLS with intercept; '+('prediction' if operation == 'predict' else 'abs(prediction-actual)'),
                'proof_scope': 'calculation and explicit coverage only; source semantics not certified'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return {'status': 'needs_input', 'outputs': {}, 'detail': str(exc)}


"""Render an explicit answer contract; never infer formatting from gold or task ID."""
import re
from decimal import Decimal, localcontext, ROUND_HALF_UP


def format_answer(value, contract):
    allowed = {'style', 'decimals', 'scale', 'suffix'}
    if set(contract)-allowed or not {'style', 'decimals', 'scale'} <= set(contract):
        raise ValueError('format requires style, decimals, scale; suffix only for unit_suffix')
    style, digits, scale = contract['style'], contract['decimals'], contract['scale']
    if style not in ('number', 'percent', 'unit_suffix') or type(digits) is not int or not 0 <= digits <= 18:
        raise ValueError('style number|percent|unit_suffix and decimals 0..18 required')
    if scale not in ('as_is', 'fraction_to_percent'):
        raise ValueError('Explicit scale as_is|fraction_to_percent required')
    if style != 'unit_suffix' and 'suffix' in contract:
        raise ValueError('Only unit_suffix accepts a suffix; number never appends units')
    suffix = contract.get('suffix', '')
    if style == 'unit_suffix' and (not isinstance(suffix, str) or not suffix.strip() or
            len(suffix) > 60 or '\n' in suffix or '%' in suffix):
        raise ValueError('unit_suffix requires a short literal unit; use percent for %')
    if type(value) not in (str, int, float):
        raise ValueError('A numeric result leaf is required; do not pass prose/reason or an entire object')
    if isinstance(value, str) and not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value.strip()):
        raise ValueError('Result must be numeric without units, commas or explanation')
    with localcontext() as context:
        context.prec = 60
        number = Decimal(str(value))
        if not number.is_finite(): raise ValueError('nonfinite result')
        if scale == 'fraction_to_percent': number *= 100
        number = number.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
        if number == 0: number = abs(number)
        answer = format(number, f'.{digits}f')
    if style == 'percent': answer += '%'
    if style == 'unit_suffix': answer += ' '+suffix.strip()
    return answer


def read_numeric(ctx, ref):
    if ctx.remaining_calls() <= 0: raise ValueError('No tool budget left to read computed result')
    result = ctx.call('read_result', {**ref, 'offset': 0, 'limit': 256})
    if not result.get('accepted') or result.get('next_offset') is not None:
        raise ValueError('Computed result is unavailable or incomplete; select a numeric leaf')
    return result['data']


"""Selected public operands -> named calculation -> fixed display -> submission."""


def run(ctx, inputs):
    try:
        if inputs['submit_answer'] is not True: raise ValueError('Explicit final submission required')
        display = inputs['format']
        format_answer('0', display)
        if display['scale'] != 'as_is':
            raise ValueError('This calculator sets numeric scale once; final format must use as_is')
        cache, evidence = {}, []
        read = lambda source: read_operand(ctx, source, cache, evidence)
        family = inputs['family']
        if family == 'polynomial':
            args={k:inputs[k] for k in ('operation','degree','unit','prediction_x')}
            args.update(decimals=18,required_periods=inputs['requested_periods'],observations=[
                {'period':r['period'],'x':r['x'],'value':read(r['source']),'unit':inputs['unit'],
                 'source_ref':r['source']['source_ref']} for r in inputs['records']])
            if 'actual_source' in inputs:
                args['actual']={'value':read(inputs['actual_source']),'unit':inputs['unit'],
                    'source_ref':inputs['actual_source']['source_ref']}
            if any(Decimal(str(r['x']))==Decimal(str(inputs['prediction_x'])) for r in inputs['records']):
                raise ValueError('Forecast date must not be included in fitting records')
            computed=calculate_polynomial(ctx,args)
            if computed['status']=='ok':
                computed['outputs']['value']=computed['outputs']['prediction' if inputs['operation']=='predict' else 'absolute_difference']
        elif family == 'statistics':
            args = {k:inputs[k] for k in ('statistic','requested_periods','unit')}
            args['decimals'] = 18
            records = []
            for row in inputs['records']:
                item = {'period':row['period'], 'value':read(row['source'])}
                if 'x_source' in row: item['x'] = read(row['x_source'])
                elif 'x' in row: item['x'] = row['x']
                records.append(item)
            args['records'] = records
            if inputs['statistic'] == 'correlation' and any('x_source' not in r for r in inputs['records']):
                raise ValueError('Correlation requires acquired x_source for each paired observation')
            if 'forecast_x' in inputs: args['forecast_x'] = inputs['forecast_x']
            if 'actual_source' in inputs: args['actual'] = read(inputs['actual_source'])
            if inputs['statistic'] in ('mean_absolute_deviation','median_absolute_deviation_mean','log_trimmed_mean','theil_t','rolling_mean_range'):
                wanted = inputs['requested_periods']
                if len(wanted) != len(set(wanted)) or len(records) != len(wanted) or [r['period'] for r in records] != wanted:
                    raise ValueError('Transformation requires exactly the requested periods in order')
                with localcontext() as precision:
                    precision.prec=50
                    values=[Decimal(r['value']) for r in records];op=inputs['statistic']
                    details={}
                    if op in ('mean_absolute_deviation','median_absolute_deviation_mean'):
                        ordered=sorted(values);n=len(values)
                        benchmark=Decimal(str(inputs['benchmark'])) if op=='mean_absolute_deviation' else (
                            ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2)
                        if not benchmark.is_finite():raise ValueError('Finite benchmark required')
                        value=sum(abs(v-benchmark) for v in values)/n
                        formula='mean(abs(value-'+('benchmark' if op=='mean_absolute_deviation' else 'median')+'))'
                        details['benchmark']=str(benchmark)
                    elif op=='log_trimmed_mean':
                        if min(values)<=0:raise ValueError('Logarithm requires positive source values')
                        proportion=Decimal(str(inputs['trim_per_tail']))
                        if not 0<=proportion<Decimal('.5'):raise ValueError('trim_per_tail must be in [0,0.5)')
                        logs=sorted(v.ln() for v in values);k=int(len(logs)*proportion)
                        retained=logs[k:len(logs)-k];value=sum(retained)/len(retained)
                        formula='mean(sort(ln(values))[floor(n*trim_per_tail):n-floor(n*trim_per_tail)])'
                        details.update(trim_count_per_tail=k,retained_count=len(retained))
                    elif op=='rolling_mean_range':
                        window=inputs['window'];groups={}
                        if type(window) is not int or window<1:raise ValueError('Positive integer window required')
                        for row,value in zip(inputs['records'],values):groups.setdefault(row['group'],[]).append(value)
                        if any(len(v)<window for v in groups.values()):raise ValueError('Each group needs a full window')
                        means=[sum(v[i-window:i])/window for v in groups.values() for i in range(window,len(v)+1)]
                        value=max(means)-min(means);formula='max(all grouped rolling means)-min(all grouped rolling means)'
                        details.update(window=window,window_count=len(means),minimum=str(min(means)),maximum=str(max(means)))
                    else:
                        if min(values)<=0:raise ValueError('Theil T requires positive source values')
                        mean=sum(values)/len(values)
                        value=sum((v/mean)*(v/mean).ln() for v in values)/len(values)
                        formula='mean((value/mean(value))*ln(value/mean(value)))'
                computed={'status':'ok','outputs':{'value':str(value),'formula':formula,
                    'selected_periods':wanted,'count':len(records),**details}}
            else:
                computed = calculate_statistics(ctx, args)
        else:
            args = {'operation':inputs['operation'],'unit':inputs['unit'],'decimals':None}
            if family == 'midpoint':
                args.update(values=[read(inputs['first']), read(inputs['second'])], output_percent=False)
            elif family == 'difference':
                args.update(from_value=read(inputs['from_source']),to_value=read(inputs['to_source']),absolute=inputs['absolute'])
            else:
                args.update(base_value=read(inputs['base']), new_value=read(inputs['end']), periods=inputs['periods'])
                args['output_percent'] = inputs['output_percent'] if inputs['operation']=='cagr' else False
            computed = calculate_growth(ctx, args)
            if computed['status']=='ok' and inputs['operation']=='annual_log_growth' and inputs['output_percent']:
                computed['outputs']['value']=str(Decimal(computed['outputs']['value'])*100)
                computed['outputs']['formula']='100*('+computed['outputs']['formula']+')'
                computed['outputs']['output_percent']=True
        if computed['status'] != 'ok': raise ValueError(computed.get('detail','Calculation rejected'))
        result = computed['outputs']
        return {'status':'ok','outputs':{'answer':format_answer(result['value'], display),
            'calculation':result,'evidence':evidence,'format':display,
            'proof_scope':'Acquired cell/result fidelity, named arithmetic and display; caller selects task semantics.'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as error:
        return {'status':'needs_input','outputs':{},'detail':str(error),
            'next_action':'Correct the indicated source selector or operation from public evidence; use native Python if the source layout is unsupported.'}
