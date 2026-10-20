METRIC_RULES = {'difference': {'required': ['operation', 'from_value', 'to_value', 'absolute', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'from_value', 'to_value', 'absolute']}, 'ratio': {'required': ['operation', 'numerator', 'denominator', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'numerator', 'denominator']}, 'percent_change': {'required': ['operation', 'base_value', 'new_value', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'base_value', 'new_value', 'periods']}, 'log_return': {'required': ['operation', 'base_value', 'new_value', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'base_value', 'new_value', 'periods']}, 'cagr': {'required': ['operation', 'base_value', 'new_value', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'base_value', 'new_value', 'periods']}, 'annual_log_growth': {'required': ['operation', 'base_value', 'new_value', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'base_value', 'new_value', 'periods']}, 'sum': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'mean': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'median': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'std_population': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'std_sample': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'variance_population': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'variance_sample': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'gini_population': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'gini_sample_adjusted': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'correlation': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'ols_slope': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'ols_intercept': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'geometric_growth': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'symmetric_difference': {'required': ['operation', 'values', 'unit', 'output_percent'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values']}, 'share_difference': {'required': ['operation', 'values', 'unit', 'output_percent', 'absolute'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'absolute']}}

"""Conservative row/column extraction from supplied Treasury text; no made-up cells."""
import re
from decimal import Decimal, InvalidOperation


def cells(line):
    s=line.strip()
    if '|' in s:return [x.strip() for x in s.strip('|').split('|')]
    return [x.strip() for x in re.split(r'\t+| {2,}',s)]


def number(s):
    t=re.sub(r'(?<=\d)[*†‡]+$','',s.strip()).replace(',','')
    if re.fullmatch(r'\(?[+-]?(?:\d+(?:\.\d*)?|\.\d+)\)?',t) is None:raise ValueError('not a unique numeric cell: '+s)
    if t.startswith('(') and t.endswith(')'):t='-'+t[1:-1]
    d=Decimal(t)
    if not d.is_finite():raise ValueError('nonfinite cell')
    return str(d)


def extract_table(ctx,inputs):
    try:
        text=inputs['text'];wanted=inputs['rows'];column=inputs['column']
        if not isinstance(text,str) or not text.strip() or not isinstance(wanted,list) or not wanted:
            raise ValueError('text and nonempty rows required')
        if any(not isinstance(x,str) or not x.strip() for x in wanted) or not isinstance(column,str):raise ValueError('row and column labels must be strings')
        raw=text.splitlines(keepends=True);offsets=[];off=0
        for line in raw:offsets.append(off);off+=len(line)
        parsed=[cells(line) for line in raw]
        norm=lambda s:' '.join(s.casefold().split())
        hints=inputs.get('header_hint','');headers=[]
        for i,row in enumerate(parsed):
            indexes=[j for j,c in enumerate(row) if norm(c)==norm(column)]
            if len(indexes)==1 and (not hints or norm(hints) in norm(' '.join(raw[max(0,i-6):i+1]))):headers.append((i,indexes[0]))
        if len(headers)!=1:raise ValueError('missing or ambiguous header; supply a tighter table window/header_hint')
        h,col=headers[0];unit=inputs.get('unit','')
        if not unit:raise ValueError('Bind the actual unit from the supplied window; use native read for missing header/unit/footnotes')
        if unit and norm(unit) not in norm(text):raise ValueError('requested unit is absent from this window')
        # Require a tight table window; repeated target labels are rejected, not merged.
        matches=[]
        for label in wanted:
            found=[]
            for i in range(h+1,len(parsed)):
                row=parsed[i]
                if len(row)<=col:continue
                if any(norm(c)==norm(label) for j,c in enumerate(row) if j!=col):
                    try:value=number(row[col])
                    except ValueError:continue
                    found.append({'row':label,'column':column,'value':value,'unit':unit,
                        'source_line':i+1,'start':offsets[i],'end':offsets[i]+len(raw[i].rstrip('\r\n')),'quote':raw[i].rstrip('\r\n')})
            if len(found)!=1:raise ValueError('missing/ambiguous row '+label+'; do not merge repeated year blocks')
            matches.append(found[0])
        return {'status':'ok','outputs':{'cells':matches,'values':[x['value'] for x in matches],
            'header':raw[h].rstrip('\r\n'),'unit':unit,'requested_rows':wanted,
            'table_header_bound':True,'target_periods_complete':True,
            'proof_scope':'requested_literal_rows_only',
            'scope':'Every requested row label matched once under this header/unit. This does not verify that your requested labels or fiscal/calendar interpretation match the question. Read the actual source if OCR is ambiguous.'}}
    except (KeyError,ValueError,TypeError,InvalidOperation) as e:return {'status':'needs_input','outputs':{},'detail':str(e)}


"""Named formulas with explicit conventions; computation, not operand provenance."""
import math
from decimal import Decimal, localcontext, ROUND_HALF_UP


def calculate_metrics(ctx,inputs):
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


"""Bound real table references to a named calculation and a final answer.

The build includes the unchanged V2 extractor/metrics and V3 polynomial functions.
This Program is an explicit final submission, never an automatic routing rule.
"""


def source_window(ctx, table):
    ref = table['source_ref']
    start = table.get('offset', 0)
    length = table.get('length', 12000)
    if type(start) is not int or start < 0 or type(length) is not int or not 1 <= length <= 24000:
        raise ValueError('source window requires offset>=0 and length 1..24000')
    # Read one preceding character to identify a partial first line.
    cursor = max(0, start-1)
    end = start+length
    chunks = []
    last = None
    while cursor < end:
        if ctx.remaining_calls() <= 0:
            raise ValueError('tool budget exhausted before requested source window was read')
        last = ctx.call('read_result', {**ref, 'offset': cursor, 'limit': min(3000, end-cursor)})
        if not last.get('accepted'):
            raise ValueError('source read refused: '+str(last.get('error')))
        if not isinstance(last.get('data'), str) or last.get('unit') != 'unicode_codepoint_0_based':
            raise ValueError('source_ref must identify acquired public text')
        if last['offset'] != cursor or last['end'] < cursor:
            raise ValueError('inconsistent source read offsets')
        chunks.append(last['data'])
        if last['next_offset'] is None or last['end'] == cursor:
            break
        cursor = last['end']
    text = ''.join(chunks)
    base = max(0, start-1)
    if start:
        if not text:
            raise ValueError('source window is empty')
        previous, text = text[0], text[1:]
        base = start
        if previous != '\n':
            newline = text.find('\n')
            if newline < 0:
                raise ValueError('no complete source line in requested window')
            base += newline+1
            text = text[newline+1:]
    if last and last['next_offset'] is not None and not text.endswith('\n'):
        # A partial numeric row must not be parsed as a smaller complete value.
        text = text[:text.rfind('\n')+1] if '\n' in text else ''
    if not text.strip():
        raise ValueError('no complete source lines in requested window')
    return text, base


def run(ctx, inputs):
    try:
        if inputs.get('submit_answer') is not True:
            raise ValueError('Choose this final-answer Program only for the complete requested calculation')
        tables = inputs['tables']
        if not 1 <= len(tables) <= 8:
            raise ValueError('1..8 table selectors required')
        groups, evidence, units, windows = {}, [], set(), {}
        for table in tables:
            key = table['key']
            if key in groups:
                raise ValueError('table keys must be distinct')
            ref = table['source_ref']
            window_key = (ref['result_id'], tuple(ref['path']), table.get('offset', 0), table.get('length', 12000))
            if window_key not in windows:
                windows[window_key] = source_window(ctx, table)
            text, base = windows[window_key]
            extracted = extract_table(ctx, {'text': text, 'rows': table['rows'],
                'column': table['column'], 'unit': table['unit'],
                'header_hint': table.get('header_hint', '')})
            if extracted['status'] != 'ok':
                raise ValueError(key+': '+extracted.get('detail', 'ambiguous table'))
            cells = extracted['outputs']['cells']
            for cell in cells:
                cell.update(source_ref=table['source_ref'], start=base+cell['start'],
                            end=base+cell['end'], table_key=key)
            groups[key] = cells
            evidence.extend(cells)
            units.add(table['unit'])
        if len(units) != 1:
            raise ValueError('mixed source units: normalize explicitly before using this Program')
        bindings = inputs['bindings']
        if not bindings or set(k for names in bindings.values() for k in names) != set(groups):
            raise ValueError('bindings must use every extracted table key and no unknown keys')
        selected = {}
        for field, names in bindings.items():
            if not names or len(set(names)) != len(names):
                raise ValueError('each binding needs distinct ordered table keys')
            selected[field] = [cell for key in names for cell in groups[key]]
        options = dict(inputs['calculation'])
        method = options.pop('method')
        options['unit'] = inputs['answer_unit']
        if method == 'series_metrics':
            if set(options) - {'operation','decimals','unit','absolute','output_percent','periods'}:
                raise ValueError('metric calculation contains polynomial-only options')
            op = options['operation']
            absolute_relative = op == 'absolute_relative_change'
            if absolute_relative:
                options['operation'] = 'percent_change'
            source_unit = next(iter(units))
            if op in {'difference','sum','mean','median','std_population','std_sample'} and options['unit'] != source_unit:
                raise ValueError('this calculation does not convert source units')
            if op in {'variance_population','variance_sample'} and options['unit'] != source_unit+'^2':
                raise ValueError('variance answer_unit must be source_unit^2')
            scalar = {'from_value','to_value','base_value','new_value','numerator','denominator'}
            if set(selected) - scalar - {'values','other_values'}:
                raise ValueError('unknown metric operand binding')
            for field, rows in selected.items():
                if field in scalar and len(rows) != 1:
                    raise ValueError(field+' requires exactly one extracted cell')
                options[field] = rows[0]['value'] if field in scalar else [r['value'] for r in rows]
            options['operand_refs'] = [r['source_ref'] for r in evidence]
            rule = METRIC_RULES.get(options['operation'])
            if not rule or set(options)-set(rule['allowed']) or set(rule['required'])-set(options):
                raise ValueError('calculation fields do not match the exact named metric contract')
            computed = calculate_metrics(ctx, options)
            if absolute_relative and computed['status'] == 'ok':
                result = computed['outputs']
                with localcontext() as precision:
                    precision.prec = 50
                    value = abs(Decimal(result['value']))
                    digits = options['decimals']
                    result.update(value=str(value),
                        answer=format(value.quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP),f'.{digits}f'),
                        formula='abs('+result['formula']+')',direction='absolute_base_relative',
                        convention='absolute_relative_change')
        elif method == 'polynomial_forecast':
            if set(options) - {'operation','degree','decimals','unit','first_x','x_step','prediction_x'}:
                raise ValueError('polynomial calculation contains unsupported options')
            if options['unit'] != next(iter(units)):
                raise ValueError('polynomial output keeps the observed source unit')
            if set(selected) - {'observations','actual'} or 'observations' not in selected:
                raise ValueError('polynomial requires observations and optional actual bindings')
            rows = selected['observations']
            first = Decimal(str(options.pop('first_x', 0)))
            step = Decimal(str(options.pop('x_step', 1)))
            if not first.is_finite() or not step.is_finite() or step == 0:
                raise ValueError('finite first_x and nonzero x_step required')
            options['required_periods'] = [r['row'] for r in rows]
            options['observations'] = [{'period': r['row'], 'x': str(first+i*step),
                'value': r['value'], 'unit': inputs['answer_unit'], 'source_ref': r['source_ref']}
                for i, r in enumerate(rows)]
            if 'actual' in selected:
                if len(selected['actual']) != 1:
                    raise ValueError('actual binding requires exactly one observed target value')
                r = selected['actual'][0]
                options['actual'] = {'value': r['value'], 'unit': inputs['answer_unit'],
                                     'source_ref': r['source_ref']}
            computed = calculate_polynomial(ctx, options)
        else:
            raise ValueError('unknown calculation method')
        if computed['status'] != 'ok':
            raise ValueError(computed.get('detail', 'calculation needs input'))
        result = computed['outputs']
        return {'status': 'ok', 'outputs': {'answer': result['answer'],
            'calculation': result, 'evidence': evidence,
            'proof_scope': 'literal source extraction and named computation; caller selects question semantics'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return {'status': 'needs_input', 'outputs': {}, 'detail': str(exc)}
