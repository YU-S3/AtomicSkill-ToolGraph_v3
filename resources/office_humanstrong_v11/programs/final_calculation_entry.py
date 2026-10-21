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
