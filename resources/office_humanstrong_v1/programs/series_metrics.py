"""Named formulas with explicit conventions; computation, not operand provenance."""
import math
from decimal import Decimal, localcontext, ROUND_HALF_UP


def run(ctx,inputs):
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
                if op not in {'ratio','percent_change','cagr','geometric_growth'}:raise ValueError('percent output is not supported for this operation')
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
