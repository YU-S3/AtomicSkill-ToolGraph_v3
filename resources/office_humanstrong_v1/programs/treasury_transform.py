"""Treasury-specific exact local transforms with explicit conventions.

Cannot identify a table, currency orientation, or task target by itself.
"""
from decimal import Decimal, localcontext, ROUND_HALF_UP
import re

SCALES={'dollars':Decimal(1),'thousands_of_dollars':Decimal(1000),
        'millions_of_dollars':Decimal(1000000),'billions_of_dollars':Decimal(1000000000)}


def run(ctx,inputs):
    try:
        op=inputs['operation']
        with localcontext() as c:
            c.prec=44
            if op=='unit_scale':
                source=inputs['source_unit'];target=inputs['target_unit']
                if source not in SCALES or target not in SCALES:raise ValueError('unsupported unit direction')
                x=Decimal(str(inputs['value']).replace(',',''));y=x*SCALES[source]/SCALES[target]
                rule=f'{source}->{target}';unit=target
            elif op=='price_32nds':
                if inputs.get('quote_basis')!='thirty_seconds':raise ValueError('source table must state quote basis as 32nds')
                raw=str(inputs['quote']).strip();m=re.fullmatch(r'(\d{1,3})\.(\d{1,2})',raw)
                if not m:raise ValueError('price quote must be an integer part and 0..31 32nd ticks')
                whole,ticks=map(int,m.groups())
                if not 0<=ticks<=31:raise ValueError('32nd ticks out of range')
                y=Decimal(whole)+Decimal(ticks)/Decimal(32)
                rule='integer + ticks/32';unit='price_points'
            elif op=='box_cox':
                x=Decimal(str(inputs['value']));lam=Decimal(str(inputs['lambda']))
                if x<=0 or not x.is_finite() or not lam.is_finite():raise ValueError('Box-Cox requires x>0 and finite lambda')
                y=x.ln() if lam==0 else ((lam*x.ln()).exp()-1)/lam
                rule='ln(x)' if lam==0 else '(x^lambda-1)/lambda';unit='transformed'
            elif op=='currency_convert':
                amount=Decimal(str(inputs['amount']));rate=Decimal(str(inputs['rate']))
                direction=inputs['rate_orientation']
                if rate<=0 or not rate.is_finite() or not amount.is_finite():raise ValueError('valid positive exchange rate required')
                if direction=='foreign_per_usd':y=amount/rate;unit='usd';rule='foreign_amount / foreign_per_usd'
                elif direction=='usd_per_foreign':y=amount*rate;unit='usd';rule='foreign_amount * usd_per_foreign'
                else:raise ValueError('exchange-rate orientation must be explicit')
            else:raise ValueError('unsupported Treasury transform')
            if not y.is_finite():raise ValueError('nonfinite transformed output')
            digits=inputs.get('decimals')
            if digits is None:ans=format(y,'f').rstrip('0').rstrip('.') if '.' in format(y,'f') else format(y,'f')
            else:
                if type(digits) is not int or not 0<=digits<=18:raise ValueError('decimals must be 0..18')
                ans=format(y.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP),f'.{digits}f')
            if ans in ('-0',''):ans='0'
            return {'status':'ok','outputs':{
                'value':str(y),'answer':ans,'unit':unit,'formula':rule,
                'proof_scope':'local_transform_only',
                'note':'Input numbers and quote/currency convention must be grounded in a real source table.'}}
    except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
        return {'status':'needs_input','outputs':{},'detail':str(exc)}
