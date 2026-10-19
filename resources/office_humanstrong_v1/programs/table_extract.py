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


def run(ctx,inputs):
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
