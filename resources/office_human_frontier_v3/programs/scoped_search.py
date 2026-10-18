"""Bounded Office retrieval using only advertised native tools and actual hits."""


def run(ctx,inputs):
    try:
        query=inputs['query']
        if not isinstance(query,str) or not query.strip():raise ValueError('nonempty query required')
        available={t['name'] for t in ctx.available_tools()}
        paths=inputs.get('paths');pattern=inputs.get('file_pattern')
        if paths is None and pattern:
            r=ctx.call('glob',{'pattern':pattern})
            if not r.get('accepted'):return {'status':'blocked','outputs':{},'detail':r.get('error','glob failed')}
            paths=r['data']
        maximum=min(4,max(1,inputs.get('max_windows',3)))
        pages=min(4,max(1,inputs.get('max_pages',2)))
        before=min(4000,max(0,inputs.get('before',1000)));after=min(8000,max(500,inputs.get('after',3500)))
        args={'pattern':query}
        if paths is not None:args['paths']=paths
        hits=[];complete=False;cursor=None;seen=set()
        for _ in range(pages):
            if ctx.remaining_calls()<=0:break
            if 'grep_page' in available:
                call={**args,'limit':40}
                if cursor is not None:call['cursor']=cursor
                r=ctx.call('grep_page',call)
                if not r.get('accepted'):return {'status':'blocked','outputs':{},'detail':r.get('error','search failed')}
                data=r['data'];hits.extend(data['hits']);complete=data['complete'];cursor=data.get('next_cursor')
                if complete or not cursor or cursor in seen or len(hits)>=maximum:break
                seen.add(cursor)
            else:
                r=ctx.call('grep',args)
                if not r.get('accepted'):return {'status':'blocked','outputs':{},'detail':r.get('error','search failed')}
                hits.extend(r['data']);complete=len(r['data'])<40;break
        windows=[];read_seen=set()
        for hit in hits:
            if len(windows)>=maximum or ctx.remaining_calls()<=0:break
            path,offset=hit['path'],hit['offset'];key=(path,offset)
            if key in read_seen:continue
            read_seen.add(key)
            if 'read_context' in available:
                r=ctx.call('read_context',{'path':path,'offset':offset,'before':before,'after':after})
                if not r.get('accepted'):continue
                data=r['data'];start=data['start_offset'];text=data['text']
            else:
                start=max(0,offset-before);r=ctx.call('read',{'path':path,'offset':start})
                if not r.get('accepted'):continue
                text=r['data']['text'][:before+after]
            windows.append({'path':path,'start_offset':start,'hit_offset':offset,'text':text})
        if not windows:return {'status':'not_found' if complete and not hits else 'needs_input','outputs':{},
            'detail':'No acquired matching window; broaden query/scope or read an actual hit. Incomplete search is not proof of absence.'}
        return {'status':'needs_input','outputs':{'windows':windows,'search_exhausted':complete,
            'target_periods_complete':None,'table_header_bound':False,'next_cursor':cursor,
            'scope':'Acquired matching windows, not a complete answer. Search exhaustion is not target-period coverage. Bind the table header, requested rows/periods and unit next.'}}
    except (KeyError,ValueError,TypeError) as e:return {'status':'needs_input','outputs':{},'detail':str(e)}
