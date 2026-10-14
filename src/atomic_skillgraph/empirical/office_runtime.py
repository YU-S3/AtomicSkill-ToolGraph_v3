"""Lossless references and bounded actual evidence for the V3 Office Runtime."""
from .contracts import digest


def evidence_view(context, goal):
    searches, pages, computations = [], [], []
    for row in context.memory.values():
        rid=row['result_id'];raw=context.results[rid]
        if not raw.get('accepted'):continue
        args=row['arguments'];data=raw.get('data')
        if row['name'] in {'grep','grep_page'}:
            hits=data if isinstance(data,list) else data.get('hits',[])
            searches.append({'query':args['pattern'],'scope':args.get('paths'), 'result_id':rid,
                'hit_count':len(hits),'complete':data.get('complete') if isinstance(data,dict) else
                    not raw.get('scope_audit',{}).get('truncated',len(hits)==40),
                'next_cursor':data.get('next_cursor') if isinstance(data,dict) else None})
        elif row['name'] in {'read','read_context'} and isinstance(data,dict):
            text=data.get('text','')
            pages.append({'path':args['path'],'anchor_offset':args['offset'],
                'text_ref':{'result_id':rid,'path':['data','text']},'actual_text_excerpt':text[:2048],
                'excerpt_end':min(2048,len(text)),'text_length':len(text),'next_offset':data.get('next_offset'),
                'interpretation':'unparsed actual table/text; headings/units/values require explicit evidence'})
        elif isinstance(data,dict) and 'computed' in data:
            computations.append({'result_id':rid,'computed':context.preview(data['computed'])})
    return {'version':'office.runtime-evidence.v3','current_goal_and_period':goal,
        'searches':searches[-8:],'acquired_pages':pages[-3:],'computed':computations[-2:],
        'omitted_counts':{'searches':max(0,len(searches)-8),'pages':max(0,len(pages)-3),'computed':max(0,len(computations)-2)},
        'missing_evidence':['numeric extraction and unit consistency are unverified until actual parsing/calculation is recorded']
            if not computations else ['whole-answer sufficiency remains the Agent decision'],
        'rule':'truncated means unfinished; never infer a missing value from a search cutoff; use read_result for full referenced text'}


def compact_recent(rows, context):
    """Historical directory/search duplicates keep identities; the latest hit page remains visible."""
    search_ids=[r.get('result_id') for r in rows if context.origin(r.get('result_id')) in
                {'native.scoped-search.v1','native.search-page.v3'}]
    names={r['result_id']:r['name'] for r in context.memory.values()}
    result=[]
    for row in rows:
        rid=row.get('result_id')
        if names.get(rid)=='glob' or rid in search_ids[:-1]:
            result.append({'result_id':rid,'name':names.get(rid),'raw_result_sha256':digest(context.results[rid]),
                'preview_omitted':'historical discovery; query/status in retrieval evidence, full result via read_result'})
        else:result.append(row)
    return result
