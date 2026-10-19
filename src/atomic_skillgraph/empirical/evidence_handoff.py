"""Bounded public evidence spans, shared by native and authored-bank finalization."""
from copy import deepcopy
import json
import re

from .contracts import digest
from .public_results import FILE_COMPUTE, PROGRAM_EXECUTION, TEXT_PAGE, TEXT_CONTEXT

VERSION = 'public-evidence-spans.v1'


def enabled(settings):
    return settings.get('evidence_handoff_version') == VERSION


def output_material(materials):
    """Keep explicit dataflow outputs, requiring their cited public spans."""
    refs = []
    def visit(value):
        if isinstance(value, dict):
            ref = value.get('source_ref') or value.get('ref')
            if not ref and isinstance(value.get('result_id'), str):
                ref = {'result_id': value['result_id'], 'path': value.get('path', [])}
            if ref:
                item = {'ref': ref}
                if 'offset' in value and 'end' in value:
                    item['ranges'] = [(value['offset'], value['end'])]
                refs.append(item)
                return {'ref': ref}
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list): return [visit(v) for v in value]
        return value
    return {k: visit(materials[k]) for k in ('completed_results', 'pending_outputs') if materials.get(k)}, refs


def text_groups(text, goal):
    """Rank exact table rows with their headers; never infer a cell or a formula."""
    lines = list(re.finditer(r'[^\n]*(?:\n|$)', text))
    lines = [m for m in lines if m.end() > m.start()]
    words = lambda s: set(re.findall(r'[a-z]{3,}|\d{4}', s.lower()))
    query = words(goal) - set('the and for from with what which how was were are this that into between rounded nearest'.split())
    groups = []
    for i, line in enumerate(lines):
        if i == 0 or not re.fullmatch(r'\s*\|?[\s:|\-]+\|\s*', line.group()):
            continue
        if '|' not in lines[i-1].group(): continue
        start = i-1
        # Include the source title, unit and date immediately before the header.
        while start > 0 and lines[i-1].start()-lines[start-1].start() < 1000 and '|' not in lines[start-1].group():
            start -= 1
        header = (lines[start].start(), line.end())
        heading_score = len(words(text[header[0]:header[1]]) & query)
        j = i+1
        while j < len(lines) and '|' in lines[j].group():
            row = lines[j]
            matches = words(row.group()) & query
            score = heading_score + len(matches) + 4*sum(w.isdigit() for w in matches)
            groups.append((score, [header, (row.start(), row.end())]))
            j += 1
    # Non-table sources remain available as exact paragraph windows.
    for m in re.finditer(r'\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*', text):
        if '|' in m.group(): continue
        for start in range(m.start(), m.end(), 2000):
            end = min(start+2000, m.end())
            groups.append((len(words(text[start:end]) & query), [(start,end)]))
    return sorted(groups, key=lambda g: -g[0])


def prepare(context, materials, *, max_bytes, output_cap, payload_builder, priority_refs=(),
            allowed_result_ids=None):
    """Whole evidence groups fit or are explicitly omitted; no sliced table rows."""
    value = deepcopy(materials)
    value['finish_evidence'] = []
    value['evidence_scope'] = 'Exact acquired spans; ranking is not semantic validation. No unseen source is fetched.'
    bound = lambda: len(json.dumps(payload_builder(value, output_cap), allow_nan=False).encode())
    if bound() > max_bytes: raise ValueError('finish_unavailable:base_input_exceeds_limit')
    included, omitted, unresolved, groups = [], [], [], []
    rows = {r['result_id']:r for r in context.memory.values()}

    def add(ref, ranges=None, rank=0, required=False):
        if allowed_result_ids is not None and ref['result_id'] not in allowed_result_ids:
            unresolved.append({**ref,'reason':'not_in_obtained_results','required':required});return
        try:
            raw = context.resolve_public(ref)
            if isinstance(raw,dict):
                for leaf in context.business_refs(ref): add(leaf,rank=rank,required=required)
                return
            if not required and (raw is None or raw == '' or raw == []):return
            size = len(raw) if isinstance(raw,(str,list)) else len(json.dumps(raw,ensure_ascii=False,allow_nan=False))
            cap = context.settings['result_preview_max_items'] if isinstance(raw,list) else context.settings['result_read_window_chars']
            ranges = ranges or [(0,max(1,size))]
            if any(b-a>cap for a,b in ranges):
                omitted.append({**ref,'reason':'whole_window_read_limit','required':required})
                return
            windows = [context.result_window(ref,a,b-a) for a,b in ranges if b>a]
            row=rows.get(ref['result_id'],{});args=row.get('arguments') or {}
            for w in windows:
                if ref['path']==['data','stdout']:
                    w['stdout_receipt']=deepcopy(context.results[ref['result_id']]['data'].get('stdout_receipt',{}))
                if args.get('path') and ref['path']==['data','text']:
                    w['source_path']=args['path']
                    data=context.results[ref['result_id']].get('data',{})
                    begin=data.get('next_offset',len(raw))-len(raw)
                    w['source_offset']=begin+w['offset']
            groups.append((rank,required,windows))
        except (KeyError,TypeError,ValueError):
            unresolved.append({**ref,'reason':'invalid_public_reference','required':required})

    for item in priority_refs:
        ref=item['ref'];quote=item.get('quote')
        ranges=item.get('ranges')
        if quote:
            try:
                raw=context.resolve_public(ref);start=raw.index(quote)
                ranges=[(max(0,start-300),min(len(raw),start+len(quote)+300))]
                for _,group in text_groups(raw,value.get('original_goal','')):
                    if len(group)==2 and group[1][0]<=start and start+len(quote)<=group[1][1]:
                        ranges=group;break
            except (ValueError,AttributeError):
                unresolved.append({**ref,'reason':'quote_not_in_source','required':True});continue
        add(ref,ranges,rank=-100,required=True)
    # Explicit read_result ranges are retained instead of resetting to offset zero.
    for read in reversed(context.local_reads):
        ref=read.get('source_ref') or {k:read[k] for k in ('result_id','path')}
        add(ref,[(read.get('offset',0),read.get('end',read.get('offset',0)+read.get('limit',2000)))],rank=-50)
    for rid in reversed(list(context.results)):
        origin=context.origin(rid)
        if origin in {TEXT_PAGE,TEXT_CONTEXT} and context.results[rid].get('accepted') is not False:
            ref={'result_id':rid,'path':['data','text']}
            raw=context.resolve_public(ref)
            # Prefer an entire already-acquired page when it fits. Row selection is
            # only a capacity fallback, never mandatory lossy preprocessing.
            add(ref,rank=-40)
            for score,ranges in text_groups(raw,value.get('original_goal','')):
                add(ref,ranges,rank=10-score)
        elif origin in {FILE_COMPUTE,PROGRAM_EXECUTION}:
            data=context.results[rid].get('data',{})
            base=['outputs'] if origin==PROGRAM_EXECUTION else ['data','computed' if 'computed' in data else 'stdout']
            try:leaves=context.business_refs({'result_id':rid,'path':base})
            except ValueError:continue
            for ref in leaves:
                raw=context.resolve_public(ref)
                # Compact calculations precede discovery lists and large search dumps.
                compact=(not any(k in ref['path'] for k in ['hits','files','matches','windows'])
                    and len(json.dumps(raw,ensure_ascii=False))<=context.settings['result_inline_max_chars'])
                add(ref,rank=-60 if compact else 30)
    seen=set();retained=[]
    def covered(window):
        for old in retained:
            if window.get('source_path'):
                if old.get('source_path')!=window['source_path']:continue
                start,end=window['source_offset'],window['source_offset']+len(window['content'])
                begin=old.get('source_offset',-1)
            else:
                if old['source_ref']!=window['source_ref']:continue
                start,end=window['offset'],window['end'];begin=old['offset']
            if begin<=start and end<=begin+len(old['content']) and old['content'][start-begin:end-begin]==window['content']:
                return True
        return False
    for rank,required,windows in sorted(groups,key=lambda g:g[0]):
        new=[]
        for w in windows:
            key=digest([w.get('source_path',w['source_ref']),w.get('source_offset',w['offset']),w['content']])
            if key not in seen and not covered(w):new.append((key,w))
        if not new:continue
        value['finish_evidence'].extend(w for _,w in new)
        if bound()>max_bytes:
            del value['finish_evidence'][-len(new):]
            omitted.extend({**{k:v for k,v in w.items() if k!='content'},'reason':'whole_group_input_limit','required':required} for _,w in new)
            continue
        for key,w in new:
            seen.add(key);retained.append(w);included.append({k:v for k,v in w.items() if k!='content'})
    required_missing=any(x.get('required') for x in [*omitted,*unresolved])
    return {'materials':value,'included_refs':included,'omitted_refs':omitted,'unresolved_refs':unresolved,
        'required_evidence_missing':required_missing,'evidence_digest':digest(value['finish_evidence']),
        'serialized_input_bytes':bound(),'input_max_bytes':max_bytes,'output_cap':output_cap,
        'projection_version':VERSION,'evidence_handoff_version':VERSION}
