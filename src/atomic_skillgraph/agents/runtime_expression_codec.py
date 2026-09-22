"""Lossless public-view codecs. Execution state and evidence are never packed."""
import copy
import hashlib
import json
from dataclasses import dataclass,asdict

DISCOVERY_FIELDS={'last_seen_revision','observed_at_revision','last_known_location',
    'public_evidence_ref','evidence_status','location_evidence_status','source_kind'}
ROWS_HELP='Rows formats preserve order: discovery rows are [entity, values in fields order]; catalog rows are [action_id, arguments in fields order]. Groups preserve their original sequence. Missing fields are absent, not null.'

def _bytes(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def _hash(v):return hashlib.sha256(_bytes(v)).hexdigest()

@dataclass(frozen=True)
class TransformAudit:
    applicable:bool
    applied:bool
    reason:str
    original_bytes:int
    packed_bytes:int
    roundtrip:bool
    source_hash:str
    packed_hash:str

def _result(raw,packed,unpack,reason='supported'):
    original=_bytes(raw)
    if packed is None:
        return copy.deepcopy(raw),TransformAudit(False,False,reason,len(original),len(original),True,_hash(raw),_hash(raw))
    if _bytes(unpack(packed))!=original:raise ValueError('runtime codec typed/order roundtrip failed')
    candidate=_bytes(packed)
    # Charge the complete explanation to each transform (conservative).
    apply=len(candidate)+len(ROWS_HELP.encode())<len(original)
    value=packed if apply else copy.deepcopy(raw)
    return value,TransformAudit(True,apply,'applied' if apply else 'no_net_saving',len(original),len(_bytes(value)),True,_hash(raw),_hash(value))

def pack_discovery_rows(raw):
    if not isinstance(raw,dict) or any(not isinstance(k,str) or not isinstance(v,dict) or set(v)-DISCOVERY_FIELDS for k,v in raw.items()):
        return _result(raw,None,unpack_discovery_rows,'unsupported_shape')
    groups=[]
    for key,record in raw.items():
        fields=list(record)
        if not groups or groups[-1]['fields']!=fields:groups.append({'fields':fields,'rows':[]})
        groups[-1]['rows'].append([key,*copy.deepcopy(list(record.values()))])
    return _result(raw,{'format':'discovery_rows_v1','groups':groups},unpack_discovery_rows)

def unpack_discovery_rows(value):
    if not isinstance(value,dict) or value.get('format')!='discovery_rows_v1':return copy.deepcopy(value)
    result={}
    for group in value['groups']:
        fields=group['fields']
        if len(fields)!=len(set(fields)) or set(fields)-DISCOVERY_FIELDS:raise ValueError('invalid discovery fields')
        for row in group['rows']:
            if len(row)!=len(fields)+1 or row[0] in result:raise ValueError('invalid discovery row')
            result[row[0]]=dict(zip(fields,copy.deepcopy(row[1:])))
    return result

def pack_catalog_rows(raw):
    if not isinstance(raw,dict) or list(raw)!=['revision','actions'] or not isinstance(raw['actions'],list):
        return _result(raw,None,unpack_catalog_rows,'unsupported_shape')
    groups=[]
    for action in raw['actions']:
        if (not isinstance(action,dict) or list(action)!=['action_id','action_type','arguments']
            or not isinstance(action['arguments'],dict) or not isinstance(action['action_id'],str)):
            return _result(raw,None,unpack_catalog_rows,'unsupported_shape')
        fields=list(action['arguments'])
        if not groups or groups[-1]['action_type']!=action['action_type'] or groups[-1]['fields']!=fields:
            groups.append({'action_type':action['action_type'],'fields':fields,'rows':[]})
        groups[-1]['rows'].append([action['action_id'],*copy.deepcopy(list(action['arguments'].values()))])
    return _result(raw,{'format':'catalog_rows_v1','revision':copy.deepcopy(raw['revision']),'groups':groups},unpack_catalog_rows)

def unpack_catalog_rows(value):
    if not isinstance(value,dict) or value.get('format')!='catalog_rows_v1':return copy.deepcopy(value)
    actions=[]
    for group in value['groups']:
        fields=group['fields']
        if len(fields)!=len(set(fields)):raise ValueError('duplicate catalog argument')
        for row in group['rows']:
            if len(row)!=len(fields)+1:raise ValueError('invalid catalog row')
            actions.append({'action_id':row[0],'action_type':group['action_type'],'arguments':dict(zip(fields,copy.deepcopy(row[1:])))})
    return {'revision':copy.deepcopy(value['revision']),'actions':actions}

def project_lean(payload,summaries=None):
    result=copy.deepcopy(payload);audit={'base_projected_hash':_hash(payload),'transforms':{},'summary_visible_refs':[],
        'summary_source_hashes':{},'missing_summary_refs':[],'removed_diagnostics':[]}
    for path,rows in [('support_atomic_candidates',result.get('support_atomic_candidates',[])),
            ('task_runtime_frame.capability_candidates',result.get('task_runtime_frame',{}).get('capability_candidates',[]))]:
        for index,row in enumerate(rows):
            original_keys=list(row)
            original_summary=copy.deepcopy(row.get('summary'))
            if 'diagnostics' in row:
                keys=list(row)
                audit['removed_diagnostics'].append({'path':path,'index':index,'ref':row.get('atomic_ref'),'keys':keys,'value':row.pop('diagnostics')})
            ref=row.get('atomic_ref')
            if ref in (summaries or {}):
                row['summary']=summaries[ref];audit['summary_visible_refs'].append(ref)
                audit['summary_source_hashes'][ref]=_hash(summaries[ref])
                audit.setdefault('summary_replacements',[]).append({'path':path,'index':index,
                    'keys':original_keys,'value':original_summary,'present':'summary' in original_keys})
            elif 'summary' in row:
                audit['summary_visible_refs'].append(ref)
                audit.setdefault('existing_summary_source_hashes', {})[ref] = _hash(row['summary'])
            else:audit['missing_summary_refs'].append(ref)
    if 'current_action_catalog' in result:
        result['current_action_catalog'],check=pack_catalog_rows(result['current_action_catalog']);audit['transforms']['catalog']=asdict(check)
    memory=result.get('exploration_memory',{})
    for field in ('historical_discoveries','observed_discoveries'):
        if field in memory:memory[field],check=pack_discovery_rows(memory[field]);audit['transforms'][field]=asdict(check)
    audit['lean_payload_hash']=_hash(result)
    if _bytes(restore_lean(result,audit))!=_bytes(payload):
        raise ValueError('lean projection is not exactly reversible')
    return result,audit

def restore_lean(payload,audit):
    result=copy.deepcopy(payload)
    if _hash(result)!=audit['lean_payload_hash']:raise ValueError('lean payload hash mismatch')
    if 'current_action_catalog' in result:result['current_action_catalog']=unpack_catalog_rows(result['current_action_catalog'])
    memory=result.get('exploration_memory',{})
    for field in ('historical_discoveries','observed_discoveries'):
        if field in memory:memory[field]=unpack_discovery_rows(memory[field])
    for path,rows in [('support_atomic_candidates',result.get('support_atomic_candidates',[])),
            ('task_runtime_frame.capability_candidates',result.get('task_runtime_frame',{}).get('capability_candidates',[]))]:
        for index,row in enumerate(rows):
            ref=row.get('atomic_ref')
            if ref in audit['summary_source_hashes']:
                if _hash(row.get('summary'))!=audit['summary_source_hashes'][ref]:raise ValueError('summary source hash mismatch')
                row.pop('summary')
            replacements=[r for r in audit.get('summary_replacements',[]) if r['path']==path and r['index']==index]
            if replacements and replacements[0]['present']:
                row['summary']=copy.deepcopy(replacements[0]['value'])
            removed=[r for r in audit['removed_diagnostics'] if r['path']==path and r['index']==index]
            if len(removed)>1:raise ValueError('ambiguous diagnostic row')
            if removed:
                record=removed[0];row['diagnostics']=copy.deepcopy(record['value'])
                restored={key:row[key] for key in record['keys']}
                row.clear();row.update(restored)
            elif replacements:
                restored={key:row[key] for key in replacements[0]['keys']}
                row.clear();row.update(restored)
    if _hash(result)!=audit['base_projected_hash']:raise ValueError('lean base restoration mismatch')
    return result
