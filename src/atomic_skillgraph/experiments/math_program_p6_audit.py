"""Read-only replay of P5 failures. Nothing here imports an old asset into P6."""
import ast
from collections import Counter
import json
from pathlib import Path

from ..empirical import math_operation_contract_p6 as c
from ..empirical import math_program_induction as old, math_program_probe as probe
from ..empirical.contracts import digest
from ..empirical.program_worker import ProgramWorker
from .canonical_manifest import sha256
from .formal_log import tree_identity
from .run_empirical import write_json


def differences(expected, actual, path):
    if type(expected)!=type(actual):return [c.error('schema_contract_mismatch',path,expected,actual)]
    if isinstance(expected,dict):
        result=[]
        for key in sorted(set(expected)|set(actual)):
            if key not in expected or key not in actual:result.append(c.error('schema_contract_mismatch',path+'.'+key,expected.get(key,'absent'),actual.get(key)))
            else:result+=differences(expected[key],actual[key],path+'.'+key)
        return result
    return [] if expected==actual else [c.error('schema_contract_mismatch',path,expected,actual)]


def replay(root, output, worker=None):
    root,output=Path(root),Path(output);before=tree_identity(root);read=lambda p:json.loads(p.read_text())
    rejected=[];families={};negative=[]
    for path in sorted((root/'learning').glob('*/learning.json')):
        row=read(path);folder=path.parent
        if row['state']=='model_invalid':
            requests=read(folder/'requests.json')['requests'];r=requests[-1];errors=[];original=[]
            calls=r.get('response',{}).get('tool_calls',[])
            if r.get('error'):
                for attempt in r.get('http_attempts',[]):
                    for choice in attempt.get('public_response',{}).get('choices',[]):
                        for i,call in enumerate(choice.get('message',{}).get('tool_calls',[])):
                            raw=call['function']['arguments'];original.append(raw)
                            try:json.loads(raw)
                            except json.JSONDecodeError as exc:
                                errors.append(c.error('response_json_invalid',f'http.response.tool_calls[{i}].function.arguments@{exc.pos}',exc.msg,raw))
                if not errors:raise ValueError('Expected saved malformed JSON evidence')
            elif calls:
                value=calls[-1]['arguments'];original.append(value)
                if r['purpose']=='math_program_spec':
                    errors=c.schema_errors(value,old.PROPOSAL)
                    if isinstance(value,dict) and isinstance(value.get('program_spec'),dict):
                        spec=value['program_spec'];io=probe.interface(spec['probe_kind'])
                        for key in ('input_schema','output_schema'):errors+=differences(io[key],spec.get(key),'program_spec.'+key)
                        errors+=c.schema_errors(spec.get('source_input'),io['input_schema'],'program_spec.source_input')
                    try:old.validate_spec(value,read(root/'selected_opportunities.json')[[o['opportunity_id'] for o in read(root/'selected_opportunities.json')].index(row['opportunity_id'])])
                    except ValueError:pass
                    else:raise ValueError('Old rejected proposal unexpectedly passed its original validator')
                else:
                    try:c.validate_source(value)
                    except c.ContractError as exc:errors=exc.diagnostics
            if not errors:raise ValueError('No exact field diagnostic for '+str(path))
            rejected.append({'unit':path.parent.name,'request_id':r['id'],'request_file':str(folder/'requests.json'),
                'request_sha256':sha256(folder/'requests.json'),'raw_arguments':original,'diagnostics':errors,
                'old_state':'model_invalid','old_result_unchanged':True})
        if (folder/'probe.json').exists():
            checked=read(folder/'probe.json');bundle=row.get('bundle',{});program=bundle.get('program')
            if checked.get('probe_passed'):
                spec=row['proposal']['program_spec'];kind=spec['probe_kind']
                fp=digest({'kind':kind,**probe.interface(kind),'semantics':probe.semantics(kind),'proof_scope':probe.LIMITATION})
                family=families.setdefault(fp,{'kind':kind,'members':[]})
                family['members'].append({'program_id':program['id'],'source_case_ids':bundle['atomic']['source_case_ids'],
                    'normalized_source_hash':digest(ast.dump(ast.parse(program['source']),include_attributes=False)),
                    'original_receipts_sha256':sha256(folder/'probe.json')})
            elif program and checked['state']=='probe_failed':
                records=[r for r in checked['receipts'] if not r['passed'] and r['role']!='missing_premise']
                for saved in records:
                    replayed=probe.execute(worker or ProgramWorker(read(root/'config.json')['program_worker']),program,
                        saved['arguments'],output/'old_worker_replay',saved['role'])
                    negative.append({'source':str(folder/'probe.json'),'source_sha256':sha256(folder/'probe.json'),
                        'original_expected':saved.get('expected_local_outputs'),'original_result':saved['result'],
                        'replayed':replayed,'unchanged_original_failure':not replayed['passed']})
    after=tree_identity(root)
    if before!=after:raise RuntimeError('Read-only P5 audit changed historical files')
    counts=Counter(e['diagnostics'][0]['code'] for e in rejected)
    result={'version':c.VERSION,'new_model_http':0,'historical_tree_before':before,'historical_tree_after':after,
        'rejected_count':len(rejected),'diagnostic_counts':dict(counts),'rejected':rejected,
        'old_capability_families':families,'old_passed_program_count':sum(len(g['members']) for g in families.values()),
        'worker_negative_replay':negative,'new_train_assets_imported':0}
    write_json(output/'p5_failure_replay.json',result);return result
