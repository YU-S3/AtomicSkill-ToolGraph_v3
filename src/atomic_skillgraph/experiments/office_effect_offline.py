"""Zero-HTTP production regression and real-corpus Docker probes, never learned assets."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from ..empirical.checkpoint import save_json
from ..empirical.contracts import PublicTask, digest, object_schema, validate_program
from ..empirical.office_effect_handoff import observed_hits, bind_read, recover_handoffs
from ..empirical.office_operation_mining import read, sha
from ..empirical.program_worker import ProgramWorker
from ..empirical.task_context import TaskContext
from ..harness.benchmarks import OfficeAdapter
from ..harness.simple_protocol import Broker
from .run_office_effect_v6 import checked, accounting
from .formal_log import utc

# Engineering fixture only. It is never inserted into any candidate or published Bank.
READ_FIXTURE = '''def run(ctx, inputs):
    result = ctx.call('read_context', inputs)
    if not result.get('accepted'): return {'status':'blocked','outputs':{},'detail':'read rejected'}
    return {'status':'ok','outputs':{'text':result['data']['text'],'source_path':inputs['path'],'source_offset':inputs['offset']}}
'''


def docker_probe(root, config, corpus_identity):
    cfg = deepcopy(config); cfg['experiment']['output_dir'] = str(root/'zero_http/worker')
    task = PublicTask('engineering_v6','engineering_v6','Read a current real search hit',{},'train')
    adapter = OfficeAdapter({task.task_id:{}},cfg); adapter.reset(task)
    broker = Broker(adapter,48,context=TaskContext(cfg['runtime']))
    try:
        # Reproduce the recorded V5 scope, then derive the new read position from a real hit.
        found = broker.call('grep_page',{'pattern':'surplus','paths':['treasury_bulletin_1960_12.txt']})
        if not found.get('accepted'): raise RuntimeError('Real corpus search failed')
        ref = observed_hits(broker.events[-1],corpus_identity)[0]
        binding = bind_read(ref,{'before':100,'after':1200},broker.events,corpus_identity)
        missing = bind_read(ref,{},broker.events,corpus_identity)
        program = {'source':READ_FIXTURE,'entry':'run',
            'input_schema':object_schema({'path':{'type':'string'},'offset':{'type':'integer'},
                'before':{'type':'integer'},'after':{'type':'integer'}},['path','offset','before','after']),
            'output_schema':object_schema({'text':{'type':'string'},'source_path':{'type':'string'},'source_offset':{'type':'integer'}},
                ['text','source_path','source_offset']), 'allowed_tools':['read_context'],
            'environment':cfg['program_environment'],'result_role':'intermediate'}
        program['id'] = validate_program(program)
        worker = ProgramWorker({**cfg['program_worker'],'record_execution_receipt':True})
        receipts = []
        for name, args, expected in [('same_hit',binding['arguments'],'ok'),
                ('scope_escape',{**binding['arguments'],'path':'../outside.txt'},'blocked')]:
            intent = root/'zero_http'/(name+'.intent.json'); output=root/'zero_http'/(name+'.json')
            identity = {'program_sha256':digest(program),'inputs':args}
            if output.exists():
                row=read(output)
                if row['identity']!=identity:raise ValueError('Worker probe identity changed')
            else:
                if intent.exists():raise RuntimeError('Worker started without receipt; no automatic replay')
                save_json(intent,identity)
                result=worker.execute(program,args,broker)
                row={'identity':identity,'result':result,'tools':deepcopy(broker.events),'expected':expected,
                    'passed':result['status']==expected,'http':0,'engineering_fixture_not_asset':True}
                save_json(output,row)
            receipt=row['result'].get('execution_receipt',{})
            if (not row['passed'] or '--network=none' not in receipt.get('command',[])
                    or receipt.get('image_digest')!=cfg['program_environment']['image_digest']):
                raise RuntimeError('Actual Docker/ABI isolation probe failed')
            receipts.append({'name':name,'sha256':sha(output),'status':row['result']['status']})
        return {'passed':True,'http':0,'binding':binding,'missing':missing,'receipts':receipts}
    finally:adapter.close()


def check(root):
    plan,cfg=checked(root,corpus=True)
    repository=Path(__file__).parents[3]
    tests=['tests/test_office_effect_v6.py','tests/test_office_trace_v5_contract.py','tests/test_office_trace_v3.py',
           'tests/test_office_interface_closure.py','tests/test_budget_revision.py','tests/test_empirical_checkpoint.py']
    tests=[p for p in tests if (repository/p).exists()]
    output=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=repository,capture_output=True,text=True)
    (root/'offline_tests.log').write_text(output.stdout+output.stderr)
    if output.returncode:raise RuntimeError('Offline production regression failed; see offline_tests.log')
    catalog=read(root/'induction.json');split=read(root/'partition.json')
    if {r['case_id'] for r in catalog}!=set(split['induction']):raise ValueError('Induction isolation violated')
    old=Path(plan['v5_run'])
    # Actual historical R2 mismatch: the source card IDs come from its natural proposal receipt.
    decision=read(old/'decisions/0.json'); candidate=decision.get('candidate',{})
    bindings=candidate.get('program_spec',{}).get('input_bindings',{})
    ids={b.get('span_id') for b in bindings.values() if isinstance(b,dict) and b.get('span_id')}
    spans=[]
    with (old/'source_span_catalog.jsonl').open() as stream:
        for line in stream:
            span=json.loads(line)
            if span['id'] in ids:spans.append(span)
    historical=recover_handoffs(spans,plan['corpus_identity']['sha256'],binding_claimed=True)
    if not any(r['reason']=='cross_file_handoff_invalid' for r in historical):
        raise RuntimeError('The actual V5 R2 cross-file counterexample was not reproduced')
    worker=docker_probe(root,cfg,plan['corpus_identity']['sha256'])
    passed={'passed':True,'http':0,'created_at':utc(),'manifest_sha256':sha(root/'manifest.json'),
        'test_log_sha256':sha(root/'offline_tests.log'),'tests':output.stdout.strip(),
        'partition_counts':{k:len(split[k]) for k in ('induction','edit_feedback','train_gate')},
        'historical_cross_file':historical,'docker':worker,'accounting':accounting(root),
        'paid_ready':not accounting(root)['blocks'],'claim':'Engineering checks only; no model utility has been established'}
    checked(root,corpus=True)
    save_json(root/'zero_http_check.json',passed);return passed
