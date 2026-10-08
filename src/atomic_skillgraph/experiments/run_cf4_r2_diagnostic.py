"""One predeclared CF4 R2 diagnostic; never resumes the formal matrix."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import re
import shutil

from ..empirical import IMPLEMENTATION_REVISION
from ..empirical.answer_status import classify_answer
from ..empirical.bank import Bank
from ..empirical.bank_view import BankView
from ..empirical.budget_governor import BudgetGovernor
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import PublicTask, digest, validate_schema_instance
from ..empirical.finish_evidence import build_finish_evidence, finalize_text
from ..empirical.program_submission import submission_contract, validate_program_declaration
from ..empirical.program_worker import program_permission_view
from ..empirical.prompts import BUILD, BUILDER_PROMPT
from ..empirical.system import EmpiricalSystem, validate_config
from ..empirical.task_context import TaskContext
from ..harness.registry import create_simple_harness
from .formal_log import tree_identity, utc
from .recover_empirical import read, rows
from .run_empirical import code_identity, load_env, write_json


CANDIDATES = {
    'spreadsheetbench': {'program_id':'program_fc9a9078cbe86007e2f23a20044d72a45d921021f709780e8c4d9df8a76ea800',
                        'task_ids':['spreadsheet:36764','spreadsheet:46167']},
    'officeqa': {'program_id':'program_a6f6161109ed6fa341806fa10f60bae09f85590474d7f11baca869abfb5fc2f0',
                 'task_ids':['officeqa:UID0060','officeqa:UID0059']}}


def applicable(benchmark, task):
    """Predeclared public-only coverage test; no evaluator or Program invocation."""
    goal = task['goal'].lower()
    if benchmark == 'spreadsheetbench':
        return 'formula' in goal and 'sum' in goal and any(w in goal for w in ('criterion','criteria','match','product id'))
    return 'compound annual growth' in goal and 'old-age' in goal and any(w in goal for w in ('appropriation','expenditure transfer'))


def settings(parent, output, bank, *, readonly=False, finish_override=False):
    config = deepcopy(parent)
    config['data_dir'] = str(bank)
    config['experiment'].update(output_dir=str(output), implementation_revision=IMPLEMENTATION_REVISION,
                                runtime_mode='frozen' if readonly else 'online')
    config['llm']['max_retries'] = 0
    if finish_override:
        config['llm']['purpose_overrides'] = {'finish_only': {'protocol': {'thinking_type':'disabled'}, 'max_completion_tokens':512}}
    return validate_config(config)


def public_bindings(benchmark, bank, candidate):
    assets = [a for a in bank.all('implementation') if a['program_id'] == candidate['program_id']]
    skill = bank.get(assets[0]['skill_id'])
    job = next(j for j in bank.jobs() if j['skill_id'] == skill['id'])
    cases = {c['task']['task_id']: c for c in bank.train_cases() if c['task']['task_id'] in candidate['task_ids']}
    bindings = []
    for tid in candidate['task_ids']:
        case = cases[tid]
        binding = deepcopy(next(b for b in job['case_bindings'] if b['case_id'] == case['task']['physical_key']))
        if benchmark == 'spreadsheetbench' and tid.endswith('36764'):
            binding['inputs'].update(target_sheet='Sheet1',data_sheet='Sheet2',criteria_column='A',
                                    criteria_cell='A2',sum_column='R',data_first_row=2,data_last_row=363)
        if benchmark == 'spreadsheetbench' and tid.endswith('46167'):
            binding['inputs'].update(target_sheet='Sheet1',data_sheet='Sheet1',criteria_column='A',
                                    criteria_cell='A2',sum_column='B',data_first_row=2,data_last_row=12)
        if benchmark == 'officeqa' and tid.endswith('0059'):
            binding['inputs']['line_item_label'] = 'Expenditure transfers to Federal Old-Age and Survivors Insurance Trust Fund'
        validate_schema_instance(binding['inputs'], skill['input_schema'])
        bindings.append(binding)
    return skill, job, cases, bindings


def prepare(review, output):
    review, output = Path(review).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('A diagnostic requires a fresh output directory')
    root = review/'runs'
    manifest = {'schema':'cf4-r2.diagnostic.v1', 'source':code_identity(), 'review_source':str(review),
        'created_at':utc(),'strategy':'historical_on_new_guidance_off',
        'budget':{'token_limit':5000000,'finish_reserve':500000,'request_limit':205},
        'D1':{},'D2':{'task_ids':['officeqa:UID0115','officeqa:UID0148'],'max_http':2}, 'D3':{},
        'D4':{'predicate_source':inspect.getsource(applicable),'predicate_hash':digest(inspect.getsource(applicable)),
              'tie_rule':'sha256("cf4-r2|" + benchmark + "|" + task_id)', 'max_http_per_episode':40},
        'stop':'D0-D4 only; no formal tail, no Test, no replacement samples or retries'}
    for benchmark,candidate in CANDIDATES.items():
        runroot=root/benchmark/'seed42'
        dataset=Path(read(runroot/'train/config.json')['harness']['evaluator_records']).parent
        public_path=dataset/'val.json'
        manifest['D4'].setdefault('public_sources',{})[benchmark]={'path':str(public_path),'sha256':digest(read(public_path))}
        bank=Bank(runroot/'train/bank',readonly=True)
        try:
            skill,job,cases,bindings=public_bindings(benchmark,bank,candidate)
            manifest['D1'][benchmark]={**candidate, 'parent_job_id':job['id'], 'skill_id':skill['id'],
                'bank_digest':bank.digest(),'bank_tree':tree_identity(bank.root), 'bindings':bindings,
                'human_assisted_diagnostic':True, 'max_builder_http':1,
                'max_pure_trials':3 if benchmark=='spreadsheetbench' else 2}
        finally: bank.close()
    for benchmark in ('searchqa','livemath'):
        runroot=root/benchmark/'seed42'
        tasks=read(runroot/'val/execution_manifest.json')['tasks']
        manifest['D3'][benchmark]={'task_ids':[t['task_id'] for t in tasks], 'max_http':len(tasks),
            'bank_tree':tree_identity(runroot/'train/frozen_bank'), 'source_run_manifest':digest(read(runroot/'run_manifest.json'))}
    write_json(output/'diagnostic_manifest.json',manifest)
    return manifest


def system_for(config, *, governor, readonly=False, view=None):
    return EmpiricalSystem(config,readonly=readonly,adapter_factory=lambda:create_simple_harness(config),
        bank_view_factory=(lambda b:BankView(b,view)) if view else None,budget_governor=governor)


def d1(benchmark, manifest, root, governor):
    declared=manifest['D1'][benchmark]; source=Path(manifest['review_source'])/'runs'/benchmark/'seed42'
    destination=root/'D1'/benchmark; shutil.copytree(source/'train/bank',destination/'bank')
    config=settings(read(source/'train/config.json'),destination,destination/'bank')
    system=system_for(config,governor=governor)
    result={'human_assisted_diagnostic':True,'trials':[],'parent_job_id':declared['parent_job_id'],
            'parent_program_id':declared['program_id'],'builder_http':0}
    try:
        skill,job,cases,_=public_bindings(benchmark,system.bank,declared)
        bindings=declared['bindings']; program=system.bank.get(declared['program_id'])
        taskmap={c['task']['physical_key']:(PublicTask(**c['task']),c['experience']) for c in cases.values()}
        system.adapter.reset(next(iter(taskmap.values()))[0]); system.task_context=TaskContext(system.config['runtime'])
        system.checkpoint=TaskCheckpoint(destination/'checkpoint'); system.audit_path=destination/'requests.json'
        system.budget_scope='diagnostic-builder:'+benchmark; system.phase='learning'
        system.request_attribution={'parent_task_id':declared['task_ids'][0], 'diagnostic_phase':'D1'}
        def trial(asset,binding):
            trial_case=system.learner._preflight_binding(binding,skill,taskmap)
            task=taskmap[binding['case_id']][0]
            record=system.test_program(asset,binding['inputs'],task,trial_id=digest([asset['id'],binding,'cf4-r2']),
                trial_case=trial_case,continuation=False)
            result['trials'].append(record)
            write_json(destination/'result.json',result)
            return record
        if benchmark=='spreadsheetbench':
            original=trial(program,bindings[1])
            if original['outcome']=='positive':
                result['program_id']=program['id']; result['repair_skipped']='Old version succeeded on the independent second task'
                result['usable']=system.bank.get(program['id'])['state']=='usable'
                system.bank.freeze(destination/'diagnostic_snapshot')
                return result
        else:
            # Public semantic audit uses already obtained grep rows, never the evaluator record.
            task_goals=[c['task']['goal'].lower() for c in cases.values()]
            semantic=all('compound annual growth' in g and '1947' in g for g in task_goals)
            tools=[e for a in system.bank.attempts(program['id']) for e in a.get('tools',[]) if e.get('state')=='finished']
            text=json.dumps([e.get('result') for e in tools],ensure_ascii=False)
            strings=[]
            def collect(value):
                if isinstance(value,str): strings.append(value)
                elif isinstance(value,dict):
                    for v in value.values(): collect(v)
                elif isinstance(value,list):
                    for v in value: collect(v)
            for e in tools: collect(e.get('result',{}))
            normalized=[''.join(re.findall('[a-z0-9]+',s.lower())) for s in strings]
            lexical={'old_suffix_matches':sum('appropriationstoffederaloldageandsurvivorsinsurancetrustfund' in s for s in normalized),
                     'correct_suffix_matches':sum('appropriationstofederaloldageandsurvivorsinsurancetrustfund' in s for s in normalized),
                     'transfer_row_matches':sum('expendituretransfer' in s and 'oldage' in s for s in normalized)}
            result['semantic_audit']={'same_parameterized_calculation':semantic,'public_event_count':len(tools),
                'different_line_item_labels':True,'public_result_digest':digest(text), **lexical}
            if not semantic:
                result.update(status='not_applicable',usable=False); return result
        permissions=program_permission_view(system.adapter.tool_definitions(),
            [*system.adapter.available_tools(),system.task_context.tool()],tool_surface=system.adapter.capabilities.tool_surface,
            workspace=getattr(system.adapter,'workspace',None),environment=config['program_environment'])
        examples=[{'case_id':b['case_id'],'fixed_binding':b,'history_domain':'runtime_agent_operations',
                   **system.learner._view(taskmap[b['case_id']][1])} for b in bindings]
        failures=result['trials'] or system.bank.attempts(program['id'])
        material=system.learner.builder_material(skill,bindings,examples,permissions,
            {'domain':'program_trial','source':program['source'],'errors':failures,
             'public_interface_findings': 'Preserve runtime bindings in the independently executed solution entry.' if benchmark=='spreadsheetbench'
             else 'Derive row matching from line_item_label and table_title_pattern; respect distinct accounting rows and units.'})
        generated=system.agent('tool_builder',BUILDER_PROMPT,material,'submit_program',BUILD,
            repair_limit=0,owner_state_version='diagnostic-revision',job_key=['cf4-r2',job['id']],repair_reason='execution')
        result['builder_http']=1
        if generated['trial_inputs'] != bindings: raise ValueError('Diagnostic Builder changed the fixed public bindings')
        asset={**{k:program[k] for k in ('entry','backend','input_schema','output_schema','allowed_tools','environment','result_role','entry_constraints') if k in program},
               'source':generated['source']}
        asset=validate_program_declaration(asset,submission_contract(system.adapter))
        saved=system.bank.put('program',asset)
        system.bank.put('implementation',{'skill_id':skill['id'],'program_id':saved['id']})
        result['program_id']=saved['id']
        for binding in bindings: trial(saved,binding)
        result['usable']=system.bank.get(saved['id'])['state']=='usable'
        system.bank.freeze(destination/'diagnostic_snapshot')
        result['snapshot_tree']=tree_identity(destination/'diagnostic_snapshot')
        return result
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),usable=False)
        return result
    finally:
        result['bank_digest']=system.bank.digest()
        write_json(destination/'result.json',result); system.close()


def d2(task_id, manifest, root, governor):
    source=Path(manifest['review_source'])/'runs/officeqa/seed42'
    state=read(source/'train/checkpoints'/task_id/'1/state.json')['executor_state']
    events=read(source/'train/checkpoints'/task_id/'1/native_events.json')
    context=TaskContext()
    for k,v in state['context'].items(): setattr(context,k,v)
    hashes={digest(e['result']) for e in events if e['state']=='finished'}
    allowed={rid for rid,value in context.results.items() if digest(value) in hashes}
    requests=read(source/'train/requests'/(task_id+'_attempt1.json'))['requests']
    original=next(r for r in reversed(requests) if r.get('purpose')=='finish_only')
    prompt=original['messages'][0]['content']; material=json.loads(original['messages'][1]['content'])
    evidence=build_finish_evidence(context,material,prompt,allowed_result_ids=allowed)
    destination=root/'D2'/task_id
    write_json(destination/'evidence.json',evidence)
    if evidence['unresolved_refs']: return {'status':'missing_evidence','unresolved_refs':evidence['unresolved_refs']}
    config=settings(read(source/'train/config.json'),destination,source/'train/bank',readonly=True,finish_override=True)
    system=system_for(config,governor=governor,readonly=True)
    try:
        system.adapter.reset(PublicTask(**read(source/'train/traces'/(task_id+'.json'))['task']))
        system.task_context=context; system.checkpoint=TaskCheckpoint(destination/'checkpoint')
        system.audit_path=destination/'requests.json'; system.phase='diagnostic_finish'; system.budget_scope=task_id
        system.request_attribution={'parent_task_id':task_id,'diagnostic_phase':'D2'}
        answer,audit=finalize_text(system.agent,context,material,prompt,owner_state_version='cf4-r2-new-finish',allowed_result_ids=allowed)
        return {'status':'completed','answer':answer,'finish_evidence':audit,
                **classify_answer(answer,system.requests[-1]['response']['finish_reason']),
                'tokens':sum(e.total_tokens for e in system.usage.events),'old_score_unchanged':True}
    except Exception as exc: return {'status':'failed','error_type':type(exc).__name__,'error':str(exc),'old_score_unchanged':True}
    finally: system.close()


def d3(benchmark, manifest, root, governor):
    source=Path(manifest['review_source'])/'runs'/benchmark/'seed42'
    tasks=read(source/'val/execution_manifest.json')['tasks']; result=[]
    destination=root/'D3'/benchmark
    config=settings(read(source/'val/config.json'),destination,source/'train/frozen_bank',readonly=True)
    system=system_for(config,governor=governor,readonly=True,view='guidance_off')
    try:
        for entry in tasks:
            task=PublicTask(**entry); original=read(source/'val/traces'/(task.task_id+'.json'))
            item={'task_id':task.task_id,'on_score':original['score'],'on_tokens':sum(e['total_tokens'] for e in original['usage'])}
            if governor.state['unknown_billing']:
                item.update(status='pair_incomplete',error='unknown_billing_stop')
                result.append(item); write_json(destination/'results.json',result)
                continue
            system.checkpoint=TaskCheckpoint(destination/'checkpoints'/task.task_id)
            system.audit_path=destination/'requests'/(task.task_id+'.json')
            system.request_attribution={'parent_task_id':task.task_id,'diagnostic_phase':'D3'}
            try:
                trace=system.run_task(task,learn=False)
                write_json(destination/'traces'/(task.task_id+'.json'),trace)
                item.update(status='paired',off_score=trace['score'],off_tokens=sum(e['total_tokens'] for e in trace['usage']),
                            answer_status=trace['execution'].get('answer_status'))
            except Exception as exc:
                item.update(status='not_run_budget_limit' if getattr(exc,'code',None)=='diagnostic_budget_exhausted' else 'pair_incomplete',
                            error_type=type(exc).__name__,error=str(exc))
            result.append(item); write_json(destination/'results.json',result)
        return result
    finally: system.close()


def verify_d3_wire(manifest):
    checks=[]
    for benchmark in manifest['D3']:
        source=Path(manifest['review_source'])/'runs'/benchmark/'seed42'
        config=settings(read(source/'val/config.json'),Path('/diagnostic'),source/'train/frozen_bank',readonly=True)
        system=system_for(config,governor=None,readonly=True)
        try:
            for entry in read(source/'val/execution_manifest.json')['tasks']:
                task=PublicTask(**entry); system.adapter.reset(task)
                old=read(source/'val/requests'/(task.task_id+'_attempt1.json'))['requests'][0]
                guidance=system.bank.retrieve_guidance(task.goal,limit=3)
                material={'goal':task.goal,'inputs':task.inputs,'guidance':[
                    {'skill_id':a['id'],'goal':a['goal'],'guidance':a['guidance']} for a in guidance],'content_parts':[]}
                prompt='Answer the question once using only the public input and any supplied guidance. '+system.adapter.answer_contract()
                messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(material,ensure_ascii=False,allow_nan=False)}]
                original=old['http_attempts'][0]['final_payload_audit']
                if messages != original['messages'] or original.get('tools',[]) != []:
                    raise ValueError('Historical D3 wire differs beyond guidance: '+task.task_id)
                provider=system.provider('runtime','runtime')
                payload=provider._build_payload(messages,[])
                if payload[provider.config.token_limit_field] != old['completion_cap']:
                    raise ValueError('Historical completion cap changed')
                checks.append({'task_id':task.task_id,'on_wire_sha256':digest(payload),'completion_cap':old['completion_cap'],
                               'only_allowed_intervention':'guidance=[]','generation_seed':'unsupported/null'})
        finally: system.close()
    return checks


def d4(benchmark, candidate, manifest, root, governor, *, select_only=False):
    if not candidate.get('usable'): return {'status':'not_applicable','reason':'Current candidate did not form a verified executable capability'}
    source=Path(manifest['review_source'])/'runs'/benchmark/'seed42'
    snapshot=root/'D1'/benchmark/'diagnostic_snapshot'
    public_source=manifest['D4']['public_sources'][benchmark]
    if digest(read(public_source['path'])) != public_source['sha256']: raise ValueError('D4 public material changed')
    public=read(public_source['path'])['tasks']
    eligible=[t['task_id'] for t in public if applicable(benchmark,t)]
    if not eligible: return {'status':'not_applicable','reason':'No public eligible Val task'}
    import hashlib
    chosen=min(eligible,key=lambda tid:hashlib.sha256(('cf4-r2|'+benchmark+'|'+tid).encode()).hexdigest())
    selection={'task_id':chosen,'eligible_ids':eligible,'snapshot_tree':tree_identity(snapshot),
               'predicate_hash':manifest['D4']['predicate_hash']}
    write_json(root/'D4'/benchmark/'selection.json',selection)
    if select_only: return {'status':'selected',**selection}
    # Runtime material uses the same canonical PublicTask, not the selector's observations.
    materialized=read(source/'val/execution_manifest.json')['tasks'] if (source/'val/execution_manifest.json').exists() else None
    if materialized is None:
        config=read(source/'train/config.json')
        dataset=Path(config['harness']['evaluator_records']).parent
        materialized=read(dataset/'val.json')['tasks']
    task=PublicTask(**next(t for t in materialized if t['task_id']==chosen))
    arms={}
    for arm in ('on','off'):
        destination=root/'D4'/benchmark/arm
        config=settings(read(source/'train/config.json'),destination,snapshot,readonly=True)
        config['harness']['evaluator_records']=config['harness']['evaluator_records'].replace('evaluator_records_train.json','evaluator_records_val.json')
        system=system_for(config,governor=governor,readonly=True,view='learned_assets_off' if arm=='off' else None)
        system.checkpoint=TaskCheckpoint(destination/'checkpoint'); system.audit_path=destination/'requests.json'
        system.request_attribution={'parent_task_id':chosen,'diagnostic_phase':'D4','episode_scope':benchmark+':'+arm,'episode_request_limit':40}
        try:
            trace=system.run_task(task,learn=False); write_json(destination/'trace.json',trace)
            arms[arm]={'status':'completed','score':trace['score'],'tokens':sum(e['total_tokens'] for e in trace['usage']),
                       'program_attempts':trace['execution'].get('attempts',[])}
        except Exception as exc: arms[arm]={'status':'failed','error_type':type(exc).__name__,'error':str(exc)}
        finally: system.close()
    if tree_identity(snapshot) != selection['snapshot_tree']: raise RuntimeError('D4 modified its frozen snapshot')
    return {'status':'completed','selection':selection,'arms':arms}


def execute(manifest_path):
    manifest_path=Path(manifest_path); manifest=read(manifest_path); root=manifest_path.parent
    if manifest['source'] != code_identity(): raise ValueError('Diagnostic source identity changed')
    for benchmark,declared in manifest['D1'].items():
        bank_path=Path(manifest['review_source'])/'runs'/benchmark/'seed42/train/bank'
        if tree_identity(bank_path) != declared['bank_tree']: raise ValueError('D1 source Bank changed')
    for benchmark,declared in manifest['D3'].items():
        bank_path=Path(manifest['review_source'])/'runs'/benchmark/'seed42/train/frozen_bank'
        if tree_identity(bank_path) != declared['bank_tree']: raise ValueError('D3 source Frozen changed')
    if (root/'diagnostic_results.json').exists() or (root/'STARTED.json').exists():
        raise ValueError('This fixed diagnostic has already started; paid slots cannot reset')
    checks=verify_d3_wire(manifest); write_json(root/'D3_wire_checks.json',checks)
    write_json(root/'STARTED.json',{'time':utc(),'manifest_hash':digest(manifest)})
    governor=BudgetGovernor(root/'budget_ledger.json',**manifest['budget'])
    result={'D0':'offline checks and zero-provider recovery reported separately','D1':{},'D2':{},'D3':{},'D4':{}}
    try:
        for benchmark in CANDIDATES:
            result['D1'][benchmark]={'status':'not_run_unknown_billing','usable':False} if governor.state['unknown_billing'] else d1(benchmark,manifest,root,governor)
            write_json(root/'diagnostic_results.json',result)
        manifest['D4']['selections']={benchmark:d4(benchmark,result['D1'][benchmark],manifest,root,governor,select_only=True)
                                    for benchmark in CANDIDATES}
        write_json(manifest_path,manifest)
        for tid in manifest['D2']['task_ids']:
            result['D2'][tid]={'status':'not_run_unknown_billing'} if governor.state['unknown_billing'] else d2(tid,manifest,root,governor)
            write_json(root/'diagnostic_results.json',result)
        for benchmark in manifest['D3']:
            result['D3'][benchmark]=[{'task_id':tid,'status':'pair_incomplete','error':'unknown_billing_stop'}
                for tid in manifest['D3'][benchmark]['task_ids']] if governor.state['unknown_billing'] else d3(benchmark,manifest,root,governor)
            write_json(root/'diagnostic_results.json',result)
        for benchmark in CANDIDATES:
            result['D4'][benchmark]={'status':'not_run_unknown_billing'} if governor.state['unknown_billing'] else d4(benchmark,result['D1'][benchmark],manifest,root,governor)
    finally:
        result['budget']=governor.state; result['ended_at']=utc(); result['formal_tail_started']=False
        write_json(root/'diagnostic_results.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-review'); parser.add_argument('--output'); parser.add_argument('--manifest'); parser.add_argument('--env-file')
    args=parser.parse_args(argv)
    if args.prepare_review:
        if not args.output: parser.error('--output required with --prepare-review')
        prepare(args.prepare_review,args.output)
    else:
        if not args.manifest: parser.error('--manifest required')
        if args.env_file: load_env(args.env_file)
        execute(args.manifest)


if __name__=='__main__': main()
