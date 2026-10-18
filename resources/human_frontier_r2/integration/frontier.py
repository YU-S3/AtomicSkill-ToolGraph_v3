"""Manual-bank bridge for AtomicSkillGraph. No learning/promotion gates.

Uses the user's native Bank, Provider, Broker, ProgramWorker, Executor and scorer.
Changes are scoped to this runner process; original repo files are never rewritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any

VERSION = 'human.frontier.runtime.r2.20261016'
KINDS = ('skill', 'program', 'implementation', 'workflow')


def read(path):
    p = Path(path)
    if p.suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        return yaml.safe_load(p.read_text(encoding='utf-8'))
    return json.loads(p.read_text(encoding='utf-8'))


def write(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(p)


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def use_repo(root):
    root=Path(root).expanduser().resolve()
    if not (root/'src/atomic_skillgraph/empirical/bank.py').is_file():
        raise ValueError('--repo must be the actual AtomicSkill-ToolGraph_v3 checkout')
    sys.path.insert(0,str(root/'src'))
    from atomic_skillgraph.experiments.run_empirical import run
    if 'bank_view_factory' not in inspect.signature(run).parameters:
        raise RuntimeError('This checkout lacks the existing bank_view_factory entry point')
    return root


def active(selector, text, benchmark):
    t=text.lower()
    if selector=='always':return True
    rules={
        'person':r'\b(who|man|woman|king|queen|author|scientist|company|organization|wife|husband|president|actor)\b',
        'quotation':r'["“”]|\b(book|novel|film|movie|song|poem|quote|title|wrote|writ|play|changed)\w*',
        'lexical_or_multiple':r'\b(two|three|both|these|called|word|term|name|define|means|blank)\b|\(\d+\)',
        'series':r'\b(each|annual|monthly|between|series|average|mean|trend|growth|years|months|correlation|regression|variance|deviation)\b',
        'calculation':r'\b(sum|total|average|mean|median|ratio|difference|percent|calculate|compute|growth|gini|deviation|variance|change|correlation|regression|coefficient)\b',
        'treasury':r'\b(treasury|bill|bond|note|tender|offered|accepted|subscription|refund|cash|security|securities|quotation)\w*',
        'units':r'\b(currency|exchange|rate|dollar|yen|pound|franc|rupee|million|billion|thousand|convert|nominal|real)\w*',
        'meta':r'stronger.*(?:result|statement)|(?:result|statement).*stronger|none.*strongest',
        'asymptotic':r'\b(bound|estimate|asymptotic|limit|convergence|uniform|rate|epsilon|exponent)\w*|\\(?:lim|epsilon)',
        'logical':r'\b(exist|unique|equivalen|implies|sufficient|necessary|every|all|counterexample|converse|congruen)\w*|\\(?:forall|exists)',
    }
    return bool(re.search(rules.get(selector,r'(?!)'),t))


def task_text(task):
    return task.goal+'\n'+'\n'.join(str(c.get('text','')) for c in task.inputs.get('choices',[]) if isinstance(c,dict))


class HumanBankView:
    """The author, not a learned qualification receipt, authorizes this isolated view."""
    force_planner=False
    def __init__(self,bank,bundle,arm='manual',programs='off',graph=None):
        if not bank.readonly:raise ValueError('Manual evaluation uses a read-only Bank')
        self.bank,self.bundle,self.arm,self.program_mode,self.graph=bank,bundle,arm,programs,graph
        self.benchmark=bundle['benchmark'];self.task=None
        self.variant=bundle.get('variant','legacy_humanfrontier')
        self.view_hash=sha([VERSION,bundle,arm,programs,graph])
        if bundle.get('assembly_order'):
            selected=[bank.get(k) for k in bundle['assembly_order']]
            if any(a is None or a.get('author_execution_permission')!='manual_eval' for a in selected):
                raise ValueError('Missing manual source segment')
            cursor=0
            for a in selected:
                span=a['source_segment']
                if span['start']!=cursor or span['end']<=cursor or span['end']-cursor!=len(a['guidance']):
                    raise ValueError('Non-contiguous source assembly')
                cursor=span['end']
            policy=''.join(a['guidance'] for a in selected)
            if hashlib.sha256(policy.encode()).hexdigest()!=bundle['raw_source_sha256']:
                raise ValueError('Assembled policy differs from original')
        self._members={a['id'] for k in KINDS for a in bank.all(k)}

    def __getattr__(self,name):return getattr(self.bank,name)
    def all(self,kind):return [] if self.arm=='off' else self.bank.all(kind)
    def get(self,key):return None if self.arm=='off' else self.bank.get(key)
    def bind_task(self,task):self.task=task
    def selected(self,task):
        if self.arm=='off':return []
        if self.bundle.get('assembly_order'):return [self.bank.get(k) for k in self.bundle['assembly_order']]
        return [a for a in self.bank.all('skill') if a.get('execution_intent')=='guidance_only'
                and active(a.get('applicability',{}).get('selector','always'),task_text(task),self.benchmark)]
    def policy(self,task):
        if self.bundle.get('assembly_order'):return ''.join(a['guidance'] for a in self.selected(task))
        return '\n\n'.join(a['goal']+': '+a['guidance'] for a in self.selected(task))
    def retrieve(self,query,limit=8):
        # A stored reasoning graph is compiled into the main prompt by default;
        # no additional LLM Planner call solely to retrieve manual guidance.
        return []
    def retrieve_guidance(self,*a,**kw):return []
    def planning_cards(self,*a,**kw):return []
    def select_guidance(self,*a,**kw):return {'selected':[],'audit':[]}
    def program_eligible(self,p):
        return bool(self.arm!='off' and self.program_mode!='off' and p and p.get('id') in self._members
                    and p.get('permission')=='author_trusted' and 'source' in p
                    and (self.bank.environment is None or p['environment']==self.bank.environment))
    def _show_programs(self):
        if self.arm=='off' or self.program_mode=='off':return False
        if self.program_mode=='on' or self.benchmark=='officeqa' or self.graph:return True
        if self.task is None:return False
        if self.benchmark=='searchqa':
            return len(self.task.inputs.get('context',''))>7500
        # Do not show toy Boolean-matrix tools for an arbitrary theorem question.
        # These two authored tools accept ordinary expressions/predicates, not truth labels.
        return bool(re.search(r'\b(?:compute|calculate|evaluate|counterexample|modulo|remainder|greatest common)\b',task_text(self.task),re.I)
                    or re.search(r'\d\s*[+*/%]\s*\d',task_text(self.task)))
    def program_options(self,*a,**kw):
        if not self._show_programs():return []
        excluded=set(kw.get('excluded',()))
        return [{'id':p['id'],'state':'callable_manual','execution_permission':'author_trusted',
            'goal':p.get('title','Authored local operation'),'input_schema':p['input_schema'],
            'output_schema':p['output_schema'],'result_role':p.get('result_role','intermediate')}
            for p in self.bank.all('program') if self.program_eligible(p) and p['id'] not in excluded]
    def routes(self,node,**kwargs):
        ids=[node['program_id']] if node.get('program_id') else [i['program_id'] for i in self.all('implementation') if i['skill_id']==node.get('skill_id')]
        return [self.get(pid) for pid in ids if self.program_eligible(self.get(pid))]


def native_import(bundle,destination,environment,office_method=None,seed=42):
    """Actual Bank.put, then a manual snapshot. No fake usable state or validation."""
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import program_digest
    root=Path(destination)
    target=deepcopy(bundle)
    identity={'bundle':sha(bundle),'environment':environment,'office_method':office_method,'seed':seed}
    if (root/'manual_manifest.json').exists():
        old=read(root/'manual_manifest.json')
        if old['identity']!=identity:raise ValueError('Use a new output folder for a changed manual Bank/config')
        return read(root/'imported_bank.json')
    if (root/'bank.sqlite3').exists():raise ValueError('Interrupted import: use a new output directory')
    bank=Bank(root,seed=seed,environment=environment);remap={}
    try:
        if office_method:bank.bind_office_method(office_method)
        for a in target['skill']:bank.put('skill',a)
        for p in target['program']:
            old=p['id'];p['environment']=deepcopy(environment);p.pop('local_validation',None)
            p['id']='program_'+program_digest(p)
            stored=bank.put('program',p);remap[old]=stored['id'];p.update(stored)
        for i in target['implementation']:
            i['program_id']=remap.get(i['program_id'],i['program_id']);bank.put('implementation',i)
        for w in target['workflow']:
            for n in w['nodes']:
                if n.get('program_id'):n['program_id']=remap.get(n['program_id'],n['program_id'])
            bank.put('workflow',w)
        bank.db.execute("INSERT OR REPLACE INTO metadata VALUES('manual_authority',?)",(VERSION,));bank.db.commit()
        manifest={'identity':identity,'native_bank_digest':bank.digest(),'authority':'author_trusted',
            'learned_promotion_required':False,'synthetic_validation_receipts':False,
            'counts':{k:len(bank.all(k)) for k in KINDS}}
        write(root/'imported_bank.json',target);write(root/'manual_manifest.json',manifest)
    finally:bank.close()
    return target


def configure(base,bundle,output,tasks_path,evaluator,corpus=None,token_budget=None,request_budget=None):
    """Fresh manual-only mode, preserving actual model and public adapter inputs."""
    from atomic_skillgraph.empirical import PROFILE,POLICY_DEFAULTS,OFFICE_TRACE_METHOD_VERSION,OFFICE_TRACE_POLICY,OFFICE_SOURCE_MATERIAL_VERSION
    from atomic_skillgraph.empirical.system import validate_config
    b=bundle['benchmark'];old=deepcopy(base);out=Path(output)
    if 'llm' not in old or not old['llm'].get('model'):raise ValueError('--config must contain the full real llm configuration')
    cfg={'mechanism_profile':PROFILE,'schema_version':'empirical.v1','data_dir':str(out/'bank'),
        'experiment':{**deepcopy(POLICY_DEFAULTS['experiment']),'seed':old.get('experiment',{}).get('seed',42),
                      'benchmark':b,'runtime_mode':'frozen','output_dir':str(out),'max_task_attempts':1},
        'harness':deepcopy(old.get('harness',{})),
        'runtime':{**deepcopy(POLICY_DEFAULTS['runtime']),'global_action_budget':48 if b=='officeqa' else 4,
                   'repeated_unchanged_failure_limit':2,'finish_input_max_bytes':65536,'dynamic_output_min_tokens':0},
        'learning':deepcopy(POLICY_DEFAULTS['learning']),
        'planning':{'retrieval_top_k':8,'structural_repair_limit':1,'task_replan_limit':1},
        'llm':deepcopy(old['llm']),'program_environment':deepcopy(old.get('program_environment') or bundle['program'][0]['environment']),
        'program_worker':deepcopy(old.get('program_worker',{})),
        'benchmark_profile':old.get('benchmark_profile',b+'.manual-public.v1'),'manifest':str(tasks_path)}
    cfg['harness'].update(adapter=b,evaluator_records=str(evaluator))
    cfg['program_environment']['adapter_abi']='simple.v2'
    cfg['program_worker']['record_execution_receipt']=True
    limits=old.get('budget',{})
    cfg['budget']={'token_limit':token_budget if token_budget is not None else limits.get('token_limit',5000000),
        'request_limit':request_budget if request_budget is not None else limits.get('request_limit',1000),
        'finish_reserve':0,'eval_task_tokens':192000 if b=='officeqa' else 128000,
        'train_task_tokens':192000 if b=='officeqa' else 128000}
    llm=cfg['llm'];llm.setdefault('max_retries',0)
    llm['purpose_overrides']={'learning_structure_repair':{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048},
        'finish_only':{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048}}
    runtime=llm.setdefault('runtime',{})
    runtime['max_completion_tokens']=32768 if b!='livemath' else 65536
    if b=='livemath':
        runtime.update(reasoning_effort='high',request_timeout_seconds=600,protocol={'thinking_type':'enabled'})
        llm['max_retries']=0
    runtime.setdefault('reasoning_effort','high');runtime.setdefault('request_timeout_seconds',600)
    runtime['max_total_tokens_per_task']=cfg['budget']['eval_task_tokens']
    for role in ('planner','extractor','tool_builder'):
        llm.setdefault(role,{}).setdefault('max_completion_tokens',8192)
    if b=='officeqa':
        corpus=corpus or cfg['harness'].get('corpus_root')
        if not corpus:raise ValueError('OfficeQA requires --corpus-root or harness.corpus_root in the config')
        cfg['harness']['corpus_root']=str(Path(corpus).expanduser().resolve())
        cfg['experiment']['officeqa_method_version']=OFFICE_TRACE_METHOD_VERSION
        cfg['learning'].update(material_version=OFFICE_SOURCE_MATERIAL_VERSION,source_program_payload_max_bytes=24576,builder_payload_max_bytes=24576)
        cfg['learning'].update(OFFICE_TRACE_POLICY)
        cfg['runtime'].update(finish_input_max_bytes=16384,dynamic_output_min_tokens=8192)
        runtime.update(max_completion_tokens=32768,reasoning_effort='high',protocol={'thinking_type':'enabled'})
    return validate_config(cfg)


def action_schema():
    from atomic_skillgraph.empirical.contracts import object_schema as obj
    strings={'type':'array','items':{'type':'string'}}
    return {'type':'object','oneOf':[
        obj({'action':{'const':'finish'},'answer':{'type':'string'},'used_results':strings},['action','answer']),
        obj({'action':{'const':'call_program'},'name':{'type':'string'},'arguments':{'type':'object'},
             'argument_refs':{'type':'object'},'used_results':strings},['action','name','arguments']),
        obj({'action':{'const':'read_result'},'result_id':{'type':'string'},'path':{'type':'array'},
             'offset':{'type':'integer','minimum':0},'limit':{'type':'integer','minimum':1}},['action','result_id','path'])]}


def compact_run(executor,task,broker,guidance,semantic):
    """Manual-only one-main-call path; at most two optional authored operations."""
    from atomic_skillgraph.empirical.contracts import validate_schema_instance
    from atomic_skillgraph.empirical.program_submission import normalize_program_result
    from atomic_skillgraph.empirical.answer_status import classify_answer
    from atomic_skillgraph.core.errors import BudgetExhausted
    s=executor.system;view=s.bank;view.bind_task(task);context=broker.context;checkpoint=s.checkpoint
    state=deepcopy(checkpoint.state.get('human_frontier',{})) if checkpoint else {}
    if state.get('finished'):return state['finished']
    if state.get('inflight'):raise RuntimeError('Interrupted HTTP/Worker: inspect saved requests; no automatic duplicate charge')
    attempts=state.get('attempts',[]);results=state.get('results',[]);reads=state.get('reads',[])
    for rid,result in state.get('raw_results',{}).items():context.results[rid]=result
    index=state.get('index',0)
    def save(inflight=False):
        state.update(index=index,attempts=attempts,results=results,reads=reads,
                     raw_results={rid:context.results[rid] for rid in results},inflight=inflight)
        if checkpoint:checkpoint.advance(checkpoint.state['stage'],human_frontier=deepcopy(state))
    def done(answer,reason):
        response=next((r.get('response',{}) for r in reversed(s.requests) if r.get('response')), {})
        value={'prediction':answer or '', 'reason':reason,'attempts':attempts,'temporary_executions':[],
               'native_calls':0,'internal_execution_count':len(attempts),'answer_protocol_version':VERSION,
               'answer_origin':'manual_compact_runtime','manual_skill_ids':[a['id'] for a in view.selected(task)],
               'workflow_mode':'prompt_compiled_not_separate_node_executions',
               **classify_answer(answer,response.get('finish_reason'),valid_format=s.adapter.valid_answer_format(answer))}
        state['finished']=value;save();s.executor.partial_execution=value;return value
    def visible_result(rid):
        raw=context.results[rid]
        return {'result_id':rid,**{k:deepcopy(raw[k]) for k in ('status','outputs','detail','error') if k in raw}}
    def finish_material():
        return {'goal':task.goal,'inputs':task.inputs,'local_results':[visible_result(r) for r in results], 'local_reads':reads[-2:]}
    def final_fallback(reason):
        nonlocal index
        if state.get('finalized'):return done('',reason)
        state['finalized']=True;index+=1;save(True)
        try:
            final_prompt='Finish the original question now. Use the best supported answer already available. '+s.adapter.answer_contract()
            final_input=finish_material()
            if getattr(s,'last_response_turn',None) is not None:
                from atomic_skillgraph.empirical.answer_executor import fit_reasoning_tail
                final_input=fit_reasoning_tail(s,final_input,final_prompt)
            a=s.agent('runtime',final_prompt,
                final_input,None,None,repair_limit=0,decision_purpose='finish_only',owner_state_version=VERSION+':final')
            save(False);return done(a,'bounded_final_'+reason)
        except (ValueError,BudgetExhausted) as exc:
            save(False)
            if isinstance(exc,ValueError) and not getattr(exc,'model_authored',False):raise
            return done('',reason+':'+str(exc))
    prompt=('Solve the supplied task. Use the attached authored strategies when relevant, not as factual evidence. '
            'Keep deliberation focused on the decisive facts or mathematical distinction. '+s.adapter.answer_contract())
    while index<5:
        cards=view.program_options(task.goal) if len(attempts)<2 else []
        material=finish_material()
        if cards:
            material['programs']=cards
            extra=(' You may answer directly as text. Only when an actual local computation helps, use answer_step '
             'action=call_program with a displayed name and arguments matching its input_schema. '
             'These human-authored Programs are callable now, with no learning/promotion prerequisite. '
             'Never invent input truth values to force a call. For finish, answer is the final text and '
             'used_results lists actually used result IDs. read_result requires an existing result_id and path. '
             'Do not invent other fields or ctx methods. You have at most two Program executions.')
        else:extra=' Answer directly; no tool calls are needed.'
        save(True)
        try:
            step=s.agent('runtime',prompt+extra,material,'answer_step' if cards else None,
                         action_schema() if cards else None,repair_limit=0,
                         completion_override=8192 if index else None,
                         owner_state_version=VERSION+':'+str(index),finish_request_material=finish_material())
        except ValueError as exc:
            save(False)
            if not getattr(exc,'model_authored',False):raise
            return final_fallback('format_or_empty')
        except BudgetExhausted:
            save(False);return final_fallback('budget')
        save(False);index+=1
        if not isinstance(step,dict):return done(step,'direct_answer') if step and s.adapter.valid_answer_format(step) else final_fallback('empty_answer')
        for a in attempts:
            if a['result_id'] in step.get('used_results',[]):a['outputs_consumed']=True
        if step.get('action')=='finish':
            answer=step.get('answer','')
            return done(answer,'agent_submitted') if answer and s.adapter.valid_answer_format(answer) else final_fallback('invalid_answer')
        if step.get('action')=='read_result':
            try:
                rid=step['result_id']
                if rid not in results or not step['path'] or step['path'][0]!='outputs':raise ValueError('Read an existing outputs field')
                value=context.read(rid,offset=step.get('offset',0),limit=step.get('limit'),path=step['path'])
                reads.append(value)
            except (ValueError,KeyError) as exc:reads.append({'error':str(exc)})
            save();continue
        try:
            p=view.get(step['name'])
            if not view.program_eligible(p):raise ValueError('Unknown or disabled authored Program ID')
            args=context.bind(step.get('arguments',{}),step.get('argument_refs',{}))
            if p.get('source_file','').endswith('evidence_windows.py') and not args:
                args={'question':task.goal,'context':task.inputs['context']}
            validate_schema_instance(args,p['input_schema'])
        except (ValueError,KeyError) as exc:
            reads.append({'error':str(exc)});save();return final_fallback('program_arguments')
        save(True)
        try:
            raw=s.worker.execute(p,args,broker)
            result=normalize_program_result(s.adapter,p,raw)
        except RuntimeError as exc:
            if 'requires Docker' not in str(exc):raise
            result={'status':'blocked','outputs':{},'detail':str(exc),'worker_status':'not_started'}
        rid=context.register('hf:'+str(index),result,name=p['id'],arguments=args)
        attempts.append({'id':'hf:'+str(index),'program_id':p['id'],'task_key':task.physical_key,'origin':'manual',
            'split':task.split,'arguments':args,'argument_refs':step.get('argument_refs',{}),'result_id':rid,'status':result['status'],
            'output_contract_status':result.get('output_contract_status'),'outputs_consumed':False,
            'result_exposed':True,'local_check':'manual_authorized','calls':result.get('calls',0),
            'worker_seconds':result.get('elapsed_seconds'),'worker_cpu_seconds':result.get('cpu_seconds'),
            'submission_by_program':False,'authority':'author_trusted_not_learned'})
        results.append(rid);save(False)
    return final_fallback('step_limit')


@contextmanager
def bridge(options):
    """Patch only this manual runner process; no global repository modifications."""
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    from atomic_skillgraph.empirical.answer_executor import AnswerExecutor
    from atomic_skillgraph.empirical.planner import Planner
    from atomic_skillgraph.empirical.task_context import TaskContext
    original_public=TaskContext.public_result
    original_agent,original_answer,original_plan=EmpiricalSystem.agent,AnswerExecutor.run,Planner.plan
    def public_result(context,raw_result,*,origin=None):
        value=original_public(context,raw_result,origin=origin)
        if origin and origin.startswith('program.') and isinstance(value,dict):
            hidden={'execution_receipt','worker_result','submission_preparation','compatibility_contract_version',
                    'diagnostic','local_check','worker_status'}
            return {k:v for k,v in value.items() if k not in hidden}
        return value
    def agent(s,stage,prompt,material,name,schema,**kwargs):
        if not isinstance(s.bank,HumanBankView):return original_agent(s,stage,prompt,material,name,schema,**kwargs)
        task=getattr(s.adapter,'task',None)
        if task:
            s.bank.bind_task(task)
            if stage in {'runtime','planner'}:
                policy=s.bank.policy(task)
                if policy:
                    prompt=prompt+'\n\n## Active authored SkillBank\n'+policy+'\n\n'+s.adapter.answer_contract()
                if s.bank.program_options(task.goal):
                    prompt+='\nThe listed human-authored Programs are callable now. They do not require training provenance or promotion receipts. Use actual task operands and their declared interfaces.'
                material=deepcopy(material)
                material['guidance']=[] # compiled once above, not duplicated as many cards
                material['human_frontier_skill_ids']=[a['id'] for a in s.bank.selected(task)]
                if stage=='runtime' and s.bank.benchmark=='officeqa':
                    if isinstance(material.get('calls'),dict):material['calls']['programs']=s.bank.program_options(task.goal)
                    else:material['programs']=s.bank.program_options(task.goal)
                elif stage=='planner':material['programs']=material.get('programs',[])
        return original_agent(s,stage,prompt,material,name,schema,**kwargs)
    def answer(executor,task,broker,guidance,semantic):
        if not isinstance(executor.system.bank,HumanBankView):return original_answer(executor,task,broker,guidance,semantic)
        system=executor.system;view=system.bank;view.bind_task(task)
        if view.graph and view.arm!='off':
            from dataclasses import replace
            graph=deepcopy(view.get(view.graph))
            if not graph or 'nodes' not in graph:raise ValueError('Unknown authored graph')
            # Invoke the native Executor, not a pretend multi-node result. The
            # adapter still evaluates exactly the original single final answer.
            old=system.adapter.capabilities
            system.adapter.capabilities=replace(old,final_submission_kind='text')
            try:
                value=system.executor.run(task,system.adapter,broker,graph)
                value['workflow_mode']='native_sequential_execution'
                value['manual_workflow_id']=view.graph
                return value
            finally:system.adapter.capabilities=old
        return compact_run(executor,task,broker,guidance,semantic)
    def plan(planner,task,adapter,feedback=None,completed_results=None):
        if isinstance(planner.bank,HumanBankView) and planner.bank.graph and not feedback:
            selected=planner.bank.get(planner.bank.graph)
            if not selected or 'nodes' not in selected:raise ValueError('Unknown authored workflow ID')
            planner.last_decision_id=None;return planner.validate(selected,completed_results or ())
        return original_plan(planner,task,adapter,feedback,completed_results)
    EmpiricalSystem.agent=agent;AnswerExecutor.run=answer;Planner.plan=plan;TaskContext.public_result=public_result
    try:yield
    finally:EmpiricalSystem.agent=original_agent;AnswerExecutor.run=original_answer;Planner.plan=original_plan;TaskContext.public_result=original_public


def main():
    ap=argparse.ArgumentParser(description='Run this authored Bank directly; no learning/promotion/approval-file gates')
    ap.add_argument('--repo',required=True);ap.add_argument('--config',required=True)
    ap.add_argument('--tasks',required=True);ap.add_argument('--evaluator');ap.add_argument('--corpus-root')
    ap.add_argument('--output',required=True);ap.add_argument('--env-file');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--arm',choices=['manual','off'],default='manual')
    ap.add_argument('--programs',choices=['auto','on','off'],default='off')
    ap.add_argument('--variant',choices=['upstream_skillopt','upstream_skillopt_lite','benchmark_prior','legacy_humanfrontier'],default='benchmark_prior')
    ap.add_argument('--limit',type=int);ap.add_argument('--token-budget',type=int);ap.add_argument('--request-budget',type=int)
    ap.add_argument('--ledger',help='Optional existing cumulative native ledger; its actual caps are preserved')
    ap.add_argument('--graph',nargs='?',const='default',help='Execute the authored graph in the native Executor; optional, usually more model calls')
    ap.add_argument('--show-policy',action='store_true',help='Print actual first-task policy and exit without model calls')
    args=ap.parse_args();use_repo(args.repo)
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import PublicTask
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    from atomic_skillgraph.experiments.run_empirical import run,load_env
    bundle=read(Path(__file__).resolve().parents[1]/('legacy_humanfrontier/bank/bank.json' if args.variant=='legacy_humanfrontier' else 'variants/'+args.variant+'/bank.json'));b=bundle['benchmark']
    print('Actual integration module:',__file__,flush=True)
    data=read(args.tasks);raw=data.get('tasks') if isinstance(data,dict) else data
    if not isinstance(raw,list):raise ValueError('--tasks requires a public materialized task list')
    if args.limit is not None:
        if args.limit<1:raise ValueError('--limit must be positive')
        raw=raw[:args.limit]
    allowed={'task_id','physical_key','goal','inputs','split'};tasks=[]
    for row in raw:
        if not isinstance(row,dict) or not {'task_id','goal','inputs'}<=set(row):
            raise ValueError('Use materialized public tasks containing goal/inputs, not an ID-only split manifest')
        fields={k:row[k] for k in allowed if k in row}
        fields.setdefault('physical_key',sha([row['task_id'],row['goal'],row['inputs']]))
        fields.setdefault('split',Path(args.tasks).stem)
        tasks.append(PublicTask(**fields))
    if not tasks:raise ValueError('No tasks supplied')
    out=Path(args.output).expanduser().resolve();base=read(args.config)
    evaluator=args.evaluator or str(Path(args.tasks).with_name('evaluator_records_'+tasks[0].split+'.json'))
    if not Path(evaluator).is_file():raise ValueError('Pass the real matching --evaluator file; it stays on the Host')
    cfg=configure(base,bundle,out,args.tasks,evaluator,args.corpus_root,args.token_budget,args.request_budget)
    cfg['harness']['split']=tasks[0].split
    if args.env_file:load_env(args.env_file)
    imported=native_import(bundle,out/'bank',cfg['program_environment'],cfg['experiment'].get('officeqa_method_version'),cfg['experiment']['seed'])
    graph=imported['default_workflow_id'] if args.graph=='default' else args.graph
    factory=lambda bank:HumanBankView(bank,imported,args.arm,args.programs,graph)
    if args.show_policy:
        bank=Bank(out/'bank',readonly=True,seed=cfg['experiment']['seed'],environment=cfg['program_environment'])
        try:
            view=factory(bank);view.bind_task(tasks[0]);print(view.policy(tasks[0]));print(json.dumps(view.program_options(''),ensure_ascii=False,indent=2))
        finally:bank.close()
        return
    identity={'runtime':VERSION,'bank':sha(imported),'arm':args.arm,'programs':args.programs,'graph':graph,'tasks':sha([asdict(t) for t in tasks])}
    identity_path=out/'manual_run.json'
    if identity_path.exists() and read(identity_path)!=identity:raise ValueError('Use another output folder for changed run settings')
    if (out/'execution_manifest.json').exists() and not args.resume:raise ValueError('Add --resume to continue this run')
    write(identity_path,identity)
    governor=None
    if args.ledger:
        ledger=read(args.ledger);governor=BudgetGovernor(args.ledger,**ledger['limits'])
    with bridge(args):
        result=run(cfg,tasks,out,resume=args.resume,readonly=True,bank_view_factory=factory,budget_governor=governor,freeze_on_complete=False)
    write(out/'manual_result.json',{'method':imported['method_version'],'origin':'external_human_edited',
        'accuracy':result['successes']/result['tasks'] if result['tasks'] else None,
        'correct':result['successes'],'completed':result['tasks'],'expected':len(tasks),'complete':result['complete'],
        'tokens':result.get('known_total_tokens'),'program_invocations':result.get('program_invocations'),
        'superiority_claimed':False,'run_identity':identity})
    print(json.dumps(read(out/'manual_result.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
