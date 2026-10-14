"""One answering Runtime with optional bounded local computation; no Planner."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from uuid import uuid4

from .contracts import digest, object_schema, validate_program, validate_schema_instance
from .prompts import single_answer_prompt
from .program_submission import normalize_program_result, prepare_program_submission
from .trial_snapshot import seal_trial_workspace, restore_trial_workspace
from . import ANSWER_PROTOCOL_VERSION

VERSION = ANSWER_PROTOCOL_VERSION


def reasoning_tail(turn, request_id, limit=16384):
    raw=(turn.reasoning_content or '').encode('utf-8')
    start=max(0,len(raw)-limit)
    while start<len(raw) and raw[start]&0xc0==0x80:start+=1
    return {'source_request_id':request_id,'response_sha256':digest(asdict(turn)),
        'reasoning_sha256':hashlib.sha256(raw).hexdigest(),'full_bytes':len(raw),
        'start_byte':start,'end_byte':len(raw),'truncated':bool(start),'text':raw[start:].decode('utf-8'),
        'qualification':'bounded_tail_of_previous_model_reasoning_not_verified_facts'}


def fit_reasoning_tail(system,material,prompt):
    from .budget_governor import input_token_bound
    from ..agents.provider import OpenAICompatibleProvider
    provider=system.provider('runtime','finish_only')
    cap=system.resolve_call_settings('runtime','finish_only')['max_completion_tokens']
    available=min(system.config['runtime']['finish_input_max_bytes'],system._remaining('runtime','finish_only')-cap)
    def fits(limit):
        tail=reasoning_tail(system.last_response_turn,system.last_response_request_id,limit)
        value={**material,'previous_reasoning_tail':tail}
        messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(value,ensure_ascii=False,allow_nan=False)}]
        wire=provider._build_payload(messages,[]) if isinstance(provider,OpenAICompatibleProvider) else {'messages':messages}
        wire[provider.config.token_limit_field if isinstance(provider,OpenAICompatibleProvider) else 'max_tokens']=cap
        return input_token_bound(wire)<=available,value
    low,high=0,16384
    while low<high:
        mid=(low+high+1)//2
        if fits(mid)[0]:low=mid
        else:high=mid-1
    # If even the empty tail cannot fit, production admission records the refusal;
    # mandatory question/options are never removed to make the call succeed.
    return fits(low)[1]
STEP = object_schema({'action':{'enum':['finish','execute_python','call_program','read_result']},
    'answer':{'type':'string'}, 'name':{'type':'string'}, 'source':{'type':'string'},
    'arguments':{'type':'object'}, 'input_schema':{'type':'object'}, 'output_schema':{'type':'object'},
    'argument_refs':{'type':'object'}, 'result_id':{'type':'string'}, 'path':{'type':'array'},
    'used_results':{'type':'array','items':{'type':'string'}},
    'replaced_results':{'type':'array','items':{'type':'string'}}}, ['action'])


def validate_math_finish(step,schema,adapter):
    """P4 finish identity/answer are mandatory; optional annotations are checked later."""
    if not isinstance(step,dict) or step.get('action')!='finish':
        validate_schema_instance(step,schema);return
    core={k:v for k,v in step.items() if k not in {'public_process','public_steps'}}
    validate_schema_instance(core,STEP)
    if not adapter.valid_answer_format(core.get('answer','')):raise ValueError('Invalid final answer')


class AnswerExecutor:
    def __init__(self, system): self.system = system

    def run(self, task, broker, guidance, semantic):
        s, context, checkpoint = self.system, broker.context, self.system.checkpoint
        public=s.config['runtime'].get('choice_guidance',{}).get('public_process',False)
        finalizer=s.config['runtime'].get('choice_bounded_finalization',{}).get('enabled',False)
        from .public_process import VERSION as PROCESS_VERSION, PUBLIC_STEP, validate_process, TRUNCATION_FINALIZATION_VERSION
        truncation_only=s.config['runtime'].get('choice_bounded_finalization',{}).get('version')==TRUNCATION_FINALIZATION_VERSION
        version=TRUNCATION_FINALIZATION_VERSION if truncation_only else PROCESS_VERSION if public else VERSION
        from . import MATH_P4_VERSION
        p4=s.config['experiment'].get('choice_method_version')==MATH_P4_VERSION
        if p4:version=MATH_P4_VERSION+'.answer.v1'
        manual=s.config['experiment'].get('manual_reference_version')
        from .manual_reference import SEARCHQA_VERSION as MANUAL_SEARCHQA_VERSION
        manual_searchqa = manual == MANUAL_SEARCHQA_VERSION
        local_limit = s.config['experiment']['manual_reference_limits']['program_executions'] if manual else 2
        if manual:version=manual+'.answer.v1'
        trial=s.config['experiment'].get('math_program_evaluation_version')
        if trial:version=trial+'.answer.v1'
        from .searchqa_guidance import VERSION as SEARCHQA_VERSION, validate_wire
        searchqa=s.config['experiment'].get('searchqa_method_version')==SEARCHQA_VERSION
        if searchqa:
            version=SEARCHQA_VERSION+'.answer.v1'
            validate_wire(guidance)
        schema=deepcopy(STEP)
        if manual:schema['properties']['action']['enum'].append('call_workflow')
        if public:
            schema['properties'].update(public_process={'type':'string','minLength':1,'maxLength':512},
                public_steps={'type':'array','items':PUBLIC_STEP,'minItems':1,'maxItems':4})
        state = deepcopy(checkpoint.state.get('answer_executor', {})) if checkpoint else {}
        attempts, temporary, results = state.get('attempts', []), state.get('temporary', []), state.get('results', [])
        workflows=state.get('workflows',[])
        if state.get('workspace'): restore_trial_workspace(s.adapter,state['workspace'])
        for key,value in state.get('context',{}).items(): setattr(context,key,value)
        if 'finished' in state: return state['finished']
        def save(finished=None,pending=None):
            s.executor.partial_execution = {'prediction':None,'reason':'answer_in_progress','attempts':deepcopy(attempts),
                'temporary_executions':deepcopy(temporary),'workflow_attempts':deepcopy(workflows),'answer_protocol_version':version}
            if not checkpoint: return
            state.update(attempts=attempts,temporary=temporary,results=results,workflows=workflows,pending_step=deepcopy(pending),
                         next_index=index if pending else index+1,
                         finalization=finalization,
                         context={k:getattr(context,k) for k in ('scope','results','sources','memory','local_reads','result_origins','window_sources','sent_windows')})
            if getattr(s.adapter,'workspace',None):
                state['workspace'] = seal_trial_workspace(s.adapter,checkpoint.root/'answer_workspace'/uuid4().hex)
            if finished is not None: state['finished'] = finished
            checkpoint.commit_decision(s.last_decision_id,'applied',answer_executor=state,program_started=False)
        def finish(answer, reason, producer=None, *, budget_censored=False, process=None):
            execution = {'prediction':answer,'reason':reason,'attempts':attempts,'temporary_executions':temporary,
                         'internal_execution_count':len(attempts)+len(temporary),'native_calls':0,
                         'submission_producer_attempt_id':producer,'answer_protocol_version':version,
                         'answer_origin':'program' if producer else 'finish_only' if reason in {'budget_finish_only','truncation_final_answer'} else 'runtime_response'}
            if manual:
                execution.update(workflow_attempts=workflows,manual_reference_version=manual,
                    internal_execution_count=len(attempts)+len(temporary)+sum(w['program_executions'] for w in workflows))
            if budget_censored: execution['budget_censored'] = True
            from .answer_status import classify_answer
            answer_request=next((r for r in reversed(s.requests) if r.get('response')), {})
            response = answer_request.get('response',{})
            execution.update(answer_request_id=answer_request.get('id'),answer_request_usage=deepcopy(response.get('usage')))
            execution.update(classify_answer(answer,response.get('finish_reason'),
                valid_format=getattr(s.adapter,'valid_answer_format',lambda a:True)(answer)))
            if response.get('finish_reason')=='length' and execution['answer_status']!='valid':
                execution.update(reason='completion_truncated_no_answer',answer_status='truncated_empty')
            if process is not None:execution['public_process_evidence']=process
            if finalizer:execution['choice_bounded_finalization']=deepcopy(finalization)
            save(execution)
            s.executor.partial_execution = execution
            return execution
        finalization=state.get('finalization',{'enabled':finalizer,'version':s.config['runtime']['choice_bounded_finalization']['version'],'attempted':False})
        def final_answer(material,reason):
            from .prompts import finish_text_prompt
            from ..core.errors import BudgetExhausted
            if finalization['attempted'] and not checkpoint:return finish('',reason)
            if truncation_only:
                stored=state.get('finalization_material')
                if stored is not None:
                    if any(stored.get(k)!=v for k,v in material.items()):raise ValueError('Finalization public input changed during recovery')
                    material=stored
                else:
                    material=fit_reasoning_tail(s,material,finish_text_prompt(s.adapter.answer_contract()))
                    state['finalization_material']=deepcopy(material)
                tail=material['previous_reasoning_tail']
                finalization['reasoning_tail']={k:v for k,v in tail.items() if k!='text'}
            finalization['attempted']=True
            if checkpoint:checkpoint.advance(checkpoint.state['stage'],answer_executor={**state,'finalization':finalization})
            start=len(s.requests)
            try:
                answer=s.agent('runtime',finish_text_prompt(s.adapter.answer_contract()),material,
                    None,None,repair_limit=0,decision_purpose='finish_only',owner_state_version=version+':finish:'+str(index))
            except (BudgetExhausted,ValueError) as exc:
                if isinstance(exc,ValueError) and not getattr(exc,'model_authored',False) and not str(exc).startswith('material_too_large'):raise
                finalization.update(reason=str(exc),request_ids=[r['id'] for r in s.requests[start:]])
                if truncation_only:finalization.update(valid_answer=False,parent_budget_exhausted=isinstance(exc,BudgetExhausted) and exc.code in {'parent_task_budget_exhausted','empirical_token_budget_exhausted','runtime_finish_reserved'})
                return finish('',reason,budget_censored=isinstance(exc,BudgetExhausted))
            valid=getattr(s.adapter,'valid_answer_format',lambda a:True)(answer)
            finalization.update(request_ids=[r['id'] for r in s.requests[start:]],reason='bounded_final_answer')
            if truncation_only:finalization.update(valid_answer=valid,parent_budget_exhausted=False)
            return finish(answer,'truncation_final_answer' if truncation_only and valid else 'finalization_invalid_answer' if truncation_only else 'budget_finish_only',budget_censored=not truncation_only)
        prompt = single_answer_prompt(getattr(s.adapter,'answer_contract',lambda:'')(),semantic=semantic) + '\n' + (
            'You may answer directly as text or submit one answer_step. finish gives the final answer. '
            'Optional execute_python runs def run(ctx, inputs) with supplied arguments and object input_schema/output_schema '
            'in the locked offline Python Worker. Return status/outputs. No network, LLM, evaluator, sympy or OCR. '
            'call_program selects a supplied usable ID and parameters. Intermediate results do not answer the question; '
            'continue reasoning from actual results. At most two local executions. Use used_results for results actually '
            'used, replaced_results for superseded work. Finite arithmetic/examples do not prove a general theorem. '
            'All question conditions and choices remain available. read_result reads an existing result_id/path.')
        if manual_searchqa:
            prompt = prompt.replace('At most two local executions.', 'At most three local executions.')
            prompt += ('\nThis is a human-authored SearchQA reference control, not learned Ours. '
                'The twelve supplied Skills are conditional procedures, not factual evidence. '
                'call_workflow uses a supplied workflow ID in name and arguments. At most one workflow; '
                'its ordered nodes count toward three total local Program executions. Dynamic nodes use the same Runtime and budget. '
                'The Host binds workflow question/context to the unchanged full public input. Provide only extra parameters such as subject/mode. '
                'Direct Programs require explicit inputs or argument_refs to real earlier results. '
                'Program matches/offset checks are intermediate evidence, not semantic entailment. '
                'If a program does not apply, solve directly. Submit <answer>short answer</answer> and mark used_results when used.')
        elif manual:
            prompt+='\nThis is an explicitly human-authored reference Bank control arm. All twelve supplied Skills are conditional procedures, not facts or proofs. '+(
                'You may call_workflow using a supplied workflow ID in name and explicit arguments. At most one Workflow and two total Program executions. '
                'Its stored ordered nodes run with actual dataflow; dynamic nodes use this same Runtime and parent budget. '
                'Only use finite programs/graphs when all domain premises are grounded in the public question. Never invent toy truth tables or option truth values. '
                'All displayed choices must be considered. If unsuitable, solve directly. Workflow results are local evidence; mark actually used result IDs.')
        if trial:
            prompt+='\nThe supplied Atomic descriptions and Programs were generated from Train traces and passed bounded local Docker probes. '+(
                'They remain train_trial_only candidates, not mathematical theorems or officially usable assets. '
                'Only call a Program when the complete finite parameters follow from this public task. '
                'Never invent a toy relation or option truth values to force a call. If unsuitable, solve directly. '
                'Mark actual consumption with used_results; an execution alone is not evidence of usefulness.')
        if public:
            prompt+='\n'+(
                'For finish you may attach public_process (at most 512 Unicode characters) describing the actual '
                'premise/quantifier/claim comparisons you performed, not private reasoning or a theorem proof. '
                'To make it auditable, include public_steps with kind check_assumptions, compare_claim_strength, '
                'inspect_quantifiers or deterministic_computation and at least two distinct references per step. '
                'References have quote plus task_path (e.g. ["goal"] or ["inputs","choices",0,"text"]), '
                'or an actually used result_id/path. Quotes must occur at those public locations. '
                'A computation requires a real successful result. Do not invent steps. Plain text answers remain valid.')
        for index in range(state.get('next_index',0),8):
            local_used=len(attempts)+len(temporary)+sum(w['program_executions'] for w in workflows)
            material = {'goal':task.goal,'inputs':task.inputs,'guidance':guidance,
                'programs':s.bank.program_options(task.goal), 'local_results':[context.view(r) for r in results],
                'remaining_local_executions':max(0,local_limit-local_used),
                'worker_resources':getattr(s.adapter,'worker_resources',{}), 'local_reads':context.local_reads[-2:],
                'content_parts':getattr(s.adapter,'content_parts',lambda:[])(), 'answer_protocol_version':version}
            if manual:material.update(workflows=s.bank.workflow_options(),remaining_workflow_calls=max(0,1-len(workflows)))
            from ..core.errors import BudgetExhausted
            try:
                step = state.get('pending_step') or s.agent('runtime',prompt,material,'answer_step',schema,repair_limit=0,owner_state_version=version+':'+str(index),
                    finish_request_material={k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}})
            except ValueError as exc:
                if not getattr(exc,'model_authored',False):raise
                if truncation_only and s.requests[-1].get('response',{}).get('finish_reason')=='length':
                    finalization['first_truncation']={'request_id':s.requests[-1]['id'],'finish_reason':'length','no_valid_answer':True}
                    return final_answer({k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}},'completion_truncated_no_answer')
                return finish('','runtime_model_failure')
            except BudgetExhausted as exc:
                if truncation_only:raise
                if exc.code not in {'parent_task_budget_exhausted','runtime_finish_reserved'} or not (results or finalizer):raise
                if finalizer:return final_answer({k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}},'token_budget_exhausted')
                from .prompts import finish_text_prompt
                try:
                    answer=s.agent('runtime',finish_text_prompt(s.adapter.answer_contract()),
                        {k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}},
                        'finish_answer',None,repair_limit=0,owner_state_version=VERSION+':finish:'+str(index))
                except BudgetExhausted:raise exc
                return finish(answer,'budget_finish_only',budget_censored=True)
            if truncation_only and not state.get('pending_step') and s.requests[-1].get('response',{}).get('finish_reason')=='length':
                valid=step.get('action')=='finish' and s.adapter.valid_answer_format(step.get('answer',''))
                finalization.setdefault('first_truncation',{'request_id':s.requests[-1]['id'],'finish_reason':'length','no_valid_answer':not valid})
                if not valid:
                    return final_answer({k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}},'completion_truncated_no_answer')
            used,replaced = set(step.get('used_results',[])),set(step.get('replaced_results',[]))
            if (used|replaced)-set(results):return finish('','unknown_result_consumption_reference')
            for a in attempts+temporary+workflows:
                if a['result_id'] in used-replaced: a['outputs_consumed']=True
                if a['result_id'] in replaced: a.update(outputs_consumed=False,replaced=True)
            if step['action']=='finish':
                answer=step.get('answer','');process=None
                if public:
                    try:
                        if p4:
                            optional={k:step[k] for k in ('public_process','public_steps') if k in step}
                            validate_schema_instance(optional,object_schema({k:schema['properties'][k] for k in ('public_process','public_steps')},[]))
                        process=validate_process(step,task,context,results,used-replaced,index,s.requests[-1]['id'] if s.requests else None)
                    except ValueError as exc:process={'verification':'rejected','reason':str(exc),'kind':'self_reported_process'}
                if finalizer and not getattr(s.adapter,'valid_answer_format',lambda a:True)(answer) and s.requests[-1].get('response',{}).get('finish_reason')=='length':
                    if truncation_only:finalization['first_truncation']={'request_id':s.requests[-1]['id'],'finish_reason':'length','no_valid_answer':True}
                    final_material={k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}}
                    if process and process.get('verification')=='source_compatible':final_material['public_process']=process['public_process']
                    return final_answer(final_material,'completion_truncated_no_answer')
                return finish(answer,'agent_submitted',process=process)
            if step['action']=='read_result':
                try:context.read(step.get('result_id',''),path=step.get('path',[]))
                except (ValueError,KeyError):return finish('','invalid_result_read')
                save(); continue
            if local_used>=local_limit:
                return finish('','internal_execution_limit')
            if manual and step['action']=='call_workflow':
                from .manual_reference import execute_workflow, WorkflowInputError
                if workflows:return finish('','manual_workflow_call_limit')
                save(pending=step)
                try:
                    try:args=context.bind(step.get('arguments',{}),step.get('argument_refs',{}))
                    except (ValueError,KeyError) as exc:raise WorkflowInputError(str(exc)) from exc
                    result=execute_workflow(s,task,broker,step.get('name',''),args,'answer_'+str(index),local_limit-local_used)
                except WorkflowInputError as exc:
                    result={'status':'needs_input','outputs':{},'reason':str(exc),'program_executions':0,'nodes':[]}
                from .public_results import MANUAL_GRAPH_EXECUTION
                rid=context.register(digest([version,index,'workflow']),result,name=step.get('name',''),
                    origin=MANUAL_GRAPH_EXECUTION if manual_searchqa else None)
                workflows.append({'workflow_id':step.get('name'),'result_id':rid,'status':result['status'],
                    'program_executions':result['program_executions'],'outputs_consumed':False,'nodes':result['nodes']})
                results.append(rid)
                if truncation_only and s.requests and s.requests[-1].get('response',{}).get('finish_reason')=='length' and result['status']!='ok':
                    finalization['first_truncation']={'request_id':s.requests[-1]['id'],'finish_reason':'length','no_valid_answer':True}
                    return final_answer({k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}},'completion_truncated_no_answer')
                save();continue
            try:
                args = context.bind(step.get('arguments',{}),step.get('argument_refs',{}))
                if step['action']=='execute_python':
                    p = {'source':step['source'],'entry':'run','input_schema':step['input_schema'],
                         'output_schema':step['output_schema'],'allowed_tools':[],
                         'environment':s.config['program_environment'],'result_role':'intermediate'}
                    p['id']='temporary_'+validate_program(p)
                    kind='temporary_python'
                else:
                    p=s.bank.get(step.get('name',''))
                    if not s.bank.program_eligible(p): raise ValueError('Program is not visible and locally validated')
                    if p['environment']!=s.config['program_environment']: raise ValueError('Program environment is incompatible')
                    if searchqa:
                        from .searchqa_program_probe import inputs
                        expected=inputs(task)
                        if not args:args=expected
                        if args!=expected:raise ValueError('R2 parameters must bind the current full public question/context')
                    kind='program'
                validate_schema_instance(args,p['input_schema'])
            except (ValueError,KeyError,SyntaxError) as exc:
                rid=context.register(uuid4().hex,{'status':'blocked','error':str(exc)})
                results.append(rid); save(); continue
            if checkpoint: checkpoint.advance(checkpoint.state['stage'],program_started=True)
            before=s.adapter.observe().get('workspace',{})
            result=normalize_program_result(s.adapter,p,s.worker.execute(p,args,broker),workspace_before=before)
            aid=uuid4().hex
            from .public_results import PROGRAM_EXECUTION
            from .public_results import MANUAL_PROGRAM_EXECUTION
            rid=context.register(aid,result,name=p['id'],arguments=args,
                origin=MANUAL_PROGRAM_EXECUTION if manual_searchqa else PROGRAM_EXECUTION)
            attempt={'id':aid,'program_id':p['id'],'task_key':task.physical_key,'origin':'online','split':task.split,
                'kind':kind,'arguments':args,'result_id':rid,'status':result['status'],'outcome':'normal',
                'basis':None,'calls':result.get('calls',0),'local_check':'unavailable','outputs_consumed':False,
                'output_contract_status':result.get('output_contract_status'),'submission_by_program':False,
                'worker_seconds':result.get('elapsed_seconds'), 'worker_cpu_seconds':result.get('cpu_seconds'),
                'input_schema':p['input_schema'],'output_schema':p['output_schema']}
            if kind=='temporary_python': attempt['source']=p['source']; temporary.append(attempt)
            else: attempts.append(attempt)
            results.append(rid)
            if kind=='program' and result['status']=='ok' and p.get('result_role')=='final_answer':
                prepared=prepare_program_submission(s.adapter,p,result,workspace_before=before)
                if prepared['status']=='ready':
                    attempt.update(outputs_consumed=True,submission_by_program=True)
                    return finish(prepared['payload'],'program_submitted',aid)
            save()
        return finish('','answer_decision_limit')
