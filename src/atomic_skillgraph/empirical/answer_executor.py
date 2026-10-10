"""One answering Runtime with optional bounded local computation; no Planner."""
from copy import deepcopy
from uuid import uuid4

from .contracts import digest, object_schema, validate_program, validate_schema_instance
from .prompts import single_answer_prompt
from .program_submission import normalize_program_result, prepare_program_submission
from .trial_snapshot import seal_trial_workspace, restore_trial_workspace
from . import ANSWER_PROTOCOL_VERSION

VERSION = ANSWER_PROTOCOL_VERSION
STEP = object_schema({'action':{'enum':['finish','execute_python','call_program','read_result']},
    'answer':{'type':'string'}, 'name':{'type':'string'}, 'source':{'type':'string'},
    'arguments':{'type':'object'}, 'input_schema':{'type':'object'}, 'output_schema':{'type':'object'},
    'argument_refs':{'type':'object'}, 'result_id':{'type':'string'}, 'path':{'type':'array'},
    'used_results':{'type':'array','items':{'type':'string'}},
    'replaced_results':{'type':'array','items':{'type':'string'}}}, ['action'])


class AnswerExecutor:
    def __init__(self, system): self.system = system

    def run(self, task, broker, guidance, semantic):
        s, context, checkpoint = self.system, broker.context, self.system.checkpoint
        state = deepcopy(checkpoint.state.get('answer_executor', {})) if checkpoint else {}
        attempts, temporary, results = state.get('attempts', []), state.get('temporary', []), state.get('results', [])
        if state.get('workspace'): restore_trial_workspace(s.adapter,state['workspace'])
        for key,value in state.get('context',{}).items(): setattr(context,key,value)
        if 'finished' in state: return state['finished']
        def save(finished=None):
            s.executor.partial_execution = {'prediction':None,'reason':'answer_in_progress','attempts':deepcopy(attempts),
                'temporary_executions':deepcopy(temporary),'answer_protocol_version':VERSION}
            if not checkpoint: return
            state.update(attempts=attempts,temporary=temporary,results=results,
                         next_index=index+1,
                         context={k:getattr(context,k) for k in ('scope','results','sources','memory','local_reads')})
            if getattr(s.adapter,'workspace',None):
                state['workspace'] = seal_trial_workspace(s.adapter,checkpoint.root/'answer_workspace'/uuid4().hex)
            if finished is not None: state['finished'] = finished
            checkpoint.commit_decision(s.last_decision_id,'applied',answer_executor=state,program_started=False)
        def finish(answer, reason, producer=None, *, budget_censored=False):
            execution = {'prediction':answer,'reason':reason,'attempts':attempts,'temporary_executions':temporary,
                         'internal_execution_count':len(attempts)+len(temporary),'native_calls':0,
                         'submission_producer_attempt_id':producer,'answer_protocol_version':VERSION}
            if budget_censored: execution['budget_censored'] = True
            from .answer_status import classify_answer
            response = s.requests[-1].get('response',{}) if s.requests else {}
            execution.update(classify_answer(answer,response.get('finish_reason'),
                valid_format=getattr(s.adapter,'valid_answer_format',lambda a:True)(answer)))
            save(execution)
            s.executor.partial_execution = execution
            return execution
        prompt = single_answer_prompt(getattr(s.adapter,'answer_contract',lambda:'')(),semantic=semantic) + '\n' + (
            'You may answer directly as text or submit one answer_step. finish gives the final answer. '
            'Optional execute_python runs def run(ctx, inputs) with supplied arguments and object input_schema/output_schema '
            'in the locked offline Python Worker. Return status/outputs. No network, LLM, evaluator, sympy or OCR. '
            'call_program selects a supplied usable ID and parameters. Intermediate results do not answer the question; '
            'continue reasoning from actual results. At most two local executions. Use used_results for results actually '
            'used, replaced_results for superseded work. Finite arithmetic/examples do not prove a general theorem. '
            'All question conditions and choices remain available. read_result reads an existing result_id/path.')
        for index in range(state.get('next_index',0),8):
            material = {'goal':task.goal,'inputs':task.inputs,'guidance':guidance,
                'programs':s.bank.program_options(task.goal), 'local_results':[context.view(r) for r in results],
                'remaining_local_executions':max(0,2-len(attempts)-len(temporary)),
                'worker_resources':getattr(s.adapter,'worker_resources',{}), 'local_reads':context.local_reads[-2:],
                'content_parts':getattr(s.adapter,'content_parts',lambda:[])(), 'answer_protocol_version':VERSION}
            from ..core.errors import BudgetExhausted
            try:
                step = s.agent('runtime',prompt,material,'answer_step',STEP,repair_limit=0,owner_state_version=VERSION+':'+str(index),
                    finish_request_material={k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}})
            except ValueError as exc:
                if not getattr(exc,'model_authored',False):raise
                return finish('','runtime_model_failure')
            except BudgetExhausted as exc:
                if exc.code not in {'parent_task_budget_exhausted','runtime_finish_reserved'} or not results:raise
                from .prompts import finish_text_prompt
                try:
                    answer=s.agent('runtime',finish_text_prompt(s.adapter.answer_contract()),
                        {k:v for k,v in material.items() if k in {'goal','inputs','local_results','local_reads'}},
                        'finish_answer',None,repair_limit=0,owner_state_version=VERSION+':finish:'+str(index))
                except BudgetExhausted:raise exc
                return finish(answer,'budget_finish_only',budget_censored=True)
            used,replaced = set(step.get('used_results',[])),set(step.get('replaced_results',[]))
            if (used|replaced)-set(results):return finish('','unknown_result_consumption_reference')
            for a in attempts+temporary:
                if a['result_id'] in used-replaced: a['outputs_consumed']=True
                if a['result_id'] in replaced: a.update(outputs_consumed=False,replaced=True)
            if step['action']=='finish': return finish(step.get('answer',''),'agent_submitted')
            if step['action']=='read_result':
                try:context.read(step.get('result_id',''),path=step.get('path',[]))
                except (ValueError,KeyError):return finish('','invalid_result_read')
                save(); continue
            if len(attempts)+len(temporary)>=2:
                return finish('','internal_execution_limit')
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
                    kind='program'
                validate_schema_instance(args,p['input_schema'])
            except (ValueError,KeyError,SyntaxError) as exc:
                rid=context.register(uuid4().hex,{'status':'blocked','error':str(exc)})
                results.append(rid); save(); continue
            if checkpoint: checkpoint.advance(checkpoint.state['stage'],program_started=True)
            before=s.adapter.observe().get('workspace',{})
            result=normalize_program_result(s.adapter,p,s.worker.execute(p,args,broker),workspace_before=before)
            aid=uuid4().hex
            rid=context.register(aid,result,name=p['id'],arguments=args)
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
