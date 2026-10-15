"""Manual-only faithful text policy execution; shared metered transport and official scoring."""
from copy import deepcopy
from pathlib import Path
import json
import re

from .checkpoint import save_json
from .contracts import digest
from .budget_governor import input_token_bound
from .math_effect_runtime import call, verify_trace
from ..agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from ..core.errors import BudgetExhausted

VERSION = 'human.frontier.runtime.r2.20261016'
RUNTIME = {'completion':65536,'thinking':'enabled','effort':'high','timeout':600,
           'parent_tokens':128000,'finish_completion':2048,'finish_thinking':'disabled',
           'retries':0,'tail_bytes':16384,'programs':'off','graph':False}
PROMPT = 'Solve the supplied mathematical multiple-choice task.'
FINISH = ('Finish the original question using the saved reasoning context. Do not use tools. '
          'Submit only the final option label.')
SUFFIX = ('Answer format for the current task: output only the supplied display label of your final '
          'choice. Labels are defined by the supplied choices; do not use their list position as a label.')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def meta_labels(task):
    """Public-text reporting only. Never passed to a model or used to choose its answer."""
    pattern = (r'(?:one of|other|remaining|above).{0,180}stronger (?:result|statement|conclusion)|'
               r'none of the above.{0,30}(?:is|are) the strongest')
    return [c['label'] for c in task.inputs['choices'] if re.search(pattern,c['text'],re.I|re.S)]


def provider(llm, finish=False):
    return OpenAICompatibleProvider(OpenAICompatibleConfig(base_url=llm['base_url'],model=llm['model'],
        api_key_env=llm['api_key_env'],dialect=llm.get('dialect','deepseek_v4_chat'),
        max_completion_tokens=2048 if finish else 65536, thinking_type='disabled' if finish else 'enabled',
        reasoning_effort='high',request_timeout_seconds=600,max_retries=0,
        generation_seed=42 if llm.get('generation_seed_supported',False) else None,
        capability_profile=llm.get('capability_profile')))


def messages(task, policy, tail=None):
    # The full policy is present once, unchanged, even on the bounded final request.
    prompt = FINISH if tail is not None else PROMPT
    if policy:prompt += '\n\n'+policy
    prompt += '\n\n'+SUFFIX
    material={'goal':task.goal,'inputs':deepcopy(task.inputs)}
    if tail is not None:material['previous_reasoning_tail']=tail
    return [{'role':'system','content':prompt},
            {'role':'user','content':json.dumps(material,ensure_ascii=False)}]


def solve(task, adapter, view, llm, root, governor, *, phase, arm, providers=None, fault=None):
    from .answer_executor import reasoning_tail
    root=Path(root);view.bind_task(task);policy=view.policy(task)
    identity=digest([VERSION,task.__dict__,view.view_hash,policy,llm,RUNTIME,str(root.resolve())])
    path=root/'trace.json'
    if path.exists():
        saved=read(path)
        if saved['identity']!=identity:raise ValueError('Completed manual episode identity changed')
        verify_trace(saved);adapter.reset(task)
        if adapter.evaluate(adapter.submit(saved['prediction']))!=saved['score']:
            raise ValueError('Official score differs from saved manual result')
        return saved
    adapter.reset(task)
    context={'session_id':identity,'stage':'runtime','phase':task.split,'human_phase':phase,'human_arm':arm,
        'method_version':VERSION,'parent_task_id':task.task_id,'parent_scope':str(root.resolve()),
        'parent_token_limit':128000,'episode_scope':digest([identity,str(root.resolve())]),
        'episode_request_limit':2,'purpose':'runtime'}
    p=(providers or {}).get('main') or provider(llm)
    turn,receipt=call(root/'main',p,messages(task,policy),governor,context,durable_received=True)
    if fault:fault('after_main_response')
    first_length=turn.finish_reason=='length';answer=turn.content;receipts=[receipt]
    censored=False;finish_state='not_needed'
    if not adapter.valid_answer_format(answer):
        p=(providers or {}).get('finish') or provider(llm,True)
        remaining=governor.headroom({**context,'purpose':'finish_only'})['tokens']-2048
        lo,hi=0,RUNTIME['tail_bytes']
        while lo<hi:
            mid=(lo+hi+1)//2
            tail=reasoning_tail(turn,receipt['request_ids'][-1],mid)
            if input_token_bound(p._build_payload(messages(task,policy,tail),[]))<=remaining:lo=mid
            else:hi=mid-1
        tail=reasoning_tail(turn,receipt['request_ids'][-1],lo)
        try:
            turn,receipt=call(root/'finish',p,messages(task,policy,tail),governor,
                {**context,'purpose':'finish_only'},durable_received=True)
            receipts.append(receipt);answer=turn.content
            finish_state='valid' if adapter.valid_answer_format(answer) else 'invalid'
        except BudgetExhausted as exc:
            if exc.code!='parent_task_budget_exhausted':raise
            censored=True;finish_state='budget_refused'
    valid=adapter.valid_answer_format(answer)
    score=adapter.evaluate(adapter.submit(answer))
    if fault:fault('after_score')
    records=[r for sub in ('main','finish') if (root/sub/'requests.json').exists()
             for r in read(root/sub/'requests.json')]
    from ..harness.scorers.livemath import parse_choice_label
    result={'schema':'human.frontier.answer.v2','identity':identity,'task':task.__dict__,
        'method_version':VERSION,'arm':arm,'phase':phase,'prediction':answer,
        'parsed_label':parse_choice_label(answer,task.inputs['choices']),'valid':valid,'score':score,
        'score_hash':digest([score,adapter.score_audit]),'scorer_audit':adapter.score_audit,
        'first_truncated':first_length,'finish_state':finish_state,'budget_censored':censored,
        'response_hashes':[r['response_hash'] for r in receipts],
        'response_models':[r['turn']['provider_metadata'].get('response_model') for r in receipts],
        'request_ids':[r['request_id'] for r in records],'raw_usage':[r['raw_usage'] for r in records],
        'tokens':sum(r['usage']['total_tokens'] for r in records if r['usage']), 'http':len(records),
        'latency_ms':sum(r['attempt_latency_ms'] for r in records),'program_calls':0,'workflow_calls':0,
        'workflow_execution':'prompt_compiled','manual_skill_ids':[a['id'] for a in view.selected(task)],
        'policy_variant_id':view.variant,'view_hash':view.view_hash,
        'raw_source_sha256':view.bundle['raw_source_sha256'] if policy else None,
        'assembled_policy_sha256':__import__('hashlib').sha256(policy.encode()).hexdigest(),
        'adapter_suffix_sha256':__import__('hashlib').sha256(SUFFIX.encode()).hexdigest(),
        'meta_option_labels':meta_labels(task),'public_input_hash':digest(task.__dict__),
        'runtime_contract_hash':digest([VERSION,RUNTIME,PROMPT,FINISH,SUFFIX,llm]),
        'request_files':{str(p.resolve()):digest(read(p)) for sub in ('main','finish')
            for name in ('request.json','response.json','requests.json','received.json')
            if (p:=root/sub/name).exists()}}
    result['trace_hash']=digest(result)
    if fault:fault('before_trace')
    save_json(path,result)
    if fault:fault('after_trace')
    return result
