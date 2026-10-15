"""Single-answer P7 wire, audited physical billing and replay without repeated HTTP."""
from dataclasses import asdict
import json
from pathlib import Path

from ..agents.protocol import AgentTurn
from ..agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from ..core.errors import BudgetExhausted
from .checkpoint import save_json
from .contracts import digest
from .math_effect_contract import ANSWER, VERSION, card
from .math_effect_source import read
from .math_effect_selection import select

PROMPT = ('Solve the public mathematical multiple-choice problem. Compare every supplied conclusion under '
    'the stated hypotheses. A stronger-looking or meta-option is a candidate, not proof. '
    'Any supplied skill is procedural guidance, not a mathematical fact or answer. '
    'Use your own reasoning. End with exactly <answer>LABEL</answer> using a supplied label.')
FINISH = ('Submit the final option for the complete problem below. The bounded previous reasoning is '
    'unverified context. Do not use tools or restart a long derivation. Output only <answer>LABEL</answer>.')


def provider(llm, stage, settings):
    s=settings['runtime'];learning=stage=='learning';finish=stage=='finish'
    return OpenAICompatibleProvider(OpenAICompatibleConfig(base_url=llm['base_url'],model=llm['model'],
        api_key_env=llm['api_key_env'],dialect=llm.get('dialect','deepseek_v4_chat'),
        max_completion_tokens=settings['learning_completion'] if learning else s['finish_completion'] if finish else s['completion'],
        thinking_type='disabled' if learning else 'enabled',reasoning_effort='low' if finish else 'high',
        request_timeout_seconds=s['timeout'],max_retries=0,
        generation_seed=42 if llm.get('generation_seed_supported',False) else None,
        capability_profile=llm.get('capability_profile')))


def messages(task, cards=(), tail=None):
    content={'goal':task.goal,'choices':task.inputs['choices']}
    if cards:content['skills']=list(cards)
    if tail is not None:content['previous_reasoning_tail']=tail
    return [{'role':'system','content':PROMPT if tail is None else FINISH},
        {'role':'user','content':json.dumps(content,ensure_ascii=False)}]


def call(root, p, messages_, governor, context, *, tools=None, tool_choice=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    wire=p._build_payload(messages_,tools or [],tool_choice);identity=digest([wire,p.config.snapshot(),context])
    request=root/'request.json';response=root/'response.json'
    if request.exists() and read(request)['identity']!=identity:raise ValueError('Paid request identity changed')
    if response.exists():
        receipt=read(response)
        if receipt['identity']!=identity or digest(receipt['turn'])!=receipt['response_hash']:raise ValueError('Response changed')
        return AgentTurn(**receipt['turn']),receipt
    if (root/'started.json').exists():
        raise RuntimeError('Interrupted physical request without committed response; no blind repeated HTTP')
    save_json(request,{'identity':identity,'payload':wire,'context':context,'provider':p.config.snapshot()})
    if governor.billing_blocked:raise RuntimeError('Unresolved physical billing; no new request')
    # Refusal before send does not leave an ambiguous started marker.
    room=governor.headroom(context)
    from .budget_governor import input_token_bound
    if room['requests']<1 or input_token_bound(wire)+p.config.max_completion_tokens>room['tokens']:
        raise BudgetExhausted('p7_budget_exhausted','P7 parent/phase/campaign admission refused')
    save_json(root/'started.json',{'identity':identity})
    p.budget_governor=governor;p.set_request_context(**context)
    start=p.request_record_count
    try:turn=p.complete(messages_,tools=tools,tool_choice=tool_choice)
    except Exception as exc:
        save_json(root/'requests.json',list(p.request_records_since(start)))
        # A metered malformed response is still a real model result, not grounds for re-payment.
        turn=getattr(exc,'usage_turn',None)
        if turn is None:raise
        turn.tool_calls=[]
    records=list(p.request_records_since(start));save_json(root/'requests.json',records)
    receipt={'identity':identity,'turn':asdict(turn),'response_hash':digest(asdict(turn)),
        'request_ids':[r['request_id'] for r in records],'request_hash':digest(records)}
    save_json(response,receipt)
    return turn,receipt


def solve(task, adapter, candidates, cfg, root, governor, *, providers=None):
    from .answer_executor import reasoning_tail
    root=Path(root);settings=cfg['effect_p7'];llm=cfg['llm']
    selection=select(candidates,task,settings['tokenizer_path'],settings['max_injection_tokens'])
    cards=[card(c['candidate']) for c in selection['selected']]
    identity=digest([task.__dict__,cards,settings,llm]);path=root/'trace.json'
    if path.exists():
        saved=read(path)
        if saved['identity']!=identity or saved['trace_hash']!=digest({k:v for k,v in saved.items() if k!='trace_hash'}):
            raise ValueError('Completed P7 task changed')
        verify_trace(saved)
        adapter.reset(task)
        if adapter.evaluate(adapter.submit(saved['prediction'])) != saved['score']:
            raise ValueError('Official score differs from committed P7 response')
        return saved
    adapter.reset(task)
    base={'session_id':identity,'stage':'runtime','phase':task.split,'parent_task_id':task.task_id,
        'parent_scope':str(root.resolve()),'parent_token_limit':settings['runtime']['parent_tokens'],
        'episode_scope':identity,'episode_request_limit':2,'purpose':'runtime'}
    p=(providers or {}).get('main') or provider(llm,'runtime',settings)
    turn,receipt=call(root/'main',p,messages(task,cards),governor,base)
    initial_length=turn.finish_reason=='length';answer=turn.content;receipts=[receipt];censored=False;finish_state='not_needed'
    if not adapter.valid_answer_format(answer):
        tail=reasoning_tail(turn,receipt['request_ids'][-1],settings['runtime']['tail_bytes'])
        p=(providers or {}).get('finish') or provider(llm,'finish',settings)
        try:
            turn,receipt=call(root/'finish',p,messages(task,cards,tail),governor,{**base,'purpose':'finish_only'})
            receipts.append(receipt);answer=turn.content;finish_state='valid' if adapter.valid_answer_format(answer) else 'invalid'
        except BudgetExhausted:
            censored=True;finish_state='budget_refused'
    valid=adapter.valid_answer_format(answer);score=adapter.evaluate(adapter.submit(answer))
    records=[r for sub in ('main','finish') if (root/sub/'requests.json').exists() for r in read(root/sub/'requests.json')]
    result={'schema':ANSWER,'identity':identity,'task':task.__dict__,'prediction':answer,'valid':valid,'score':score,
        'score_hash':digest([score,adapter.score_audit]),'scorer_audit':adapter.score_audit,'first_truncated':initial_length,
        'finish_state':finish_state,'budget_censored':censored,'response_hashes':[r['response_hash'] for r in receipts],
        'request_ids':[r['request_id'] for r in records],'raw_usage':[r['raw_usage'] for r in records],
        'tokens':sum(r['usage']['total_tokens'] for r in records if r['usage']), 'http':len(records),
        'latency_ms':sum(r['attempt_latency_ms'] for r in records),
        'selection':{**selection,'selected':[c['id'] for c in selection['selected']]},
        'public_input_hash':digest({'goal':task.goal,'choices':task.inputs['choices']}),
        'runtime_contract_hash':digest([ANSWER,PROMPT,FINISH,llm,settings['runtime']]),'program_calls':0,
        'request_files':{str(p.resolve()):digest(read(p)) for sub in ('main','finish') for name in ('request.json','response.json','requests.json')
            if (p:=root/sub/name).exists()}}
    result['trace_hash']=digest(result);save_json(path,result);return result


def verify_trace(trace):
    if trace['trace_hash']!=digest({k:v for k,v in trace.items() if k!='trace_hash'}):raise ValueError('Trace hash changed')
    for path,h in trace['request_files'].items():
        if digest(read(path))!=h:raise ValueError('Physical request/response changed: '+path)
    records=[r for path in trace['request_files'] if path.endswith('/requests.json') for r in read(path)]
    if ([r['request_id'] for r in records]!=trace['request_ids'] or [r['raw_usage'] for r in records]!=trace['raw_usage'] or
        sum(r['usage']['total_tokens'] for r in records if r['usage'])!=trace['tokens']):raise ValueError('Trace usage changed')
    if digest([trace['score'],trace['scorer_audit']])!=trace['score_hash']:raise ValueError('Score evidence changed')


def system_task(system,task,learn):
    if learn:raise ValueError('P7 learning is an explicit Train-trace batch, never per-task Runtime learning')
    from .math_effect_contract import checked_effect_asset
    assets=system.bank.all('skill')
    if any(not checked_effect_asset(a) for a in assets):raise ValueError('P7 Runtime requires qualified Frozen assets')
    return solve(task,system.adapter,[a['effect_candidate'] for a in assets],system.config,
        Path(system.config['experiment']['output_dir'])/task.physical_key,system.budget_governor)
