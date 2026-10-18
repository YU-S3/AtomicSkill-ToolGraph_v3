"""R3.1 production integration for one human-authored LiveMath episode.

Uses the repository's actual Docker ProgramWorker for deterministic nodes and
human_frontier_runtime.solve for ONE dynamic final judgment, when needed.
Keep this as an independent manual arm; no Ours learner/qualification mutation.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
from .native_programmatic_runtime import digest,docker_worker,run_programmatic_graph,load_bundle


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    from atomic_skillgraph.empirical.checkpoint import save_json
    save_json(path,obj)


class DynamicPolicyView:
    def __init__(self,bank,bundle,handoff,mode):
        self.bank=bank;self.source_bundle=bundle;self.arm='manual'
        self.variant='r31_'+mode
        self.handoff=deepcopy(handoff)
        self.mode=mode
        self.base=Path(__file__).resolve().parents[1]
        skill=next(a for a in bank.all('skill') if a['id']=='skill_lm_r31_dynamic_theorem')
        self.bundle={'raw_source_sha256':hashlib.sha256(skill['guidance'].encode()).hexdigest()}
        self.view_hash=digest([bundle['method_version'],mode,bundle['source_basis'],handoff])
    def bind_task(self,task):self.task=task
    def selected(self,task):
        return [next(a for a in self.bank.all('skill') if a['id']=='skill_lm_r31_dynamic_theorem')]
    def policy(self,task):
        # Execute the actual new Atomic's guidance, not an unused reference checkpoint.
        guidance=self.selected(task)[0]['guidance']
        hints={'features':self.handoff.get('features',[]),
               'contrasts':self.handoff.get('contrasts',[]),
               'limitation':self.handoff.get('limitation')}
        diagnostic=json.dumps(hints,ensure_ascii=False,separators=(',',':'))
        return (guidance+'\n\nProgrammatic Atomic observation (original public option wording only; '
                'not a proof or an answer certificate):\n'+diagnostic)


def solve_one(task,adapter,bank,bundle,llm,root,governor,*,variant='benchmark_prior',
              phase='manual_eval',providers=None):
    """Return an independent R3.1 trace. The source R2 trace is never rewritten.

    Caller owns canonical tasks, private scorer, shared Governor, authorized budget,
    and identity. Program-only meta path has 0 model HTTP and reports it honestly.
    """
    if not getattr(bank,'readonly',False):
        raise ValueError('R3.1 manual episode requires read-only Native Bank')
    auth=bank.db.execute("SELECT value FROM metadata WHERE key='manual_authority'").fetchone()
    if auth!=(bundle['method_version'],):
        raise ValueError('Native Bank lacks matching human-authored Program permission')
    root=Path(root)
    prior=root/'r31_result.json'
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.human_frontier_runtime import solve as model_solve
    from atomic_skillgraph.empirical.human_frontier_runtime import RUNTIME
    identity=digest([task.__dict__,bank.digest(),digest(bundle),variant,llm,RUNTIME,str(root.resolve())])
    if prior.exists():
        x=json.loads(prior.read_text())
        if x['identity']!=identity or x['trace_hash']!=digest({k:v for k,v in x.items() if k!='trace_hash'}):
            raise ValueError('Saved R3.1 episode identity mismatch')
        for filename,hash_value in x['artifact_hashes'].items():
            if hashlib.sha256((root/filename).read_bytes()).hexdigest()!=hash_value:
                raise ValueError('Completed R3.1 request or Program receipt changed')
        adapter.reset(task)
        if adapter.evaluate(adapter.submit(x['prediction']))!=x['score']:
            raise ValueError('Official saved score changed')
        return x
    adapter.reset(task)
    execute=docker_worker(adapter)
    model_trace=None
    def dynamic(current_task,handoff,node):
        nonlocal model_trace
        view=DynamicPolicyView(bank,bundle,handoff,variant)
        model_trace=model_solve(current_task,adapter,view,llm,root/'dynamic',governor,
                                phase=phase,arm='human_r31_dynamic',providers=providers)
        # This is a real scored, metered model result. Never infer a different label.
        return model_trace['parsed_label'],{
            'relative_trace_path':'dynamic/trace.json','trace_hash':model_trace['trace_hash'],
            'request_ids':model_trace['request_ids'],'tokens':model_trace['tokens'],
            'http':model_trace['http'],'answer':model_trace['parsed_label']}
    graph=run_programmatic_graph(task,bank,variant=variant,run_worker=execute,
          dynamic_answer=dynamic,stage_dir=root/'program_stages')
    if graph['status'] not in {'ok','invalid_answer'}:
        raise RuntimeError('Graph did not reach a final scored outcome')
    answer=graph['answer'];prediction=model_trace['prediction'] if model_trace else answer
    adapter.reset(task)
    scored=adapter.evaluate(adapter.submit(prediction))
    valid=adapter.valid_answer_format(prediction)
    dynamic_receipt=graph.get('dynamic_model_receipt') or {}
    result={'schema':'human.frontier.r31.programmatic.trace.v1','identity':identity,
       'task_id':task.task_id,'status':'completed','variant':variant,
       'answer':answer,'prediction':prediction,'valid':valid,'score':scored,'correct':bool(scored.get('hard')),
       'task':task.__dict__,'phase':phase,'scorer_audit':adapter.score_audit,
       'graph':graph,'source_bank_digest':bank.digest(),'programmatic_benchmark_prior':graph['decision_origin']=='programmatic_benchmark_prior',
       'model_tokens':dynamic_receipt.get('tokens',0),'model_http':dynamic_receipt.get('http',0),
       'request_ids':model_trace['request_ids'] if model_trace else [],
       'raw_usage':model_trace['raw_usage'] if model_trace else [],
       'first_truncated':model_trace['first_truncated'] if model_trace else False,
       'finish_state':model_trace['finish_state'] if model_trace else 'not_needed',
       'budget_censored':model_trace['budget_censored'] if model_trace else False,
       'workflow_execution':'programmatic','program_invocations_total':graph['program_invocations_total'],
       'program_failures':sum(n['status']!='ok' for n in graph['program_nodes'] if n['mode']=='program'),
       'program_cpu_seconds':sum(n['result'].get('cpu_seconds') or 0 for n in graph['program_nodes'] if n['mode']=='program'),
       'program_wall_seconds':sum(n['result'].get('elapsed_seconds') or 0 for n in graph['program_nodes'] if n['mode']=='program'),
       'program_worker_invocations':graph['worker_invocations_this_run'],
       'program_dataflow_consumed':graph['program_results_consumed'],
       'llm_reasoning_consumed_program_hints':'unknown_without_explicit_model_declaration',
       'cost_note':'Model billing is in original Governor; Docker CPU/wall costs separate.'}
    result['artifact_hashes']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for sub in ('program_stages','dynamic') for p in (root/sub).rglob('*.json') if p.is_file()}
    result['trace_hash']=digest(result)
    save(prior,result)
    return result
