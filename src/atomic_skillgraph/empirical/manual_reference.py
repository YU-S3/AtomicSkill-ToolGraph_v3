"""Explicit, read-only human-authored control arm. No Ours qualification is changed."""
from copy import deepcopy
import ast
import json
from pathlib import Path

from .contracts import digest, object_schema, validate_schema_instance, validate_workflow
from .checkpoint import save_json

VERSION = 'manual-reference.graph-test.v1'
LIMITS = {'workflow_calls': 1, 'program_executions': 2, 'dynamic_nodes': 3}
SEARCHQA_VERSION = 'human.searchqa.reference-test.v1'
SEARCHQA_LIMITS = {'workflow_calls': 1, 'program_executions': 3, 'dynamic_nodes': 1}


class WorkflowInputError(ValueError):
    pass


def validate_config(config):
    e = config['experiment']
    from .manual_math_v2 import VERSION as V2, FINALIZATION, LIMITS as V2_LIMITS
    if e.get('manual_reference_version') == V2:
        if (e.get('benchmark') != 'livemath' or e.get('runtime_mode') != 'frozen'
                or e.get('choice_method_version') or e.get('manual_reference_limits') != V2_LIMITS
                or not e.get('manual_reference_manifest') or type(e.get('manual_reference_enabled', True)) is not bool):
            raise ValueError('V2 requires its independent read-only LiveMath contract')
        from .math_capability import validate_runtime_contract
        copy = deepcopy(config)
        from .public_process import TRUNCATION_FINALIZATION_VERSION
        if config['runtime']['choice_bounded_finalization'] != {'enabled':True,'version':FINALIZATION}:
            raise ValueError('V2 requires bounded finish after truncation or tool failure')
        copy['runtime']['choice_bounded_finalization']['version'] = TRUNCATION_FINALIZATION_VERSION
        validate_runtime_contract(copy)
        return
    if e.get('manual_reference_version') == SEARCHQA_VERSION:
        llm = config['llm']
        if (e.get('benchmark') != 'searchqa' or config['harness']['adapter'] != 'searchqa'
                or e.get('runtime_mode') != 'frozen' or e.get('searchqa_method_version')
                or e.get('choice_method_version') or e.get('manual_reference_limits') != SEARCHQA_LIMITS
                or not e.get('manual_reference_manifest') or config['budget']['eval_task_tokens'] != 96000
                or llm['runtime']['max_completion_tokens'] != 32768
                or llm['runtime']['reasoning_effort'] != 'high'
                or {**llm.get('protocol', {}), **llm['runtime'].get('protocol', {})}.get('thinking_type') != 'enabled'
                or config['runtime']['choice_bounded_finalization']['enabled']):
            raise ValueError('SearchQA reference requires its independent frozen Test contract')
        return
    if (e.get('manual_reference_version') != VERSION or e.get('benchmark') != 'livemath'
            or e.get('runtime_mode') != 'frozen' or e.get('choice_method_version')
            or e.get('manual_reference_limits') != LIMITS or not e.get('manual_reference_manifest')):
        raise ValueError('Manual reference requires its independent read-only graph evaluation contract')
    from .math_capability import validate_runtime_contract
    validate_runtime_contract(config)


class ReferenceView:
    """Candidates remain candidates in SQLite; execution authority is the explicit control-arm receipt."""
    def __init__(self, bank, manifest_path):
        if not bank.readonly: raise ValueError('Manual reference view must be read-only')
        self.bank = bank
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        from .manual_math_v2 import VERSION as V2, CapabilityRouter
        self.version = manifest['version']
        if manifest['version'] not in {VERSION, V2, SEARCHQA_VERSION} or manifest['bank_digest'] != bank.digest():
            raise ValueError('Manual Bank identity changed')
        self.qualified = set()
        for pid, records in manifest['program_receipts'].items():
            program = bank.get(pid)
            if not program or program['allowed_tools'] or digest(program) != manifest['program_hashes'][pid]:
                raise ValueError('Manual Program identity changed')
            if {r['role'] for r in records} != {'positive', 'variation', 'invalid'}:
                raise ValueError('Manual Program needs independent positive, variation and rejection probes')
            for ref in records:
                data = (path.parent / ref['file']).read_bytes()
                import hashlib
                if hashlib.sha256(data).hexdigest() != ref['sha256']:
                    raise ValueError('Manual Worker receipt changed')
                row = json.loads(data); receipt = row['result'].get('execution_receipt', {})
                expected_ok=(row['result']['status']=='needs_input' if row['expected'] is None else
                    row['result']['status']==row.get('expected_status','ok') and row['result'].get('outputs',{})==row['expected'])
                if (not row['passed'] or not expected_ok or row['role']!=ref['role'] or row['program_id'] != pid or row['rpc_calls']
                        or receipt.get('exit_code') != 0 or receipt.get('image_digest') != program['environment']['image_digest']
                        or '--network=none' not in receipt.get('command', [])):
                    raise ValueError('Manual Program has no real locked Docker qualification')
            self.qualified.add(pid)
        if self.qualified != {p['id'] for p in bank.all('program')}:
            raise ValueError('The complete manual Program catalog must be checked before Test')
        self.router = CapabilityRouter(self,manifest['capability_index'],manifest['theorem_registry']) if self.version==V2 else None

    def __getattr__(self, name): return getattr(self.bank, name)

    def program_eligible(self, program):
        return bool(program and program['id'] in self.qualified and self.bank.get(program['id']) == program)

    def program_options(self, query, **kwargs):
        cards = [{'id':p['id'], 'state':'manual_reference_docker_checked',
            'documentation':ast.get_docstring(ast.parse(p['source'])) or '',
            'entry_constraints':p['entry_constraints'], 'result_role':p['result_role'],
            'input_schema':p['input_schema'], 'output_schema':p['output_schema'],
            'capabilities':[{'skill_id':i['skill_id'], 'goal':self.get(i['skill_id'])['goal']}
                            for i in self.all('implementation') if i['program_id']==p['id']]}
            for p in self.all('program') if self.program_eligible(p)
                and (self.router is None or p['id'] in self.router.programs)]
        if self.router:
            for card in cards: card['verified_public_binding'] = self.router.binding(card['id'])
        return cards

    def workflow_options(self):
        return [deepcopy(w) for w in self.all('workflow') if self.router is None or w['id'] in self.router.workflows]


NODE_PROMPT = '''Execute only the current node of this supplied human-authored workflow.
The complete public question and all choices are supplied. Do not read any evaluator or hidden answer.
Return exactly one JSON object: {"status":"ok"|"needs_input", "outputs":{...}, "references":[...] }.
For ok, provide every requested output field with the stated types. references contain task_path and an
exact quote from the public question/choices. A source node must cite at least one exact public reference.
For mathematical grounding, construct a relation, truth table or AST only when justified by the actual
public premises; never invent a finite toy instance, an option truth value or domain completeness.
An incomplete/infinite domain is needs_input. Finite computations never prove an unrelated domain theorem.
Interpret only actual preceding outputs. No tools or code execution in this response. Do not guess missing fields.'''

SEARCHQA_NODE_PROMPT = '''Execute only the current node of this human-authored SearchQA workflow.
Use the complete public question/context and actual predecessor outputs. Return exactly one JSON object:
{"status":"ok"|"needs_input", "outputs":{...}, "references":[{"task_path":["inputs","context"],"quote":"exact source text"}]}.
Meet the supplied output schema. Cite exact public text for an ok answer; lexical matches alone are not
semantic entailment. Resolve the question's entity, relation and time constraints. Abstain if unsupported.
No tools, hidden answers or invented execution. This is an authored reference, not learned Ours.'''


def execute_workflow(system, task, broker, workflow_id, arguments, key, remaining_programs):
    """Resume node results and physical responses, preserving real ordered dataflow."""
    from ..agents.protocol import parse_json_strict
    from .math_program_probe import PureBroker
    from .contracts import ValueStore, handoff_requirements
    s, cp = system, system.checkpoint
    version = s.config['experiment']['manual_reference_version']
    limits = s.config['experiment']['manual_reference_limits']
    workflow = s.bank.get(workflow_id)
    if not workflow or 'nodes' not in workflow: raise WorkflowInputError('Unknown supplied Workflow')
    validate_workflow(workflow, bank=s.bank)
    if len([n for n in workflow['nodes'] if n['execution_mode']=='program']) > remaining_programs:
        raise WorkflowInputError('Workflow exceeds remaining local Program execution allowance')
    if len([n for n in workflow['nodes'] if n['execution_mode']=='dynamic']) > limits['dynamic_nodes']:
        raise WorkflowInputError('Workflow exceeds frozen dynamic-node allowance')
    if set(arguments) & {'goal','inputs'}:
        raise WorkflowInputError('Workflow arguments cannot replace the public question')
    arguments = deepcopy(arguments)
    if getattr(s.bank,'router',None):
        try: arguments = s.bank.router.workflow_arguments(workflow_id,arguments)
        except ValueError as exc: raise WorkflowInputError(str(exc)) from exc
    if version == SEARCHQA_VERSION:
        for name, value in {'question':task.goal, 'context':task.inputs['context']}.items():
            if name in arguments and arguments[name] != value:
                raise WorkflowInputError('Workflow cannot replace the canonical public '+name)
            arguments[name] = value
    identity = digest([workflow, arguments, task.goal, task.inputs])
    all_states = deepcopy(cp.state.get('manual_workflows', {})) if cp else {}
    state = all_states.get(key, {'identity':identity,'nodes':[],'results':{},'program_executions':0})
    if state['identity'] != identity: raise ValueError('Workflow identity changed across recovery')
    if 'finished' in state: return state['finished']
    values = ValueStore(task); values.task.update(deepcopy(arguments)); values.results=deepcopy(state['results'])
    def persist(decision=None):
        state['results']=deepcopy(values.results);all_states[key]=state
        system.executor.partial_execution['active_workflow']=deepcopy(state)
        if cp: cp.commit_decision(decision,'applied',manual_workflows=all_states)
    def finish(status, outputs, reason=None):
        result={'status':status,'outputs':outputs,'workflow_id':workflow_id,'nodes':deepcopy(state['nodes']),
            'program_executions':state['program_executions'],'authority':version,'reason':reason}
        state['finished']=result;persist();return result
    for index,node in enumerate(workflow['nodes']):
        if index < len(state['nodes']): continue
        args,missing=values.resolve(node.get('args',{}))
        if missing:return finish('needs_input',{},'Missing graph inputs: '+str(sorted(missing)))
        decision=None
        if node['execution_mode']=='program':
            p=s.bank.get(node['program_id'])
            if not s.bank.program_eligible(p):raise ValueError('Unqualified manual Program')
            try: validate_schema_instance(args,p['input_schema'])
            except ValueError as exc:return finish('needs_input',{},str(exc))
            receipt_path=cp.root/'manual_worker'/(digest([key,node['id'],args])+'.json') if cp else None
            claim=receipt_path.with_suffix('.started.json') if receipt_path else None
            if receipt_path and receipt_path.exists():result=json.loads(receipt_path.read_text())
            else:
                if claim and claim.exists():raise RuntimeError('Unresolved manual Worker invocation; review before replay')
                if claim:save_json(claim,{'program_id':p['id'],'input_hash':digest(args)})
                pure=PureBroker(s.adapter,1,context=broker.context)
                result=s.worker.execute(p,args,pure)
                if pure.events:raise ValueError('Manual pure Program attempted RPC')
                if receipt_path:save_json(receipt_path,result)
            state['program_executions']+=1
        else:
            required=handoff_requirements(workflow,node['id'])['required_fields']
            fields={}
            for consumer in workflow['nodes']:
                if consumer.get('program_id'):
                    schema=s.bank.get(consumer['program_id'])['input_schema']['properties']
                    for name,ref in consumer.get('args',{}).items():
                        if ref.get('from')==node['id']:fields[ref['field']]=schema[name]
            for name in required:fields.setdefault(name,{'type':'string'})
            output_schema=object_schema(fields,required)
            material={'goal':task.goal,'inputs':task.inputs,'node':node,'bound_arguments':args,
                'actual_predecessor_outputs':values.results,'required_outputs':output_schema,
                'consumer_programs':[p for p in s.bank.program_options(task.goal)
                    if any(n.get('program_id')==p['id'] and node['id'] in n.get('after',[]) for n in workflow['nodes'])],
                'authority':'human_authored_reference_not_learned_ours'}
            try:
                raw=s.agent('runtime',SEARCHQA_NODE_PROMPT if version == SEARCHQA_VERSION else NODE_PROMPT,material,None,None,repair_limit=0,
                    completion_override=4096 if getattr(s.bank,'router',None) else None,
                    owner_state_version=version+':'+key+':'+node['id'])
            except ValueError as exc:
                if not getattr(exc,'model_authored',False):raise
                return finish('needs_input',{},str(exc))
            decision=s.last_decision_id
            try:
                result=parse_json_strict(raw)
                validate_schema_instance(result,object_schema({'status':{'enum':['ok','needs_input']},
                    'outputs':{'type':'object'},'references':{'type':'array','items':{'type':'object'}}},['status','outputs','references']))
                if result['status']=='ok':
                    validate_schema_instance(result['outputs'],output_schema)
                    if getattr(s.bank,'router',None) and set(result['outputs']) & {'relation','truth_values','domain_complete','premises_satisfied','answer'}:
                        raise ValueError('Dynamic interpretation cannot certify mathematical premises or submit an answer')
                    if (not node.get('after') or version == SEARCHQA_VERSION) and not result['references']:raise ValueError('Missing source grounding')
                    for ref in result['references']:
                        text={'goal':task.goal,'inputs':task.inputs}
                        for part in ref['task_path']:text=text[part]
                        if not isinstance(text,str) or not ref['quote'] or ref['quote'] not in text:
                            raise ValueError('Unknown public grounding reference')
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                result={'status':'needs_input','outputs':{},'error':str(exc),'raw_response':raw}
        state['nodes'].append({'node_id':node['id'],'mode':node['execution_mode'],
            'program_id':node.get('program_id'),'arguments':args,'result':result,'decision_id':decision})
        if result['status']=='ok':values.results[node['id']]=result['outputs']
        persist(decision)
        if result['status']!='ok':return finish(result['status'],{},'Node stopped: '+node['id'])
    outputs,missing=values.resolve(workflow.get('outputs',{}))
    return finish('needs_input' if missing else 'ok',outputs,'Missing final fields' if missing else None)
