"""Public-input routing and strict action contracts for the authored math control."""
from copy import deepcopy
import json
import re

from .contracts import digest, object_schema, validate_schema_instance

VERSION = 'manual-reference.graph-test.v2'
FINALIZATION = VERSION + '.bounded-finish'
LIMITS = {'workflow_calls': 1, 'program_executions': 2, 'dynamic_nodes': 1}


def action_schema():
    text = {'type': 'string', 'minLength': 1}
    interface = {'type':'object','properties':{'type':{'const':'object'}},'required':['type']}
    fields = {
        'finish': {'answer': text, 'used_results': {'type': 'array', 'items': text}},
        'call_program': {'name': text, 'arguments': {'type': 'object'}, 'argument_refs': {'type': 'object'}},
        'execute_python': {'source': text, 'input_schema': interface,
                           'output_schema': interface, 'arguments': {'type': 'object'}},
        'call_workflow': {'name': text, 'arguments': {'type': 'object'}},
        'read_result': {'result_id': text, 'path': {'type': 'array'}},
    }
    return {'type':'object','oneOf': [object_schema({'action': {'const': action}, **value},
        ['action', *[k for k in value if k not in {'used_results', 'argument_refs'}]])
        for action, value in fields.items()]}


def action_errors(step):
    """Same branches as the wire Schema, with useful bounded feedback to the model."""
    if not isinstance(step, dict):
        return [{'field_path': '$', 'required': 'object'}]
    branch = next((s for s in action_schema()['oneOf'] if s['properties']['action']['const'] == step.get('action')), None)
    if branch is None:
        return [{'field_path': 'action', 'required': 'one registered action'}]
    errors = [{'field_path': k, 'required': 'top-level ' + branch['properties'][k].get('type', 'value')}
              for k in branch['required'] if k not in step]
    errors += [{'field_path': k, 'required': 'not allowed for ' + step['action']}
               for k in step if k not in branch['properties']]
    if not errors:
        try:
            validate_schema_instance(step, branch)
            if step['action'] == 'execute_python':
                for k in ('input_schema', 'output_schema'):
                    if step[k].get('type') != 'object':
                        errors.append({'field_path': k, 'required': 'object JSON Schema'})
        except ValueError as exc:
            errors.append({'field_path': '$', 'required': str(exc)[:500]})
    return errors


def public_spans(task):
    yield ['goal'], task.goal
    for i, choice in enumerate(task.inputs.get('choices', [])):
        if isinstance(choice, dict) and isinstance(choice.get('text'), str):
            yield ['inputs', 'choices', i, 'text'], choice['text']


def reference(path, text, match):
    return {'task_path': path, 'quote': match.group(0), 'start': match.start(), 'end': match.end(),
            'source_sha256': digest(text), 'qualification': 'public_expression_not_proven_proposition'}


def grounded_operations(task):
    """Parse only closed literal expressions; never infer a relation or theorem truth."""
    operations = {}
    fraction = r'(?:\\frac\{(-?\d{1,6})\}\{(-?\d{1,6})\}|\(?(-?\d{1,6})/(-?\d{1,6})\)?)'
    pattern = re.compile(r'(?<![\w/])' + fraction + r'\s*([+*]|\\times)\s*' + fraction + r'(?![\w/])')
    for path, text in public_spans(task):
        for m in pattern.finditer(text):
            a = [m[1] or m[3], m[2] or m[4]]
            b = [m[6] or m[8], m[7] or m[9]]
            if int(a[1]) and int(b[1]):
                operations.setdefault('exact_rational', {'arguments': {
                    'operation': 'add' if m[5] == '+' else 'multiply',
                    'a': dict(zip(('numerator','denominator'),map(int,a))),
                    'b': dict(zip(('numerator','denominator'),map(int,b)))}, 'references': [reference(path,text,m)]})
        for m in re.finditer(r'(?<!\w)(?:\\?gcd\s*\(\s*(-?\d{1,6})\s*,\s*(-?\d{1,6})\s*\)|(-?\d{1,6})\s+(?:mod|\\bmod)\s+([1-9]\d{0,5}))(?!\w)',text):
            operations.setdefault('exact_integer', {'arguments': {'operation': 'gcd' if m[1] else 'mod',
                'a': int(m[1] or m[3]), 'b': int(m[2] or m[4])}, 'references': [reference(path,text,m)]})
        for m in re.finditer(r'\\begin\{([pb])matrix\}(.{1,512}?)\\end\{\1matrix\}',text,re.S):
            rows=[[v.strip() for v in row.split('&')] for row in m[2].split('\\\\')]
            if (1<=len(rows)<=4 and all(len(row)==len(rows) for row in rows)
                    and all(re.fullmatch(r'-?\d{1,6}(?:/[1-9]\d{0,5})?',v) for row in rows for v in row)):
                operations.setdefault('exact_matrix',{'arguments':{'matrix':rows},'references':[reference(path,text,m)]})
    # Explicit typed public structures are accepted without an LLM declaring their truth.
    # They are operands of local operations, never certificates about all choices.
    structures = task.inputs.get('reference_inputs', {})
    if isinstance(structures, dict):
        for name in ('exact_rational', 'exact_integer', 'exact_matrix', 'finite_entailment', 'finite_quantifiers'):
            if isinstance(structures.get(name),dict):
                operations.setdefault(name, {'arguments': deepcopy(structures[name]),
                    'references': [{'task_path':['inputs','reference_inputs',name],
                        'value_sha256':digest(structures[name]), 'qualification':'explicit_public_local_operands'}]})
    return operations


# These patterns identify a stated obligation, not the answer or a theorem's truth.
OBLIGATIONS = {
    'LM01': r'\b(?:prove|disprove|counterexample)\b',
    'LM02': r'\\forall|\\exists|\bfor (?:every|all|each)\b|\bthere exists\b',
    'LM04': r'\b(?:unique(?:ness)?|existence)\b',
    'LM05': r'\b(?:counterexample|disprove)\b',
    'LM11': r'\b(?:converg(?:e|es|ence)|asymptotic|uniform(?:ly)?)\b|\\lim\b',
    'LM12': r'\b(?:implies|equivalent|necessary|sufficient)\b|\\(?:implies|iff)\b',
}


class CapabilityRouter:
    def __init__(self, view, index, registry):
        self.view, self.index, self.registry = view, index, registry
        self.task = None
        self.routes, self.audit = {}, {}

    def bind(self, task, context, *, enabled=True):
        self.task, self.context = task, context
        operations = grounded_operations(task)
        self.routes = {}
        guidance, programs, workflows, matched = [], [], [], []
        for meta in self.index['capabilities']:
            refs = []
            pattern = OBLIGATIONS.get(meta['code'])
            if pattern:
                for path,text in public_spans(task):
                    m = re.search(pattern,text,re.I)
                    if m: refs.append(reference(path,text,m))
            binding = operations.get(meta.get('operation_kind'))
            if binding: refs += binding['references']
            ready = False
            if binding and meta.get('program_id'):
                p = self.view.get(meta['program_id'])
                try: validate_schema_instance(binding['arguments'],p['input_schema']);ready=True
                except ValueError: pass
            status = 'ready' if ready or refs and not meta.get('program_id') else 'requires_grounding' if refs else 'not_applicable'
            self.routes[meta['capability_id']] = {'status':status,'references':refs, 'binding':binding if ready else None}
            if refs: matched.append(meta['capability_id'])
            if enabled and status == 'ready':
                guidance.append(meta['skill_id'])
                if ready and meta['program_id'] not in programs: programs.append(meta['program_id'])
        programs = programs[:2]
        if enabled:
            for w in self.index['workflows']:
                if w.get('program_id') in programs and w.get('routable'):
                    workflows.append(w['id'])
        self.selected = []
        knowledge_count = len(self.knowledge()) if enabled else 0
        self.selected = list(dict.fromkeys(guidance))[:max(0,2-knowledge_count)]
        self.programs, self.workflows = programs, workflows[:1]
        self.audit = {'version':VERSION, 'public_input_hash':digest([task.goal,task.inputs]),
            'catalog_available':{'skills':len(self.index['capabilities']), 'programs':len(self.view.qualified),
                                 'workflows':len(self.view.all('workflow'))},
            'matched':matched, 'input_ready':programs, 'injected':{'skills':self.selected,'programs':programs,'workflows':self.workflows},
            'routes':deepcopy(self.routes), 'result_refs_available':list(context.results),
            'attempted':[], 'executed':[], 'used':[], 'failed':[]}
        return [self.view.get(s) for s in self.selected]

    def binding(self, pid):
        for meta in self.index['capabilities']:
            route = self.routes.get(meta['capability_id'],{})
            if meta.get('program_id') == pid and route.get('binding'):
                return deepcopy(route['binding'])
        raise ValueError('Program inputs require verified public grounding')

    def validate_call(self, pid, arguments):
        if pid not in self.programs: raise ValueError('Program not input-ready for this public task')
        if arguments != self.binding(pid)['arguments']:
            raise ValueError('Arguments differ from exact public operands; model hypotheses are not verified inputs')

    def workflow_arguments(self, wid, arguments):
        if wid not in self.workflows: raise ValueError('Workflow not routable from current public inputs')
        meta = next(w for w in self.index['workflows'] if w['id']==wid)
        bound = self.binding(meta['program_id'])['arguments']
        if arguments and arguments != bound: raise ValueError('Workflow cannot override grounded public inputs')
        return bound

    def knowledge(self):
        """Only source-checked entries, as conditional obligations; no premise certification."""
        out = []
        for theorem in self.registry:
            if theorem['verification_status'] != 'source_checked': continue
            refs = []
            for path,text in public_spans(self.task):
                for phrase in theorem['public_triggers']:
                    m = re.search(re.escape(phrase),text,re.I)
                    if m: refs.append(reference(path,text,m))
            if not refs: continue
            out.append({'matching_theorems':[theorem['theorem_id']], 'statement':theorem['statement'],
                'verified_assumptions':[], 'unverified_assumptions':theorem['assumptions'],
                'possible_consequence':theorem['conclusion'], 'evidence_refs':refs,
                'source_identifier':theorem['source_identifier'], 'source_location':theorem['source_location'],
                'coverage_status':'insufficient_evidence', 'known_limitations':theorem['known_limitations'],
                'qualification':'source_checked_conditional_theorem_not_verified_task_application'})
        return out[:max(0,2-len(self.selected))]


def budget_snapshot(system):
    used, cap = system._budget('runtime')
    return {'metered_task_tokens':used, 'task_limit':cap,
        'ledger_headroom':system.budget_governor.headroom({**system.request_attribution,'purpose':'finish_only'})
            if system.budget_governor else None,
        'finish_completion_cap':system.resolve_call_settings('runtime','finish_only')['max_completion_tokens']}


def tool_headroom(system, task):
    """Admission of a complete bounded finish before a tool, using production serialization."""
    from .budget_governor import input_token_bound
    from .prompts import finish_text_prompt
    provider = system.provider('runtime','finish_only')
    material = {'goal':task.goal,'inputs':task.inputs,'local_results':[],'local_reads':[]}
    wire = provider._build_payload([{'role':'system','content':finish_text_prompt(system.adapter.answer_contract())},
        {'role':'user','content':json.dumps(material,ensure_ascii=False,allow_nan=False)}],[])
    cap = system.resolve_call_settings('runtime','finish_only')['max_completion_tokens']
    wire[provider.config.token_limit_field] = cap
    return input_token_bound(wire)+cap+2048 <= system._remaining('runtime','finish_only')
