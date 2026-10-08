"""One Learner decision and one durable realization job per completed Train task."""
from copy import deepcopy
import json

from .contracts import PublicTask, TrialCase, digest, validate_schema_instance
from .prompts import BUILD, BUILDER_PROMPT, LEARNING, LEARNER_PROMPT, GUIDANCE_LEARNING, GUIDANCE_LEARNER_PROMPT
from .task_context import TaskContext
from .program_worker import program_permission_view
from .model_view import project, model_task
from .program_submission import validate_program_declaration, submission_contract, ProgramContractError


class Learner:
    def __init__(self, system):
        self.system = system
        self.cases = [(PublicTask(**row['task']), row['experience']) for row in system.bank.train_cases()]

    def _experience(self, task, trace):
        return {'task': {'goal': task.goal, 'inputs': task.inputs}, 'events': trace.get('tools', []),
                'score': trace.get('score'), 'submission': trace.get('execution', {}).get('prediction'),
                'termination_reason': trace.get('execution', {}).get('reason'),
                'program_calls': trace.get('execution', {}).get('attempts', [])}

    def _view(self, experience):
        context = getattr(self.system, 'task_context', None) or TaskContext(self.system.config['runtime'])
        self.system.task_context = context
        view = {k: v for k, v in experience.items() if k != 'events'}
        original = experience.get('task')
        if original:
            view['task'] = model_task(PublicTask('', '', original['goal'], original.get('inputs', {})), self.system.adapter)
        view['events'] = []
        for index, event in enumerate(experience.get('events', [])):
            result_id = context.register(str(index), event.get('result', {}))
            view['events'].append({'name': event['name'], 'arguments': context.preview(event['arguments']),
                                   'result': context.view(result_id)})
        return view

    def _receive(self, key, stage, prompt, material, name, schema, **kwargs):
        s = self.system
        if s.checkpoint:
            snapshot = s.checkpoint.state.get(key + '_request_material')
            if snapshot is None:
                s.checkpoint.advance(s.checkpoint.state['stage'], **{
                    key + '_request_material': material, key + '_request_schema': schema})
            else:
                material = snapshot
                # A completed decision retains its original dynamic ID enum,
                # even when applying that decision has added assets to Bank.
                schema = s.checkpoint.state.get(key + '_request_schema', schema)
        semantics = digest([stage, prompt, material, name, schema,
            {k:v for k,v in kwargs.items() if k in ('completion_override','repair_limit','repair_reason','job_key')},
            s.config['experiment']['implementation_revision']])
        if s.checkpoint and key in s.checkpoint.state:
            if s.checkpoint.state.get(key + '_semantics') != semantics:
                raise ValueError('Cached learning response semantics changed')
            return s.checkpoint.state[key]
        value = s.agent(stage, prompt, material, name, schema, owner_state_version=key, **kwargs)
        if s.checkpoint:
            s.checkpoint.commit_decision(s.last_decision_id, 'applied', **{key: value, key + '_semantics': semantics})
        return value

    def _skill_references(self, proposal):
        known = {asset['id'] for asset in self.system.bank.all('skill')}
        if proposal.get('existing_skill_id') is not None and proposal['existing_skill_id'] not in known:
            raise ValueError('existing_skill_id must be an existing Skill ID')
        references = [node['skill_id'] for node in proposal.get('workflow', {}).get('nodes', []) if 'skill_id' in node]
        references += [ref for node in proposal.get('workflow', {}).get('nodes', []) for ref in node.get('reference_skill_ids', [])]
        if proposal.get('realization_request'):
            references.append(proposal['realization_request']['skill_id'])
        for reference in references:
            if reference == '$new' and proposal.get('skill'):
                continue
            if reference not in known:
                raise ValueError('Skill reference must be $new with a new Skill, or an existing Skill ID')

    def _proposal_skill(self, proposal):
        if proposal.get('skill'):
            skill = deepcopy(proposal['skill'])
            requested = proposal.get('generate_program', proposal['decision'] == 'propose_skill_and_program_spec')
            skill.setdefault('execution_intent', 'program_requested' if requested and
                self.system.adapter.capabilities.interaction != 'single_answer' else 'guidance_only')
            skill.setdefault('result_role', 'intermediate')
            skill = validate_program_declaration(skill, submission_contract(self.system.adapter))
            skill.setdefault('id', 'skill_' + digest(skill))
            return skill
        return self.system.bank.get(proposal.get('existing_skill_id') or '')

    def _resolve_realization_request(self, proposal):
        """Classify against actual assets without merging bindings or writing."""
        self._skill_references(proposal)
        bank, skill = self.system.bank, self._proposal_skill(proposal)
        request = proposal.get('realization_request')
        if request:
            skill = skill if request['skill_id'] == '$new' else bank.get(request['skill_id'])
        job = next((j for j in bank.jobs() if skill and j['id'] == digest(['realization', skill['id']])), None)
        program = bank.get(job.get('program_id') or '') if job else None
        associated = skill and job and job['skill_id'] == skill['id'] and job['skill_version'] == skill['id'] and any(
            i['skill_id'] == skill['id'] and i['program_id'] == job.get('program_id') for i in bank.all('implementation'))
        return {'skill': skill, 'job': job, 'already_usable': bool(request and associated and program and
            program['state'] == 'usable' and request['action'] in {'build', 'trial'})}

    def validate_learning_proposal(self, proposal, cases=None):
        resolved = self._resolve_realization_request(proposal)
        request, skill, job = proposal.get('realization_request'), resolved['skill'], resolved['job']
        if skill: validate_program_declaration(skill, submission_contract(self.system.adapter))
        if job and job.get('contract_quarantine'):
            raise ProgramContractError('program_quarantined', 'Create a validated new declaration version', 'declaration')
        if not request or resolved['already_usable']: return resolved
        if not skill or 'input_schema' not in skill: raise ValueError('Unknown realization Skill')
        if cases is None:
            cases = {r['task']['physical_key']: (PublicTask(**r['task']), r['experience'])
                     for r in self.system.bank.train_cases()}
        bindings, seen = request['case_bindings'], set()
        if len(bindings) > 2: raise ValueError('A realization request has at most two applicable Train bindings')
        fixed = {b['case_id']: b for b in job['case_bindings']} if job else {}
        for index, binding in enumerate(bindings):
            try:
                if binding['case_id'] in seen: raise ValueError('Duplicate trial case')
                seen.add(binding['case_id'])
                self._preflight_binding(binding, skill, cases)
                if binding['case_id'] in fixed and fixed[binding['case_id']] != binding and any(
                    a['task_key'] == binding['case_id'] and a['origin'] == 'train_test' and a['outcome'] != 'inapplicable'
                    for a in self.system.bank.attempts(job.get('program_id') or '')):
                    raise ValueError('An executed trial binding cannot be changed')
            except ValueError as exc:
                feedback = {'code': 'trial_binding_invalid', 'path': f'realization_request.case_bindings[{index}].inputs',
                    'case_id': binding['case_id'], 'missing_fields': sorted(set(skill['input_schema'].get('required', [])) - set(binding['inputs'])),
                    'expected_schema': skill['input_schema'], 'detail': str(exc)}
                error = ValueError(json.dumps(feedback, ensure_ascii=False))
                error.feedback = feedback
                error.repair_material = {'skill': skill, 'binding': binding, 'error': feedback,
                    'completed_train_case': self._view(cases[binding['case_id']][1]) if binding['case_id'] in cases else None}
                raise error from exc
        if len(set(fixed) | seen) > 2: raise ValueError('A realization job has at most two fixed slots')
        return resolved

    def related_candidates(self, task):
        bank = self.system.bank
        jobs = bank.jobs()
        related = []
        priority = []
        for job in jobs:
            program = bank.get(job.get('program_id') or '')
            linked = any(i['skill_id'] == job['skill_id'] and i['program_id'] == job.get('program_id')
                         for i in bank.all('implementation'))
            positives = {a['task_key'] for a in bank.attempts(job.get('program_id') or '') if a['outcome'] == 'positive'}
            tried = {a['task_key'] for a in bank.attempts(job.get('program_id') or '') if a['outcome'] != 'inapplicable'}
            if linked and program and program['state'] == 'candidate' and len(positives) == 1 and job['state'] not in {'done','deferred'} and not job.get('contract_quarantine') and (
                    len(job['case_bindings']) < 2 or any(b['case_id'] not in tried for b in job['case_bindings'])):
                asset = bank.get(job['skill_id'])
                priority.append((job, {**asset, 'current_program': program, 'current_job': job,
                    'used_physical_tasks': sorted(tried), 'independent_results': bank.attempts(program['id'])}))
        priority.sort(key=lambda row: (-len(bank.words(task.goal) & bank.words(row[1]['goal'])), row[0]['id']))
        ordered = [a for _, a in priority]
        ordered += [a for a in bank.retrieve(task.goal) if a['id'] not in {x['id'] for x in ordered}]
        for asset in ordered[:8]:
            pending = [j for j in jobs if j['skill_id'] == asset['id'] and j['state'] != 'done']
            related.append({**asset, 'pending': pending,
                'execution_intent': asset.get('execution_intent', 'program_requested' if pending else 'guidance_only'),
                'usable_programs': [p['id'] for p in bank.routes({'execution_mode': 'skill', 'skill_id': asset['id']})]
                if 'guidance' in asset else []})
        return related

    def learn(self, task, trace, *, focus=None):
        s, bank = self.system, self.system.bank
        experience = self._experience(task, trace)
        new_experience = not any(r['task']['physical_key'] == task.physical_key for r in bank.train_cases())
        if not any(t.physical_key == task.physical_key for t, _ in self.cases):
            self.cases.append((task, experience))
        for case_task, case in self.cases:
            bank.save_case(case_task, case)
        if s.adapter.capabilities.interaction == 'single_answer':
            return self._learn_guidance(task, experience)
        cases = {r['task']['physical_key']: (PublicTask(**r['task']), r['experience']) for r in bank.train_cases()}
        jobs = bank.jobs()
        related = self.related_candidates(task)
        tools = getattr(s.adapter, 'tool_definitions', s.adapter.available_tools)()
        if s.adapter.capabilities.interaction != 'single_answer':
            tools = [*tools, TaskContext(s.config['runtime']).tool()]
        log = {'decision': None, 'program': None, 'tests': [], 'errors': []}
        schema = deepcopy(LEARNING)
        skill_ids = [asset['id'] for asset in bank.all('skill')]
        if skill_ids:
            schema['properties']['existing_skill_id'] = {'type': 'string', 'enum': skill_ids}
        for properties in [schema['properties']['realization_request']['properties'],
                           schema['properties']['workflow']['properties']['nodes']['items']['properties']]:
            properties['skill_id'] = {'type': 'string', 'enum': ['$new', *skill_ids]}
        schema['properties']['workflow']['properties']['nodes']['items']['properties']['reference_skill_ids']['items'] = {
            'type': 'string', 'enum': ['$new', *skill_ids]}
        try:
            proposal = self._receive('learning_proposal', 'extractor', LEARNER_PROMPT, project('extractor',
                {'experience': self._view(experience), 'related': related, 'tools': tools,
                 'program_submission_contract': submission_contract(s.adapter),
                 'completed_train_cases': [{'case_id': key, 'task': case.get('task', {'goal': t.goal, 'inputs': t.inputs}),
                     'action_prefix': [{'name': e['name'], 'arguments': e['arguments']} for e in case.get('events', []) if e.get('backend_invoked', True)]}
                     for key, (t, case) in sorted(cases.items(), key=lambda row: (
                         -len(bank.words(task.goal) & bank.words(row[1][0].goal)), row[0]))[:8]],
                 'selected_local_goal': focus}, task=task, adapter=s.adapter, context=s.task_context), 'submit_learning', schema,
                validator=lambda p: self.validate_learning_proposal(p, cases), repair_limit=1)
            resolved = self.validate_learning_proposal(proposal, cases)
        except ValueError as exc:
            log.update(decision='rejected', errors=[str(exc)])
            return log
        log['decision'] = proposal['decision']
        skill = None
        if proposal.get('skill'):
            asset = self._proposal_skill(proposal)
            skill = bank.put('skill', asset)
        elif proposal.get('existing_skill_id'):
            skill = bank.get(proposal['existing_skill_id'])
            if not skill or 'input_schema' not in skill: raise ValueError('Unknown existing Skill')
        new_skill = skill if proposal.get('skill') else None
        request = proposal.get('realization_request')
        if request:
            selected_id = skill['id'] if request['skill_id'] == '$new' and skill else request['skill_id']
            skill = bank.get(selected_id)
            if not skill or 'input_schema' not in skill: raise ValueError('Unknown realization Skill')
        requested = skill and skill.get('execution_intent') != 'guidance_only' and (proposal.get('generate_program', proposal['decision'] == 'propose_skill_and_program_spec') or
                               skill.get('execution_intent') == 'program_requested' or request)
        if skill and skill.get('execution_intent') == 'guidance_only' and request:
            log['ignored_realization_request'] = 'guidance_only has no Program job'
        if requested:
            job_id = digest(['realization', skill['id']])
            job = next((j for j in jobs if j['id'] == job_id), None)
            if job is None:
                job = {'id': job_id, 'skill_id': skill['id'], 'skill_version': skill['id'], 'kind': 'build',
                       'state': 'waiting_example', 'program_id': None, 'case_bindings': [], 'repair_used': False,
                       'generation_count': 0, 'epoch': 0, 'last_error_kind': None, 'trigger_task': task.physical_key}
            if request:
                if resolved['already_usable']:
                    job['state'] = 'done'
                    log['realization_skipped'] = 'already_usable'
                else:
                    self._merge_request(job, request, skill, cases, experience, new_experience, log)
            bank.save_job(job)
        ready = [j for j in bank.jobs() if j['state'] == 'ready' and (j['skill_id'] == (skill or {}).get('id') or
            bank.words(task.goal) & bank.words((bank.get(j['skill_id']) or {}).get('goal','')))]
        ready.sort(key=lambda j: ({'trial': 0, 'repair': 1, 'build': 2}[j['kind']],
            j['id'] not in {old['id'] for old in jobs},
            -len(bank.words(task.goal) & bank.words((bank.get(j['skill_id']) or {}).get('goal',''))), j['id']))
        if ready:
            from ..core.errors import BudgetExhausted
            try: self._realize(ready[0], task, cases, tools, log)
            except BudgetExhausted:
                job = next(j for j in bank.jobs() if j['id'] == ready[0]['id'])
                job.update(state='deferred', last_error_kind='learning_budget_exhausted')
                bank.save_job(job)
                raise
        if proposal.get('workflow'):
            workflow = deepcopy(proposal['workflow'])
            for node in workflow['nodes']:
                if node.get('skill_id') == '$new':
                    node['skill_id'] = new_skill['id']
                if '$new' in node.get('reference_skill_ids', []):
                    node['reference_skill_ids'] = [new_skill['id'] if ref == '$new' else ref for ref in node['reference_skill_ids']]
            try: log['workflow'] = bank.put('workflow', workflow)['id']
            except ValueError as exc: log['errors'].append('Workflow rejected: ' + str(exc))
        return log

    def validate_guidance(self, proposal):
        validate_schema_instance(proposal, GUIDANCE_LEARNING)
        decision, old_id = proposal['decision'], proposal.get('existing_skill_id')
        old = None
        if 'existing_skill_id' in proposal or decision == 'reuse_existing':
            old = next((a for a in self.system.bank.all('skill') if a['id'] == old_id), None)
            if not old or old.get('execution_intent') != 'guidance_only' or not isinstance(old.get('guidance'), str) or not old['guidance'].strip():
                raise ValueError('existing_skill_id must identify a real nonempty guidance Skill')
        if decision == 'no_change' and ('existing_skill_id' in proposal or 'guidance_skill' in proposal):
            raise ValueError('no_change must not supply asset fields')
        if decision == 'reuse_existing' and 'guidance_skill' in proposal:
            raise ValueError('reuse_existing must not replace content')
        if decision == 'upsert_guidance':
            guidance = proposal.get('guidance_skill')
            if not guidance or any(not guidance[k].strip() for k in ('goal', 'guidance')):
                raise ValueError('upsert_guidance requires nonempty goal and guidance')
        return old

    def _learn_guidance(self, task, experience):
        s, bank = self.system, self.system.bank
        if s.checkpoint and 'guidance_learning_result' in s.checkpoint.state:
            return s.checkpoint.state['guidance_learning_result']
        related = bank.retrieve_guidance(task.goal, limit=8)
        log = {'decision': None, 'program': None, 'tests': [], 'errors': [], 'persisted_skill_id': None,
               'reused_skill_id': None, 'parent_skill_id': None, 'learning_rejected_reason': None,
               'retrieved_guidance_ids': [a['id'] for a in related],
               'injected_guidance_ids': [a['id'] for a in related]}
        try:
            proposal = self._receive('guidance_learning_proposal', 'extractor', GUIDANCE_LEARNER_PROMPT,
                {'experience': self._view(experience), 'related_guidance': related}, 'submit_learning',
                GUIDANCE_LEARNING, validator=self.validate_guidance, repair_limit=1)
            old = self.validate_guidance(proposal)
            log['decision'] = proposal['decision']
            if proposal['decision'] == 'reuse_existing':
                log['reused_skill_id'] = old['id']
            elif proposal['decision'] == 'upsert_guidance':
                guidance = proposal['guidance_skill']
                if old and all(old.get(k) == guidance[k] for k in ('goal', 'guidance')):
                    log['reused_skill_id'] = old['id']
                else:
                    asset = {**guidance, 'execution_intent': 'guidance_only', 'result_role': 'final_answer',
                             'input_schema': {'type': 'object'}, 'output_schema': {'type': 'object'}}
                    if old: asset['parent_skill_id'] = old['id']
                    saved = bank.put('skill', asset)
                    log.update(persisted_skill_id=saved['id'], parent_skill_id=saved.get('parent_skill_id'))
        except ValueError as exc:
            log.update(decision='rejected', rejected=True, errors=[str(exc)], learning_rejected_reason=str(exc))
        if s.checkpoint:
            s.checkpoint.advance(s.checkpoint.state['stage'], guidance_learning_result=log)
        return log

    def _merge_request(self, job, request, skill, cases, experience, new_experience, log):
        bank = self.system.bank
        if job.get('contract_quarantine'):
            raise ProgramContractError('program_quarantined', 'Quarantined job cannot be reactivated', 'declaration')
        if request['action'] == 'repair': job['stage'] = 'source_repair'
        job['kind'] = request['action'] if request['action'] != 'defer' else job['kind']
        existing = {b['case_id']: b for b in job['case_bindings']}
        new_binding = any(b['case_id'] not in existing for b in request['case_bindings'])
        existing.update({b['case_id']: deepcopy(b) for b in request['case_bindings']})
        job['case_bindings'] = list(existing.values())
        related_new = new_experience and bool(bank.words(experience['task']['goal']) & bank.words(skill['goal']))
        errors = [a for a in experience['program_calls'] if a.get('program_id') == job.get('program_id') and
                  (a.get('status') == 'execution_error' or a.get('outcome') == 'execution_failure')]
        error_signal = digest(errors) if errors else None
        new_error = error_signal and error_signal != job.get('error_signal')
        eligible = related_new or new_binding or new_error
        if new_error: job['error_signal'] = error_signal
        if request['action'] == 'defer': job['state'] = 'waiting_example'
        elif job['state'] != 'deferred' or eligible:
            if job['generation_count'] and request['action'] in {'build','repair'} and job['state'] in {'done','deferred','waiting_example'} and eligible:
                job.update(epoch=job['epoch']+1, generation_count=0, repair_used=False, last_error_kind=None)
                job.pop('pending_generation', None)
            job['state'] = 'ready' if job['case_bindings'] else 'waiting_example'

    def _preflight_binding(self, binding, skill, cases):
        if binding['case_id'] not in cases: raise ValueError('TrialCase must be a completed Train case')
        validate_schema_instance(binding['inputs'], skill['input_schema'])
        task, experience = cases[binding['case_id']]
        if task.split != 'train': raise ValueError('TrialCase must be a completed Train case')
        if binding['start_mode'] == 'reset' and binding['prefix']:
            raise ValueError('Reset trial cannot contain an action prefix')
        case = TrialCase(binding['case_id'], task.physical_key, task.task_id, deepcopy(binding['inputs']),
                         binding['start_mode'], tuple(binding['prefix']))
        actual = [{'name': e['name'], 'arguments': e['arguments']} for e in experience.get('events', [])
                  if e.get('backend_invoked', True)]
        if case.prefix and list(case.prefix) != actual[:len(case.prefix)]:
            raise ValueError('Trial prefix must be a real action prefix from this case')
        return case

    def _realize(self, job, task, cases, tools, log):
        s, bank = self.system, self.system.bank
        skill = bank.get(job['skill_id'])
        try:
            validate_program_declaration(skill, submission_contract(s.adapter))
        except ProgramContractError as exc:
            job.update(state='deferred', last_error_kind='declaration_contract')
            bank.save_job(job)
            log['errors'].append(str(exc))
            return
        bindings = job['case_bindings']
        if not bindings:
            job['state'] = 'waiting_example'; bank.save_job(job); return
        for binding in bindings: self._preflight_binding(binding, skill, cases)
        log['test_subjects'] = [b['case_id'] for b in bindings]
        examples = [{'case_id': b['case_id'], 'fixed_binding': b, 'history_domain': 'runtime_agent_operations',
                     **self._view(cases[b['case_id']][1])} for b in bindings]
        permissions = program_permission_view(tools, [*s.adapter.available_tools(), TaskContext(s.config['runtime']).tool()],
            tool_surface=s.adapter.capabilities.tool_surface, workspace=getattr(s.adapter, 'workspace', None),
            environment=s.config['program_environment'])
        def build_material(previous_failure=None):
            return self.builder_material(skill, bindings, examples, permissions, previous_failure)
        material = build_material()
        if job['kind'] == 'repair' and job.get('program_id'):
            previous = bank.get(job['program_id'])
            material = build_material({'domain': 'program_trial', 'source': previous['source'],
                'errors': [r.get('result', {}) for r in bank.attempts(previous['id'])[-2:]]})
        program = bank.get(job.get('program_id') or '') if job.get('program_id') and (
            job['kind'] == 'trial' or job.get('stage') == 'program_ready' or job['kind'] == 'build') else None
        if job.get('pending_candidate'):
            program = bank.put('program', job['pending_candidate'])
            bank.put('implementation', {'skill_id': skill['id'], 'program_id': program['id']})
            job.update(program_id=program['id'], stage='program_ready')
            job.pop('pending_candidate')
            bank.save_job(job)
        while True:
            if program is None:
                generation = job.get('pending_generation', job['generation_count'])
                if generation >= 2:
                    job['state'] = 'deferred'; break
                if 'pending_generation' not in job:
                    job.update(pending_generation=generation, generation_count=generation+1, trigger_task=task.physical_key)
                    bank.save_job(job)
                cap = s.config['learning']['builder_truncation_recovery_max_completion_tokens'] if job.get('last_error_kind') == 'length' else None
                try:
                    generated = self._receive('builder_' + job['id'] + '_' + str(job['epoch']) + '_' + str(generation),
                        'tool_builder', BUILDER_PROMPT, material, 'submit_program', BUILD, repair_limit=0,
                        completion_override=cap, repair_reason=job.get('last_error_kind'), job_key=[job['id'],job['epoch'],generation])
                    generated_bindings = {b['case_id']: b for b in generated['trial_inputs']}
                    if len(generated_bindings) != len(bindings) or set(generated_bindings) != {b['case_id'] for b in bindings}:
                        raise ValueError('Builder must bind exactly the fixed Train cases')
                    if generated_bindings != {b['case_id']: b for b in bindings}:
                        raise ValueError('Builder changed fixed trial inputs/start')
                    candidate = {'source': generated['source'], 'entry': 'run', 'backend': 'sandbox_python_v1',
                        'input_schema': skill['input_schema'], 'output_schema': skill['output_schema'],
                        'allowed_tools': permissions['allowed_names'],
                        'environment': s.config['program_environment'], 'result_role': skill.get('result_role','intermediate'),
                        'entry_constraints': skill.get('entry_constraints','undeclared')}
                    candidate = validate_program_declaration(candidate, submission_contract(s.adapter))
                    job['pending_candidate'] = candidate
                    bank.save_job(job)
                    program = bank.put('program', candidate)
                    bank.put('implementation', {'skill_id': skill['id'], 'program_id': program['id']})
                    job.update(program_id=program['id'], stage='program_ready', next_trial=0)
                    job.pop('pending_generation', None)
                    job.pop('pending_candidate', None)
                    bank.save_job(job)
                    log['trial_inputs'] = bindings
                    log['example_inputs'] = bindings[0]['inputs']
                except (ValueError, SyntaxError) as exc:
                    log['errors'].append(str(exc))
                    job.pop('pending_generation', None)
                    if job['repair_used']:
                        job.update(state='deferred', last_error_kind=getattr(exc,'finish_reason',None) or 'structure'); break
                    job.update(repair_used=True, last_error_kind=getattr(exc,'finish_reason',None) or 'structure')
                    bank.save_job(job)
                    material = build_material({'domain': 'builder_generation', 'error': str(exc)[:2048]})
                    continue
            log['program'] = program['id']
            failures = []
            for trial_index, binding in enumerate(bindings):
                if any(a['task_key'] == binding['case_id'] and a['origin']=='train_test' and a['outcome']!='inapplicable'
                       for a in bank.attempts(program['id'])): continue
                trial_case = self._preflight_binding(binding, skill, cases)
                test_task = cases[binding['case_id']][0]
                trial = s.test_program(program, trial_case.inputs, test_task,
                    trial_id=digest([program['id'], test_task.physical_key, binding, 'train_test']), trial_case=trial_case)
                log['tests'].append(trial)
                job['next_trial'] = trial_index + 1
                bank.save_job(job)
                if trial['outcome'] == 'execution_failure': failures.append(trial)
            if failures and not job['repair_used']:
                job.update(repair_used=True, last_error_kind='execution', stage='source_repair')
                bank.save_job(job)
                material = build_material({'domain': 'program_trial', 'source': program['source'], 'errors': failures})
                program = None
                continue
            actual = [a for a in bank.attempts(program['id']) if a['origin']=='train_test' and a['outcome']!='inapplicable']
            job['state'] = 'done' if len({a['task_key'] for a in actual}) >= 2 else 'waiting_example'
            job['kind'] = 'trial'
            if failures: job['state'] = 'deferred'
            break
        bank.save_job(job)
        if s.checkpoint: s.checkpoint.advance(s.checkpoint.state['stage'], realization_job=job)

    def builder_material(self, skill, bindings, examples, permissions, previous_failure=None):
        value = {'build_request': {'skill': skill, 'entry': 'def run(ctx, inputs)', 'fixed_bindings': bindings},
                 'submission_contract': {'tool_name': 'submit_program', 'input_schema': BUILD},
                 'program_submission_contract': submission_contract(self.system.adapter),
                 'future_program_api': {k:v for k,v in permissions.items() if k != 'workspace_capabilities'},
                 'workspace_capabilities': permissions['workspace_capabilities'], 'examples': examples}
        if previous_failure is not None: value['previous_failure'] = previous_failure
        return value
