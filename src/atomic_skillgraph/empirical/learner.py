"""One Learner decision and one durable realization job per completed Train task."""
from copy import deepcopy
from pathlib import Path
import json

from .contracts import PublicTask, TrialCase, digest, validate_schema_instance
from .prompts import BUILD, BUILDER_PROMPT, LEARNING, LEARNER_PROMPT, GUIDANCE_LEARNING, GUIDANCE_LEARNER_PROMPT
from .task_context import TaskContext
from .program_worker import program_permission_view
from .model_view import project, model_task, pack_material, learning_preview, learning_case, builder_example
from .program_submission import validate_program_declaration, submission_contract, ProgramContractError
from .local_validation import evidence_from_trace
from . import (LEARNING_MATERIAL_VERSION, GUIDANCE_POLICY_VERSION, CHOICE_GUIDANCE_MATERIAL_VERSION,
               CHOICE_GUIDANCE_POLICY_VERSION)


class Learner:
    def __init__(self, system):
        self.system = system
        self.cases = [(PublicTask(**row['task']), row['experience']) for row in system.bank.train_cases()]

    def _experience(self, task, trace):
        return {'task': {'goal': task.goal, 'inputs': task.inputs}, 'events': trace.get('tools', []),
                'score': trace.get('score'), 'submission': trace.get('execution', {}).get('prediction'),
                'termination_reason': trace.get('execution', {}).get('reason'),
                **{k: trace.get('execution', {}).get(k) for k in
                   ('answer_status', 'empty_answer', 'completion_truncated', 'provider_finish_reason')},
                'program_calls': trace.get('execution', {}).get('attempts', []),
                'result_store':deepcopy(trace.get('result_store',{})),
                'local_evidence':evidence_from_trace(
                    task,trace,self.system.config['program_environment'])}

    def _view(self, experience, *, source_id):
        if experience.get('local_evidence'):
            return learning_case(experience,source_id)
        context = getattr(self.system, 'task_context', None) or TaskContext(self.system.config['runtime'])
        self.system.task_context = context
        view = {k: v for k, v in experience.items() if k not in {'events','result_store','local_evidence','program_calls'}}
        view['local_evidence'] = deepcopy(experience.get('local_evidence', [])[-3:])
        view['program_calls'] = deepcopy(experience.get('program_calls', [])[-2:])
        view['projection_audit'] = {'original_event_count':len(experience.get('events',[])),
            'original_evidence_count':len(experience.get('local_evidence',[])),
            'original_utf8_bytes':len(json.dumps(experience,ensure_ascii=False).encode())}
        snapshot = digest(experience.get('events', []))
        view['case_source'] = {'physical_case_id': source_id, 'events_snapshot_sha256': snapshot,
                               'material_version': LEARNING_MATERIAL_VERSION}
        original = experience.get('task')
        if original:
            view['task'] = model_task(PublicTask('', '', original['goal'], original.get('inputs', {})), self.system.adapter)
        view['events'] = []
        events = experience.get('events', [])
        for index, event in list(enumerate(events))[-6:]:
            result_id = context.register(f'learner_case:{source_id}:{snapshot}:{index}',
                                        deepcopy(event.get('result', {})), verify_source=True)
            view['events'].append({'name': event['name'], 'arguments': learning_preview(context, event['arguments']),
                'result': {'result_id': result_id, **{k:learning_preview(context,v) for k,v in context.results[result_id].items()}}})
        return view

    def _receive(self, key, stage, prompt, material, name, schema, *, material_version=None, **kwargs):
        s = self.system
        version = material_version or LEARNING_MATERIAL_VERSION
        material = {**material, 'learning_material_version': version}
        if stage=='extractor' and 'image' in s.adapter.capabilities.input_modalities:
            material['content_parts']=s.adapter.content_parts()
        # Old paid responses remain attached to their original material. Migration is explicit.
        if s.checkpoint and key in s.checkpoint.state:
            raise ValueError('Legacy learning response requires explicit recovery/defer; no cross-version reuse')
        if s.checkpoint and key+'_request_material' in s.checkpoint.state:
            paid = any(d.get('owner_state_version') == key and d.get('status') == 'response_received'
                       for d in s.checkpoint.state.get('decisions', {}).values())
            audit = getattr(s, 'audit_path', None)
            if audit and Path(audit).exists():
                prior = json.loads(Path(audit).read_text())
                paid = paid or any(q.get('owner_state_version') == key and q.get('http_attempts')
                                   for q in prior.get('requests', []))
            if paid: raise ValueError('Paid legacy material requires explicit recovery/defer')
        key += '_' + version.replace('.', '_').replace('-', '_')
        if s.checkpoint:
            snapshot = s.checkpoint.state.get(key + '_request_material')
            if snapshot is None:
                s.checkpoint.advance(s.checkpoint.state['stage'], **{
                    key + '_request_material': material, key + '_request_schema': schema})
            else:
                if version == CHOICE_GUIDANCE_MATERIAL_VERSION and (snapshot != material or
                    s.checkpoint.state.get(key + '_request_schema') != schema):
                    raise ValueError('Choice guidance source/material/schema changed across recovery')
                material = snapshot
                # A completed decision retains its original dynamic ID enum,
                # even when applying that decision has added assets to Bank.
                schema = s.checkpoint.state.get(key + '_request_schema', schema)
        semantics = digest([stage, prompt, material, name, schema,
            {k:v for k,v in kwargs.items() if k in ('completion_override','repair_limit','repair_reason','job_key','decision_purpose')},
            s.config['experiment']['implementation_revision'], version, GUIDANCE_POLICY_VERSION])
        if version == CHOICE_GUIDANCE_MATERIAL_VERSION:
            semantics = digest([semantics, s.config['learning']['choice_guidance'],
                                s.config['llm'], s.resolve_call_settings(stage, kwargs.get('decision_purpose'))])
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
            skill.setdefault('execution_intent', 'program_requested' if requested else 'guidance_only')
            skill.setdefault('result_role', 'intermediate')
            skill = validate_program_declaration(skill, submission_contract(self.system.adapter))
            if self.system.config['learning'].get('choice_guidance',{}).get('enabled') and 'choices' in getattr(
                    getattr(self.system.adapter,'task',None),'inputs',{}):
                skill['guidance'] = ''
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
            bank.program_eligible(program) and request['action'] in {'build', 'trial'})}

    def validate_learning_proposal(self, proposal, cases=None):
        shape=deepcopy(LEARNING)
        binding_shape=shape['properties']['realization_request']['properties']['case_bindings']['items']
        binding_shape['required']=['case_id','inputs','local_evidence_ref','reference_fields']
        binding_shape['additionalProperties']=True
        validate_schema_instance(proposal,shape)
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
        bindings, seen, canonical, errors = request['case_bindings'], set(), [], []
        if len(bindings) > 2: raise ValueError('A realization request has at most two applicable Train bindings')
        fixed = {b['case_id']: b for b in job['case_bindings']} if job else {}
        for index, binding in enumerate(bindings):
            try:
                if binding['case_id'] in seen: raise ValueError('Duplicate trial case')
                seen.add(binding['case_id'])
                if binding['case_id'] not in cases:raise ValueError('TrialCase must be a completed Train case')
                from .local_validation import canonical_binding
                locked=canonical_binding(binding,cases[binding['case_id']][1],skill)
                self._preflight_binding(locked, skill, cases)
                canonical.append(locked)
                if binding['case_id'] in fixed and fixed[binding['case_id']] != locked and any(
                    a['task_key'] == binding['case_id'] and a['origin'] == 'train_test' and a['outcome'] != 'inapplicable'
                    for a in self.system.bank.attempts(job.get('program_id') or '')):
                    raise ValueError('An executed trial binding cannot be changed')
            except ValueError as exc:
                feedback=getattr(exc,'feedback',{'code':'trial_binding_invalid','errors':[{'code':'invalid_binding','detail':str(exc)}]})
                errors.append({'binding_index':index,'case_id':binding['case_id'],**feedback})
        if errors:
            error=ValueError(json.dumps(errors,ensure_ascii=False))
            error.feedback=errors
            from .model_view import structure_repair_material
            error.repair_material=structure_repair_material(proposal,errors,
                [self._view(cases[b['case_id']][1],source_id=b['case_id'])['local_evidence']
                 for b in bindings if b['case_id'] in cases])
            raise error
        if len(set(fixed) | seen) > 2: raise ValueError('A realization job has at most two fixed slots')
        resolved['canonical_bindings']=canonical
        return resolved

    def related_candidates(self, task):
        bank = self.system.bank
        jobs = bank.jobs()
        related = []
        ordered = bank.retrieve(task.goal)
        for asset in ordered[:8]:
            pending = [j for j in jobs if j['skill_id'] == asset['id'] and j['state'] != 'done']
            current = next((j for j in jobs if j['skill_id']==asset['id']),None)
            if current:
                program=bank.get(current.get('program_id') or '')
                attempts=bank.attempts(program['id']) if program else []
                asset={**asset,'current_job':current,'current_program':program,'independent_results':attempts,
                    'used_physical_tasks':sorted({a['task_key'] for a in attempts})}
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
        if s.config['learning'].get('choice_guidance',{}).get('enabled') and 'choices' in task.inputs:
            bank.update_choice_statistics()
        if s.adapter.capabilities.interaction == 'single_answer' and not experience['local_evidence']:
            if s.config['learning'].get('choice_guidance', {}).get('enabled') is True and 'choices' in task.inputs:
                return self._learn_grounded_choice_guidance(task, experience)
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
                {'experience': self._view(experience, source_id=task.physical_key), 'related': related, 'tools': tools,
                 'program_submission_contract': submission_contract(s.adapter),
                 'completed_train_cases': [learning_case(case,key)
                     for key, (t, case) in sorted(cases.items(), key=lambda row: (
                         -len(bank.words(task.goal) & bank.words(row[1][0].goal)), row[0])) if key!=task.physical_key][:2],
                 'selected_local_goal': focus}, task=task, adapter=s.adapter, context=s.task_context), 'submit_learning', schema,
                validator=lambda p: self.validate_learning_proposal(p, cases), repair_limit=1)
            resolved = self.validate_learning_proposal(proposal, cases)
        except ValueError as exc:
            log.update(decision='rejected', reason=getattr(exc,'code','proposal_invalid'), errors=[str(exc)])
            return log
        log['decision'] = proposal['decision']
        skill = None
        if proposal.get('skill'):
            asset = self._proposal_skill(proposal)
            # Choice guidance retains its checked-source contract. General operation
            # learning may publish code, but cannot bypass that contract with new text.
            if s.config['learning'].get('choice_guidance',{}).get('enabled') and 'choices' in task.inputs:
                log['text_publication'] = 'withheld_unchecked_choice_guidance'
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
                    self._merge_request(job, {**request,'case_bindings':resolved['canonical_bindings']}, skill, cases, experience, new_experience, log)
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

    @staticmethod
    def _guidance_evidence(experience):
        """Select evidence by its public shape, not benchmark or guessed gold."""
        if experience.get('empty_answer') or experience.get('completion_truncated') or not experience.get('submission'):
            return {'eligible': False, 'kind': 'insufficient_reusable_evidence'}
        if (experience.get('score') or {}).get('hard') is True:
            return {'eligible': True, 'kind': 'correct_public_submission', 'qualification': 'advisory_not_proof'}
        # Real checked operation results remain learnable; a scalar wrong-label signal is insufficient.
        checked = [e for e in experience.get('events', []) if e.get('backend_invoked') and
                   isinstance(e.get('result'), dict) and type(e['result'].get('accepted')) is bool and
                   any(e['result'].get(k) for k in ('data', 'outputs', 'error'))]
        return {'eligible': bool(checked), 'kind': 'public_checked_process' if checked else 'insufficient_reusable_evidence'}

    def _learn_guidance(self, task, experience):
        s, bank = self.system, self.system.bank
        result_key = 'guidance_learning_result_' + GUIDANCE_POLICY_VERSION.replace('.', '_')
        if s.checkpoint and result_key in s.checkpoint.state:
            return s.checkpoint.state[result_key]
        related = bank.retrieve_guidance(task.goal, limit=8)
        log = {'decision': None, 'program': None, 'tests': [], 'errors': [], 'persisted_skill_id': None,
               'reused_skill_id': None, 'parent_skill_id': None, 'learning_rejected_reason': None,
               'retrieved_guidance_ids': [a['id'] for a in related],
               'injected_guidance_ids': [a['id'] for a in related]}
        evidence = {**self._guidance_evidence(experience), 'source_task_id': task.task_id,
                    'physical_key': task.physical_key, 'policy_version': GUIDANCE_POLICY_VERSION,
                    'creation_order': len(bank.train_cases()) - 1, 'origin': 'host'}
        log['evidence'] = evidence
        if not evidence['eligible']:
            log.update(decision='no_change', decision_origin='host', reason='insufficient_reusable_evidence')
            if s.checkpoint: s.checkpoint.advance(s.checkpoint.state['stage'], **{result_key: log})
            return log
        try:
            proposal = self._receive('guidance_learning_proposal', 'extractor', GUIDANCE_LEARNER_PROMPT,
                {'experience': self._view(experience, source_id=task.physical_key), 'related_guidance': related,
                 'host_evidence': evidence}, 'submit_learning', GUIDANCE_LEARNING, validator=self.validate_guidance,
                repair_limit=s.config['learning']['guidance_repair_limit'], decision_purpose='guidance_learning')
            old = self.validate_guidance(proposal)
            log['decision'] = proposal['decision']
            if proposal['decision'] == 'reuse_existing':
                log['reused_skill_id'] = old['id']
            elif proposal['decision'] == 'upsert_guidance':
                guidance = proposal['guidance_skill']
                if old and all(old.get(k) == guidance[k] for k in ('goal', 'guidance')):
                    log['reused_skill_id'] = old['id']
                else:
                    asset = {**guidance, 'evidence_source': evidence, 'execution_intent': 'guidance_only', 'result_role': 'final_answer',
                             'input_schema': {'type': 'object'}, 'output_schema': {'type': 'object'}}
                    if old: asset['parent_skill_id'] = old['id']
                    saved = bank.put('skill', asset)
                    log.update(persisted_skill_id=saved['id'], parent_skill_id=saved.get('parent_skill_id'))
        except ValueError as exc:
            log.update(decision='rejected', rejected=True, errors=[str(exc)], learning_rejected_reason=str(exc))
        if s.checkpoint:
            s.checkpoint.advance(s.checkpoint.state['stage'], **{result_key: log})
        return log

    def _learn_grounded_choice_guidance(self, task, experience):
        from .choice_guidance import build_verified_public_source, validate_proposal, validate_check, render, words
        from .prompts import CHOICE_PROPOSAL, GUIDANCE_CHECK, CHOICE_LEARNER_PROMPT, GUIDANCE_CHECK_PROMPT
        s, bank = self.system, self.system.bank
        key = 'choice_guidance_result_v1'
        if s.checkpoint and key in s.checkpoint.state: return s.checkpoint.state[key]
        source = build_verified_public_source(task, experience, {
            'projection_version': s.request_attribution.get('projection_version') or
                                  s.config['experiment'].get('choice_projection_version'),
            'source_run_id': s.request_attribution.get('source_run_id') or s.config['experiment'].get('run_id') or
                             getattr(s.observer, 'manifest', {}).get('run_id'),
            'source_record_hash': s.request_attribution.get('source_record_hash')})
        log = {'decision': None, 'program': None, 'tests': [], 'errors': [], 'evidence': source,
               'guidance_policy_version': CHOICE_GUIDANCE_POLICY_VERSION, 'persisted_skill_id': None}
        def finish():
            if s.checkpoint: s.checkpoint.advance(s.checkpoint.state['stage'], **{key: log})
            return log
        if not source['eligible']:
            log.update(decision='no_change', decision_origin='host', reason=source['reason'])
            return finish()
        policy = {**s.config['runtime'].get('choice_guidance', {}),
                  'max_items': s.config['learning']['choice_guidance']['max_related_assets'], 'max_total_chars': 2400}
        selection = bank.select_guidance(task, policy)
        related = selection['selected']
        context = {'source': source, 'related': related, 'selection': selection}
        if s.checkpoint:
            saved = s.checkpoint.state.get('choice_guidance_context_v1')
            if saved:
                if saved['source'] != source: raise ValueError('Choice source changed across recovery')
                context = saved
            else: s.checkpoint.advance(s.checkpoint.state['stage'], choice_guidance_context_v1=context)
        related, selection = context['related'], context['selection']
        log['selection_audit'] = selection
        schema = deepcopy(CHOICE_PROPOSAL)
        if related: schema['properties']['existing_skill_id']['enum'] = [a['id'] for a in related]
        else: schema['properties'].pop('existing_skill_id')
        validator = lambda p: validate_proposal(p, source, related, schema)
        material = {'source': deepcopy(source), 'public_choices': deepcopy(task.inputs['choices']),
                    'source_context': deepcopy(task.inputs), 'related_guidance': [render(a) for a in related],
                    'normalized_source_topic_words': sorted(words(task.goal)), 'normalizer_version': 'choice-guidance.normalizer.v1'}
        try:
            proposal = self._receive('choice_guidance_proposal', 'extractor', CHOICE_LEARNER_PROMPT, material,
                'submit_learning', schema, validator=validator, repair_limit=s.config['learning']['guidance_repair_limit'],
                decision_purpose='guidance_learning', material_version=CHOICE_GUIDANCE_MATERIAL_VERSION)
            old = validator(proposal)
            log.update(decision=proposal['decision'], proposal=proposal, proposal_hash=digest(proposal))
            if proposal['decision'] == 'reuse_existing':
                log.update(reused_skill_id=old['id'], source_association={'source': source, 'skill_id': old['id'],
                           'qualification': 'association_not_additional_proof'})
            elif proposal['decision'] == 'upsert_guidance':
                check = self._receive('choice_guidance_grounding_' + digest(proposal), 'extractor', GUIDANCE_CHECK_PROMPT,
                    {'source': deepcopy(source), 'public_choices': deepcopy(task.inputs['choices']), 'proposal': proposal,
                     'proposal_hash': digest(proposal), 'policy_version': CHOICE_GUIDANCE_POLICY_VERSION},
                    'submit_guidance_check', GUIDANCE_CHECK,
                    validator=lambda c: validate_check(c, proposal, source, GUIDANCE_CHECK), repair_limit=0,
                    decision_purpose='guidance_grounding', material_version=CHOICE_GUIDANCE_MATERIAL_VERSION)
                log['grounding_check'] = check
                if not validate_check(check, proposal, source, GUIDANCE_CHECK):
                    log.update(decision='rejected', rejected=True, reason='grounding_' + check['status'])
                else:
                    asset = {k: deepcopy(proposal[k]) for k in ('goal', 'guidance', 'scope_terms', 'applicability')}
                    asset.update(evidence_source=deepcopy(source), grounding_check={**check, 'source_hash': digest(source)},
                        grounded_proposal=deepcopy(proposal), guidance_policy_version=CHOICE_GUIDANCE_POLICY_VERSION,
                        execution_intent='guidance_only', result_role='final_answer',
                        input_schema={'type': 'object'}, output_schema={'type': 'object'})
                    if old: asset['parent_skill_id'] = old['id']
                    saved = bank.put('skill', asset)
                    log.update(persisted_skill_id=saved['id'], parent_skill_id=saved.get('parent_skill_id'))
        except ValueError as exc:
            log.update(decision='rejected', rejected=True, reason='proposal_or_grounding_invalid', errors=[str(exc)])
        return finish()

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
                job.update(epoch=job['epoch']+1, last_error_kind=None)
                job.pop('pending_generation', None)
            job['state'] = 'ready' if job['case_bindings'] else 'waiting_example'
        if not job.get('program_id') and len({cases[b['case_id']][0].physical_key for b in job['case_bindings']}) < self.system.config['learning']['min_distinct_train_cases_before_first_build']:
            job['state'] = 'waiting_example'

    def _preflight_binding(self, binding, skill, cases):
        if binding['case_id'] not in cases: raise ValueError('TrialCase must be a completed Train case')
        validate_schema_instance(binding['inputs'], skill['input_schema'])
        task, experience = cases[binding['case_id']]
        from .local_validation import resolve_evidence
        resolve_evidence(binding,experience,skill)
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
        if s.budget_governor and not s.budget_governor.validation_available():
            job.update(state='deferred', last_error_kind='local_validation_budget_exhausted')
            bank.save_job(job)
            log['realization_skipped'] = 'local_validation_budget_exhausted'
            return
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
        try:
            for binding in bindings: self._preflight_binding(binding, skill, cases)
        except ValueError as exc:
            job.update(state='deferred', last_error_kind='local_binding_invalid')
            bank.save_job(job)
            log['errors'].append(str(exc))
            return
        if not job.get('program_id') and len({cases[b['case_id']][0].physical_key for b in bindings}) < s.config['learning']['min_distinct_train_cases_before_first_build']:
            job['state'] = 'waiting_example'; bank.save_job(job); return
        log['test_subjects'] = [b['case_id'] for b in bindings]
        from .local_validation import resolve_evidence
        examples = [builder_example(b,resolve_evidence(b,cases[b['case_id']][1],skill)) for b in bindings]
        permissions = program_permission_view(tools, [*s.adapter.available_tools(), TaskContext(s.config['runtime']).tool()],
            tool_surface=s.adapter.capabilities.tool_surface, workspace=getattr(s.adapter, 'workspace', None),
            environment=s.config['program_environment'])
        def build_material(previous_failure=None):
            return self.builder_material(skill, bindings, examples, permissions, previous_failure)
        material = build_material()
        if job['kind'] == 'repair' and job.get('program_id'):
            previous = bank.get(job['program_id'])
            material = build_material({'domain': 'program_trial', 'program_id': previous['id'], 'source': previous['source'],
                'errors': bank.attempts(previous['id'])[-2:]})
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
                cap = s.config['learning']['builder_truncation_recovery_max_completion_tokens'] if job.get('repair_used') or job['kind']=='repair' else None
                try:
                    generated = self._receive('builder_' + job['id'] + '_' + str(job['epoch']) + '_' + str(generation),
                        'tool_builder', BUILDER_PROMPT, material, 'submit_program', BUILD, repair_limit=0,
                        completion_override=cap, repair_reason=job.get('last_error_kind'), job_key=[job['id'],job['epoch'],generation])
                    if generated.get('binding_hash',digest(bindings)) != digest(bindings):
                        raise ValueError('Builder changed Host binding identity')
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
                    if getattr(exc,'code',None)=='material_too_large':
                        job.update(state='deferred',last_error_kind='material_too_large'); break
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
                    trial_id=digest([program['id'], test_task.physical_key, binding, 'train_test']), trial_case=trial_case,
                    local_binding=binding,source_experience=cases[binding['case_id']][1])
                log['tests'].append(trial)
                job['next_trial'] = trial_index + 1
                bank.save_job(job)
                if trial['outcome'] == 'execution_failure': failures.append(trial)
            if failures and not job['repair_used']:
                job.update(repair_used=True, last_error_kind='execution', stage='source_repair')
                bank.save_job(job)
                material = build_material({'domain': 'program_trial', 'program_id': program['id'], 'source': program['source'], 'errors': failures})
                program = None
                continue
            actual = [a for a in bank.attempts(program['id']) if a['origin']=='train_test' and a['outcome']!='inapplicable']
            job['state'] = 'done'
            job['kind'] = 'trial'
            if failures: job['state'] = 'deferred'
            break
        bank.save_job(job)
        if s.checkpoint: s.checkpoint.advance(s.checkpoint.state['stage'], realization_job=job)

    def builder_material(self, skill, bindings, examples, permissions, previous_failure=None):
        value = {'build_request': {'skill': skill, 'entry': 'def run(ctx, inputs)', 'binding_hash':digest(bindings)},
                 'program_submission_contract': submission_contract(self.system.adapter),
                 'future_program_api': {k:v for k,v in permissions.items() if k != 'workspace_capabilities'},
                 'workspace_capabilities': permissions['workspace_capabilities'], 'examples': examples}
        if 'image' in self.system.adapter.capabilities.input_modalities:
            from ..harness.benchmarks import image_content_parts
            paths=list(dict.fromkeys(path for b in bindings for t,case in self.cases if t.physical_key==b['case_id']
                                    for path in t.inputs.get('images',[])))
            value['content_parts']=image_content_parts(paths)
        if previous_failure is not None: value['previous_failure'] = self._failure_view(previous_failure)
        return pack_material(value)

    def _failure_view(self, failure):
        """Keep one source and bounded, readable evidence; full trial audit stays unchanged."""
        def bounded(v):
            raw = json.dumps(v, ensure_ascii=False)
            return v if len(raw) <= 2048 else {'preview': raw[:2048], 'chars': len(raw), 'truncated': True}
        def walk(v, key=''):
            if key in {'tools','history'} and isinstance(v, list):
                return {'count': len(v), 'operations': [{k:bounded(e[k]) for k in
                    ('name','arguments','state','error','progress_before','progress_after') if k in e} for e in v],
                    'last_results': [bounded(e.get('result', {})) for e in v[-2:]]}
            if isinstance(v, dict): return {k:walk(x,k) for k,x in v.items() if k != 'source'}
            if isinstance(v, list): return [walk(x,key) for x in v[-2:]] if key == 'errors' else bounded(v)
            return bounded(v)
        return {**walk(failure), **({'source': failure['source']} if 'source' in failure else {})}
