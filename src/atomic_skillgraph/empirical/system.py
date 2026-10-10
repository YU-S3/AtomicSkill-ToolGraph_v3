"""The empirical production chain; no legacy System is constructed."""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import json
import os
from uuid import uuid4

from ..agents.protocol import AgentTurn, NativeToolSpec, validate_schema_instance
from ..agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider, ProviderAgentProtocolError, ProviderProtocolError
from ..agents.usage import UsageLedger
from ..agents.session import repair_messages
from ..core.errors import BudgetExhausted, FailureLayer
from ..harness.simple_protocol import Broker, EpisodeResult
from . import (PROFILE, POLICY_DEFAULTS, CHOICE_GUIDANCE_POLICY_VERSION, CHOICE_GUIDANCE_MATERIAL_VERSION,
               CHOICE_GUIDANCE_SELECTION_VERSION, SINGLE_ANSWER_PROMPT_VERSION)
from .bank import Bank
from .contracts import PublicTask, RuntimeDecision, digest
from .task_context import TaskContext
from .executor import Executor
from .learner import Learner
from .planner import Planner, dynamic
from .program_worker import ProgramWorker
from .program_submission import normalize_program_result, positive_eligible
from .trial_snapshot import seal_trial_workspace, restore_trial_workspace, exception_details, host_call


STAGE_BUCKETS = {"planner": "planner_p1", "runtime": "runtime_dynamic", "extractor": "extractor_e1",
                "tool_builder": "tool_builder_evolution"}


def validate_config(config):
    if config.get("mechanism_profile") != PROFILE:
        raise ValueError("Empirical System requires its explicit profile")
    config = deepcopy(config)
    overrides = config.get('llm', {}).get('purpose_overrides', {})
    config['llm'].setdefault('purpose_overrides',overrides)
    overrides.setdefault('learning_structure_repair',{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048})
    if overrides['learning_structure_repair'] != {'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048}:
        raise ValueError('Structural learning repair requires disabled thinking and 2048 completion')
    if set(overrides) - {'finish_only', 'guidance_learning', 'guidance_grounding', 'learning_structure_repair'}:
        raise ValueError('Unsupported purpose override')
    for settings in overrides.values():
        if set(settings) - {'protocol', 'max_completion_tokens', 'reasoning_effort'}:
            raise ValueError('Unsupported purpose setting')
        if set(settings.get('protocol', {})) - {'thinking_type'} or settings.get('protocol', {}).get('thinking_type', 'enabled') not in {'enabled','disabled'}:
            raise ValueError('Invalid purpose protocol')
        if type(settings.get('max_completion_tokens', 512)) is not int or settings.get('max_completion_tokens', 512) < 1:
            raise ValueError('Invalid purpose completion cap')
    for section, defaults in POLICY_DEFAULTS.items():
        config[section] = {**defaults, **config.get(section, {})}
        for key, value in defaults.items():
            if config[section][key] != value:
                raise ValueError('Unsupported execution policy ' + section + '.' + key)
    learning = config['learning']
    learning.setdefault('guidance_repair_limit', 1)
    if type(learning['guidance_repair_limit']) is not int or learning['guidance_repair_limit'] not in {0, 1}:
        raise ValueError('Guidance repair limit must be 0 or 1')
    if type(learning['min_distinct_train_cases_before_first_build']) is not int or learning['min_distinct_train_cases_before_first_build'] != 1:
        raise ValueError('First build requires one real locally bound Train case')
    if config.get("schema_version") != "empirical.v1":
        raise ValueError("Empirical System requires a fresh empirical.v1 schema")
    choices = {
        'learning': {'enabled': False, 'policy_version': CHOICE_GUIDANCE_POLICY_VERSION,
                     'material_version': CHOICE_GUIDANCE_MATERIAL_VERSION, 'max_related_assets': 2,
                     'source_check_required': True},
        'runtime': {'enabled': False, 'selection_version': CHOICE_GUIDANCE_SELECTION_VERSION,
                    'prompt_version': SINGLE_ANSWER_PROMPT_VERSION, 'max_items': 2, 'max_total_chars': 2400}}
    for section, defaults in choices.items():
        supplied = config[section].get('choice_guidance')
        if supplied is None: continue
        if not isinstance(supplied, dict) or set(supplied) - set(defaults):
            raise ValueError('Invalid ' + section + '.choice_guidance fields')
        supplied = {**defaults, **supplied}
        for key, expected in defaults.items():
            value = supplied[key]
            if type(value) is not type(expected) or (key != 'enabled' and value != expected):
                raise ValueError('Invalid choice_guidance policy: ' + key)
        config[section]['choice_guidance'] = supplied
    if (config['learning'].get('choice_guidance', {}).get('enabled') is True and
            config['runtime'].get('choice_guidance', {}).get('enabled') is not True):
        raise ValueError('Choice guidance learning requires choice guidance runtime selection')
    if config['learning'].get('choice_guidance', {}).get('enabled'):
        for purpose, cap in [('guidance_learning', 2048), ('guidance_grounding', 1536)]:
            settings = overrides.get(purpose, {})
            if settings.get('protocol', {}).get('thinking_type') != 'disabled' or settings.get('max_completion_tokens') != cap:
                raise ValueError('Choice guidance requires bounded disabled-thinking override: ' + purpose)
    allowed = {'mechanism_profile', 'schema_version', 'data_dir', 'experiment', 'harness', 'runtime',
               'learning', 'planning', 'program_worker', 'program_environment', 'llm', 'benchmark_profile', 'manifest', 'budget'}
    if set(config) - allowed:
        raise ValueError('Legacy or unsupported configuration keys: ' + ', '.join(sorted(set(config) - allowed)))
    if not config.get('llm', {}).get('model') or not config['llm'].get('api_key_env'):
        raise ValueError('Actual model ID and key environment variable name are required')
    for stage in STAGE_BUCKETS:
        if config['llm'].get(stage,{}).get('model',config['llm']['model']) != config['llm']['model']:
            raise ValueError('All roles must use the same configured backbone')
    if 'budget' in config:
        limits = config['budget']
        if (not isinstance(limits,dict) or set(limits)-{'token_limit','request_limit','finish_reserve','validation_limit','train_task_tokens','eval_task_tokens','train_solve_tokens','train_learning_tokens'}
                or any(type(v) is not int or v < (0 if k=='finish_reserve' else 1) for k,v in limits.items())
                or not all(k in limits for k in ('token_limit','request_limit','finish_reserve'))
                or limits['finish_reserve'] >= limits['token_limit']):
            raise ValueError('Invalid explicit whole-run budget')
        pools=('train_solve_tokens','train_learning_tokens')
        if any(k in limits for k in pools) and (not all(k in limits for k in (*pools,'train_task_tokens')) or
                sum(limits[k] for k in pools)>limits['train_task_tokens']):
            raise ValueError('Train pools must fit the explicit parent budget')
    experiment = config.get('experiment', {})
    if experiment.get('seed') not in {42, 43, 44} or experiment.get('runtime_mode') not in {'online', 'frozen'} or not experiment.get('output_dir'):
        raise ValueError('Explicit seed, output directory and online/frozen mode are required')
    if not config.get('harness', {}).get('adapter') or not config.get('runtime', {}).get('global_action_budget'):
        raise ValueError('Benchmark adapter and native call budget are required')
    if not config.get('benchmark_profile') or not config.get('manifest'):
        raise ValueError('Explicit benchmark profile and fixed manifest are required')
    environment = config.get('program_environment', {})
    if not environment.get('adapter_abi') or not environment.get('image_digest'):
        raise ValueError('Lock the Adapter ABI and container image before running')
    if environment['adapter_abi'] not in {'simple.v1','simple.v2'}:
        raise ValueError('Unsupported Adapter ABI')
    environment['adapter_abi'] = 'simple.v2'
    for section, limits in {"learning": {"max_new_programs_per_task": 1, "builder_repair_limit": 1,
        "max_train_test_cases_per_program_version": 2}, "planning": {"retrieval_top_k": 8,
        "structural_repair_limit": 1, "task_replan_limit": 1}, "runtime": {"repeated_unchanged_failure_limit": 2}}.items():
        for key, expected in limits.items():
            if config.get(section, {}).get(key, expected) != expected:
                raise ValueError("Unsupported empirical policy " + section + "." + key)
    return config


class EmpiricalSystem:
    def __init__(self, config, *, harness=None, provider=None, readonly=None, adapter_factory=None,
                 bank_view_factory=None, budget_governor=None):
        self.config = validate_config(config)
        self.readonly = bool(self.config.get("experiment", {}).get("runtime_mode") == "frozen") if readonly is None else readonly
        self.bank = Bank(self.config["data_dir"], readonly=self.readonly,
                         seed=self.config.get("experiment", {}).get("seed", 42),environment=self.config['program_environment'])
        if bank_view_factory:
            if not self.readonly: raise ValueError('Bank views require readonly evaluation')
            self.bank = bank_view_factory(self.bank)
        from .budget_governor import BudgetGovernor
        budget = self.config.get('budget')
        self.budget_governor = budget_governor or (BudgetGovernor(Path(self.config['experiment']['output_dir'])/'budget.json',
            **{k:v for k,v in budget.items() if k in {'token_limit','request_limit','finish_reserve','validation_limit'}}) if budget else None)
        self.request_attribution = {}
        self.adapter_factory = adapter_factory
        if harness is None:
            from ..harness.registry import create_simple_harness
            self.adapter_factory = self.adapter_factory or (lambda: create_simple_harness(self.config))
            harness = self.adapter_factory()
        self.adapter = harness
        if 'image' in self.adapter.capabilities.input_modalities and 'image' not in self.config['llm'].get('input_modalities',['text']):
            self.bank.close()
            raise ValueError('Model capability lock does not support image input')
        self.provider_override = provider
        self.providers = {}
        self.usage = UsageLedger()
        self.requests = []
        self.planner = Planner(self.bank, self.agent)
        self.worker = ProgramWorker(self.config.get("program_worker"))
        self.executor = Executor(self.bank, self.agent, self.worker, self.planner, frozen=self.readonly)
        self.learner = Learner(self)
        self._task_start = 0
        self._runtime_start = 0
        self._learning_start = None
        self.audit_path = None
        self._request_start = 0
        self.checkpoint = None
        self.phase = 'train'
        self.budget_scope = None
        self.prior_usage = []
        self.observer = None
        self.task_context = None
        self.last_decision_id = None

    def _budget(self, stage):
        if stage == 'runtime':
            prefix, cap = {'runtime_dynamic'}, self.config['llm'].get('runtime', {}).get('max_total_tokens_per_task', 600000)
        elif stage == 'planner':
            prefix, cap = {'planner_p1'}, self.config['llm'].get('planner', {}).get('max_total_tokens_per_phase', 120000)
        else:
            prefix, cap = {'extractor_e1','tool_builder_evolution'}, self.config['llm'].get('extractor', {}).get('max_total_tokens_per_task', 262144)
        usage = {e['event_id']: e for e in self.prior_usage}
        usage.update({e.event_id: e.to_dict() for e in self.usage.events[self._task_start:]})
        used = sum(e['total_tokens'] for e in usage.values() if e['bucket'] in prefix and
            e.get('provider_metadata', {}).get('budget_scope') == self.budget_scope)
        return used, cap

    def _save_requests(self):
        if self.audit_path is not None:
            path = Path(self.audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.tmp')
            prior = json.loads(path.read_text()) if path.exists() else {'requests': [], 'usage': []}
            requests = {r['id']: r for r in prior['requests']}
            for record in self.requests[self._request_start:]:
                requests[record['id']] = {**requests.get(record['id'], {}), **record}
            usage = {e['event_id']: e for e in prior['usage']}
            usage.update({e.event_id: {**e.to_dict(), 'phase': e.provider_metadata.get('phase')} for e in self.usage.events[self._task_start:]})
            temporary.write_text(json.dumps({'requests': list(requests.values()),
                'usage': list(usage.values())}, ensure_ascii=False), encoding='utf-8')
            os.replace(temporary, path)
            if self.observer:
                self.observer.requests(list(requests.values()))

    def resolve_call_settings(self, stage, purpose):
        llm = self.config['llm']
        settings = {**llm, **llm.get(stage, {})}
        override = llm.get('purpose_overrides', {}).get(purpose, {})
        settings.update({k:v for k,v in override.items() if k != 'protocol'})
        settings['protocol'] = {**llm.get('protocol', {}), **llm.get(stage, {}).get('protocol', {}), **override.get('protocol', {})}
        if settings['model'] != llm['model']:
            raise ValueError('All roles must use the same configured backbone')
        return settings

    def provider(self, stage, purpose=None):
        if self.provider_override is not None:
            return self.provider_override.get(stage) if isinstance(self.provider_override, dict) else self.provider_override
        settings = self.resolve_call_settings(stage, purpose)
        key = (stage, purpose, digest(settings))
        if key not in self.providers:
            llm = self.config["llm"]
            self.providers[key] = OpenAICompatibleProvider(OpenAICompatibleConfig(
                base_url=llm["base_url"], model=llm["model"], api_key_env=llm["api_key_env"],
                dialect=llm.get('dialect', 'deepseek_v4_chat'),
                input_modalities=tuple(llm.get('input_modalities', ['text'])),
                token_limit_field=llm.get('token_limit_field', 'max_tokens'),
                generation_seed=self.config['experiment']['seed'] if llm.get('generation_seed_supported') is True else None,
                max_completion_tokens=settings["max_completion_tokens"],
                thinking_type=settings.get("protocol", {}).get("thinking_type", "enabled"),
                reasoning_effort=settings.get("reasoning_effort", "high"),
                request_timeout_seconds=settings.get("request_timeout_seconds", 180),
                max_retries=settings.get("max_retries", 4), capability_profile=settings.get('capability_profile')))
        provider = self.providers[key]
        provider.budget_governor = self.budget_governor
        return provider

    def agent(self, stage, prompt, materials, name, schema, *, validator=None, repair_limit=0,
              completion_override=None, repair_reason=None, job_key=None, owner_state_version=None,
              decision_purpose=None, finish_request_material=None):
        from .model_view import callable_tools
        if self.checkpoint and self.task_context:
            self.checkpoint.advance(self.checkpoint.state['stage'], model_context={k: getattr(self.task_context, k)
                for k in ('scope', 'results', 'sources', 'memory')})
        purpose = decision_purpose or ('finish_only' if name == 'finish_answer' else {'tool_builder': 'builder'}.get(stage, stage))
        if purpose=='finish_only':name=None
        original_purpose=purpose
        tools = [NativeToolSpec(name, "Submit the requested result", schema)] if name else []
        choice_learning = (self.config['learning'].get('choice_guidance', {}).get('enabled') is True and
                           self.adapter.capabilities.interaction == 'single_answer' and
                           materials.get('learning_material_version') == CHOICE_GUIDANCE_MATERIAL_VERSION and
                           purpose in {'guidance_learning', 'guidance_grounding'})
        def repair_available():
            return not choice_learning or not self.budget_governor or self.budget_governor.repair_available()
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(materials, ensure_ascii=False, allow_nan=False)}]
        content_parts = materials.get('content_parts') if isinstance(materials, dict) else None
        if content_parts:
            if 'image' not in self.config.get('llm', {}).get('input_modalities', ['text']):
                raise ValueError('Model capability lock does not support image input')
            messages[1]['content'] = [{'type': 'text', 'text': json.dumps(
                {k: v for k, v in materials.items() if k != 'content_parts'}, ensure_ascii=False)}, *content_parts]
        scope = json.dumps([str(self.checkpoint.root) if self.checkpoint else '', self.budget_scope, job_key])
        decision_id = self.checkpoint.prepare_decision(scope, purpose, owner_state_version) if self.checkpoint else uuid4().hex
        self.last_decision_id = decision_id
        runtime_owned = (name in {'runtime_step','answer_step'} or purpose == 'finish_only') and owner_state_version is not None
        def commit(status, repair):
            if self.checkpoint:
                self.checkpoint.commit_decision(decision_id, status, repair_index=repair)
        def accepted(value, repair):
            if owner_state_version is None:
                commit('applied', repair)
            return value
        for repair in range(repair_limit + 1):
            purpose='learning_structure_repair' if repair and stage=='extractor' and name=='submit_learning' and not choice_learning else original_purpose
            round_schema=deepcopy(schema)
            if purpose=='learning_structure_repair':
                pending=json.loads(messages[1]['content']).get('proposal') or {}
                if 'workflow' not in pending:round_schema['properties'].pop('workflow',None)
            tools=[NativeToolSpec(name,'Submit the requested result',round_schema)] if name else []
            settings=self.resolve_call_settings(stage,purpose)
            provider=self.provider(stage,purpose)
            tool_choice=({'type':'function','function':{'name':name}} if name and
                (choice_learning or purpose=='learning_structure_repair') and settings.get('protocol',{}).get('thinking_type')=='disabled' else None)
            used, cap = self._budget(stage)
            if used >= cap:
                raise BudgetExhausted('empirical_token_budget_exhausted', 'Role token budget exhausted', layer=FailureLayer.RUNTIME_AGENT)
            turn = None
            completion_cap = settings.get('max_completion_tokens',2048) if purpose=='learning_structure_repair' else completion_override or settings.get('max_completion_tokens', 32768)
            if stage == 'tool_builder': completion_cap = min(completion_cap, cap-used)
            payload_max=None
            if stage in {'extractor','tool_builder'}:
                key='structural_repair_payload_max_bytes' if purpose=='learning_structure_repair' else 'builder_payload_max_bytes' if stage=='tool_builder' else 'extractor_payload_max_bytes'
                payload_max=self.config['learning'][key]
                if isinstance(provider,OpenAICompatibleProvider):
                    from .model_view import pack_material, expand_material
                    from .budget_governor import input_token_bound
                    while True:
                        payload=provider._build_payload(messages,tools,tool_choice)
                        payload[provider.config.token_limit_field]=completion_cap
                        if input_token_bound(payload)<=payload_max:break
                        # Only optional complete history/candidate records may be removed.
                        try: projected=expand_material(json.loads(messages[1]['content']))
                        except (TypeError,ValueError):break
                        removable=next((k for k in ('completed_train_cases','related','related_guidance') if projected.get(k)),None)
                        if removable is None:break
                        projected[removable].pop()
                        messages[1]={'role':'user','content':json.dumps(pack_material(projected),ensure_ascii=False,separators=(',',':'))}
            if stage == 'runtime' and self.adapter.capabilities.final_submission_kind == 'text' and self.adapter.capabilities.interaction != 'single_answer':
                from .budget_governor import input_token_bound
                reserve = 0 if purpose == 'finish_only' else input_token_bound({'goal': materials.get('original_task',{}).get('goal',''), 'answer_contract':getattr(self.adapter,'answer_contract',lambda:'')()}) + self.resolve_call_settings(stage,'finish_only').get('max_completion_tokens',2048)
                if used + input_token_bound({'messages': messages, 'tools': [t.to_openai() for t in tools]}) + completion_cap + reserve > cap:
                    raise BudgetExhausted('runtime_finish_reserved', 'Runtime admission preserves a bounded finish', layer=FailureLayer.RUNTIME_AGENT)
            response_key = digest({'scope': scope, 'purpose': purpose, 'logical_decision_id': decision_id, 'repair_index': repair,
                                   'stage': stage, 'phase': self.phase, 'budget_scope': self.budget_scope,
                                   'messages': messages, 'tools': [t.to_openai() for t in tools],
                                   'completion_cap': completion_cap, 'repair_reason': repair_reason, 'job_key': job_key,
                                   'model_identity': self.config['llm'],
                                   'effective_call_settings': settings, 'tool_choice':tool_choice,
                                   'policy': self.config['runtime']['plan_execution_policy'],
                                   'implementation_revision': self.config['experiment']['implementation_revision']})
            if choice_learning:
                response_key = digest([response_key, tool_choice, self.config['learning']['choice_guidance'],
                                       self.config['runtime'].get('choice_guidance')])
            recovered = self.checkpoint.response(response_key) if self.checkpoint else None
            request_id = recovered['request_id'] if recovered else uuid4().hex
            provider_offset = getattr(provider, "request_record_count", 0)
            record = {"id": request_id, "stage": stage, "repair": repair,
                      'logical_decision_id': decision_id, 'decision_scope': scope,
                      'purpose': purpose, 'owner_state_version': owner_state_version,
                      "phase": self.phase, 'budget_scope': self.budget_scope,
                      "messages": messages, "tools": [t.to_openai() for t in tools]}
            record.update(completion_cap=completion_cap, repair_reason=repair_reason, job_key=job_key)
            pool='learning' if self._learning_start is not None or self.phase=='trial' or stage in {'extractor','tool_builder'} else 'solve'
            pool_limit=self.config.get('budget',{}).get('train_'+pool+'_tokens') if self.request_attribution.get('train_parent') else None
            record.update(budget_pool=pool,budget_pool_limit=pool_limit,payload_max_bytes=payload_max,
                          effective_call_settings=settings,tool_choice=tool_choice)
            if choice_learning:
                record.update(tool_choice=tool_choice, decision_purpose=('repair' if repair else
                    'grounding' if purpose == 'guidance_grounding' else 'proposal'))
            if stage == 'extractor' and 'candidate_view_version' in materials:
                record.update({k: deepcopy(materials[k]) for k in ('candidate_view_version', 'candidate_view_audit')})
            record['effective_call_settings_hash'] = digest(settings)
            record['material_audit'] = {'projected_utf8_bytes':len(json.dumps(materials,ensure_ascii=False).encode()),
                'local_evidence_count':len(materials.get('experience',{}).get('local_evidence',[])),
                **materials.get('experience',{}).get('projection_audit',{})}
            # Trace messages omit replay-private reasoning. The immediate repair
            # retains the actual assistant envelope in process only.
            record["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
            self.requests.append(record)
            self._save_requests()
            try:
                if recovered:
                    request_id = recovered['request_id']
                    record['id'] = request_id
                    turn = AgentTurn(**recovered['turn'])
                    record['recovered_response'] = True
                    if 'http_attempts' in recovered:
                        record['http_attempts'] = recovered['http_attempts']
                    if recovered.get('protocol_error'):
                        raise ProviderAgentProtocolError('runtime_agent_schema_error', recovered['protocol_error'], usage_turn=turn)
                else:
                    if hasattr(provider, 'set_request_context'):
                        from .budget_governor import input_token_bound
                        finish_reserve = 0
                        if pool=='solve' and stage=='runtime' and purpose!='finish_only' and self.request_attribution.get('parent_token_limit') and self.adapter.capabilities.final_submission_kind in {'text','single_answer'}:
                            from .prompts import finish_text_prompt
                            finish_prompt=finish_text_prompt(self.adapter.answer_contract())
                            finish_material=finish_request_material or {k:v for k,v in materials.items() if k in {
                                'goal','inputs','local_results','local_reads'}}
                            if self.adapter.capabilities.interaction!='single_answer':
                                from .finish_evidence import build_finish_evidence
                                finish_material=build_finish_evidence(self.task_context,finish_material,finish_prompt)['materials']
                            finish_provider=self.provider('runtime','finish_only')
                            finish_messages=[{'role':'system','content':finish_prompt},{'role':'user','content':json.dumps(finish_material,ensure_ascii=False)}]
                            finish_cap=self.resolve_call_settings('runtime','finish_only').get('max_completion_tokens',2048)
                            wire=finish_provider._build_payload(finish_messages,[]) if isinstance(finish_provider,OpenAICompatibleProvider) else {'messages':finish_messages,'max_tokens':finish_cap}
                            finish_reserve=input_token_bound(wire)+finish_cap
                        provider.set_request_context(session_id=request_id, stage=stage, repair=repair,
                            purpose=purpose, logical_decision_id=decision_id, parent_task_id=self.request_attribution.get('parent_task_id'),
                            trial_id=self.budget_scope if self.phase == 'trial' else None,
                            trial_execution_id=self.checkpoint.state.get('execution_id') if self.phase == 'trial' and self.checkpoint else None,
                            **{k:v for k,v in self.request_attribution.items() if k != 'parent_task_id' and v is not None},
                            parent_finish_reserve=finish_reserve,
                            budget_pool=pool,budget_pool_limit=pool_limit,payload_max_bytes=payload_max,
                            **({'decision_purpose': record['decision_purpose']} if choice_learning else {}))
                    kwargs = {'max_completion_tokens': completion_cap} if isinstance(provider, OpenAICompatibleProvider) or completion_override is not None else {}
                    if tool_choice is not None: kwargs['tool_choice'] = tool_choice
                    turn = provider.complete(messages, tools=tools, **kwargs)
                    if self.checkpoint:
                        records = getattr(provider, 'request_records_since', None)
                        self.checkpoint.save_response(response_key, {'turn': asdict(turn), 'request_id': request_id,
                            'http_attempts': list(records(provider_offset)) if records else []})
                commit('response_received', repair)
            except Exception as exc:
                invalid_finish_text = (purpose == 'finish_only' and isinstance(exc, ProviderProtocolError)
                                       and 'assistant content must be a string or null' in str(exc))
                turn = getattr(exc, "usage_turn", None)
                if turn is not None:
                    turn.provider_metadata['phase'] = self.phase
                    turn.provider_metadata['budget_scope'] = self.budget_scope
                    if choice_learning: turn.provider_metadata['decision_purpose'] = record['decision_purpose']
                    prior = json.loads(Path(self.audit_path).read_text())['usage'] if self.audit_path and Path(self.audit_path).exists() else []
                    if not any(e['session_id'] == request_id for e in prior) and not any(e.session_id == request_id for e in self.usage.events):
                        self.usage.record_turn(session_id=request_id, turn_index=0, bucket=STAGE_BUCKETS[stage], turn=turn)
                    record['response'] = {'content': turn.content, 'finish_reason': turn.finish_reason,
                        'tool_calls': [{'id': c.call_id, 'name': c.name, 'arguments': c.arguments} for c in turn.tool_calls],
                        'usage': {k: getattr(turn, k) for k in ('prompt_tokens','completion_tokens','total_tokens','reasoning_tokens')}}
                    if self.checkpoint:
                        records = getattr(provider, 'request_records_since', None)
                        self.checkpoint.save_response(response_key, {'turn': asdict(turn), 'request_id': request_id,
                            'protocol_error': 'finish_only_invalid_text' if invalid_finish_text else
                                str(exc) if isinstance(exc, ProviderAgentProtocolError) else None,
                            'http_attempts': recovered.get('http_attempts', []) if recovered else list(records(provider_offset)) if records else []})
                record["error"] = str(exc)
                self._save_requests()
                if isinstance(exc, ProviderAgentProtocolError) or invalid_finish_text:
                    if repair < repair_limit and turn.finish_reason != 'length' and repair_available():
                        if name=='submit_learning' and stage=='extractor' and not choice_learning:
                            # A missing/invalid envelope cannot supply a candidate to repair.
                            if len(turn.tool_calls)!=1 or not isinstance(turn.tool_calls[0].arguments,dict):
                                failure=ValueError('learning_structure_unrecoverable: '+str(exc))
                                failure.model_authored=True
                                failure.finish_reason=turn.finish_reason
                                if not runtime_owned:commit('rejected',repair)
                                raise failure from exc
                            from .model_view import structure_repair_material
                            short=structure_repair_material(turn.tool_calls[0].arguments,
                                [{'binding_index':None,'errors':[{'code':'output_type','detail':str(exc)[:200]}]}],[])
                            messages=[{'role':'system','content':'Repair only this proposal structure. Preserve its goal; submit one submit_learning ToolCall. Do not solve the task again.'},
                                      {'role':'user','content':json.dumps(short,ensure_ascii=False,separators=(',',':'))}]
                            commit('prepared',repair+1)
                            continue
                        commit('prepared', repair + 1)
                        messages.append({'role': 'user', 'content': 'Repair only the invalid ToolCall JSON: ' + str(exc)[:2048]})
                        continue
                    failure = ValueError('finish_only_invalid_text' if invalid_finish_text else str(exc))
                    failure.model_authored = True
                    failure.finish_reason = turn.finish_reason
                    failure.logical_decision_id = decision_id
                    if not runtime_owned: commit('rejected', repair)
                    raise failure from exc
                raise
            finally:
                records = getattr(provider, "request_records_since", None)
                if records and not recovered:
                    record["http_attempts"] = list(records(provider_offset))
                self._save_requests()
            if self.budget_governor and (self.budget_governor.state['unknown_billing'] or sum(
                a.get('accounted_tokens', a['reserved_tokens']) for a in self.budget_governor.state['attempts'].values()
            ) > self.budget_governor.limits['token_limit']):
                raise BudgetExhausted('diagnostic_budget_exhausted', 'Batch stopped before applying an unmetered or over-budget response')
            prior_usage = json.loads(Path(self.audit_path).read_text())['usage'] if self.audit_path and Path(self.audit_path).exists() else []
            if not any(e['session_id'] == request_id for e in prior_usage) and not any(
                    e.session_id == request_id for e in self.usage.events):
                turn.provider_metadata['phase'] = self.phase
                turn.provider_metadata['budget_scope'] = self.budget_scope
                if choice_learning: turn.provider_metadata['decision_purpose'] = record['decision_purpose']
                self.usage.record_turn(session_id=request_id, turn_index=0, bucket=STAGE_BUCKETS[stage], turn=turn)
            record["response"] = {"content": turn.content, "finish_reason": turn.finish_reason,
                "tool_calls": [{"id": c.call_id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
                "usage": {k: getattr(turn, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")}}
            from .answer_status import classify_answer
            record.update(classify_answer(turn.content, turn.finish_reason))
            self._save_requests()
            if self._budget(stage)[0] > cap:
                raise BudgetExhausted('empirical_token_budget_exhausted', 'Metered turn exceeded role budget', layer=FailureLayer.RUNTIME_AGENT)
            try:
                if choice_learning and turn.finish_reason == 'length':
                    raise ValueError('Truncated choice guidance response')
                if name == 'answer_step' and not turn.tool_calls:
                    return accepted({'action':'finish','answer':turn.content or ''},repair)
                if not name:
                    if purpose == 'finish_only':
                        if turn.tool_calls:
                            raise ValueError('finish_only_unexpected_tool_call')
                        if not isinstance(turn.content, str):
                            raise ValueError('finish_only_invalid_text')
                        if not turn.content.strip():
                            raise ValueError('finish_only_empty_answer')
                    return accepted(turn.content, repair)
                if name == 'runtime_step' and turn.tool_calls:
                    actions = []
                    for call in turn.tool_calls:
                        if call.name != name: raise ValueError('Unexpected Runtime ToolCall name')
                        validate_schema_instance(call.arguments, schema)
                        self.task_context.bind(call.arguments.get('arguments', {}), call.arguments.get('argument_refs', {})) if self.task_context else None
                        actions.append(call.arguments)
                    if len(actions) > 1:
                        specs = {t['name']: t for t in callable_tools(materials)}
                        if len(actions) > self.config['runtime']['read_batch_max_calls'] or any(
                            a['action'] != 'call_tool' or not specs.get(a.get('name'), {}).get('batchable') or
                            specs.get(a.get('name'), {}).get('effect') != 'read_only' for a in actions):
                            raise ValueError('Batch must contain at most 3 independent approved read-only calls')
                        for action in actions:
                            args = self.task_context.bind(action.get('arguments', {}), action.get('argument_refs', {})) if self.task_context else action.get('arguments', {})
                            validate_schema_instance(args, specs[action['name']]['input_schema'])
                    for action in actions:
                        if validator: validator(action)
                    return accepted(RuntimeDecision(tuple(actions), tuple(c.call_id for c in turn.tool_calls), request_id), repair)
                if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != name:
                    if name == 'submit_program':
                        raise ValueError('builder_submission_tool_mismatch: expected=[submit_program], received=' +
                                         json.dumps([call.name for call in turn.tool_calls]))
                    details = ''
                    if choice_learning and isinstance(turn.content, str):
                        try:
                            from ..agents.protocol import parse_json_strict
                            text_value = parse_json_strict(turn.content)
                            if validator: validator(text_value)
                            else: validate_schema_instance(text_value, schema)
                        except (ValueError, TypeError) as field_error:
                            details = '; text arguments: ' + str(field_error)
                    raise ValueError("Expected one " + name + " ToolCall" + details)
                value = turn.tool_calls[0].arguments
                if choice_learning and validator:
                    validator(value)
                else:
                    if validator and name=='submit_learning':validator(value)
                    validate_schema_instance(value, round_schema)
                    if validator and name!='submit_learning': validator(value)
                return accepted(value, repair)
            except ValueError as exc:
                if repair == repair_limit or (turn.finish_reason == 'length' and not turn.tool_calls) or not repair_available():
                    exc.model_authored = True
                    exc.finish_reason = turn.finish_reason
                    exc.logical_decision_id = decision_id
                    if not runtime_owned: commit('rejected', repair)
                    raise
                commit('prepared', repair + 1)
                if name=='submit_learning' and stage=='extractor' and not choice_learning:
                    from .model_view import structure_repair_material
                    repaired=getattr(exc,'repair_material',None)
                    if repaired is None:
                        repaired=structure_repair_material(turn.tool_calls[0].arguments if len(turn.tool_calls)==1 else {},
                            [{'binding_index':None,'errors':[{'code':'output_type','detail':str(exc)[:200]}]}],[])
                    messages=[{'role':'system','content':'Repair only this learning proposal structure and binding references. Preserve the local goal. Submit exactly one submit_learning ToolCall. Host supplies replay start and identity; do not solve the task again.'},
                              {'role':'user','content':json.dumps(repaired,ensure_ascii=False,separators=(',',':'))}]
                    continue
                if hasattr(exc, 'repair_material'):
                    messages[1] = {'role': 'user', 'content': json.dumps(exc.repair_material, ensure_ascii=False)}
                messages.extend(repair_messages(turn, exc))
                messages.append({"role": "user", "content": "Repair only the invalid structure. Preserve the requested goal and valid content. "
                    "If you returned JSON as text, submit those same arguments through the requested ToolCall: " + str(exc)[:2048]})

    def run_task(self, task, *, learn=None, attempt_id=None):
        if not isinstance(task, PublicTask):
            raise TypeError("Empirical System accepts only PublicTask")
        learn = not self.readonly if learn is None else learn
        if learn and (self.readonly or task.split != "train"):
            raise RuntimeError("Only Train may update the empirical Bank")
        identity = {'config_hash':digest(self.config),'implementation_revision':self.config['experiment']['implementation_revision'],
            'learning_material_version':self.config['learning']['material_version'],
            'local_validation_policy':self.config['learning']['local_validation_policy'],
            'answer_protocol_version':self.config['runtime']['answer_protocol_version'],
            'binding_policy':self.config['learning']['binding_policy'],
            'file_effect_version':self.config['learning']['file_effect_version'],
            'budget_policy':self.config['experiment']['budget_policy'],
            'budget':{'path':str(self.budget_governor.path),'limits':self.budget_governor.limits} if self.budget_governor else None}
        if self.checkpoint:
            if self.checkpoint.state.get('method_identity',identity) != identity:
                raise ValueError('Checkpoint method identity changed')
            self.checkpoint.advance(self.checkpoint.state['stage'],method_identity=identity)
        before = self.bank.digest()
        self._task_start, request_start = len(self.usage.events), len(self.requests)
        self._runtime_start = self._task_start
        self._request_start = request_start
        trace = {"schema": "empirical.trace.v1", "task": asdict(task), "attempt_id": attempt_id or uuid4().hex,
                 "knowledge_before": before, "tools": [], "execution": {}, "score": None,
                 "solve_status": "not_started", "learning_status": "not_started", "learning_error": None,
                 'method_identity':identity}
        self.budget_scope = trace['attempt_id']
        budget = self.config.get('budget',{})
        self.request_attribution.update(parent_scope=str(self.config['experiment']['output_dir'])+':'+task.physical_key,
            parent_token_limit=budget.get('train_task_tokens' if task.split=='train' else 'eval_task_tokens'),train_parent=task.split=='train')
        self.request_attribution['parent_task_id'] = task.task_id
        self.last_decision_id = None
        self.prior_usage = json.loads(Path(self.audit_path).read_text())['usage'] if self.audit_path and Path(self.audit_path).exists() else []
        self.phase = task.split
        self.planner.checkpoint = self.checkpoint
        if self.checkpoint and self.checkpoint.state['stage'] in {'task_execution_finished', 'learning_finished', 'task_committed'}:
            trace = self.checkpoint.state['trace']
            if learn and self.checkpoint.state['stage'] == 'task_execution_finished':
                receipt = self.checkpoint.state.get('learning_workspace')
                if receipt:
                    self.adapter.reset(task)
                    restore_trial_workspace(self.adapter, receipt)
                elif getattr(self.adapter, 'workspace', None):
                    raise RuntimeError('Saved learning boundary lacks a public workspace; use explicit recovery')
                self.task_context = TaskContext(self.config['runtime'])
                context_state = self.checkpoint.state.get('executor_state', {}).get('context', self.checkpoint.state.get('model_context', {}))
                for key in ('scope','results','sources','memory','local_reads'):
                    if key in context_state: setattr(self.task_context,key,context_state[key])
                self.learn_trace(task, trace)
                self.checkpoint.advance('learning_finished', trace=trace)
            trace['knowledge_after'] = self.bank.digest()
            self._save_requests()
            if self.audit_path and Path(self.audit_path).exists():
                trace.update(json.loads(Path(self.audit_path).read_text()))
            return trace
        self.adapter.reset(task)
        self.task_context = TaskContext(self.config['runtime'])
        if self.checkpoint:
            for key, value in self.checkpoint.state.get('model_context', {}).items():
                setattr(self.task_context, key, value)
        broker = Broker(self.adapter, self.config.get('runtime', {}).get('global_action_budget', 100),
            step_limit=self.config.get('runtime', {}).get('environment_step_budget'),
            journal=self.checkpoint.native_events if self.checkpoint else None,
            observer=self.observer.native_observer(trace['attempt_id']) if self.observer else None,
            context=self.task_context)
        self.executor.checkpoint = self.checkpoint
        self.executor.partial_execution = {'attempts': []}
        if self.checkpoint and hasattr(self.adapter, 'evaluation_receipts'):
            self.adapter.evaluation_receipts = self.checkpoint.root/'evaluation_receipts'
        if self.checkpoint and self.checkpoint.state.get('program_started'):
            from ..harness.simple_protocol import UnknownSideEffect
            raise UnknownSideEffect('Interrupted Program/workspace execution requires explicit reconstruction')
        native_path = self.checkpoint.root/'native_events.json' if self.checkpoint else None
        if native_path and native_path.exists():
            prior_events = json.loads(native_path.read_text())
            specs = {t['name']: t for t in broker.available_tools()}
            sealed_workspace = self.checkpoint.state.get('executor_finished_workspace')
            if self.adapter.capabilities.checkpoint_mode != 'workspace_copy' or any(
                e['state'] != 'finished' or (not sealed_workspace and specs.get(e['name'], {}).get('effect') != 'read_only') for e in prior_events):
                from ..harness.simple_protocol import UnknownSideEffect
                raise UnknownSideEffect('Interrupted stateful or unknown operation requires explicit reconstruction')
            broker.events = prior_events
            broker.environment_steps = sum(e.get('environment_step', 0) for e in prior_events)
        try:
            if self.adapter.capabilities.interaction == "single_answer":
                from .choice_guidance import valid_choices, render
                from .answer_executor import AnswerExecutor
                semantic = self.config['runtime'].get('choice_guidance', {}).get('enabled') is True and valid_choices(task)
                selection = self.bank.select_guidance(task,self.config['runtime']['choice_guidance']) if semantic else None
                guidance = selection['selected'] if selection else self.bank.retrieve_guidance(task.goal,limit=3)
                trace['retrieved_guidance_ids'] = [a['id'] for a in guidance]
                trace['injected_guidance_ids'] = trace['retrieved_guidance_ids'][:]
                trace['guidance_retrieval_audit'] = selection if selection else self.bank.guidance_retrieval_audit(task.goal,guidance)
                execution = AnswerExecutor(self).run(task,broker,[render(a) if semantic else
                    {'skill_id':a['id'],'goal':a['goal'],'guidance':a['guidance']} for a in guidance],semantic)
            else:
                plan = self.checkpoint.state['executor_state']['plan'] if self.checkpoint and self.checkpoint.state.get('executor_state') else (
                    self.checkpoint.state.get('initial_plan') if self.checkpoint else None) or self.planner.plan(task, self.adapter)
                if self.checkpoint and not self.checkpoint.state.get('executor_state'):
                    self.checkpoint.commit_decision(getattr(self.planner, 'last_decision_id', None), 'applied', initial_plan=plan)
                trace["initial_plan"] = plan
                execution = self.executor.run(task, self.adapter, broker, plan)
            self.complete_execution(task, trace, broker, execution, learn=learn)
        except BudgetExhausted as exc:
            if exc.code == 'diagnostic_budget_exhausted':
                raise
            if trace.get('solve_status') == 'completed':
                raise
            trace["error"] = {"code": exc.code, "message": str(exc)}
            partial = deepcopy(getattr(self.executor, 'partial_execution', {}))
            partial.setdefault('prediction',None)
            partial.update(reason='token_budget_exhausted',budget_censored=True)
            partial.setdefault('attempts', [])
            self.complete_execution(task, trace, broker, partial, learn=learn, decision_status='rejected')
        finally:
            native_ids = {e.get('result_id') for e in broker.events}
            trace['result_store'] = {'native_index': [{'result_id': e.get('result_id'), 'event_id': e.get('event_id'), 'index': e['index']}
                for e in broker.events if e.get('result_id')],
                'program_results': {rid: value for rid, value in broker.context.results.items() if rid not in native_ids},
                'local_reads': broker.context.local_reads}
            trace["usage"] = [e.to_dict() for e in self.usage.events[self._task_start:]]
            trace["requests"] = self.requests[request_start:]
            self._save_requests()
            trace["knowledge_after"] = self.bank.digest()
            if self.observer:
                trace['scoring_audit'] = getattr(self.adapter, 'score_audit', {})
            if self.audit_path and Path(self.audit_path).exists():
                trace['usage'] = json.loads(Path(self.audit_path).read_text())['usage']
            if self.readonly and trace["knowledge_after"] != before:
                raise RuntimeError("Frozen Bank changed during evaluation")
        return trace

    def learn_from_completed_record(self, task, experience, identity):
        """Explicit paid recompile event; inherited submission is never a new Runtime answer."""
        if self.readonly or task.split != 'train' or self.adapter.capabilities.interaction != 'single_answer':
            raise ValueError('Recompile requires a writable single-answer Train event')
        if not self.checkpoint or not self.audit_path: raise ValueError('Recompile requires a durable event checkpoint and audit')
        if self.checkpoint.state.get('source_identity') not in (None, identity): raise ValueError('Recompile source identity changed')
        self.checkpoint.advance(self.checkpoint.state['stage'], source_identity=identity)
        self._task_start = self._runtime_start = len(self.usage.events)
        self._request_start = len(self.requests)
        self._learning_start = None
        self.phase, self.budget_scope = 'train', 'recompile:' + identity['source_record_hash']
        self.prior_usage = json.loads(Path(self.audit_path).read_text())['usage'] if Path(self.audit_path).exists() else []
        self.task_context = TaskContext(self.config['runtime'])
        self.request_attribution = {**identity, 'parent_task_id': task.task_id}
        if self.checkpoint.state.get('recompile_result'): return self.checkpoint.state['recompile_result']
        trace = {'execution': {'prediction': experience['submission'], 'attempts': [],
                 **{k: experience.get(k) for k in ('answer_status', 'empty_answer', 'completion_truncated', 'provider_finish_reason')}},
                 'score': experience['score'], 'tools': [], 'inherited_experience': deepcopy(experience)}
        try:
            self.learn_trace(task, trace)
            self._save_requests()
            audit = json.loads(Path(self.audit_path).read_text()) if Path(self.audit_path).exists() else {'requests': [], 'usage': []}
            result = {**trace, **audit, 'source_identity': identity, 'derivation_mode': 'offline_recompile_from_completed_train'}
            self.checkpoint.advance('learning_finished', recompile_result=result)
            return result
        finally:
            self._save_requests()

    def complete_execution(self, task, trace, broker, execution, *, learn, decision_status='applied',
                           sealed=None, score=None):
        """One durable solve/score boundary for normal, budget and explicit recovery endings."""
        sealed = self.adapter.submit(execution.get('prediction')) if sealed is None else sealed
        score = self.adapter.evaluate(sealed) if score is None else score
        producer = execution.get('submission_producer_attempt_id')
        if producer:
            execution['submission_seal'] = {'producer_attempt_id': producer, 'sealed_digest': digest(sealed),
                'workspace': self.adapter.observe().get('workspace', {})}
        trace['result_store'] = {'program_results':deepcopy(broker.context.results), 'local_reads':deepcopy(broker.context.local_reads)}
        trace.update(execution=execution, score=score, tools=broker.events, solve_status='completed',
            native_call_attempts=len(broker.events), environment_steps=broker.environment_steps,
            scoring_audit=getattr(self.adapter, 'score_audit', {}))
        trace['episode_result'] = asdict(EpisodeResult(task.task_id, task.split, sealed, score,
            execution['reason'], cost_path=str(self.audit_path) if self.audit_path else None))
        if self.checkpoint:
            receipt = self.checkpoint.state.get('learning_workspace')
            if learn and getattr(self.adapter, 'workspace', None) and not receipt:
                receipt = seal_trial_workspace(self.adapter, self.checkpoint.root/'learning_workspace')
            decision = self.checkpoint.state['decisions'].get(self.last_decision_id, {})
            key = self.last_decision_id if decision.get('status') in {'prepared','response_received'} else None
            self.checkpoint.commit_decision(key, decision_status, stage='task_execution_finished',
                trace=trace, learning_workspace=receipt)
        if learn: self.learn_trace(task, trace)
        else: trace['learning_status'] = 'frozen' if self.readonly else 'deferred'
        if self.checkpoint and learn: self.checkpoint.advance('learning_finished', trace=trace)

    def learn_trace(self, task, trace):
        if self.readonly or task.split != 'train':
            raise RuntimeError('Only Train may learn')
        learning_observation = self.observer.learning_start(task) if self.observer else None
        self._learning_start = len(self.usage.events)
        try:
            for attempt in trace["execution"]["attempts"]:
                if attempt["status"] == "ok" and attempt.get('output_contract_status') == 'valid' and (attempt["outputs_consumed"] or attempt.get("terminal_by_program", False) or attempt.get('submission_by_program', False)) and trace["score"]["hard"] and attempt["local_check"] == "unavailable":
                    attempt.update(outcome="positive", basis="task_outcome")
                host_call('bank_record', self.bank.record, attempt, trial_context={'stage': 'learning'})
            trace["learning"] = self.learner.learn(task, trace)
            result = trace['learning'] or {}
            trace['learning_status'] = ('skipped_policy' if result.get('decision_origin') == 'host' else
                'rejected' if result.get('rejected') or result.get('decision') == 'rejected' else 'completed')
            if trace.get('learning_error'): trace['learning_error'] = None
        except BudgetExhausted as exc:
            if exc.code == 'diagnostic_budget_exhausted': raise
            trace['learning'] = {'error': str(exc), 'reason': 'budget_unavailable', 'decision_origin': 'host'}
            trace['learning_status'] = 'deferred_budget'
        except (ValueError, SyntaxError) as exc:
            trace["learning"] = {"error": str(exc), "rejected": True}
            trace['learning_status'] = 'rejected'
        except Exception as exc:
            trace['learning_status'] = 'failed_engineering'
            trace['learning_error'] = exception_details(exc, 'learning')
            trace['learning'] = {'error': str(exc), 'exception': trace['learning_error']}
            if self.checkpoint:
                host_call('checkpoint_save', self.checkpoint.advance, 'task_execution_finished', trace=trace,
                          trial_context=getattr(exc, 'trial_context', {'stage': 'learning'}))
            raise
        finally:
            if self.observer:
                self.observer.learning_end(learning_observation, trace.get('learning'))
            self._learning_start = None

    def test_program(self, program, inputs, task, *, trial_id, trial_case=None,
                     continuation=False, new_execution=False, local_binding=None, source_experience=None):
        if local_binding is not None:
            from .local_validation import validate_on_source
            return validate_on_source(self,program,local_binding,task,source_experience,trial_id)
        if self.readonly or task.split != 'train': raise ValueError('Program trials require Train')
        if self.adapter_factory is None:
            return {"id": trial_id, "program_id": program["id"], "task_key": task.physical_key,
                    "origin": "train_test", "outcome": "inapplicable", "error": "No isolated Adapter factory"}
        stage, execution_id = 'trial_lookup', None
        def host(operation, function, *args, **kwargs):
            return host_call(operation, function, *args, trial_context={
                'stage': stage, 'trial_id': trial_id, 'trial_execution_id': execution_id}, **kwargs)
        existing = next((a for a in self.bank.attempts(program['id']) if a['id'] == trial_id), None)
        if existing and not new_execution:
            return existing
        from .checkpoint import TaskCheckpoint
        logical = host('checkpoint_load', TaskCheckpoint, self.checkpoint.root / 'trials' / trial_id) if self.checkpoint else None
        if logical and logical.state.get('record') and not new_execution:
            host('bank_record', self.bank.record, logical.state['record'])
            return logical.state['record']
        if logical and logical.state.get('new_execution_authorization'):
            new_execution = True
        execution_id = (logical.state.get('execution_id') if logical and not new_execution else None) or uuid4().hex
        trial_checkpoint = host('checkpoint_load', TaskCheckpoint, logical.root / 'executions' / execution_id) if logical else None
        distinct_execution = new_execution or bool(trial_checkpoint and trial_checkpoint.state.get('new_execution'))
        if trial_checkpoint and trial_checkpoint.state.get('record'):
            record = trial_checkpoint.state['record']
            host('bank_record', self.bank.record, record)
            host('checkpoint_save', logical.advance, 'bank_recorded', record=record)
            return record
        resume_worker = bool(trial_checkpoint and trial_checkpoint.state.get('worker_result'))
        stage = 'worker_finished' if resume_worker else 'trial_started'
        if trial_checkpoint and trial_checkpoint.path.exists() and not resume_worker:
            raise RuntimeError('Interrupted trial has unknown side effects; an explicit new execution is required')
        if resume_worker and not trial_checkpoint.state.get('artifacts'):
            raise RuntimeError('Interrupted trial workspace is unrecoverable; no worker replay or positive credit')
        if logical:
            host('checkpoint_save', logical.advance, 'trial_started', execution_id=execution_id,
                 new_execution_authorization=None)
            if not resume_worker:
                host('checkpoint_save', trial_checkpoint.advance, 'trial_started', trial_id=trial_id, execution_id=execution_id,
                                         program_id=program['id'], task_key=task.physical_key, inputs=inputs,
                                         new_execution=distinct_execution)
        def finish(record):
            nonlocal stage
            stage = 'trial_finished'
            record['logical_trial_id'] = trial_id
            if distinct_execution: record['id'] = trial_id + ':' + execution_id
            record['trial_execution_id'] = execution_id
            if trial_checkpoint:
                host('checkpoint_save', trial_checkpoint.advance, 'trial_finished', record=record)
            host('bank_record', self.bank.record, record)
            if logical:
                host('checkpoint_save', trial_checkpoint.advance, 'bank_recorded', record=record)
                host('checkpoint_save', logical.advance, 'bank_recorded', record=record)
            return record
        previous_phase = self.phase
        previous_checkpoint = self.checkpoint
        previous_planner_checkpoint = getattr(self.planner, 'checkpoint', None)
        previous_scope = self.budget_scope
        previous_context = self.task_context
        self.task_context = TaskContext(self.config['runtime'])
        self.phase = 'trial'
        self.checkpoint = trial_checkpoint
        self.planner.checkpoint = trial_checkpoint
        self.budget_scope = previous_scope
        adapter, broker, trial_observation = None, None, None
        previous_runtime_start = self._runtime_start
        self._runtime_start = len(self.usage.events)
        try:
            adapter = host('adapter_create', self.adapter_factory)
            trial_observation = host('trial_start', self.observer.trial_start, trial_id + ':' + execution_id, task) if self.observer else None
            broker = host('broker_create', Broker, adapter, self.config.get('runtime', {}).get('global_action_budget', 100),
                step_limit=self.config.get('runtime', {}).get('environment_step_budget'),
                journal=trial_checkpoint.native_events if trial_checkpoint else None,
                observer=self.observer.native_observer(trial_id + ':' + execution_id) if self.observer else None,
                context=self.task_context)
            inherit = getattr(adapter, 'inherit_discovery', None)
            if inherit:
                host('inherit_discovery', inherit, self.adapter)
            host('adapter_reset', adapter.reset, task)
            if trial_observation:
                trial_observation['consumed'] = True
            if resume_worker:
                host('restore_workspace', restore_trial_workspace, adapter, trial_checkpoint.state.get('artifacts'))
                native_path = trial_checkpoint.root / 'native_events.json'
                broker.events[:] = host('native_events_restore', lambda: json.loads(native_path.read_text())) if native_path.exists() else []
            if trial_case and not resume_worker:
                if trial_case.physical_task_key != task.physical_key or task.split != 'train':
                    raise ValueError('TrialCase physical task/split mismatch')
                for action in trial_case.prefix:
                    if broker.done or not broker.call(action['name'], action['arguments']).get('accepted'):
                        record = {'id': trial_id, 'program_id': program['id'], 'task_key': task.physical_key,
                                  'origin': 'train_test', 'outcome': 'inapplicable', 'basis': None,
                                  'error': 'Prefix could not reconstruct an active start state'}
                        return finish(record)
                if broker.done:
                    record = {'id': trial_id, 'program_id': program['id'], 'task_key': task.physical_key,
                              'origin': 'train_test', 'outcome': 'inapplicable', 'basis': None,
                              'error': 'Trial prefix reached terminal'}
                    return finish(record)
            start = len(broker.events)
            was_terminal = broker.done
            workspace_before = host('adapter_observe', adapter.observe).get('workspace', {})
            try:
                validate_schema_instance(inputs, program["input_schema"])
                if len(json.dumps(inputs, ensure_ascii=False).encode()) > self.worker.settings['max_rpc_message_bytes']:
                    raise ValueError('Program inputs exceed RPC limit')
                tools = getattr(adapter, 'tool_definitions', adapter.available_tools)()
                if not set(program['allowed_tools']).issubset({t['name'] for t in tools} | {'read_result'}):
                    raise ValueError('Required public tools are unavailable for this case')
            except ValueError as exc:
                outcome, result, basis = "inapplicable", {"error": str(exc)}, None
            else:
                if resume_worker:
                    result = trial_checkpoint.state['worker_result']
                    start = trial_checkpoint.state['native_event_start']
                    workspace_before = trial_checkpoint.state['workspace_before']
                else:
                    result = normalize_program_result(adapter, program, host('worker_execute', self.worker.execute, program, inputs, broker),
                                                      workspace_before=workspace_before)
                    result['terminal_by_program'] = not was_terminal and broker.done
                    stage = 'worker_finished'
                    if trial_checkpoint:
                        artifacts = host('seal_workspace', seal_trial_workspace, adapter, trial_checkpoint.root / 'artifacts')
                        host('checkpoint_save', trial_checkpoint.advance, 'worker_finished', worker_result=result, artifacts=artifacts,
                            native_event_start=start, workspace_before=workspace_before)
                local = host('local_check', broker.check_local, inputs, result.get('outputs', {}), start) if positive_eligible(result) else "unavailable"
                result['local_check'] = local
                basis, outcome = None, "normal"
                if result["status"] == "execution_error":
                    outcome = "execution_failure"
                elif result['terminal_by_program'] and positive_eligible(result):
                    score = host('evaluate', adapter.evaluate, host('submit', adapter.submit, result.get('outputs', {})))
                    result['score'] = score
                    if score['hard'] and result['status'] == 'ok': basis, outcome = 'task_outcome', 'positive'
                    elif result['status'] == 'execution_error': outcome = 'execution_failure'
                elif local == "failed":
                    outcome = "execution_failure"
                elif result["status"] == "ok" and local == "passed":
                    basis, outcome = "local_check", "positive"
                elif result["status"] == "ok":
                    role = program.get('result_role', 'intermediate')
                    preparation = result.get('submission_preparation', {})
                    ready = preparation.get('status') == 'ready'
                    if ready:
                        sealed = host('submit', adapter.submit, preparation['payload'])
                        score = host('evaluate', adapter.evaluate, sealed)
                        result.update(score=score, submission='direct_program_output')
                        if score['hard']: basis, outcome = 'task_outcome', 'positive'
                    if role in {'final_answer', 'final_files'}:
                        record = {"id": trial_id, "program_id": program["id"], "task_key": task.physical_key,
                            "origin": "train_test", "split": "train", "outcome": outcome, "basis": basis,
                            "calls": len(broker.events), "result": result, "tools": broker.events}
                        return finish(record)
                    if not continuation:
                        record = {'id': trial_id, 'program_id': program['id'], 'task_key': task.physical_key,
                                  'origin': 'train_test', 'split': 'train', 'outcome': 'inapplicable',
                                  'basis': None, 'error': 'Pure trial requires a terminal Program',
                                  'result': result, 'tools': broker.events, 'calls': len(broker.events)}
                        return finish(record)
                    # Without a local oracle, run the remainder from the actual
                    # Train state; this is billed training, not replay credit.
                    executor = Executor(self.bank, self.agent, self.worker, self.planner)
                    executor.checkpoint = trial_checkpoint
                    continuation = dynamic(task)
                    continuation["nodes"][0]["args"] = {k: {"literal": v} for k, v in result.get("outputs", {}).items()}
                    rest = executor.run(task, adapter, broker, continuation)
                    result["continuation"] = rest
                    score = host('evaluate', adapter.evaluate, host('submit', adapter.submit, rest['prediction']))
                    result["score"] = score
                    node_id = continuation['nodes'][0]['id']
                    if score["hard"] and result.get('outputs') and node_id in rest['input_reads'] and not rest['replaced_inputs'].get(node_id):
                        basis, outcome = "task_outcome", "positive"
            record = {"id": trial_id, "program_id": program["id"], "task_key": task.physical_key,
                      "origin": "train_test", "split": "train", "outcome": outcome,
                      "basis": basis, "calls": len(broker.events), "result": result, "tools": broker.events}
            return finish(record)
        except Exception as exc:
            if trial_checkpoint:
                host('checkpoint_save', trial_checkpoint.advance, 'trial_exception', exception=exception_details(exc, stage))
            raise
        finally:
            try:
                if self.observer:
                    host('trial_end', self.observer.trial_end, trial_observation, broker.events if broker else [],
                         locals().get('record'), locals().get('result'), getattr(adapter, 'score_audit', {}))
            finally:
                self._runtime_start = previous_runtime_start
                self.phase = previous_phase
                self.checkpoint = previous_checkpoint
                self.planner.checkpoint = previous_planner_checkpoint
                self.budget_scope = previous_scope
                self.task_context = previous_context
                if adapter: host('adapter_close', adapter.close)

    def close(self):
        self.adapter.close()
        self.bank.close()
