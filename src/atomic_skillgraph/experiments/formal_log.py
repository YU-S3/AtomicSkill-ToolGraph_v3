"""Passive formal records alongside native checkpoints; no model calls."""
from datetime import datetime, timezone
from collections import Counter
import json
from pathlib import Path
import time
from uuid import uuid4

from ..empirical.contracts import digest
from .canonical_manifest import sha256
from .run_empirical import write_json


LOGS = ('episodes', 'llm_calls', 'interactions', 'training_events', 'validation_events',
        'artifacts', 'scorer_outputs', 'errors')


def utc(timestamp=None):
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp, timezone.utc).isoformat()


def tree_identity(root):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}
    return {'sha256': digest(files), 'files_sha256': files,
            'size_bytes': sum((root / name).stat().st_size for name in files)}


class FormalLog:
    def __init__(self, root, identity, config, *, resume=False):
        self.root, self.identity = Path(root), identity
        self.root.mkdir(parents=True, exist_ok=True)
        self.ids = {}
        for name in LOGS:
            path = self.root / (name + '.jsonl')
            path.touch(exist_ok=True)
            self.ids[name] = {r['event_id'] for r in self.rows(name)}
        self.manifest_path = self.root / 'run_manifest.json'
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest['identity'] != identity or not resume:
                raise ValueError('Formal run identity mismatch, or --resume required')
            if self.manifest['config_hash'] != digest(config):
                raise ValueError('Formal resolved config changed')
            self.manifest.update(run_status='running', end_time=None)
            self.emit('errors', {'event_id': 'resume:' + utc(), 'timestamp': utc(), 'event_type': 'resume',
                                'normalized_error': None, 'native_checkpoint_resume': True})
        else:
            self.manifest = {**identity, 'identity': identity, 'run_id': uuid4().hex,
                             'start_time': utc(), 'end_time': None, 'run_status': 'running',
                             'config_hash': digest(config), 'initial_artifact_hash': None}
            write_json(self.root / 'resolved_config.json', config)
        write_json(self.manifest_path, self.manifest)
        self.current_task = None
        self.training_index = len(self.rows('training_events'))
        self.consumed = sum(r['train_examples_seen_this_unit'] for r in self.rows('training_events'))
        self.versions = {}
        for row in self.rows('artifacts'):
            self.versions[row['logical_asset_id']] = row
        self.requests_seen = []
        self.trial_tasks = {}
        self.active_learning = None
        self.task_call_counts = Counter((r['task_id'], r['phase']) for r in self.rows('llm_calls')
                                       if r['stage'] in {'planner', 'runtime'})
        self.infrastructure_tasks = {r.get('task_id') for r in self.rows('errors') if r.get('infrastructure_error')}

    def rows(self, name):
        return [json.loads(line) for line in (self.root / (name + '.jsonl')).read_text().splitlines() if line]

    def emit(self, name, record):
        if record['event_id'] in self.ids[name]:
            return
        with (self.root / (name + '.jsonl')).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'run_id': self.manifest['run_id'], **record}, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
        self.ids[name].add(record['event_id'])
        if name == 'llm_calls' and hasattr(self, 'task_call_counts') and record['stage'] in {'planner', 'runtime'}:
            self.task_call_counts[(record['task_id'], record['phase'])] += 1
        if name == 'errors' and hasattr(self, 'infrastructure_tasks') and record.get('infrastructure_error'):
            self.infrastructure_tasks.add(record.get('task_id'))

    def begin_task(self, task, index, attempt_id, metadata):
        self.current_task = {'task_id': task.task_id, 'split': task.split, 'task_order_index': index,
                             'attempt_id': attempt_id, 'task_metadata': metadata, 'task_start_time': utc()}
        path = self.root / 'task_starts' / (attempt_id + '.json')
        if path.exists():
            self.current_task = json.loads(path.read_text())
        else:
            write_json(path, self.current_task)

    def requests(self, requests):
        self.requests_seen = requests
        for request in requests:
            if self.active_learning and request['stage'] == 'tool_builder':
                material = json.loads(request['messages'][1]['content'])
                self.active_learning['subjects'].update(r['case_id'] for r in material.get('examples', []))
            task_id = self.trial_tasks.get(request.get('budget_scope'), self.current_task['task_id'] if self.current_task else None)
            for attempt in request.get('http_attempts', []):
                raw = attempt.get('raw_usage')
                usage = raw if isinstance(raw, dict) else {}
                response = attempt.get('public_response') or {}
                choices = response.get('choices') or []
                message = choices[0].get('message', {}) if choices else {}
                code = attempt.get('error_code')
                status = 'success' if attempt['outcome'] == 'success' else (
                    'timeout' if code == 'provider_timeout' else 'rate_limit' if attempt.get('http_status') == 429
                    else 'other_error' if code == 'runtime_agent_schema_error' else 'provider_error')
                call_id = attempt['request_id']
                self.emit('llm_calls', {'event_id': call_id, 'call_id': call_id,
                    'task_id': task_id,
                    'stage': request['stage'], 'phase': request['phase'], 'budget_scope': request.get('budget_scope'),
                    'logical_decision_id': request.get('logical_decision_id'), 'decision_scope': request.get('decision_scope'),
                    'owner_state_version': request.get('owner_state_version'), 'purpose': request.get('purpose'),
                    'model_id': attempt['model_id'], 'request_messages': attempt['final_payload_audit']['messages'],
                    'request_tools': attempt['final_payload_audit'].get('tools', []),
                    'private_reasoning_redacted': attempt['final_payload_audit'].get('private_reasoning_redacted', False),
                    'actual_payload_sha256': attempt['payload_fingerprint'], 'response_text': message.get('content'),
                    'tool_calls': message.get('tool_calls'), 'public_response': response,
                    'raw_usage': raw, 'prompt_tokens': usage.get('prompt_tokens'),
                    'completion_tokens': usage.get('completion_tokens'),
                    'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                    'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens', usage.get('prompt_cache_hit_tokens')),
                    'request_start_time': utc(attempt['started_at']), 'request_end_time': utc(attempt['ended_at']),
                    'latency_ms': attempt['attempt_latency_ms'], 'provider_request_id': attempt.get('provider_request_id') or None,
                    'retry_index': attempt['retry_count'], 'structural_repair_index': request['repair'], 'call_status': status})
                if status != 'success':
                    self.emit('errors', {'event_id': call_id, 'task_id': task_id,
                        'timestamp': utc(attempt['ended_at']), 'event_type': 'llm_request_error', 'normalized_error': code,
                        'message': attempt.get('sanitized_error'), 'retry_index': attempt['retry_count'],
                        'http_status': attempt.get('http_status'),
                        'infrastructure_error': code != 'runtime_agent_schema_error'})

    def native_observer(self, scope):
        def observe(event, started_at, ended_at):
            result = event.get('result', {})
            self.emit('interactions', {'event_id': scope + ':' + str(event['index']) + ':' + event['state'], 'scope_id': scope,
                'task_id': self.trial_tasks.get(scope, self.current_task['task_id']), 'interaction_index': event['index'],
                'action_or_tool_name': event['name'], 'action_arguments': event['arguments'],
                'observation': result, 'action_status': event['state'] if event['state'] != 'finished' else
                    ('success' if result.get('accepted') else 'rejected'),
                'timestamp': utc(started_at), 'end_time': utc(ended_at) if ended_at else None,
                **{key: event.get(key) for key in ['backend_invoked','tool_call_consumed','environment_step',
                    'batch_id','call_id','result_id','local_result_read','progress_before','progress_after','state_update']}})
        return observe

    def training(self, event_id, unit_type, start, end, examples, status, error=None, **details):
        if event_id in self.ids['training_events']:
            return
        self.training_index += 1
        self.consumed += examples
        self.emit('training_events', {'event_id': event_id, 'training_event_id': event_id,
            'task_id': self.current_task['task_id'], 'training_unit_type': unit_type,
            'training_unit_index': self.training_index, 'train_examples_seen_this_unit': examples,
            'cumulative_train_examples_seen': self.consumed, 'training_event_start_time': start,
            'training_event_end_time': end, 'training_event_status': status, 'training_event_error': error, **details})

    def trial_start(self, trial_id, task):
        self.trial_tasks[trial_id] = task.task_id
        return {'id': trial_id, 'task_id': task.task_id, 'start': utc()}

    def learning_start(self, task):
        event_id = self.current_task['attempt_id'] + ':learning'
        path = self.root / 'learning_starts' / (event_id + '.json')
        if path.exists():
            self.active_learning = json.loads(path.read_text())
            self.active_learning['subjects'] = set(self.active_learning['subjects'])
        else:
            self.active_learning = {'id': event_id, 'start': utc(), 'subjects': {task.physical_key}}
            write_json(path, {**self.active_learning, 'subjects': sorted(self.active_learning['subjects'])})
        return self.active_learning

    def learning_end(self, observation, result):
        self.training(observation['id'], 'learning_update', observation['start'], utc(), len(observation['subjects']),
                      'rejected' if result and result.get('rejected') else 'completed',
                      result.get('error') if result else None, consumed_physical_keys=sorted(observation['subjects']))
        self.active_learning = None

    def trial_end(self, observation, events, record, result, audit):
        self.training(observation['id'], 'program_trial', observation['start'], utc(), int(observation.get('consumed', False)),
                      'completed' if record else 'exception', None if record else 'See native trial checkpoint',
                      consumed_task_id=observation['task_id'])
        if result and result.get('score'):
            self.scorer(observation['task_id'], observation['id'], result['score'], phase='trial', audit=audit)

    def asset(self, kind, asset):
        content_hash = digest(asset)
        previous = self.versions.get(asset['id'])
        if previous and previous['artifact_hash'] == content_hash:
            return
        parent = previous['artifact_id'] if previous else None
        if not parent and kind == 'program' and self.requests_seen:
            request = self.requests_seen[-1]
            material = json.loads(request['messages'][1]['content'])
            if material.get('source'):
                for row in self.versions.values():
                    if row['artifact_kind'] == 'program' and json.loads(Path(row['artifact_path']).read_text()).get('source') == material['source']:
                        parent = row['artifact_id']
                        break
        artifact_id = asset['id'] + ':' + content_hash
        path = self.root / 'artifact_versions' / (content_hash + '.json')
        write_json(path, asset)
        row = {'event_id': artifact_id, 'artifact_id': artifact_id, 'logical_asset_id': asset['id'],
               'artifact_kind': kind, 'artifact_version': previous['artifact_version'] + 1 if previous else 1,
               'parent_artifact_id': parent, 'created_at_training_unit': self.active_learning['id'] if self.active_learning else self.current_task['attempt_id'],
               'artifact_path': str(path), 'artifact_hash': content_hash, 'file_sha256': sha256(path),
               'artifact_size_bytes': path.stat().st_size, 'created_at': utc(),
               'accepted': asset.get('state') == 'usable' if kind == 'program' else True,
               'is_final_frozen': False}
        self.emit('artifacts', row)
        self.versions[asset['id']] = row

    def scorer(self, task_id, scope, score, phase, audit):
        self.emit('scorer_outputs', {'event_id': scope, 'task_id': task_id, 'phase': phase,
            'scorer_name': score['scorer'], 'scorer_version': audit.get('scorer_version'),
            'raw_scorer_output': audit.get('raw_scorer_output'), 'official_score': score['raw_score'], 'scoring_error': None})

    def end_task(self, trace):
        task = self.current_task
        self.requests(trace.get('requests', []))
        self.scorer(task['task_id'], task['attempt_id'], trace['score'], task['split'], audit=trace.get('scoring_audit', {}))
        calls = self.task_call_counts[(task['task_id'], task['split'])]
        infrastructure = task['task_id'] in self.infrastructure_tasks
        error = trace.get('error')
        self.emit('episodes', {'event_id': task['attempt_id'], **task, 'task_end_time': utc(),
            'official_score': trace['score']['raw_score'], 'success': trace['score']['hard'],
            'terminal_reason': trace['execution']['reason'], 'environment_step_count': trace.get('environment_steps', 0),
            'task_llm_call_count': calls, 'task_infrastructure_error': infrastructure,
            'task_error_type': error.get('code') if error else None, 'task_error_message': error.get('message') if error else None})
        if task['split'] == 'train':
            self.training(task['attempt_id'], 'trajectory', task['task_start_time'], utc(), 1, 'completed',
                          learning_result=trace.get('learning'))

    def task_error(self, exc):
        from ..agents.provider import AgentProviderError
        from ..harness.simple_protocol import UnknownSideEffect
        from ..core.errors import FailureLayer
        infrastructure = isinstance(exc, (AgentProviderError, UnknownSideEffect, OSError)) or getattr(exc, 'layer', None) == FailureLayer.INFRASTRUCTURE
        code = getattr(exc, 'code', type(exc).__name__)
        self.emit('errors', {'event_id': 'error:' + utc(), 'task_id': self.current_task['task_id'] if self.current_task else None,
            'timestamp': utc(), 'event_type': 'task_error', 'normalized_error': code,
            'message': str(exc), 'infrastructure_error': infrastructure})
        if self.current_task and not getattr(exc, 'model_authored', False):
            self.emit('episodes', {'event_id': self.current_task['attempt_id'], **self.current_task,
                'task_end_time': utc(), 'official_score': None, 'success': None,
                'terminal_reason': 'infrastructure_error' if infrastructure else 'execution_error',
                'environment_step_count': None, 'task_llm_call_count': self.task_call_counts[(self.current_task['task_id'], self.current_task['split'])],
                'task_infrastructure_error': infrastructure, 'task_error_type': code, 'task_error_message': str(exc)})
        return infrastructure

    def freeze(self, path, *, newly_created):
        target = self.root / 'final_frozen_manifest.json'
        if target.exists():
            frozen = json.loads(target.read_text())
            if tree_identity(path)['sha256'] != frozen['final_artifact_hash']:
                raise RuntimeError('Frozen snapshot changed')
            return frozen
        if not newly_created:
            raise RuntimeError('Freeze occurred without its formal timestamp; inspect native state before resuming')
        identity = tree_identity(path)
        frozen = {'final_artifact_id': 'snapshot:' + identity['sha256'], 'final_artifact_hash': identity['sha256'],
                  'final_artifact_path': str(path), 'freeze_time': utc(), 'test_started_after_freeze': False,
                  'artifact_hash_after_test': None, 'files_sha256': identity['files_sha256']}
        write_json(target, frozen)
        self.emit('artifacts', {'event_id': frozen['final_artifact_id'], 'artifact_id': frozen['final_artifact_id'],
            'logical_asset_id': 'snapshot', 'artifact_kind': 'snapshot', 'artifact_version': 1,
            'parent_artifact_id': None, 'created_at_training_unit': self.current_task['attempt_id'] if self.current_task else None,
            'artifact_path': str(path), 'artifact_hash': identity['sha256'], 'artifact_size_bytes': identity['size_bytes'],
            'accepted': True, 'is_final_frozen': True})
        return frozen

    def finish(self, status):
        self.manifest.update(run_status=status, end_time=utc())
        write_json(self.manifest_path, self.manifest)
