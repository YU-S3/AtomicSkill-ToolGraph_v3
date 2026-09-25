"""Shared B0/B1/B3/B5 text loop. Rejected tuples never reach env.step."""
import json
from pathlib import Path
import time
import uuid
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.core.serialization import atomic_write_json
from experiments.scienceworld_manifest import task_from_entry
from experiments.baselines.common.model_client import append_event
from .protocol import ScienceWorldTextPolicyProtocol as Protocol, PolicyRejection

class ScienceWorldTextEpisodeRunner:
    def __init__(self, chat, *, max_actions=100, harness_factory=ScienceWorldAdapter):
        if not 1 <= max_actions <= 100:
            raise ValueError('Policy action cap must be in 1..100')
        self.chat, self.max_actions, self.harness_factory = chat, max_actions, harness_factory

    def run(self, entry, skill_text, output):
        from .parallel import environment_slot
        with environment_slot():
            return self._run(entry, skill_text, output)

    def _run(self, entry, skill_text, output):
        episode = Path(output)
        if (episode / 'result.json').exists():
            raise FileExistsError(episode)
        # Keep failed prefixes immutable. The paid provider audit remains at
        # episode scope, while each environment reset has its own action path.
        attempt_id = uuid.uuid4().hex
        output = episode / 'attempts' / attempt_id
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_json(output / 'attempt.json', {'attempt_id': attempt_id,
            'source_identity': entry, 'started_at_unix': time.time()})
        started = time.monotonic()
        harness = self.harness_factory(split=entry['source_split'])
        actions, rejections, decisions, conversation = [], 0, 0, []
        try:
            harness.reset(task_from_entry(entry, ''))
            instruction = Protocol.instruction
            if skill_text:
                instruction += '\n\nReusable skill knowledge:\n' + skill_text
            messages = [{'role': 'system', 'content': instruction}]
            termination = 'action_budget'
            while not harness.validator_channel().done and len(actions) < self.max_actions:
                frame = Protocol.frame(harness)
                messages.append({'role': 'user', 'content': json.dumps(frame, ensure_ascii=False)})
                revision = harness.validator_channel().revision
                selected = None
                for repair in range(2):
                    # Same paid/audited backend for normal and repair requests.
                    text = self.chat(messages=messages, repair=bool(repair), task_id=entry['task_id'])
                    decisions += 1
                    messages.append({'role': 'assistant', 'content': text})
                    append_event(output / 'policy_responses.jsonl', {'request_sequence': decisions,
                        'revision': revision, 'repair': bool(repair), 'content': text})
                    try:
                        selected = Protocol.parse(text, harness=harness, revision=revision)
                        break
                    except PolicyRejection as exc:
                        rejections += 1
                        append_event(output / 'protocol_rejections.jsonl', {'revision': revision,
                            'repair': bool(repair), 'errors': exc.errors})
                        if not repair:
                            messages.append({'role': 'user', 'content': str(exc) + '\nRepair once using the same current catalog.'})
                if selected is None:
                    termination = 'protocol_repair_exhausted'
                    break
                result = harness.execute_action(selected.action_id, revision)
                event = {'step_index': len(actions), 'action_type': selected.action_type,
                    'arguments': selected.arguments, 'raw_action': selected.raw_action,
                    'accepted': result.accepted, 'observation': harness._frame['observation'],
                    'score': result.benchmark_score, 'reward_delta': result.benchmark_reward,
                    'done': result.done, 'moves': result.metadata['environment_moves']}
                actions.append(event)
                conversation.append({'role': 'assistant', 'content': json.dumps({'action_type': selected.action_type,
                    'arguments': selected.arguments})})
                conversation.append({'role': 'user', 'content': result.observation})
                append_event(output / 'environment_actions.jsonl', event)
            channel = harness.validator_channel()
            if channel.done:
                termination = 'perfect' if channel.won else 'environment_terminal'
            result = {'task_id': entry['task_id'], 'task_type': entry['task_type'],
                'macro_type': entry['macro_type'], 'variation_idx': entry['variation_idx'],
                'benchmark': 'scienceworld', 'official_score': channel.score,
                'normalized_score': channel.score / 100.0, 'perfect_success': channel.score == 100,
                'official_success': channel.score == 100, 'environment_done': channel.done,
                'environment_moves': harness._info['moves'], 'environment_actions': len(actions),
                'protocol_rejections': rejections, 'target_decision_requests': decisions,
                'termination_reason': termination, 'wall_time_ms': int(1000 * (time.monotonic()-started)),
                'source_identity': entry, 'infrastructure_failure': False,
                'selected_attempt_id': attempt_id,
                'canonical_actions_path': str(output / 'environment_actions.jsonl')}
            atomic_write_json(output / 'conversation.json', conversation)
            atomic_write_json(output / 'messages.json', messages)
            atomic_write_json(output / 'result.json', result)
            # Existing optimizer adapters consume these two episode-level files.
            # They are published only after one complete terminal attempt.
            atomic_write_json(episode / 'conversation.json', conversation)
            atomic_write_json(episode / 'result.json', result)
            return result
        except Exception as exc:
            atomic_write_json(output / 'infrastructure_failure.json', {'error_type': type(exc).__name__,
                'source_identity': entry, 'completed_actions': len(actions),
                'attempt_id': attempt_id})
            raise
        finally:
            harness._close_backend()
