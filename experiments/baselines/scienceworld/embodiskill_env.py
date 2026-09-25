"""ScienceWorld public boundary for the unchanged upstream TeamSolver."""
import json
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from experiments.scienceworld_manifest import task_from_entry
from experiments.baselines.common.model_client import append_event
from .protocol import ScienceWorldTextPolicyProtocol as Protocol, PolicyRejection


def make_environment(entry, output, *, max_trials=100):
    from agentkit.agents import Env

    class Environment(Env):
        def __init__(self):
            self.harness = ScienceWorldAdapter(split=entry['source_split'])
            self.harness.reset(task_from_entry(entry, ''))
            self.initial_observation = json.dumps(Protocol.frame(self.harness), ensure_ascii=False)
            self.max_trials = min(max_trials, 100)
            self.actions, self.rejections = [], 0
            self.done, self.won, self._reset_consumed = False, False, False
            append_event(output / 'reset.jsonl', {'source_identity': entry, 'public_frame': Protocol.frame(self.harness)})

        def reset(self):
            if self._reset_consumed:
                raise RuntimeError('An isolated episode cannot reset twice')
            self._reset_consumed = True
            return self.initial_observation

        def process_action(self, text):
            # Pure parsing; upstream transport evidence may invoke this too.
            if getattr(self, '_exhausted_text', None) == text:
                return text
            selected = Protocol.parse(text, harness=self.harness, revision=self.harness.validator_channel().revision)
            return json.dumps({'action_type': selected.action_type, 'arguments': selected.arguments})

        def step(self, text):
            if self.done or len(self.actions) >= self.max_trials:
                raise RuntimeError('Step beyond episode boundary')
            try:
                selected = Protocol.parse(text, harness=self.harness, revision=self.harness.validator_channel().revision)
            except PolicyRejection as exc:
                # Transport already used the sole repair. Never send an invalid tuple to the JVM.
                self.done = True
                return str(exc), 0.0, True
            result = self.harness.execute_action(selected.action_id, self.harness.validator_channel().revision)
            self.done, self.won = result.done, self.harness.validator_channel().won
            self.last_matched_action = text
            self.actions.append(selected.raw_action)
            append_event(output / 'environment_actions.jsonl', {'step_index': len(self.actions),
                'raw_action': selected.raw_action, 'action_type': selected.action_type,
                'arguments': selected.arguments, 'observation': result.observation,
                'official_score': result.benchmark_score, 'reward_delta': result.benchmark_reward,
                'done': result.done, 'accepted': result.accepted})
            return json.dumps(Protocol.frame(self.harness), ensure_ascii=False), result.benchmark_reward / 100.0, self.done or len(self.actions) >= self.max_trials

        def feedback(self):
            score = self.harness.validator_channel().score
            return score / 100.0, score == 100, json.dumps({'official_score': score, 'perfect_success': score == 100})

        def close(self):
            self.harness._close_backend()

    return Environment()


def install_scienceworld_prompts():
    """Benchmark vocabulary only; retain upstream reflection schemas and revision logic."""
    import importlib
    p = importlib.import_module('agentkit.skill.embodiskill_skill.prompt')
    m = importlib.import_module('agentkit.skill.embodiskill_skill.EmbodiSkill')
    for module in (p, m):
        module.normalize_manual_profile = lambda task: 'scienceworld'
        module.get_manual_display_name = lambda profile: 'ScienceWorld Skill Manual'
    p.EmbodiSkillPrompts.extract_true_traj_user_prompt = (
        'Extract critical steps faithfully from the supplied successful trajectory. '
        'Do not add actions, hidden scientific answers, or unobserved outcomes. '
        'Retain exact entity identities and distinguish action rejection from successful observation. '
        'Output concise named procedural steps.\nTask:\n{task}\nTrajectory:\n{trajectory}\nOutput:\n')
    p.EmbodiSkillPrompts.detect_mistakes_system_prompt = (
        'Compare the public observations in the failed trajectory with the stated task. '
        'Preserve exact entity identities, numbers, units and observed state. '
        'Separate missing evidence, execution failure and unsupported scientific inference. '
        'Do not invent hidden state or infer correctness from an object name. Return a concise diagnosis.')
