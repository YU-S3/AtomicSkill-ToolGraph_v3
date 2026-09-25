"""Isolated original EmbodiSkill operation, with a ScienceWorld Env adapter."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from atomic_skillgraph.core.serialization import atomic_write_json
from experiments.baselines.common.model_client import AuditedChatClient, append_event
from .protocol import ScienceWorldTextPolicyProtocol as Protocol, PolicyRejection


def run(job):
    # The upstream agentkit import prints this alias; never expose it.
    os.environ.pop('OPENAI_API_KEY', None)
    os.environ.update(TOKENIZERS_PARALLELISM='false', ANONYMIZED_TELEMETRY='False')
    sys.path.insert(0, job['source'])
    os.chdir(job['source'])
    import torch
    import numpy as np
    torch.set_num_threads(1)
    random.seed(job['seed'])
    np.random.seed(job['seed'])
    from agentkit.skill import EmbodiSkill
    from agentkit.utils import EmbeddingFunc
    from agentkit.reasoning import ReasoningIO
    from tasks.workflow.team.team import TeamSolver
    from experiments.baselines.b4_embodiskill.model_adapter import MethodTransport, AuditedEmbedding, instrument_method
    from .embodiskill_env import make_environment, install_scienceworld_prompts
    install_scienceworld_prompts()
    output, state = Path(job['output']), Path(job['state'])
    started = time.monotonic()
    readonly = job['phase'] in ('dev', 'test')
    client = AuditedChatClient(output=output / 'provider_calls.jsonl', identity=job['identity'], model=job['model'])

    class Transport(MethodTransport):
        env = None

        def __call__(self, messages, **kwargs):
            if self.stage not in ('solver', 'stuck_recovery'):
                return super().__call__(messages, **kwargs)
            rows = [dict(role=m.role, content=m.content) for m in messages]
            rows.append({'role': 'user', 'content': json.dumps(Protocol.frame(self.env.harness), ensure_ascii=False)})
            for repair in range(2):
                text = self.client.chat(messages=rows, role='target', stage='scienceworld_protocol_repair' if repair else self.stage,
                    method_output_token_hint=kwargs.get('max_tokens', 512), temperature=kwargs.get('temperature', .1))
                try:
                    return self.env.process_action(text)
                except PolicyRejection as exc:
                    self.env.rejections += 1
                    append_event(output / 'protocol_rejections.jsonl', {'repair': bool(repair), 'errors': exc.errors})
                    rows.extend([{'role': 'assistant', 'content': text}, {'role': 'user', 'content': str(exc) + '\nRepair once using the current catalog.'}])
            self.env._exhausted_text = text
            return text

    transport = Transport(client, output, readonly=readonly)
    delegate = EmbeddingFunc(job['embedding_path'])
    delegate.model_type = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding = AuditedEmbedding(delegate, output)
    skill = EmbodiSkill(namespace='EmbodiSkill', global_config={**job['config'],
        'working_dir': str(state.parent), 'persist_dir': str(state), 'task': 'scienceworld',
        'current_epoch_id': job['epoch']}, llm_model=transport, embedding_func=embedding)
    before = skill.get_active_manual_data()
    result = {'phase': job['phase'], 'operation': job['operation'], 'manual_before': before}
    if job['phase'] == 'revision':
        instrument_method(skill, None, transport, output, readonly=False)
        result['revision'] = skill.revise_manual(epoch_id=job['epoch'], success_rate=job['train_score'])
    else:
        entry = job['entry']
        env = make_environment(entry, output, max_trials=100)
        transport.env = env
        try:
            task = {'task_main': env.harness._frame['task_description'], 'task_description': env.initial_observation, 'few_shots': []}
            team = TeamSolver()
            team.build_system(ReasoningIO(transport), skill, env, job['config'])
            for agent in team.agents_team.values():
                agent.add_task_instruction(Protocol.instruction)
            instrument_method(skill, team, transport, output, readonly=readonly)
            reward, won, trajectory = team.schedule(task, update_skill=not readonly)
            channel = env.harness.validator_channel()
            if reward != channel.score / 100 or won != (channel.score == 100):
                raise RuntimeError('Upstream reward disagrees with official score')
            result.update(task_id=entry['task_id'], task_type=entry['task_type'], macro_type=entry['macro_type'],
                variation_idx=entry['variation_idx'], benchmark='scienceworld', source_identity=entry,
                official_score=channel.score, normalized_score=channel.score/100, perfect_success=won,
                official_success=won, environment_done=channel.done, environment_moves=env.harness._info['moves'],
                environment_actions=len(env.actions), protocol_rejections=env.rejections, trajectory=trajectory,
                infrastructure_failure=False, update_skill=not readonly)
            if skill.get_active_manual_data() != before:
                raise RuntimeError('Manual changed within episode')
        finally:
            env.close()
    result.update(manual_after=skill.get_active_manual_data(), trajectory_records=skill.skill_size,
        embedding_calls=embedding.calls, wall_time_ms=int((time.monotonic()-started)*1000))
    atomic_write_json(output / 'result.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    args = p.parse_args()
    run(json.loads(Path(args.job).read_text()))
