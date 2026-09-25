"""ScienceWorld EnvAdapter for the unchanged pinned ReflACT trainer."""
import json
from pathlib import Path
from skillopt.envs.base import EnvAdapter
from experiments.baselines.b3_skillopt.common_alfworld_adapter import CommonSkillOptDataLoader
from atomic_skillgraph.core.serialization import atomic_write_json
from .runner import ScienceWorldTextEpisodeRunner

class ScienceWorldDataLoader(CommonSkillOptDataLoader):
    def __init__(self, train, dev):
        self.train_items = [{**r, 'id': r['task_id'], 'phase': 'train'} for r in train]
        self.val_items = [{**r, 'id': r['task_id'], 'phase': 'validation'} for r in dev]
        self.test_items = []  # Test cannot be queried by optimizer or slow update.

class ScienceWorldSkillOptAdapter(EnvAdapter):
    def __init__(self, train, dev, chat_factory, *, max_steps=100, seed=42):
        self.dataloader = ScienceWorldDataLoader(train, dev)
        self.chat_factory, self.max_steps, self.seed = chat_factory, max_steps, seed
        self.analyst_workers, self.failure_only, self.minibatch_size, self.edit_budget = 1, False, 8, 4

    def setup(self, cfg):
        super().setup(cfg)
        self.analyst_workers = cfg['analyst_workers']
        self.minibatch_size, self.edit_budget = cfg['minibatch_size'], cfg['edit_budget']

    def get_dataloader(self): return self.dataloader
    def get_task_types(self): return sorted({r['task_type'] for r in self.dataloader.train_items})
    def build_reference_text(self, item): return ''
    def build_env_from_batch(self, batch, **kwargs): return list(batch.payload)
    def build_train_env(self, batch_size, seed, **kwargs):
        return self.build_env_from_batch(self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs))
    def build_eval_env(self, env_num, split, seed, **kwargs):
        return self.build_env_from_batch(self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs))

    def rollout(self, env_manager, skill_content, out_dir, **kwargs):
        from .parallel import ordered_map
        output, rows = Path(out_dir), []
        def evaluate(entry):
            episode = output / 'predictions' / entry['task_id']
            chat = self.chat_factory(episode, entry)
            result = ScienceWorldTextEpisodeRunner(chat, max_actions=self.max_steps).run(entry, skill_content, episode)
            row = {'id': entry['task_id'], 'task_type': entry['task_type'],
                'hard': int(result['perfect_success']), 'soft': result['normalized_score'],
                'official_score': result['official_score'], 'conversation_path': str(episode / 'conversation.json')}
            return row
        rows=ordered_map(evaluate,env_manager)
        atomic_write_json(output / 'results.json', rows)
        return rows
