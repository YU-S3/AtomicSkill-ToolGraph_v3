"""Calls the pinned learning algorithms; Test is never supplied to them."""
import json
from pathlib import Path
import yaml
from atomic_skillgraph.core.serialization import atomic_write_json

ROOT = Path(__file__).resolve().parents[3]

def train_skillopt(root, train, dev, chat_factory, model, seed, *, smoke=False):
    from experiments.baselines.b3_skillopt.worker import _flat_train_cfg, _configure_model
    from experiments.baselines.common.model_config import ModelConfig
    from experiments.baselines.b3_skillopt.provider_observer import install_provider_observer, uninstall_provider_observer
    from .skillopt_adapter import ScienceWorldSkillOptAdapter
    from skillopt.engine.trainer import ReflACTTrainer
    cfg = yaml.safe_load((ROOT / 'configs/baselines/b3_skillopt.yaml').read_text())
    cfg['evaluation'].update(gate_metric='soft', sel_env_num=len(dev), eval_test=False)
    cfg['env'].update(name='scienceworld', workers=1, max_api_workers=1)
    cfg['gradient']['analyst_workers'] = 1
    cfg['train']['train_size'] = len(train)
    if smoke:
        cfg['train'].update(num_epochs=1, batch_size=len(train))
        cfg['optimizer']['slow_update_samples'] = len(train)
    identity = ModelConfig.from_mapping(model)
    _configure_model(identity, sdk_max_retries=0)
    observer = install_provider_observer(output_path=root / 'provider_calls.jsonl', method='b3_skillopt',
        phase='train', model=identity.model, reasoning_effort=identity.reasoning_effort,
        run_id=root.parent.name, run_seed=seed, application_retry_limit=5,
        retry_delays_seconds=[2,5,10,20], expected_sdk_max_retries=0)
    adapter = ScienceWorldSkillOptAdapter(train, dev, chat_factory, seed=seed)
    flat = _flat_train_cfg(config=cfg, model=identity, run_seed=seed, out_root=root,
        skill_init_path=str(ROOT / 'experiments/baselines/assets/scienceworld_initial.md'),
        train_size=len(train), selection_size=len(dev))
    flat['env'] = 'scienceworld'
    atomic_write_json(root / 'actual_algorithm_config.json', flat)
    try:
        summary = ReflACTTrainer(flat, adapter).train()
        observer.raise_if_active_episode_failed()
        atomic_write_json(root / 'algorithm_result.json', summary)
    finally:
        uninstall_provider_observer(observer)
    return (root / 'best_skill.md').read_text()

def train_gepa(root, train, dev, chat_factory, evolution_client, seed, *, smoke=False):
    from gepa import optimize
    from .gepa_adapter import ScienceWorldGEPAAdapter
    cfg = yaml.safe_load((ROOT / 'configs/baselines/b5_gepa.yaml').read_text())['gepa']
    keys = ('candidate_selection_strategy', 'frontier_type', 'skip_perfect_score', 'batch_sampler',
        'reflection_minibatch_size', 'perfect_score', 'module_selector', 'use_merge', 'max_metric_calls',
        'cache_evaluation', 'val_evaluation_policy', 'acceptance_criterion')
    options = {k: cfg[k] for k in keys}
    if smoke:
        options.update(reflection_minibatch_size=1, max_metric_calls=len(dev) + 2 + len(dev))
    atomic_write_json(root / 'actual_algorithm_config.json', options)
    def reflection(messages):
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        return evolution_client.chat(messages=messages, stage='gepa_reflection', role='evolution')
    result = optimize(seed_candidate={'skill_text': (ROOT / 'experiments/baselines/assets/scienceworld_initial.md').read_text()},
        trainset=train, valset=dev, adapter=ScienceWorldGEPAAdapter(root / 'evaluations', chat_factory),
        reflection_lm=reflection, run_dir=str(root / 'optimizer'), seed=seed, raise_on_exception=True,
        **options)
    atomic_write_json(root / 'algorithm_result.json', {'candidates': result.candidates,
        'parents': result.parents, 'best_idx': result.best_idx,
        'val_aggregate_scores': result.val_aggregate_scores, 'total_metric_calls': result.total_metric_calls})
    skill = result.candidates[result.best_idx]['skill_text']
    (root / 'best_skill.md').write_text(skill, encoding='utf-8')
    return skill
