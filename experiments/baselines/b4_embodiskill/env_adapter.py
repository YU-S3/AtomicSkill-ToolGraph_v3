"""Exact-gamefile benchmark boundary; upstream action/reward semantics retained."""
from pathlib import Path

from experiments.baselines.common.formal_validation import verify_observed_gamefile
from experiments.baselines.common.model_client import append_event


def make_environment(task, data_root, source, *, max_trials, seed, output):
    import yaml
    import alfworld.agents.environment
    from tasks.envs.alfworld_env import AlfworldEnv

    class ExactEnvironment(AlfworldEnv):
        def __init__(self):
            config = yaml.safe_load((source / "tasks/env_configs/alfworld_config.yaml").read_text())
            config["dataset"].update(data_path=str(data_root / "json_2.1.1/train"),
                eval_id_data_path=str(data_root / "json_2.1.1/valid_seen"),
                eval_ood_data_path=str(data_root / "json_2.1.1/valid_unseen"))
            config["logic"].update(domain=str(data_root / "logic/alfred.pddl"),
                                   grammar=str(data_root / "logic/alfred.twl2"))
            config["env"].update(domain_randomization=False, goal_desc_human_anns_prob=0.0)
            config["rl"]["training"]["max_nb_steps_per_episode"] = 100
            split = {"train": "train", "valid_seen": "eval_in_distribution",
                     "valid_unseen": "eval_out_of_distribution"}[task.source_split]
            self.main_env = alfworld.agents.environment.get_environment("AlfredTWEnv")(config, train_eval=split)
            self.main_env.game_files = [str(data_root / task.gamefile_rel)]
            self.env = self.main_env.init_env(batch_size=1)
            self.env.seed(seed)
            observations, infos = self.env.reset()
            self.gamefile = str(verify_observed_gamefile(task, infos["extra.gamefile"][0], alfworld_data=data_root))
            self.initial_observation = str(observations[0])
            self.max_trials, self.actions, self.done, self.won = max_trials, [], False, False
            self._reset_consumed = False
            append_event(output / "reset.jsonl", dict(observation=self.initial_observation,
                observed_gamefile=self.gamefile, official_won=False))

        def reset(self):
            # TeamSolver's reset consumes the exact reset used to form its visible task.
            if self._reset_consumed:
                raise RuntimeError("An isolated episode cannot reset twice")
            self._reset_consumed = True
            return self.initial_observation

        def step(self, action):
            if self.done or self.won or len(self.actions) >= 100:
                raise RuntimeError("Step beyond terminal/action boundary")
            sent = self.process_action(action)
            observations, rewards, dones, infos = self.env.step([sent])
            raw = str(observations[0])
            self.done, self.won = bool(dones[0]), bool(infos["won"][0])
            observation = raw[raw.find('. ')+2:] if raw.startswith('You arrive at loc ') else raw
            reward = -1 if 'think:' in sent or observation == 'Nothing happens.' else int(self.won)
            if 'think:' in sent:
                observation = 'OK.'
            self.actions.append(sent)
            append_event(output / "environment_actions.jsonl", dict(episode_task_id=task.task_id,
                step_index=len(self.actions), action=sent, env_feedback=raw,
                method_observation=observation, reward=float(rewards[0]),
                method_reward=reward, done=self.done, won=self.won))
            return observation, reward, self.done or self.won or len(self.actions) >= 100

        def feedback(self):
            return float(self.won), self.won, ("You successfully finished this task!" if self.won else "You failed the task.")

    return ExactEnvironment()
