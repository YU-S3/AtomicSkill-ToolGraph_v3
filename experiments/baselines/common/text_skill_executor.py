"""Public text executor: delegate unchanged to the B3/B5 pinned rollout."""
from experiments.baselines.b3_skillopt.episode_runner import SkillOptTextEpisodeRunner


class TextSkillALFWorldExecutor(SkillOptTextEpisodeRunner):
    def run_episode(self, *, task, skill_text, run_seed, phase,
                    output_dir, rollout_id, max_environment_actions=100):
        if run_seed != self.seed or max_environment_actions != self.max_actions:
            raise ValueError("Episode settings differ from executor identity")
        # Upstream _build_skill_prompt(None) omits the entire Skill Knowledge
        # section. No initial.md or other persistent knowledge is loaded here.
        return self.run(task, skill_text, str(output_dir), rollout_id=rollout_id)
