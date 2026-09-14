"""One isolated direct upstream task or manual-revision operation."""
from __future__ import annotations
import argparse
import os
import random
import sys
import time
import traceback
from pathlib import Path

from experiments.baselines.b4_embodiskill.state import read_json, write_json
from experiments.baselines.common.source_identity import sanitize_error_text


def run(job):
    output, state, source = (Path(job[k]) for k in ("output", "state", "source"))
    # agentkit.llm prints OPENAI_API_KEY at import. Never expose that alias.
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ.pop("EMBODISKILL_MAX_TOKEN", None)
    os.environ.update(TOKENIZERS_PARALLELISM="false", ANONYMIZED_TELEMETRY="False")
    os.chdir(source)
    sys.path.insert(0, str(source))
    import torch
    import numpy as np
    torch.set_num_threads(1)
    random.seed(job["run_seed"])
    np.random.seed(job["run_seed"])
    from agentkit.skill import EmbodiSkill
    from agentkit.utils import EmbeddingFunc
    from agentkit.reasoning import ReasoningIO
    from tasks.workflow.team.team import TeamSolver
    from tasks.prompts import get_dataset_system_prompt, get_task_few_shots
    from experiments.baselines.common.model_client import AuditedChatClient
    from experiments.baselines.common.provider_gate import CampaignProviderGate
    from experiments.baselines.common.manifest import ManifestTask
    from experiments.baselines.b4_embodiskill.env_adapter import make_environment
    from experiments.baselines.b4_embodiskill.manifest_adapter import visible_task
    from experiments.baselines.b4_embodiskill.model_adapter import MethodTransport, AuditedEmbedding, instrument_method

    started = time.monotonic()
    cfg, phase = job["config"], job["phase"]
    identity = {k: job[k] for k in ("run_id", "run_seed", "phase", "operation")}
    identity["task_id"] = job.get("task", {}).get("task_id")
    gate = CampaignProviderGate(gate_dir=Path(job["gate_dir"]), campaign_id=job["campaign_id"],
                                max_inflight=job["provider_cap"])
    client = AuditedChatClient(output=output / "provider_calls.jsonl", identity=identity, model=cfg["model"], gate=gate)
    readonly = phase in {"validation", "test", "load_probe"}
    transport = MethodTransport(client, output, readonly=readonly)
    delegate = EmbeddingFunc(job["embedding_path"])
    delegate.model_type = cfg["embedding"]["model"]
    embedding = AuditedEmbedding(delegate, output)
    skill = EmbodiSkill(namespace="EmbodiSkill", global_config={**cfg["embodiskill"],
        "working_dir": str(state.parent), "persist_dir": str(state), "task": "alfworld",
        "current_epoch_id": job["epoch"]}, llm_model=transport, embedding_func=embedding)
    before = skill.get_active_manual_data()
    result = dict(phase=phase, operation=job["operation"], manual_before=before,
                  method_specific_alfworld_prior=True, trainer_constructed=not readonly)
    if phase == "revision":
        instrument_method(skill, None, transport, output, readonly=False)
        result["revision"] = skill.revise_manual(epoch_id=job["epoch"], success_rate=job["train_success_rate"])
    else:
        task = ManifestTask.from_dict(job["task"])
        env = make_environment(task, Path(job["alfworld_data"]), source,
            max_trials=cfg["embodiskill"]["max_trials"], seed=job["run_seed"], output=output)
        try:
            if phase == "load_probe":
                # Hold real embedding/Chroma/ALFWorld simultaneously until controller samples memory.
                write_json(output / "ready.json", dict(pid=os.getpid(), gamefile=env.gamefile))
                deadline = time.monotonic() + 7200
                while not Path(job["release_file"]).exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Load-probe release did not arrive")
                    time.sleep(.5)
                result["provider_probe"] = client.chat(messages=[dict(role="user", content="Reply OK.")],
                    role="target", stage="load_probe", max_tokens=512)
            else:
                config = visible_task(task, env.initial_observation)
                config["few_shots"] = get_task_few_shots("alfworld", config, cfg["embodiskill"]["static_few_shots"])
                team = TeamSolver()
                team.build_system(ReasoningIO(transport), skill, env, cfg["embodiskill"])
                for agent in team.agents_team.values():
                    agent.add_task_instruction(get_dataset_system_prompt("alfworld", config))
                instrument_method(skill, team, transport, output, readonly=readonly)
                if phase == "role_probe":
                    trajectory = live_role_probe(team, skill, env, config)
                    won = env.won
                else:
                    reward, won, trajectory = team.schedule(config, update_skill=not readonly)
                if bool(won) != env.won:
                    raise RuntimeError("Upstream success label differs from official won")
                result.update(task=task.to_dict(), official_success=env.won, done=env.done,
                    actions=env.actions, gamefile=env.gamefile, trajectory=trajectory,
                    update_skill=not readonly, termination_reason="won" if env.won else "done" if env.done else "method_max_trials")
                if skill.get_active_manual_data() != before:
                    raise RuntimeError("Manual changed within an episode")
        finally:
            env.env.close()
    result.update(manual_after=skill.get_active_manual_data(),
        trajectory_records=skill.skill_size, task_graph_nodes=skill.task_layer.graph.number_of_nodes(),
        task_graph_edges=skill.task_layer.graph.number_of_edges(), embedding_calls=embedding.calls,
        wall_time_ms=int((time.monotonic()-started)*1000))
    write_json(output / "result.json", result)
    return result


def live_role_probe(team, skill, env, config):
    """Exercise official recovery and diagnosis on an isolated partial episode.

    This is transport qualification, not a training example or task-failure label.
    No save/reflection/manual hook is called and its state is never reused.
    """
    from tasks.workflow.format import format_task_prompt_with_skills
    from agentkit.skill.common import AgentMessage
    env.reset()
    skill.init_task_context(config["task_main"], config["task_description"])
    agent = team.get_agent(team.ground_truth_name)
    prompt = format_task_prompt_with_skills(retrieved_skills=[],
        task_description=skill.summarize(), skills=[], few_shots=config["few_shots"])
    action = env.process_action(agent.response(prompt, team.reasoning_config))
    if not action:
        raise RuntimeError("Live recovery probe returned no consumable action")
    skill.add_agent_node(AgentMessage(agent_name=agent.name,
        system_instruction=agent.system_instruction, user_instruction=prompt, message=action),
        upstream_agent_ids=[])
    observation, reward, done = env.step(action)
    skill.move_skill_state(action, observation, reward=reward)
    context = skill.current_task_context
    context.add_extra_field("clean_traj", f"> {action}\n{observation}\n")
    diagnosis = skill._detect_mistakes(context)
    if not diagnosis.strip():
        raise RuntimeError("Live diagnosis probe returned no content")
    return dict(probe_kind="independent_live_role", partial_episode=True,
        counted_as_train=False, recovery_action=action, diagnosis=diagnosis)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    args = p.parse_args()
    job = read_json(args.job)
    try:
        run(job)
        return 0
    except Exception as exc:
        infrastructure = isinstance(exc, (OSError, TimeoutError, ImportError)) or type(exc).__name__ in {"LLMRequestError", "ProviderFailure"}
        write_json(Path(job["output"]) / "rollout_failure.json", dict(
            failure_kind=getattr(exc, "failure_kind", "infrastructure_failure" if infrastructure else "protocol_failure"),
            failure_code=getattr(exc, "failure_code", None), error_type=type(exc).__name__,
            error=sanitize_error_text(exc), traceback=sanitize_error_text(traceback.format_exc()),
            task_id=job.get("task", {}).get("task_id"), phase=job["phase"]))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
