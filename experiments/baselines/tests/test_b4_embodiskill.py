import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.baselines.b4_embodiskill.campaign import load_config, validate_config
from experiments.baselines.b4_embodiskill.controller import usage
from experiments.baselines.b4_embodiskill.manifest_adapter import train_chunks, visible_task
from experiments.baselines.b4_embodiskill.model_adapter import instrument_method
from experiments.baselines.b4_embodiskill.state import (copy_state, read_json, write_json,
    publish_checkpoint, load_checkpoint)
from experiments.baselines.common.model_client import AuditedChatClient, ProviderFailure
from experiments.baselines.common.manifest import TaskManifestSet

REPO = Path(__file__).resolve().parents[3]


def test_matched_chunks_use_120_unique_once_and_seed_changes_order():
    chunks = train_chunks(list(range(120)), seed=42, epochs=4, chunk_size=30)
    assert list(map(len, chunks)) == [30]*4
    assert sorted(x for c in chunks for x in c) == list(range(120))
    assert chunks != train_chunks(list(range(120)), seed=43, epochs=4, chunk_size=30)
    with pytest.raises(ValueError):
        train_chunks(list(range(120)), seed=42, epochs=4, chunk_size=40)


def test_visible_task_has_no_reference_or_hidden_state():
    task = TaskManifestSet.load(REPO / "data/baseline_manifests/train_6_smoke.json").tasks[0]
    visible = "You are in a room.\nYour task is to: put an apple in a bowl."
    config = visible_task(task, visible)
    assert config["task_description"] == visible
    assert config["task_type"] == "examine"
    assert set(config) == {"task_main", "task_description", "task_type", "env_name"}


def test_failed_operation_cannot_publish_partial_state(tmp_path):
    state = tmp_path / "attempt/state"
    copy_state(None, state)
    write_json(state / "manual.json", {"version": 0})
    assert load_checkpoint(tmp_path, "train_0") is None
    with pytest.raises(FileNotFoundError):
        publish_checkpoint(tmp_path, "train_0", state.parent, state)
    write_json(state.parent / "result.json", {"done": True})
    publish_checkpoint(tmp_path, "train_0", state.parent, state)
    write_json(state / "manual.json", {"version": 1})
    with pytest.raises(RuntimeError, match="changed"):
        load_checkpoint(tmp_path, "train_0")


def test_eval_hooks_fail_before_upstream_mutation(tmp_path):
    calls = []
    skill = NS(**{name: lambda *a, **k: calls.append("mutation") for name in
        ("retrieve_skill", "save_task_context", "reflect_episode", "revise_manual")})
    instrument_method(skill, None, None, tmp_path, readonly=True)
    for name in ("save_task_context", "reflect_episode", "revise_manual"):
        with pytest.raises(RuntimeError, match="Read-only"):
            getattr(skill, name)()
    assert not calls


def response(text, prompt=12, completion=15):
    return NS(usage=NS(prompt_tokens=prompt, completion_tokens=completion,
        completion_tokens_details=NS(reasoning_tokens=8)), choices=[NS(message=NS(content=text), finish_reason="stop")])


def test_billed_empty_evolution_attempt_is_not_lost(tmp_path, monkeypatch):
    replies = iter([response(""), response("{}")] )
    client = NS(chat=NS(completions=NS(create=lambda **kw: next(replies))))
    monkeypatch.setattr("experiments.baselines.common.model_client.time.sleep", lambda _: None)
    path = tmp_path / "calls.jsonl"
    transport = AuditedChatClient(output=path, identity={"run_id":"test", "run_seed":42},
        model=load_config(False)["model"], client=client)
    assert transport.chat(messages=[], stage="reflection", role="evolution", max_tokens=512) == "{}"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    cost = usage(rows)
    assert cost["evolution"]["prompt_tokens"] == 24
    assert cost["evolution"]["completion_tokens"] == 30
    assert cost["physical_attempts"] == 2 and cost["provider_retries"] == 1


def test_empty_target_response_is_not_transport_retry(tmp_path):
    replies = iter([response("")])
    client = NS(chat=NS(completions=NS(create=lambda **kw: next(replies))))
    transport = AuditedChatClient(output=tmp_path/"calls.jsonl", identity={}, model=load_config(False)["model"], client=client)
    assert transport.chat(messages=[], stage="solver", role="target", max_tokens=512) == ""


def test_missing_usage_is_null_not_zero():
    rows = [dict(role="target",logical_call_id="1",attempt=1,status="failed",usage_status="unavailable",
                 prompt_tokens=None,completion_tokens=None,reasoning_tokens=None)]
    cost = usage(rows)
    assert cost["target"]["prompt_tokens"] is None
    assert not cost["usage_complete"]


def test_frozen_formal_config_and_fixed_current_machine_parallelism():
    validate_config(load_config(False))
    for w in (12,16):
        cfg = load_config(False)
        cfg["parallel"].update(episode_workers_per_seed=w,test_workers_per_seed=w,campaign_provider_max_inflight=3*w)
        with pytest.raises(ValueError, match="3 x 8"):
            validate_config(cfg)
    cfg["parallel"]["seed_lanes"] = 1
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = load_config(False)
    cfg["embodiskill"]["manual_reflection_max_tokens"] = 1024
    with pytest.raises(ValueError, match="frozen"):
        validate_config(cfg)


def test_independent_evaluations_return_input_order(tmp_path, monkeypatch):
    import time
    from experiments.baselines.b4_embodiskill.controller import SeedController
    controller = SeedController({"output":str(tmp_path), "config":load_config(True)},42,[])
    def operation(name, phase, **kw):
        time.sleep(.02 if kw["task"] == "first" else 0)
        return kw["task"]
    monkeypatch.setattr(controller, "operation", operation)
    assert controller.evaluate("validation", ["first","second"],tmp_path,0) == ["first","second"]


def test_worker_protocol_failure_keeps_classification_and_cost(tmp_path, monkeypatch):
    from experiments.baselines.b4_embodiskill.controller import SeedController, WorkerFailure
    def failed(command, **kw):
        job = read_json(command[-1])
        write_json(Path(job["output"])/"rollout_failure.json", {"failure_kind":"protocol_failure"})
        return NS(returncode=1)
    monkeypatch.setattr("experiments.baselines.b4_embodiskill.controller.subprocess.run",failed)
    controller = SeedController({"output":str(tmp_path), "config":load_config(True),
        "repo":str(REPO),"campaign_id":"test","worker_python":"python"},42,[])
    with pytest.raises(WorkerFailure) as exc:
        controller.operation("train_00", "train")
    assert exc.value.failure_kind == "protocol_failure"
    failure = read_json(controller.root/"failure.json")
    assert failure["failure_kind"] == "protocol_failure"
    assert failure["usage"]["physical_attempts"] == 0


def test_training_chunk_revision_selection_and_freeze(tmp_path, monkeypatch):
    from experiments.baselines.b4_embodiskill.controller import SeedController
    cfg = load_config(True)
    cfg["train"] = dict(train_size=4,num_epochs=2,train_chunk_size=2)
    manifests = [TaskManifestSet.load(REPO/"data/baseline_manifests"/(name+"_6_smoke.json"))
                 for name in ("train","validation","test")]
    spec = dict(output=str(tmp_path),config=cfg,campaign_id="unit",
        source_receipt={"repo":"scripted-test","declared_commit":"fixture"},git_state={"commit":"fixture"},
        identity={"dependencies":{"distributions":{"alfworld":"fixture"}}})
    controller = SeedController(spec,42,manifests)
    operations = []
    def operation(name, phase, *, source=None, task=None, epoch=0, rate=0):
        operations.append((phase,epoch,task.task_id if task else None))
        attempt = tmp_path/"attempts"/name
        state = attempt/"state"
        copy_state(source,state)
        manual_path = state/"skill/current.json"
        before = read_json(manual_path) if manual_path.exists() else dict(version=1,sections=[],execution_notes=[])
        after = {**before, "version":before["version"]+int(phase=="revision")}
        write_json(manual_path,after)
        write_json(state/f"skill/versions/skill_v{after['version']}.json",after)
        result = dict(manual_before=before,manual_after=after,embedding_calls=0,
                      official_success=True,update_skill=phase=="train",actions=[])
        if task:
            result["task"] = task.to_dict()
        write_json(attempt/"result.json",result)
        return publish_checkpoint(controller.root,name,attempt,state)
    monkeypatch.setattr(controller,"operation",operation)
    monkeypatch.setattr(controller,"report",lambda phase,receipts:{"tasks":len(receipts)})
    monkeypatch.setattr("experiments.baselines.b4_embodiskill.controller.smoke_checks",lambda *args:{"simulated":True})
    result = controller.run()
    assert [p for p,e,t in operations[:6]] == ["role_probe","train","train","revision","validation","validation"]
    assert len({t for p,e,t in operations if p=="train"}) == 4
    assert result["best_epoch"] == 0  # Equal Val score must keep the earlier state.
    assert result["frozen_unchanged"]
    assert result["train_episodes"] == 4 and result["validation_episodes"] == 4
    frozen_manual = read_json(controller.root/"frozen/artifact/state/skill/current.json")
    assert frozen_manual["version"] == 2


def test_pinned_core_train_revision_reload_readonly_with_scripted_provider(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    source = REPO/".external/embodiskill"
    monkeypatch.chdir(source)
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    from agentkit.skill import EmbodiSkill
    from agentkit.reasoning import ReasoningIO
    from tasks.workflow.team.team import TeamSolver
    from experiments.baselines.b4_embodiskill.model_adapter import MethodTransport
    from experiments.baselines.common.artifact_digest import digest_directory
    stages = []
    requests = []
    class ScriptedClient:
        def chat(self, **kw):
            stages.append(kw["stage"])
            requests.append({k:v for k,v in kw.items() if k != "content_parser"})
            if kw["stage"] == "episode_reflection":
                return json.dumps(dict(is_empty=False, reflection_type="S_NEW",target_section="Search",
                    content="Inspect visible surfaces before opening containers.",target_skill=""))
            if kw["stage"] == "manual_sections":
                return json.dumps(dict(sections=[dict(title="Search",items=["Inspect visible surfaces first."])]))
            if kw["stage"] == "trajectory_reranking":
                return "10"
            return "look" if kw["role"] == "target" else "Inspect visible surfaces."
    class Embedding:
        def embed_query(self, text):
            return [1.0]+[0.0]*383
        def embed_documents(self,texts):
            return [self.embed_query(text) for text in texts]
    class VisibleEnv:
        max_trials=30
        def reset(self):
            pass
        def process_action(self, action):
            return action
        def step(self, action):
            assert action == "look"
            return "You completed the task.",1,True
        def feedback(self):
            return 1,True,"Success"
    def build(state,readonly,transport_cls=MethodTransport):
        output=state.parent/"events"
        transport=transport_cls(ScriptedClient(),output,readonly=readonly)
        skill=EmbodiSkill(namespace="EmbodiSkill",global_config={**load_config(False)["embodiskill"],
            "working_dir":str(state.parent),"persist_dir":str(state),"task":"alfworld","current_epoch_id":0},
            llm_model=transport,embedding_func=Embedding())
        team=TeamSolver()
        team.build_system(ReasoningIO(transport),skill,VisibleEnv(),load_config(False)["embodiskill"])
        instrument_method(skill,team,transport,output,readonly=readonly)
        return skill,team
    state=tmp_path/"train/state"
    skill,team=build(state,False)
    task=dict(task_main="put-an-apple-in-bowl",task_description="Your task is to: put an apple in a bowl.")
    version=skill.get_active_manual_data()["version"]
    _,won,trajectory=team.schedule(task,update_skill=True)
    assert won and skill.skill_size == 1
    assert skill.get_active_manual_data()["version"] == version
    assert trajectory["episode_reflection"]["reflection_type"] == "S_NEW"
    skill.revise_manual(epoch_id=0,success_rate=1)
    assert skill.get_active_manual_data()["version"] == version+1
    assert (state/"task/task_layer_graph.pkl").is_file()
    before=copy_state(state,tmp_path/"eval/state")
    readonly_skill,readonly_team=build(tmp_path/"eval/state",True)
    stages.clear()
    requests.clear()
    readonly_team.schedule({**task,"task_main":"put-another-apple-in-bowl"},update_skill=False)
    assert set(stages) <= {"solver","trajectory_reranking","stuck_recovery"}
    assert "trajectory_reranking" in stages
    assert all(r["role"] == "target" for r in requests)
    fixed_requests = list(requests)
    assert readonly_skill.skill_size == 1
    assert readonly_skill.get_active_manual_data()["version"] == version+1
    assert digest_directory(state) == before
    class LegacyAuditTransport(MethodTransport):
        # Exact pre-v2.3 readonly mapping; scripted responses and upstream core are unchanged.
        def __call__(self,messages,temperature=.1,max_tokens=512,stop_strs=None,num_comps=1,**kwargs):
            role = "target" if self.stage in {"solver","stuck_recovery"} else "evolution"
            return self.client.chat(messages=[dict(role=m.role,content=m.content) for m in messages],
                stage=self.stage, role=role, method_output_token_hint=max_tokens,
                temperature=temperature, stop=stop_strs,
                content_parser=self.action_parser if role=="target" else None)
    copy_state(state,tmp_path/"legacy/state")
    legacy_skill,legacy_team=build(tmp_path/"legacy/state",True,LegacyAuditTransport)
    requests.clear()
    legacy_team.schedule({**task,"task_main":"put-another-apple-in-bowl"},update_skill=False)
    assert [{k:v for k,v in r.items() if k != "role"} for r in requests] == [
        {k:v for k,v in r.items() if k != "role"} for r in fixed_requests]
    assert legacy_skill.get_active_manual_data() == readonly_skill.get_active_manual_data()
    assert legacy_skill.skill_size == readonly_skill.skill_size
    assert digest_directory(state) == before
    # Typed exhaustion must escape the real TeamSolver without its three empty-action retries.
    from agentkit.llm import LLMRequestError
    from experiments.baselines.common.model_client import CompletionBudgetExhausted
    calls = []
    def exhausted(**kw):
        calls.append(kw)
        raise CompletionBudgetExhausted("COMPLETION_BUDGET_EXHAUSTED")
    failed_skill,failed_team=build(tmp_path/"budget_failure/state",False)
    failed_skill.llm_model.client.chat=exhausted
    with pytest.raises(LLMRequestError) as failure:
        failed_team.schedule(task,update_skill=True)
    assert failure.value.failure_kind == "protocol_failure"
    assert failure.value.failure_code == "COMPLETION_BUDGET_EXHAUSTED"
    assert len(calls) == 1 and failed_skill.skill_size == 0


@pytest.mark.parametrize("done,won", [(True, False), (True, True), (False, False)])
def test_real_adapter_labels_reflection_by_won_not_done(tmp_path, monkeypatch, done, won):
    pytest.importorskip("sentence_transformers")
    source = REPO / ".external/embodiskill"
    monkeypatch.chdir(source)
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    import alfworld.agents.environment as environment
    from experiments.baselines.b4_embodiskill.env_adapter import make_environment
    from experiments.baselines.common.manifest import ManifestTask
    import hashlib
    original = TaskManifestSet.load(REPO / "data/baseline_manifests/train_6_smoke.json").tasks[0]
    path = tmp_path / original.gamefile_rel
    path.parent.mkdir(parents=True)
    path.write_text("fixture")
    task = ManifestTask.from_dict({**original.to_dict(), "gamefile_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    class FakeTW:
        def __init__(self, cfg, **kw):
            self.cfg = cfg
        def init_env(self, batch_size):
            assert self.game_files == [str(path)] and batch_size == 1
            return self
        def seed(self, seed):
            pass
        def reset(self):
            return ["Your task is to: examine an apple."], {"extra.gamefile":[str(path)]}
        def step(self, actions):
            assert actions == ["look"]
            return ["Visible feedback"], [int(won)], [done], {"won":[won]}
    monkeypatch.setattr(environment, "get_environment", lambda name: FakeTW)
    env = make_environment(task, tmp_path, source, max_trials=30,seed=42,output=tmp_path/"audit")
    env.reset()
    env.step("look")
    assert env.feedback()[:2] == (float(won),won)
    assert env.done == done
    row = json.loads((tmp_path/"audit/environment_actions.jsonl").read_text())
    assert row["done"] == done and row["won"] == won
