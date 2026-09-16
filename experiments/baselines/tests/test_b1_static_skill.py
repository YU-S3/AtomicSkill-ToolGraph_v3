from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from experiments.baselines.b1_static_skill import campaign
from experiments.baselines.b1_static_skill.driver import StaticSkillDriver, usage_totals, completed_outcome, file_hash
from experiments.baselines.b4_embodiskill.state import write_json
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.manifest import sha256_json


def config():
    return yaml.safe_load((campaign.REPO/"configs/baselines/b1_static_skill.yaml").read_text())


def test_frozen_config():
    campaign.validate_config(config())


@pytest.mark.parametrize("key,value", [("max_environment_actions",101),("seeds",[42]),
    ("train",{}),("validation",{}),("skill_text","prior"),("bank","ours")])
def test_no_method_drift(key,value):
    cfg=config()
    cfg[key]=value
    with pytest.raises(ValueError):
        campaign.validate_config(cfg)


SOURCE = campaign.REPO / ".external/skillopt"


def test_no_train_and_exact_initial_freeze(tmp_path):
    from experiments.baselines.b1_static_skill.driver import initial_authority, frozen_skill
    driver=StaticSkillDriver()
    assert driver.train()["llm_calls"]==0
    with pytest.raises(ValueError):
        driver.train(train_manifest=object())
    with pytest.raises(ValueError):
        driver.train(validation_manifest=object())
    frozen=driver.freeze(tmp_path,SOURCE)
    authority=initial_authority(SOURCE)
    assert driver.freeze(tmp_path,SOURCE).digest==frozen.digest
    assert (frozen.root/"initial.md").read_bytes()==Path(authority["source_path"]).read_bytes()
    assert frozen.metadata["external_commit"]==authority["external_commit"]
    assert frozen_skill(frozen,authority)==Path(authority["source_path"]).read_bytes().decode("utf-8")
    (frozen.root/"skill.md").write_text("illegal")
    with pytest.raises(ValueError):
        driver.freeze(tmp_path,SOURCE)


@pytest.mark.parametrize("mutation",["source","method","train","validation","metadata","artifact"])
def test_initial_authority_and_frozen_drift_rejected(tmp_path,mutation):
    from experiments.baselines.b1_static_skill.driver import initial_authority, INITIAL_REL
    from experiments.baselines.b4_embodiskill.state import read_json
    source=tmp_path/"source"
    skill=source/INITIAL_REL
    skill.parent.mkdir(parents=True)
    skill.write_bytes((SOURCE/INITIAL_REL).read_bytes())
    driver=StaticSkillDriver()
    lane=tmp_path/"lane"
    frozen=driver.freeze(lane,source)
    if mutation=="source":
        skill.write_text("modified initial")
    elif mutation=="artifact":
        (frozen.root/"initial.md").write_text("modified frozen")
    else:
        path=lane/"frozen/digest.json"
        doc=read_json(path)
        if mutation=="method":
            doc["method_id"]="b0_dynamic"
        elif mutation=="train":
            doc["source_train_manifest_hash"]="trained"
        elif mutation=="validation":
            doc["source_validation_manifest_hash"]="selected"
        else:
            doc["metadata"]["external_commit"]="other"
        write_json(path,doc)
    with pytest.raises(ValueError):
        driver.freeze(lane,source)


def test_static_prompt_upstream_and_fallback_golden(tmp_path,monkeypatch):
    from skillopt.envs.alfworld import rollout
    from experiments.baselines.common.text_skill_executor import TextSkillALFWorldExecutor
    from experiments.baselines.b1_static_skill.driver import INITIAL_REL
    skill=(SOURCE/INITIAL_REL).read_bytes().decode("utf-8")
    prompts=[]
    class Env:
        def reset(self,*a):
            return {"text":["visible observation; use <action>"],"anchor":["Your task is to: place item"]}, [{"extra.gamefile":"pick_and_place/game.tw-pddl"}]
        def step(self,actions):
            assert actions==["<think>missing action tag</think><action>look</action>"]
            return {"text":["done"],"anchor":["Nothing happens"]}, [0], [True], [{"won":False}]
    monkeypatch.setattr(rollout,"chat_target",lambda **kw: (prompts.append(kw) or "no action",{}))
    rows=rollout.run_alfworld_batch(Env(),skill,max_steps=100,out_root=str(tmp_path),max_api_workers=1)
    assert rows[0]["hard"]==0
    assert prompts[0]["system"]=="You are an expert agent operating in the ALFRED Embodied Environment."
    assert prompts[0]["user"]==rollout._build_skill_prompt(skill)+"\nvisible observation; use <action>"
    assert "Training Readout" not in prompts[0]["user"]
    observed=[]
    monkeypatch.setattr(TextSkillALFWorldExecutor,"run",lambda self,*a,**kw:observed.append((a,kw)))
    TextSkillALFWorldExecutor(seed=42).run_episode(task={},skill_text=skill,run_seed=42,
        phase="test",output_dir=tmp_path,rollout_id="r")
    assert observed[0][0][1]==skill


def test_partial_billed_usage_never_zeroed():
    events=[dict(role="target",physical_attempt_usage=[dict(prompt_tokens=10,completion_tokens=20,
        reasoning_tokens=18,visible_completion_tokens=2),dict(prompt_tokens=None,completion_tokens=None)])]
    totals=usage_totals(events)
    assert totals["physical_attempts"]==2 and totals["logical_calls"]==1
    assert totals["completion_tokens"] is None
    assert totals["completion_tokens_known_subtotal"]==20
    with pytest.raises(ValueError):
        usage_totals([dict(role="evolution",physical_attempt_usage=[{}])])


def test_checkpoint_commit_before_replay_and_tamper_rejection(tmp_path):
    task={"task_id":"one"}
    assert completed_outcome(tmp_path,task) is None
    attempt=tmp_path/"attempts/attempt_0001"
    write_json(attempt/"outcome.json",dict(task=task,infrastructure_failure=False))
    write_json(tmp_path/"episode_checkpoint.json",dict(attempt="attempts/attempt_0001",digest=digest_directory(attempt)))
    assert completed_outcome(tmp_path,task)["task"]==task
    write_json(attempt/"provider_calls.jsonl",{"modified":True})
    with pytest.raises(ValueError,match="changed"):
        completed_outcome(tmp_path,task)


def test_qualification_binds_evidence_config_runtime(tmp_path):
    cfg=config()
    identity={"code":"frozen","runtime":"venv"}
    proof=tmp_path/"evidence.json"
    write_json(proof,{"passed":True})
    receipt=tmp_path/"smoke_qualification.json"
    write_json(receipt,dict(passed=True,identity=identity,config_hash=sha256_json(cfg),
                           real_episodes=6,seeds=[42,43,44],checks_passed=True,
                           evidence_hashes={proof.name:file_hash(proof)}))
    campaign.verify_receipt(receipt,identity,cfg)
    with pytest.raises(ValueError):
        campaign.verify_receipt(receipt,{"runtime":"other"},cfg)
    write_json(proof,{"passed":False})
    with pytest.raises(ValueError,match="changed"):
        campaign.verify_receipt(receipt,identity,cfg)


def test_worktrees_share_method_lease(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,"REPO",tmp_path/"isolated")
    monkeypatch.setattr(campaign,"git",lambda *a:"../original/.git")
    assert campaign.lease_root()==tmp_path/"original"


@pytest.mark.parametrize("mutation",["none","missing_prompt","wrong_skill","learning","unknown_usage","training"])
def test_smoke_requires_actual_static_prompt_and_complete_usage(tmp_path,mutation):
    from skillopt.envs.alfworld.rollout import _build_skill_prompt
    from experiments.baselines.b1_static_skill.driver import initial_authority, frozen_skill
    from experiments.baselines.common.post_evaluator import _write_jsonl_atomic
    authority=initial_authority(SOURCE)
    write_json(tmp_path/"campaign_lock.json",{"identity":{"initial_skill":authority}})
    for i,seed in enumerate((42,43,44)):
        lane=tmp_path/f"seed_{seed}"
        frozen=StaticSkillDriver().freeze(lane,SOURCE)
        skill=frozen_skill(frozen,authority)
        cost=StaticSkillDriver().train()
        if mutation=="training":
            cost["llm_calls"]=1
        write_json(lane/"summary.json",{"training_cost":cost})
        events=[]
        for j in range(2):
            call_id=f"call_{i}_{j}"
            episode=lane/"smoke/episodes"/f"task_{j}"
            write_json(episode/"record.json",dict(task_type=f"family_{i}_{j}",run_seed=seed,
                method_metrics={"initial_skill_sha256":authority["initial_skill_sha256"]},
                evolution_llm_calls=1 if mutation=="learning" else 0))
            physical=dict(provider_completion_cap=65536,budget_exhausted=False,
                prompt_tokens=12,completion_tokens=10,reasoning_tokens=8,visible_completion_tokens=2)
            if mutation=="unknown_usage":
                physical["prompt_tokens"]=None
            events.append(dict(call_id=call_id,role="target",physical_attempt_usage=[physical]))
            prefix=_build_skill_prompt("wrong" if mutation=="wrong_skill" else skill)+"\n"
            if mutation!="missing_prompt":
                _write_jsonl_atomic(episode/"attempts/attempt_0001/model_responses.jsonl",
                    [dict(logical_call_id=call_id,messages=[dict(role="user",content=prefix+"observation")])],overwrite=False)
        _write_jsonl_atomic(lane/"smoke/provider_calls.jsonl",events,overwrite=False)
    assert campaign.smoke_checks(tmp_path)==(mutation=="none")


def test_complete_lane_reports_and_resume_without_execution(tmp_path,monkeypatch):
    from experiments.baselines.common.manifest import ManifestTask
    from experiments.baselines.common.schema import CommonEpisodeRecord
    seed=42
    lane=tmp_path/"seed_42"
    frozen=StaticSkillDriver().freeze(lane,SOURCE)
    tasks=[]
    for index,family in enumerate(("family_a","family_b")):
        task=ManifestTask(index=index,task_id=f"task_{index}",task_type=family,source_split="valid_unseen",
            env_index=index,gamefile_rel=f"game_{index}",gamefile_sha256="a"*64,task_signature=f"{index:064x}")
        tasks.append(task)
        path=lane/"test/episodes"/f"task_{index:04d}"
        write_json(path/"attempts/attempt_0001/outcome.json",dict(wall_time_ms=10))
        record=CommonEpisodeRecord(method="b1_static_skill",phase="test",run_seed=seed,
            task_id=task.task_id,task_type=family,manifest_index=index,gamefile=task.gamefile_rel,
            gamefile_hash=task.gamefile_sha256,official_success=True,contract_consistency=True,
            common_strict_success=True,target_llm_calls=1,target_prompt_tokens=10,target_completion_tokens=2,
            invalid_actions=0,artifact_digest_before=frozen.digest,artifact_digest_after=frozen.digest)
        write_json(path/"record.json",record.to_dict())
        write_json(path/"completion.json",dict(passed=True,record_hash=file_hash(path/"record.json"),
            evidence_digest=digest_directory(path/"attempts")))
    def forbidden(*a,**kw):
        raise AssertionError("Completed lane must not execute an episode")
    monkeypatch.setattr(StaticSkillDriver,"evaluate",forbidden)
    from experiments.baselines.b1_static_skill.driver import initial_authority
    spec=dict(output=str(tmp_path),identity={"initial_skill":initial_authority(SOURCE)},
        source=str(SOURCE),config=config(),phase="test",cap=24,campaign_id="test_complete")
    result=campaign.lane_run(spec,seed,tasks)
    assert result["passed"] and result["test"]["tasks"]==2
    assert result["test"]["common_strict_success"]==2
    assert result["training_cost"]["tokens"]==0
    assert campaign.lane_run(spec,seed,tasks)==result
    assert (lane/"test_report.json").is_file()


@pytest.mark.parametrize("report_fails",[False,True])
def test_formal_controller_publishes_only_complete_reports(tmp_path,monkeypatch,report_fails):
    from contextlib import nullcontext
    from experiments.baselines.common.manifest import ManifestTask
    from experiments.baselines.common.formal_validation import ALFWORLD_FORMAL_TASK_TYPES
    from experiments.baselines.common.post_evaluator import load_rows_jsonl,summarize_rows
    from experiments.baselines.tests import test_report_campaign as fixtures
    from experiments.baselines import report_campaign
    monkeypatch.setenv("MODEL_API_KEY","UNIT_ONLY_CREDENTIAL_NOT_PERSISTED")
    monkeypatch.setenv("ALFWORLD_DATA",str(tmp_path))
    monkeypatch.setattr(campaign,"source_identity",lambda source:{"commit":"scripted"})
    monkeypatch.setattr(campaign,"verify_formal_manifest",lambda *a,**kw:None)
    monkeypatch.setattr(campaign,"verify_receipt",lambda *a,**kw:None)
    monkeypatch.setattr(campaign,"_method_campaign_lease",lambda *a,**kw:nullcontext())
    monkeypatch.setattr(campaign,"preflight",lambda *a,**kw:{"selected_cap":24})
    tasks=[ManifestTask(index=i,task_id=f"task_{i}",task_type=family,source_split="valid_unseen",
        env_index=i,gamefile_rel=f"game_{i}.tw-pddl",gamefile_sha256=f"{i+1:064x}",task_signature=f"{i:064x}")
        for i,family in enumerate(ALFWORLD_FORMAL_TASK_TYPES)]
    monkeypatch.setattr(campaign.TaskManifestSet,"load",lambda *a:NS(tasks=tasks,digest="manifest"))
    monkeypatch.setattr(fixtures,"_FAMILIES",list(ALFWORLD_FORMAL_TASK_TYPES))
    def execute(spec,selected):
        assert selected==tasks and spec["phase"]=="test"
        results=[]
        for seed in (42,43,44):
            lane=Path(spec["output"])/f"seed_{seed}"
            fixtures._write_test_run(lane,method="b1_static_skill",seed=seed,successes=[True]*6)
            summary=summarize_rows(load_rows_jsonl(lane/"test/task_rows.jsonl"),task_types=list(ALFWORLD_FORMAL_TASK_TYPES))
            results.append(dict(passed=True,seed=seed,test=summary))
        return results
    monkeypatch.setattr(campaign,"execute",execute)
    root=tmp_path/"formal"
    args=NS(config=str(campaign.REPO/"configs/baselines/b1_static_skill.yaml"),skillopt_root=str(tmp_path),
        output=str(root),mode="formal",qualification=str(tmp_path/"smoke.json"))
    if report_fails:
        def broken(*a,**kw):
            raise ValueError("Report evidence rejected")
        monkeypatch.setattr(report_campaign,"build_campaign_report",broken)
        with pytest.raises(ValueError,match="Report evidence"):
            campaign.run(args)
        assert not (root/"campaign_summary.json").exists()
    else:
        assert campaign.run(args)==0
        assert (root/"REPORT.md").is_file() and (root/"paper_report.json").is_file()
        from experiments.baselines.b4_embodiskill.state import read_json
        assert read_json(root/"paper_report.json")["methods"]["b1_static_skill"]["transfer_vs_b0"]["status"]=="unavailable_without_b0_artifacts"
