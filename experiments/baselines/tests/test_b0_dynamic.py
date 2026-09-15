import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from experiments.baselines.b0_dynamic import campaign
from experiments.baselines.b0_dynamic.driver import PureDynamicDriver, usage_totals, completed_outcome, file_hash
from experiments.baselines.b4_embodiskill.state import write_json
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.manifest import sha256_json


def config():
    return yaml.safe_load((campaign.REPO/"configs/baselines/b0_dynamic.yaml").read_text())


def test_frozen_config():
    campaign.validate_config(config())


@pytest.mark.parametrize("key,value", [("max_environment_actions",101),("seeds",[42]),
    ("train",{}),("validation",{}),("skill_text","prior"),("bank","ours")])
def test_no_method_drift(key,value):
    cfg=config()
    cfg[key]=value
    with pytest.raises(ValueError):
        campaign.validate_config(cfg)


def test_no_train_or_persistent_knowledge(tmp_path):
    driver=PureDynamicDriver()
    assert driver.train()["llm_calls"]==0
    with pytest.raises(ValueError):
        driver.train(train_manifest=object())
    with pytest.raises(ValueError):
        driver.train(validation_manifest=object())
    frozen=driver.freeze(tmp_path)
    assert driver.freeze(tmp_path).digest==frozen.digest
    (frozen.root/"skill.md").write_text("illegal")
    with pytest.raises(ValueError):
        driver.freeze(tmp_path)


def test_empty_prompt_upstream_and_fallback_golden(tmp_path,monkeypatch):
    from skillopt.envs.alfworld import rollout
    from experiments.baselines.common.text_skill_executor import TextSkillALFWorldExecutor
    prompts=[]
    class Env:
        def reset(self,*a):
            return {"text":["visible observation; use <action>"],"anchor":["Your task is to: place item"]}, [{"extra.gamefile":"pick_and_place/game.tw-pddl"}]
        def step(self,actions):
            assert actions==["<think>missing action tag</think><action>look</action>"]
            return {"text":["done"],"anchor":["Nothing happens"]}, [0], [True], [{"won":False}]
    monkeypatch.setattr(rollout,"chat_target",lambda **kw: (prompts.append(kw) or "no action",{}))
    rows=rollout.run_alfworld_batch(Env(),None,max_steps=100,out_root=str(tmp_path),max_api_workers=1)
    assert rows[0]["hard"]==0
    assert prompts[0]["system"]=="You are an expert agent operating in the ALFRED Embodied Environment."
    assert prompts[0]["user"]=="visible observation; use <action>"
    assert rollout._build_skill_prompt(None)==""
    assert "## Skill Knowledge" in rollout._build_skill_prompt("known skill")
    observed=[]
    monkeypatch.setattr(TextSkillALFWorldExecutor,"run",lambda self,*a,**kw:observed.append((a,kw)))
    TextSkillALFWorldExecutor(seed=42).run_episode(task={},skill_text=None,run_seed=42,
        phase="test",output_dir=tmp_path,rollout_id="r")
    assert observed[0][0][1] is None


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


def test_canonical_b0_automatically_used_for_transfer(tmp_path):
    from experiments.baselines.tests.test_report_campaign import _runs, _FAMILIES
    from experiments.baselines.report_campaign import build_campaign_report
    b0=_runs(tmp_path,"b0_dynamic",[False]*6)
    ours=_runs(tmp_path,"ours",[True]*6)
    report=build_campaign_report({"b0_dynamic":b0,"ours":ours},task_types=_FAMILIES,bootstrap_samples=20)
    assert report["passed"]
    assert "b0_dynamic" in report["ours_pairwise"]["comparisons"]


def test_complete_lane_reports_and_resume_without_execution(tmp_path,monkeypatch):
    from experiments.baselines.common.manifest import ManifestTask
    from experiments.baselines.common.schema import CommonEpisodeRecord
    seed=42
    lane=tmp_path/"seed_42"
    frozen=PureDynamicDriver().freeze(lane)
    tasks=[]
    for index,family in enumerate(("family_a","family_b")):
        task=ManifestTask(index=index,task_id=f"task_{index}",task_type=family,source_split="valid_unseen",
            env_index=index,gamefile_rel=f"game_{index}",gamefile_sha256="a"*64,task_signature=f"{index:064x}")
        tasks.append(task)
        path=lane/"test/episodes"/f"task_{index:04d}"
        write_json(path/"attempts/attempt_0001/outcome.json",dict(wall_time_ms=10))
        record=CommonEpisodeRecord(method="b0_dynamic",phase="test",run_seed=seed,
            task_id=task.task_id,task_type=family,manifest_index=index,gamefile=task.gamefile_rel,
            gamefile_hash=task.gamefile_sha256,official_success=True,contract_consistency=True,
            common_strict_success=True,target_llm_calls=1,target_prompt_tokens=10,target_completion_tokens=2,
            invalid_actions=0,artifact_digest_before=frozen.digest,artifact_digest_after=frozen.digest)
        write_json(path/"record.json",record.to_dict())
        write_json(path/"completion.json",dict(passed=True,record_hash=file_hash(path/"record.json"),
            evidence_digest=digest_directory(path/"attempts")))
    def forbidden(*a,**kw):
        raise AssertionError("Completed lane must not execute an episode")
    monkeypatch.setattr(PureDynamicDriver,"evaluate",forbidden)
    spec=dict(output=str(tmp_path),identity={},config=config(),phase="test",cap=24,campaign_id="test_complete")
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
            fixtures._write_test_run(lane,method="b0_dynamic",seed=seed,successes=[True]*6)
            summary=summarize_rows(load_rows_jsonl(lane/"test/task_rows.jsonl"),task_types=list(ALFWORLD_FORMAL_TASK_TYPES))
            results.append(dict(passed=True,seed=seed,test=summary))
        return results
    monkeypatch.setattr(campaign,"execute",execute)
    root=tmp_path/"formal"
    args=NS(config=str(campaign.REPO/"configs/baselines/b0_dynamic.yaml"),skillopt_root=str(tmp_path),
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
