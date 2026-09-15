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
