from pathlib import Path
from types import SimpleNamespace as NS
import copy
import json
import sys

import pytest

from experiments.baselines.repeat_test import authority, launch, run, text_worker
from experiments.baselines.b4_embodiskill.state import write_json, read_json
from experiments.baselines.common.freeze import freeze_files


def test_ten_independent_commands_keep_seed42_and_distinct_roots(tmp_path):
    jobs=launch.round_commands(tmp_path,smoke=False,qualification=tmp_path/"q.json")
    assert len(jobs)==10
    assert len({str(j[2]) for j in jobs})==10
    assert {(j[0],j[1]) for j in jobs}=={(m,r) for m in authority.METHODS for r in (1,2)}
    assert all("--smoke" not in j[3] and "--seed" not in j[3] and "--resume" not in j[3] for j in jobs)


def test_processes_detached_no_sibling_wait(tmp_path,monkeypatch):
    seen=[]
    monkeypatch.setattr(launch.subprocess,"Popen",lambda cmd,**kw:seen.append((cmd,kw)) or NS(pid=123))
    result=launch.launch_process(["python","fake"],tmp_path/"log")
    assert result.pid==123 and seen[0][1]["start_new_session"] is True
    assert seen[0][1]["stdin"]==launch.subprocess.DEVNULL


def test_failed_attempt_excluded_but_successful_retry_counted():
    failed=dict(prompt_tokens=None,completion_tokens=None,reasoning_tokens=None)
    success=dict(prompt_tokens=10,completion_tokens=4,reasoning_tokens=3)
    event=dict(status="succeeded",role="target",logical_call_id="a",
        physical_attempt_usage=[failed,success])
    usage=run.successful_usage([event])
    assert usage["total_tokens"]==14 and usage["logical_calls"]==1
    assert usage["reasoning_tokens"]==3
    with pytest.raises(ValueError,match="Duplicate"):
        run.successful_usage([event,event])
    event["physical_attempt_usage"][-1]["completion_tokens"]=None
    with pytest.raises(ValueError,match="unknown"):
        run.successful_usage([event])


def test_b4_physical_events_success_only():
    base=dict(logical_call_id="a",role="target",prompt_tokens=10,completion_tokens=4,reasoning_tokens=2)
    usage=run.successful_usage([dict(base,status="failed"),dict(base,status="succeeded")])
    assert usage["total_tokens"]==14


@pytest.mark.parametrize("method",["b0_dynamic","b1_static_skill","b3_skillopt","b5_gepa"])
def test_exact_copy_and_skill_without_training(tmp_path,method):
    name=authority.SKILL_FILES[method] or "empty_descriptor.json"
    source=tmp_path/name
    source.write_text('{"persistent_skill":null}' if method=="b0_dynamic" else "original fixed skill")
    frozen=freeze_files(method_id=method,source_files={name:source},destination=tmp_path/"original",
        source_train_manifest_hash="",source_validation_manifest_hash=None)
    a=dict(method=method,original_frozen=str(tmp_path/"original"),frozen_digest=frozen.digest)
    lane=tmp_path/"repeat"
    lane.mkdir()
    copied=run.copy_frozen(a,lane)
    assert copied.digest==frozen.digest
    assert text_worker.skill_from_frozen(copied,a)==(None if method=="b0_dynamic" else "original fixed skill")
    (copied.root/name).write_text("tampered")
    with pytest.raises(ValueError):
        run.copy_frozen(a,lane)
    assert (frozen.root/name).read_bytes()==source.read_bytes()


def test_round_locks_are_independent_and_duplicate_is_rejected(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"
    a.mkdir();b.mkdir()
    with run.round_lease(a):
        with run.round_lease(b):
            with pytest.raises(RuntimeError,match="already running"):
                with run.round_lease(a):
                    pass
    assert not list(a.glob("failure*.json"))


def test_failure_is_recorded_without_sibling_stop(tmp_path):
    a,b=tmp_path/"a",tmp_path/"b"
    a.mkdir();b.mkdir()
    with run.round_lease(b):
        with pytest.raises(ValueError):
            with run.round_lease(a):
                raise ValueError("scripted task failure")
        assert not list(b.glob("failure*.json"))
    assert len(list(a.glob("failure*.json")))==1


def test_original_reports_never_relabeled_new_training_seeds():
    cfg=authority.yaml.safe_load(authority.CONFIG.read_text())
    assert cfg["seed"]==42 and cfg["repeats"]==[1,2]
    assert set(cfg["sources"])==set(authority.METHODS)
    assert len({v["digest"] for v in cfg["sources"].values()})==5


def test_decoding_env_override_rejected(monkeypatch):
    monkeypatch.setenv("TARGET_OPENAI_COMPATIBLE_TEMPERATURE","0.8")
    with pytest.raises(ValueError,match="temperature|TEMPERATURE"):
        authority.guard_environment()


@pytest.mark.parametrize("method,cap",[
    ("b0_dynamic",65536),("b1_static_skill",65536),("b3_skillopt",16384),("b5_gepa",65536)])
def test_wire_cap_preserves_historical_method_policy(tmp_path,monkeypatch,method,cap):
    from experiments.baselines.tests.test_reasoning_budget import reply
    from experiments.baselines.b3_skillopt.provider_observer import ProviderCallObserver
    import skillopt.model.openai_compatible_backend as backend
    payloads=[]
    monkeypatch.setattr(backend,"_get_client",lambda role:NS(chat=NS(completions=NS(
        create=lambda **kw:payloads.append(kw) or reply("<action>look</action>",20,10,"stop")))))
    monkeypatch.setattr(backend.TARGET_CONFIG,"deployment","deepseek-v4-flash")
    monkeypatch.setattr(backend.TARGET_CONFIG,"max_tokens",32768)
    monkeypatch.setattr(backend.TARGET_CONFIG,"temperature",None)
    observer=ProviderCallObserver(output_path=tmp_path/"calls.jsonl",method=method,phase="test",
        model="deepseek-v4-flash",reasoning_effort="high",run_id="test",run_seed=42)
    observer.install()
    try:
        backend._chat_messages_impl([],16384,5,"rollout",role="target")
        assert payloads[0]["max_tokens"]==cap
        assert payloads[0]["reasoning_effort"]=="high"
        assert "temperature" not in payloads[0]
    finally:
        observer.uninstall()


def test_qualification_rejects_other_code_and_tampered_evidence(tmp_path):
    a={"method":"b3_skillopt"}
    proof=tmp_path/"proof.json";write_json(proof,{"passed":True})
    q=tmp_path/"q.json"
    code={"commit":"one"}
    write_json(q,dict(passed=True,independent_smoke_rounds=10,
        identity=dict(code=code,sources={"b3_skillopt":authority.sha256_json(a)}),
        evidence_hashes={"proof.json":authority.file_hash(proof)}))
    run.verify_qualification(q,"b3_skillopt",a,code)
    with pytest.raises(ValueError):
        run.verify_qualification(q,"b3_skillopt",a,{"commit":"other"})
    write_json(proof,{"passed":False})
    with pytest.raises(ValueError):
        run.verify_qualification(q,"b3_skillopt",a,code)


@pytest.mark.parametrize("fails",[False,True])
def test_round_completes_and_resume_does_not_repeat_api(tmp_path,monkeypatch,fails):
    source=tmp_path/"initial.md";source.write_text("fixed")
    frozen=freeze_files(method_id="b1_static_skill",source_files={"initial.md":source},
        destination=tmp_path/"original",source_train_manifest_hash="",source_validation_manifest_hash=None)
    a=dict(method="b1_static_skill",original_frozen=str(tmp_path/"original"),frozen_digest=frozen.digest,
        original_test=str(tmp_path/"original_test"),original_result={"tasks":134,"official":113,"strict":111},
        model={"api_key_env":"MODEL_API_KEY"})
    monkeypatch.setenv("MODEL_API_KEY","UNIT_TEST_CREDENTIAL")
    monkeypatch.setenv("ALFWORLD_DATA",str(tmp_path))
    monkeypatch.setattr(run,"audit_source",lambda m:copy.deepcopy(a))
    monkeypatch.setattr(run,"code_identity",lambda:{"commit":"fixture"})
    monkeypatch.setattr(run,"guard_environment",lambda:None)
    monkeypatch.setattr(run,"assert_source_unchanged",lambda a:None)
    monkeypatch.setattr(run,"verify_qualification",lambda *a:None)
    monkeypatch.setattr(run,"tasks_for_mode",lambda m:(NS(digest="testmanifest"),[NS(task_id="task")]))
    calls=[]
    def evaluate(spec,lane,tasks,frozen):
        calls.append(1)
        if fails:
            raise RuntimeError("scripted infra failure")
        write_json(lane/"task_evidence.json",{"real_fixture":True})
        return dict(tasks=1,infrastructure_failed_episodes=0,official_success=0,common_strict_success=0),[
            dict(status="succeeded",role="target",logical_call_id="one",
                 physical_attempt_usage=[dict(prompt_tokens=10,completion_tokens=2,reasoning_tokens=1,visible_completion_tokens=1)])]
    monkeypatch.setattr(run,"text_evaluate",evaluate)
    args=NS(method="b1_static_skill",repeat=1,output=str(tmp_path/"repeat"),smoke=False,
            resume=False,qualification=str(tmp_path/"q"))
    if fails:
        with pytest.raises(RuntimeError):
            run.execute(args)
        assert not (tmp_path/"repeat/completion.json").exists()
        assert list((tmp_path/"repeat").glob("failure_*.json"))
    else:
        assert run.execute(args)==0
        args.resume=True
        assert run.execute(args)==0 and len(calls)==1
        (tmp_path/"repeat/seed_42/task_evidence.json").write_text("changed")
        with pytest.raises(ValueError,match="evidence changed"):
            run.execute(args)
