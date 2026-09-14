import json
import sys
from types import SimpleNamespace as NS, ModuleType
from pathlib import Path

import pytest

from experiments.baselines.b4_embodiskill.controller import usage, reranking_usage
from experiments.baselines.b4_embodiskill.model_adapter import MethodTransport, TARGET_STAGES
from experiments.baselines.b5_gepa import qualification as q
from experiments.baselines.b5_gepa import run_seed_campaign as campaign


def transport_stubs(monkeypatch):
    llm = ModuleType("agentkit.llm")
    llm.LLMRequestError = RuntimeError
    prompt = ModuleType("agentkit.skill.embodiskill_skill.prompt")
    prompt.EmbodiSkillPrompts = NS(detect_mistakes_system_prompt="diagnosis")
    monkeypatch.setitem(sys.modules, "agentkit.llm", llm)
    monkeypatch.setitem(sys.modules, "agentkit.skill.embodiskill_skill.prompt", prompt)


def test_roles_parsers_and_physical_token_conservation(tmp_path, monkeypatch):
    transport_stubs(monkeypatch)
    calls = []
    def chat(**kw):
        calls.append(kw)
        return "10"
    transport = MethodTransport(NS(chat=chat), tmp_path, readonly=False)
    transport.action_parser = lambda x: "ACTION:" + x
    transport.json_parser = json.loads
    stages = ["solver", "stuck_recovery", "trajectory_reranking", "trajectory_condensation",
              "failure_diagnosis", "episode_reflection", "manual_revision",
              "manual_sections_refactor", "manual_execution_notes_revision", "manual_execution_notes_refactor"]
    events = []
    for i, stage in enumerate(stages):
        with transport.scope(stage):
            assert transport([NS(role="user", content="input")]) == "10"
        expected = "target" if stage in TARGET_STAGES else "evolution"
        assert calls[-1]["role"] == expected
        if stage == "trajectory_reranking":
            assert calls[-1]["content_parser"] is None
        events.append(dict(role=expected, stage=stage, logical_call_id=str(i), attempt=1,
            status="succeeded", usage_status="known", prompt_tokens=10+i,
            completion_tokens=20+i, reasoning_tokens=12+i, visible_completion_tokens=8))
    totals = usage(events)
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "visible_completion_tokens"):
        assert sum(totals[role][key+"_known_subtotal"] for role in ("target", "evolution")) == sum(e[key] for e in events)
    rerank = reranking_usage(events)
    assert rerank["trajectory_reranking_calls"] == 1
    assert rerank["trajectory_reranking_completion_tokens"] == 22
    events[2]["completion_tokens"] = None
    assert reranking_usage(events)["trajectory_reranking_completion_tokens"] is None
    transport.readonly = True
    with transport.scope("trajectory_reranking"):
        assert transport([NS(role="user", content="input")]) == "10"
    for stage in set(stages)-TARGET_STAGES:
        with transport.scope(stage), pytest.raises(RuntimeError, match="Forbidden"):
            transport([NS(role="user", content="input")])


def test_reranking_role_change_keeps_empty_response_retries(tmp_path, monkeypatch):
    from experiments.baselines.common.model_client import AuditedChatClient
    from experiments.baselines.b4_embodiskill.campaign import load_config
    monkeypatch.setattr("experiments.baselines.common.model_client.time.sleep", lambda _: None)
    def response(content):
        return NS(usage=NS(prompt_tokens=10, completion_tokens=12, completion_tokens_details=NS(reasoning_tokens=8)),
                  choices=[NS(message=NS(content=content), finish_reason="stop")])
    payloads = []
    for role in ("target", "evolution"):
        responses = iter([response(""), response("10")])
        def create(**kw):
            payloads.append(kw)
            return next(responses)
        client = AuditedChatClient(output=tmp_path/role/"provider_calls.jsonl", identity={},
            model=load_config(False)["model"], client=NS(chat=NS(completions=NS(create=create))))
        assert client.chat(messages=[], stage="trajectory_reranking", role=role,
                           method_output_token_hint=512) == "10"
    assert len(payloads) == 4 and all(p == payloads[0] for p in payloads)


def memory(cap, passed=True):
    return dict(target_workers=cap, passed=passed, memory_load_passed=passed,
                memory_rejected=not passed, workers_released=True,
                loaded_workers=cap if passed else 1, workers=[{}]*cap if passed else [{}],
                exact_gamefile_reset=passed, memory_total=1000, reserve_bytes=150,
                minimum_mem_available=400, samples=[dict(workers=1,available=400)])


def receipt_fixture(tmp_path, cap=36):
    tmp_path.mkdir(exist_ok=True)
    mpath, ppath = tmp_path/"memory.json", tmp_path/"provider.json"
    mpath.write_text(json.dumps(memory(cap)))
    ppath.write_text(json.dumps(dict(passed=True, concurrency=cap, memory_load_report=str(mpath))))
    payload = {k: "identity" for k in q.IDENTITY_KEYS}
    payload.update(controller_commit="commit", formal_config_digest="resolved", source_formal_config_digest="source",
        campaign_provider_max_inflight=cap, seed_lanes=3,
        parallel=dict(seed_lanes=3, episode_workers_per_seed=cap//3, test_workers_per_seed=cap//3,
                      campaign_provider_max_inflight=cap), provider_probe={"report_path": str(ppath)})
    q.write_load_receipt(NS(output_dir=tmp_path), payload)
    return payload


@pytest.mark.parametrize("field", ["missing", "failed", "source", "config", "cap", "workers", "memory", "provider"])
def test_load_receipt_rejects_missing_failure_identity_or_tampering(tmp_path, field):
    payload = receipt_fixture(tmp_path)
    assert q.verify_load_receipt(payload, tmp_path)["selected_global_cap"] == 36
    path = Path(payload["load_probe_receipt_path"])
    if field == "missing":
        path.unlink()
    elif field == "failed":
        r = q.read(path)
        r["passed"] = False
        path.write_text(json.dumps(r))
        payload["load_probe_receipt_hash"] = q._sha256_file(path)
    elif field == "source": payload["controller_code_digest"] = "other"
    elif field == "config": payload["formal_config_digest"] = "other"
    elif field == "cap": payload["campaign_provider_max_inflight"] = 24
    elif field == "workers": payload["parallel"]["test_workers_per_seed"] = 8
    else: (tmp_path/(field+".json")).write_text("{}")
    with pytest.raises(ValueError): q.verify_load_receipt(payload, tmp_path)


@pytest.mark.parametrize("selected", [48,36,24,None])
def test_real_gate_fallback_order_is_memory_only(tmp_path, selected):
    from experiments.baselines.run_seed_campaign import _run_provider_probe_with_fallback
    import yaml
    (tmp_path/"configs/baselines").mkdir(parents=True)
    (tmp_path/"configs/baselines/common.yaml").write_text("{}")
    cfg = tmp_path/"method.yaml"
    cfg.write_text(yaml.safe_dump(dict(parallel={"seed_lanes":3}, provider_probe={})))
    spec = campaign.GEPACampaignSpec("b5_gepa", (42,43,44), cfg,cfg,cfg,cfg,tmp_path/"out",Path(sys.executable),tmp_path)
    payload = dict(model_identity=dict(base_url="url",model="m",api_key_env="key",reasoning_effort="high"),
        provider_probe=dict(concurrency=48,requests=96,max_completion_tokens=65536),
        retry_policy=dict(sdk_max_retries=0,attempts=5,delays=[2],jitter_ratio=.1),
        parallel=dict(seed_lanes=3,campaign_provider_max_inflight=48), campaign_provider_max_inflight=48,
        provider_probe_python=sys.executable,campaign_id="c",campaign_run_id="r",provider_gate_dir=str(tmp_path/"gate"))
    seen=[]
    def run(command, **kwargs):
        cap=int(command[command.index("--concurrency")+1]); seen.append(cap)
        out=Path(command[command.index("--output-dir")+1]);out.mkdir(parents=True)
        mpath=out.parent/(out.name+"_memory.json");mpath.write_text(json.dumps(memory(cap,cap==selected)))
        report=dict(passed=cap==selected,memory_load_passed=cap==selected,memory_load_report=str(mpath))
        if cap==selected:
            calls=out/"calls.jsonl";calls.write_text("calls")
            report.update(probe_kind="campaign_provider_load",campaign_id="c",run_id="r_probe",model="m",
                reasoning_effort="high",concurrency=cap,requests=96,max_completion_tokens=65536,
                campaign_provider_max_inflight=cap,logical_calls_recorded=96,provider_evidence_complete=True,
                provider_calls_path=str(calls),provider_calls_sha256=q._sha256_file(calls),
                completed_logical_calls=96,exhausted_provider_calls=0,permanent_provider_errors=0)
        (out/"provider_load_probe.json").write_text(json.dumps(report))
        return 0 if cap==selected else 1
    if selected is None:
        with pytest.raises(RuntimeError):
            _run_provider_probe_with_fallback(spec,payload,command_runner=q.guarded_probe_runner(run),caps=(48,36,24))
    else:
        _, lock, _ = _run_provider_probe_with_fallback(spec,payload,command_runner=q.guarded_probe_runner(run),caps=(48,36,24))
        assert lock["campaign_provider_max_inflight"]==selected
    expected = [48,36,24][:(48,36,24).index(selected)+1] if selected else [48,36,24]
    assert seen == expected
    assert not (spec.output_dir/"campaign_lock.json").exists()
    assert not list(spec.output_dir.glob("seed_*"))


def test_b3_metadata_does_not_modify_rows_or_input_evidence(tmp_path):
    from experiments.baselines.tests.test_report_campaign import _write_test_run, _FAMILIES
    from experiments.baselines.report_campaign import build_campaign_report
    roots = [_write_test_run(tmp_path/str(s), method="b3_skillopt",seed=s,successes=[True,False]*3) for s in (42,43,44)]
    before={p:p.read_bytes() for r in roots for p in r.rglob("*") if p.is_file()}
    report=build_campaign_report({"b3_skillopt":roots},task_types=_FAMILIES,bootstrap_samples=10)
    assert report["methods"]["b3_skillopt"]["comparability_note"]["result_status"]=="retained_formal_result"
    assert len(report["paired_task_rows"])==18
    assert sum(r["official_success"] for r in report["paired_task_rows"])==9
    assert all(p.read_bytes()==data for p,data in before.items())


def test_formal_missing_smoke_stops_before_any_process_or_lock(tmp_path):
    cfg=tmp_path/"config.yaml";cfg.write_text("{}")
    spec=campaign.GEPACampaignSpec("b5_gepa",(42,43,44),cfg,cfg,cfg,cfg,tmp_path/"out",Path(sys.executable),tmp_path)
    commands=[]
    with pytest.raises(ValueError,match="smoke-receipt"):
        campaign.run_campaign(spec,command_runner=lambda *a,**kw:commands.append(a),
            source_inspector=lambda _:dict(dirty=False),
            lock_builder=lambda *a:dict(formal_config_digest="x"))
    assert not commands and not spec.output_dir.exists()


@pytest.mark.parametrize("field", ["failed","controller_commit","formal_config","evidence"])
def test_smoke_receipt_fails_closed(tmp_path, field):
    payload={k:"same" for k in q.IDENTITY_KEYS}
    payload["formal_config_digest"]="config"
    evidence=[]
    for i in range(3):
        p=tmp_path/str(i);p.write_text("evidence")
        evidence.append(dict(path=str(p),sha256=q._sha256_file(p)))
    receipt=dict(passed=True,identity={k:payload[k] for k in q.IDENTITY_KEYS},
                 formal_config_digest="config",evidence=evidence)
    if field=="failed":receipt["passed"]=False
    elif field=="controller_commit":receipt["identity"][field]="different"
    elif field=="formal_config":receipt["formal_config_digest"]="different"
    else:(tmp_path/"0").write_text("changed")
    path=tmp_path/"smoke.json";path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):q.verify_smoke_receipt(path,payload)


def test_guard_does_not_downgrade_for_provider_failure(tmp_path):
    out=tmp_path/"probe";out.mkdir()
    mp=tmp_path/"memory.json";mp.write_text(json.dumps(memory(48)))
    (out/"provider_load_probe.json").write_text(json.dumps(dict(passed=False,memory_load_report=str(mp))))
    with pytest.raises(RuntimeError,match="not a memory fallback"):
        q.guarded_probe_runner(lambda *a,**kw:1)(["--output-dir",str(out),"--concurrency","48"])
