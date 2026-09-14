import json
from types import SimpleNamespace as NS
import pytest

from experiments.baselines.common.model_client import AuditedChatClient, CompletionBudgetExhausted
from experiments.baselines.common.reasoning_budget import response_evidence
from experiments.baselines.b4_embodiskill.campaign import load_config
from experiments.baselines.b4_embodiskill.controller import usage, stage_qualification


def reply(content, completion=40010, reasoning=40000, finish="stop"):
    return NS(usage=NS(prompt_tokens=100,completion_tokens=completion,
        completion_tokens_details=NS(reasoning_tokens=reasoning)),
        choices=[NS(finish_reason=finish,message=NS(content=content,reasoning_content="PRIVATE REASONING"))])


@pytest.mark.parametrize("hint", [512,2048])
def test_independent_cap_content_only_and_real_usage(tmp_path,hint):
    payloads=[]
    def create(**kw):
        payloads.append(kw)
        return reply("open fridge 1")
    path=tmp_path/"provider_calls.jsonl"
    client=AuditedChatClient(output=path,identity={},model=load_config(False)["model"],
        client=NS(chat=NS(completions=NS(create=create))))
    assert client.chat(messages=[],stage="solver",role="target",max_tokens=hint)=="open fridge 1"
    assert payloads[0]["max_tokens"]==65536 and "max_completion_tokens" not in payloads[0]
    row=json.loads(path.read_text())
    assert row["method_output_token_hint"]==hint and row["provider_completion_cap"]==65536
    assert row["visible_completion_tokens"]==10 and row["reasoning_tokens"]==40000
    assert row["content_parse_success"] and row["usage_status"]=="known"
    assert "PRIVATE REASONING" not in path.with_name("model_responses.jsonl").read_text()


@pytest.mark.parametrize("content,parser", [("",None),('{"sections": [',json.loads)])
def test_exhaustion_failfast_preserves_billed_attempt(tmp_path,content,parser):
    calls=[]
    def create(**kw):
        calls.append(kw)
        return reply(content,65536,65536 if not content else 65520,"length")
    path=tmp_path/"provider_calls.jsonl"
    client=AuditedChatClient(output=path,identity={},model=load_config(False)["model"],
        client=NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(CompletionBudgetExhausted) as error:
        client.chat(messages=[],stage="manual_revision",role="evolution",max_tokens=2048,content_parser=parser)
    assert error.value.failure_kind=="protocol_failure" and len(calls)==1
    row=json.loads(path.read_text())
    assert row["failure_code"]=="COMPLETION_BUDGET_EXHAUSTED"
    assert usage([row])["evolution"]["completion_tokens"]==65536
    assert not stage_qualification([row])["passed"]


def test_non_length_empty_is_not_budget_exhaustion():
    row=response_evidence(reply("",65536,65536,"stop"),cap=65536,hint=512)
    assert not row["budget_exhausted"] and not row["content_parse_success"]


def test_missing_reasoning_is_not_assumed_visible():
    row=response_evidence(reply("go",100,None),cap=65536,hint=512)
    assert row["usage_status"]=="partial" and row["visible_completion_tokens"] is None


def test_formal_gate_rejects_missing_or_changed_evidence(tmp_path):
    from experiments.baselines.b4_embodiskill.campaign import verify_smoke_qualification
    from experiments.baselines.b4_embodiskill.state import write_json
    from experiments.baselines.common.reasoning_budget import policy_metadata
    from experiments.baselines.common.manifest import sha256_json
    cfg=load_config(False)
    with pytest.raises(ValueError,match="requires"):
        verify_smoke_qualification(None,{},cfg)
    identity=dict(code_hash="code",upstream_tree="upstream",worker_runtime="python",dependencies="deps",embedding="embedding")
    summary=dict(passed=True)
    receipt=dict(passed=True,identity=identity,summary_sha256=sha256_json(summary),
        formal_config_hash=sha256_json(cfg),**policy_metadata(cfg["model"]))
    write_json(tmp_path/"campaign_summary.json",summary)
    write_json(tmp_path/"smoke_qualification.json",receipt)
    verify_smoke_qualification(tmp_path/"smoke_qualification.json",identity,cfg)
    cfg["model"]["base_url"]="https://different-provider.invalid"
    with pytest.raises(ValueError,match="configuration changed"):
        verify_smoke_qualification(tmp_path/"smoke_qualification.json",identity,cfg)
