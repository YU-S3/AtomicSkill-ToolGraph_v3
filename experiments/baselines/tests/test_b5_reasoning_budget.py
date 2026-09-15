import json
from types import SimpleNamespace as NS
import pytest

pytest.importorskip("skillopt")
from experiments.baselines.b3_skillopt.provider_observer import ProviderCallObserver, ProviderCallExhausted
from experiments.baselines.tests.test_reasoning_budget import reply


@pytest.mark.parametrize("method,exhausted", [("b5_gepa",True),("b5_gepa",False),("b0_dynamic",True),("b0_dynamic",False),("b3_skillopt",False)])
def test_actual_skillopt_backend_budget_and_frozen_b3(tmp_path,monkeypatch,method,exhausted):
    import skillopt.model.openai_compatible_backend as backend
    payloads=[]
    def create(**kw):
        payloads.append(kw)
        return reply("" if exhausted else "<action>look</action>",
            65536 if exhausted else 40010,65536 if exhausted else 40000,"length" if exhausted else "stop")
    monkeypatch.setattr(backend,"_get_client",lambda role:NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(backend.TARGET_CONFIG,"deployment","deepseek-v4-flash")
    monkeypatch.setattr(backend.time,"sleep",lambda _:None)
    observer=ProviderCallObserver(output_path=tmp_path/"calls.jsonl",method=method,phase="smoke",
        model="deepseek-v4-flash",reasoning_effort="high",run_id="fixture",run_seed=42,application_retry_limit=5)
    observer.install()
    try:
        if exhausted:
            with pytest.raises(ProviderCallExhausted) as error:
                backend._chat_messages_impl([],512,5,"rollout",role="target")
            assert error.value.failure_code=="COMPLETION_BUDGET_EXHAUSTED"
            assert not error.value.infrastructure_failure
        else:
            result,_=backend._chat_messages_impl([],512,5,"rollout",role="target")
            assert result=="<action>look</action>"
        assert len(payloads)==1
        if method in {"b5_gepa", "b0_dynamic"}:
            assert payloads[0]["max_tokens"]==65536 and "max_completion_tokens" not in payloads[0]
            row=observer.events()[0]
            assert row["method_output_token_hint"]==512
            assert row["reasoning_tokens"]==(65536 if exhausted else 40000)
            assert row["budget_exhaustion_count"]==int(exhausted)
            if method == "b0_dynamic":
                assert "PRIVATE REASONING" not in (tmp_path/"model_responses.jsonl").read_text()
        else:
            assert payloads[0]["max_tokens"]==512 and "max_completion_tokens" not in payloads[0]
    finally:
        observer.uninstall()
