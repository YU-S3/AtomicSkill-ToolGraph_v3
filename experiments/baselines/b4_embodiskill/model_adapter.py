"""Inject audited transport/embedding without replacing upstream reasoning."""
from contextlib import contextmanager
from functools import wraps

from experiments.baselines.common.model_client import append_event, ProviderFailure


class MethodTransport:
    def __init__(self, client, output, *, readonly):
        self.client, self.output, self.readonly = client, output, readonly
        self.stage = "trajectory_condensation"

    @contextmanager
    def scope(self, stage):
        previous, self.stage = self.stage, stage
        try:
            yield
        finally:
            self.stage = previous

    def __call__(self, messages, temperature=0.1, max_tokens=512, stop_strs=None, num_comps=1, **kwargs):
        from agentkit.llm import LLMRequestError
        from agentkit.skill.embodiskill_skill.prompt import EmbodiSkillPrompts
        if num_comps != 1 or kwargs:
            raise ValueError("Unexpected upstream generation parameters")
        rows = [dict(role=m.role, content=m.content) for m in messages]
        stage = self.stage
        if rows[0]["content"] == EmbodiSkillPrompts.detect_mistakes_system_prompt:
            stage = "failure_diagnosis"
        role = "target" if stage in {"solver", "stuck_recovery"} else "evolution"
        if self.readonly and role == "evolution" and stage != "trajectory_reranking":
            raise RuntimeError(f"Forbidden evaluation learning call: {stage}")
        try:
            return self.client.chat(messages=rows, stage=stage, role=role,
                max_tokens=max_tokens, temperature=temperature, stop=stop_strs)
        except ProviderFailure as exc:
            # Upstream explicitly propagates this class; do not multiply its retries.
            raise LLMRequestError(str(exc)) from exc


def instrument_method(skill, team, transport, output, *, readonly):
    def wrap(owner, name, stage, forbidden=False):
        original = getattr(owner, name)
        @wraps(original)
        def observed(*args, **kwargs):
            if readonly and forbidden:
                raise RuntimeError(f"Read-only evaluation invoked {name}")
            append_event(output / "method_events.jsonl", dict(event="upstream_enter", method=name, stage=stage))
            with transport.scope(stage):
                result = original(*args, **kwargs)
            event = dict(event="upstream_exit", method=name, stage=stage)
            if name == "retrieve_skill":
                event.update(retrieved_successful=[t.task_main for t in result[0]],
                             retrieved_failed=[t.task_main for t in result[1]])
            if name in {"reflect_episode", "revise_manual"}:
                event["result"] = result
            append_event(output / "method_events.jsonl", event)
            return result
        setattr(owner, name, observed)

    for name, stage, forbidden in [
        ("retrieve_skill", "trajectory_reranking", False),
        ("save_task_context", "trajectory_condensation", True),
        ("reflect_episode", "episode_reflection", True),
        ("revise_manual", "manual_revision", True),
    ]:
        wrap(skill, name, stage, forbidden)
    if hasattr(skill, "_attempt_manual_revision_phase"):
        original_revision_phase = skill._attempt_manual_revision_phase
        @wraps(original_revision_phase)
        def revision_phase(*args, **kwargs):
            if readonly:
                raise RuntimeError("Read-only evaluation invoked a manual revision phase")
            with transport.scope("manual_" + kwargs["phase"]):
                return original_revision_phase(*args, **kwargs)
        skill._attempt_manual_revision_phase = revision_phase
    if team is not None:
        wrap(team.get_agent(team.solver_name), "response", "solver")
        wrap(team.get_agent(team.ground_truth_name), "response", "stuck_recovery")


class AuditedEmbedding:
    def __init__(self, delegate, output):
        self.delegate, self.output, self.calls = delegate, output, 0

    def embed_documents(self, texts):
        result = self.delegate.embed_documents(texts)
        self.calls += len(texts)
        append_event(self.output / "embedding_calls.jsonl", dict(kind="documents", calls=len(texts),
            model=self.delegate.model_type))
        return result

    def embed_query(self, query):
        result = self.delegate.embed_query(query)
        self.calls += 1
        append_event(self.output / "embedding_calls.jsonl", dict(kind="query", calls=1,
            model=self.delegate.model_type))
        return result
