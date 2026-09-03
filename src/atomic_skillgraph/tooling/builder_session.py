"""ToolBuilder is the only v3.2 Tool Program author.

The Runtime main agent never receives ``create_tool``.  Runtime automation and
Success Evolution both funnel their already-proposed Atomic contracts through
this sub-agent, and the returned ToolProposal is only a submission: code-side
static validation, compilation, admission, and lifecycle remain authoritative.
"""

from __future__ import annotations

from typing import Any

from ..agents.context_builder import ContextBuilder
from ..agents.structured_submission import (
    TOOL_PROPOSAL_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.contracts import AbstractAtomicSkill
from .proposal import ToolProposal, ToolProvenance, tool_proposal_from_dict


class ToolBuilderSession:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.submissions = StructuredSubmissionClient()
        self.context = ContextBuilder()

    def build(
        self,
        *,
        atomic: AbstractAtomicSkill,
        provenance: ToolProvenance,
        evidence_support: list[dict[str, Any]] | None = None,
        semantic_delta: dict[str, Any] | None = None,
        harness_interface: dict[str, Any] | None = None,
        near_match_interfaces: list[dict[str, Any]] | None = None,
        local_failures: list[dict[str, Any]] | None = None,
        bucket: str = "tool_builder_evolution",
    ) -> ToolProposal:
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket(bucket)
        prompt = self.context.tool_builder(
            atomic=atomic,
            provenance=provenance,
            evidence_support=evidence_support or [],
            semantic_delta=semantic_delta or {},
            harness_interface=harness_interface or {},
            near_match_interfaces=near_match_interfaces or [],
            local_failures=local_failures or [],
        )
        submission = self.submissions.request(
            self.session,
            prompt=prompt,
            tool_name="create_tool",
            description=(
                "Submit either a complete declarative ToolProposal with a bounded "
                "ACTION/IF/FOR_EACH/STOP_WHEN/RETURN IR program, or decision=no_tool."
            ),
            schema=TOOL_PROPOSAL_SCHEMA,
        )
        return tool_proposal_from_dict(submission.value)


__all__ = ["ToolBuilderSession"]
