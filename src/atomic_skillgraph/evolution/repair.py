"""Immutable, replay-gated local repair proposals."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.serialization import to_primitive
from ..knowledge.database import StateDatabase


ALLOWED_OPERATIONS = frozenset({
    "revise_atomic_contract", "revise_guideline", "revise_implementation_mapping",
    "revise_grounding_constraint", "replace_tool_body", "add_tool_test", "specialize_tool",
    "split_tool", "revise_composite_sequence", "remove_redundant_occurrence", "insert_missing_occurrence",
    "split_atomic", "merge_atomic", "specialize_implementation", "add_replay",
    "alternative", "revise_composite_insight",
})


@dataclass
class RepairProposal:
    proposal_id: str
    target_ref: str
    target_layer: str
    operation: str
    proposed_patch: dict[str, Any]
    source_failure_ids: list[str]
    status: str = "proposed"
    replay_result: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation not in ALLOWED_OPERATIONS:
            raise ValueError(f"unsupported repair operation: {self.operation}")
        if self.status not in {"proposed", "replaying", "admitted", "rejected"}:
            raise ValueError(f"invalid repair status: {self.status}")

    @classmethod
    def create(cls, target_ref: str, target_layer: str, operation: str, patch: dict[str, Any], failures: list[str]) -> "RepairProposal":
        return cls(f"repair_{uuid.uuid4().hex}", target_ref, target_layer, operation, patch, failures)


class RepairStore:
    """Checkpoint-safe queue plus immutable resolved repair history.

    Only ``proposed``/``replaying`` items live in the mutable queue key.  Once
    admitted or rejected they move to history, allowing a frozen snapshot to
    exclude the train-time queue without losing the long-term audit result.
    """

    _METADATA_KEY = "evolution.repair_proposals"
    _HISTORY_KEY = "evolution.repair_history"

    def __init__(self, database: StateDatabase) -> None:
        self.database = database

    def _read(self, key: str) -> list[RepairProposal]:
        row = self.database.execute(
            "SELECT value FROM metadata WHERE key=?", (key,),
        ).fetchone()
        if row is None:
            return []
        import json

        payload = json.loads(str(row["value"]))
        if not isinstance(payload, list):
            raise ValueError("repair proposal metadata must be a list")
        return [RepairProposal(**item) for item in payload]

    def pending(self) -> list[RepairProposal]:
        return self._read(self._METADATA_KEY)

    def history(self) -> list[RepairProposal]:
        return self._read(self._HISTORY_KEY)

    def list(self) -> list[RepairProposal]:
        return [*self.pending(), *self.history()]

    @staticmethod
    def _encoded(items: list[RepairProposal]) -> str:
        import json

        return json.dumps(
            to_primitive(items),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def save(self, proposal: RepairProposal) -> None:
        if self.database.readonly:
            raise RuntimeError("frozen repair store is read-only")
        pending = [
            item for item in self.pending() if item.proposal_id != proposal.proposal_id
        ]
        history = [
            item for item in self.history() if item.proposal_id != proposal.proposal_id
        ]
        if proposal.status in {"proposed", "replaying"}:
            pending.append(proposal)
        else:
            history.append(proposal)
        with self.database.transaction() as connection:
            if pending:
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (self._METADATA_KEY, self._encoded(pending)),
                )
            else:
                connection.execute(
                    "DELETE FROM metadata WHERE key=?", (self._METADATA_KEY,),
                )
            if history:
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (self._HISTORY_KEY, self._encoded(history)),
                )

    def reject_pending_without_concrete_patch(self) -> list[RepairProposal]:
        """Resolve diagnostic-only proposals without activating any edit."""
        resolved: list[RepairProposal] = []
        for proposal in self.pending():
            if proposal.status != "proposed" or not proposal.proposed_patch.get(
                "requires_concrete_patch"
            ):
                continue
            proposal.status = "rejected"
            proposal.replay_result = {
                "passed": False,
                "failure_code": "concrete_patch_unavailable",
            }
            self.save(proposal)
            resolved.append(proposal)
        return resolved

    def assert_queue_empty(self) -> None:
        pending = self.pending()
        if pending:
            raise RuntimeError(
                "unresolved repair proposals remain: "
                + ", ".join(item.proposal_id for item in pending)
            )

    def remove_queue_metadata(self) -> None:
        """Delete only an already-empty mutable queue key before freezing."""
        if self.database.readonly:
            raise RuntimeError("frozen repair store is read-only")
        self.assert_queue_empty()
        with self.database.transaction() as connection:
            connection.execute(
                "DELETE FROM metadata WHERE key=?", (self._METADATA_KEY,),
            )

    def replay_and_admit(
        self, proposal: RepairProposal, replay: Callable[[RepairProposal], dict[str, Any]],
        admit: Callable[[RepairProposal, dict[str, Any]], bool],
    ) -> RepairProposal:
        proposal.status = "replaying"
        self.save(proposal)
        try:
            result = replay(proposal)
            proposal.replay_result = result
            proposal.status = "admitted" if result.get("passed") and admit(proposal, result) else "rejected"
        except AtomicSkillGraphError as exc:
            if exc.layer is FailureLayer.INFRASTRUCTURE:
                raise
            proposal.replay_result = {"passed": False, "error": str(exc)}
            proposal.status = "rejected"
        except (KeyError, TypeError, ValueError) as exc:
            proposal.replay_result = {"passed": False, "error": str(exc)}
            proposal.status = "rejected"
        self.save(proposal)
        return proposal
