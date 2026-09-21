"""Append-only Runtime support observations, never an executable registry."""
from __future__ import annotations
import json
from typing import Any
from ..core.refs import content_hash

DDL = """
CREATE TABLE IF NOT EXISTS runtime_support_observations (
 observation_id TEXT PRIMARY KEY, contract_signature TEXT NOT NULL,
 task_id TEXT NOT NULL, trace_id TEXT NOT NULL, draft_id TEXT NOT NULL,
 harness_profile TEXT NOT NULL, atomic_payload_json TEXT NOT NULL,
 tool_proposal_json TEXT NOT NULL, trial_summary_json TEXT NOT NULL,
 content_hash TEXT NOT NULL, created_at REAL NOT NULL,
 UNIQUE(contract_signature, task_id)
);
"""


class RuntimeSupportStore:
    def __new__(cls, database: Any, data_dir=None):
        if data_dir is not None:
            from .execution_observations import ExecutionObservationStore
            return ExecutionObservationStore(database, data_dir)
        return super().__new__(cls)

    def __init__(self, database: Any) -> None:
        self.database = database
        if not database.readonly:
            with database.transaction() as connection:
                connection.execute(DDL)
        info = database.rows('PRAGMA table_info("runtime_support_observations")')
        if not info and database.readonly:
            return  # Historical frozen banks have no R10 staging extension.
        expected = ("observation_id", "contract_signature", "task_id", "trace_id", "draft_id",
                    "harness_profile", "atomic_payload_json", "tool_proposal_json", "trial_summary_json",
                    "content_hash", "created_at")
        if tuple(row["name"] for row in info) != expected or not info[0]["pk"]:
            raise RuntimeError("runtime_support_observation_schema_mismatch")
        unique_keys = {tuple(row["name"] for row in database.rows(f'PRAGMA index_info("{index["name"]}")'))
                       for index in database.rows('PRAGMA index_list("runtime_support_observations")') if index["unique"]}
        if ("contract_signature", "task_id") not in unique_keys:
            raise RuntimeError("runtime_support_observation_unique_key_missing")

    def observations(self, signature: str) -> list[dict]:
        result = []
        for row in self.database.rows(
            "SELECT trial_summary_json, content_hash FROM runtime_support_observations WHERE contract_signature=? ORDER BY created_at, observation_id",
            (signature,),
        ):
            value = json.loads(row["trial_summary_json"])
            if content_hash(value) != row["content_hash"]:
                raise RuntimeError("runtime_support_observation_integrity_mismatch")
            result.append(value)
        return result

    def append(self, observation: dict) -> bool:
        if self.database.readonly:
            raise RuntimeError("Frozen cannot stage Runtime support observations")
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO runtime_support_observations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (observation["observation_id"], observation["contract_signature"],
                 observation["task_id"], observation["trace_id"], observation["draft_id"],
                 observation["harness_profile"], json.dumps(observation["bundle"]["atomic"], sort_keys=True),
                 json.dumps(observation["tool_proposal"], sort_keys=True),
                 json.dumps(observation, sort_keys=True), content_hash(observation), observation["created_at"]),
            )
            return cursor.rowcount == 1
