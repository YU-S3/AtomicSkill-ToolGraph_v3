"""Atomic Trace storage; successful save precedes all EvidenceLedger writes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..core.serialization import atomic_create_json, read_json
from .schema import NodeTraceRecord, TaskRecord, TraceRecord


class TraceStore:
    def __init__(self, data_dir: str | Path, *, readonly: bool = False) -> None:
        self.root = Path(data_dir) / "traces"
        self.readonly = readonly
        if not readonly:
            self.root.mkdir(parents=True, exist_ok=True)

    def save_atomic(self, trace: TraceRecord) -> Path:
        if self.readonly:
            raise RuntimeError("trace store is read-only")
        if trace.schema_version != 3:
            raise ValueError("only v3 traces can be persisted")
        return atomic_create_json(self.root / f"{trace.trace_id}.json", trace)

    save = save_atomic

    def load_payload(self, trace_id: str) -> dict:
        return read_json(self.root / f"{trace_id}.json")

    def load(self, trace_id: str) -> TraceRecord:
        payload = self.load_payload(trace_id)
        task = TaskRecord(**payload.pop("task"))
        payload["node_records"] = [NodeTraceRecord(**item) for item in payload.get("node_records", [])]
        # Governance consumes the stable scalar/list fields. Less common nested
        # record types remain dict payloads after cross-process resume.
        return TraceRecord(task=task, **payload)

    def iter_payloads(self) -> Iterator[dict]:
        for path in sorted(self.root.glob("trace_*.json")):
            yield read_json(path)

    def exists(self, trace_id: str) -> bool:
        return (self.root / f"{trace_id}.json").is_file()
