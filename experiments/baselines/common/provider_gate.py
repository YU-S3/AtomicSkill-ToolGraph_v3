"""Campaign-wide, cross-process provider concurrency authority.

Every real HTTP attempt acquires exactly one file-lock slot immediately before
the SDK boundary and releases it immediately afterwards.  Retry backoff is
therefore deliberately outside this module and never occupies a slot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


_CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_GATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProviderLease:
    """Evidence returned for one acquired real-provider slot."""

    slot_id: int
    queue_wait_ms: int
    acquired_at_unix: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CampaignProviderGate:
    """Bound real HTTP in-flight calls across every process in one campaign."""

    def __init__(
        self,
        *,
        gate_dir: Path,
        campaign_id: str,
        max_inflight: int,
    ) -> None:
        self.gate_dir = Path(gate_dir).expanduser().resolve()
        self.campaign_id = str(campaign_id).strip()
        self.max_inflight = int(max_inflight)
        if not _CAMPAIGN_ID_RE.fullmatch(self.campaign_id):
            raise ValueError("campaign_id is empty or contains unsafe characters")
        if not 1 <= self.max_inflight <= 256:
            raise ValueError("max_inflight must be within 1..256")
        self.gate_dir.mkdir(parents=True, exist_ok=True)
        self._bind_identity()
        self._slot_paths = tuple(
            self.gate_dir / f"slot_{slot_id:03d}.lock"
            for slot_id in range(self.max_inflight)
        )
        for path in self._slot_paths:
            path.touch(exist_ok=True)

    @contextmanager
    def acquire(
        self,
        *,
        run_id: str,
        seed: int,
        role: str,
        stage: str,
        logical_call_id: str,
    ) -> Iterator[ProviderLease]:
        """Wait for a campaign slot and hold it only for the caller's request."""

        metadata = {
            "run_id": str(run_id).strip(),
            "seed": int(seed),
            "role": str(role).strip(),
            "stage": str(stage).strip(),
            "logical_call_id": str(logical_call_id).strip(),
        }
        if any(not str(metadata[key]) for key in ("run_id", "role", "stage", "logical_call_id")):
            raise ValueError("provider lease metadata must be non-empty")

        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - formal runtime is Linux
            raise RuntimeError(
                "CampaignProviderGate requires Linux fcntl.flock"
            ) from exc

        started = time.perf_counter()
        rotation = int.from_bytes(
            hashlib.sha256(
                (
                    f"{self.campaign_id}\0{metadata['run_id']}\0"
                    f"{metadata['logical_call_id']}"
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        ) % self.max_inflight
        handle = None
        slot_id = -1
        while handle is None:
            for offset in range(self.max_inflight):
                candidate_id = (rotation + offset) % self.max_inflight
                candidate = self._slot_paths[candidate_id].open("a+b")
                try:
                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                except BaseException:
                    candidate.close()
                    raise
                handle = candidate
                slot_id = candidate_id
                break
            if handle is None:
                time.sleep(0.01)

        lease = ProviderLease(
            slot_id=slot_id,
            queue_wait_ms=max(0, int((time.perf_counter() - started) * 1000)),
            acquired_at_unix=time.time(),
        )
        try:
            yield lease
        finally:
            assert handle is not None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def identity(self) -> dict[str, object]:
        return {
            "schema_version": _GATE_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "max_inflight": self.max_inflight,
        }

    def _bind_identity(self) -> None:
        """Fail closed if another process configured this directory differently."""

        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - formal runtime is Linux
            raise RuntimeError(
                "CampaignProviderGate requires Linux fcntl.flock"
            ) from exc

        lock_path = self.gate_dir / "gate_identity.lock"
        identity_path = self.gate_dir / "gate_identity.json"
        expected = self.identity()
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                if identity_path.exists():
                    try:
                        actual = json.loads(identity_path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            f"provider gate identity is unreadable: {identity_path}"
                        ) from exc
                    if actual != expected:
                        raise ValueError(
                            "provider gate directory is already bound to a different "
                            f"campaign policy: expected {expected!r}, got {actual!r}"
                        )
                    return
                temporary = identity_path.with_name(
                    f".{identity_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    with temporary.open("x", encoding="utf-8") as handle:
                        json.dump(expected, handle, ensure_ascii=False, indent=2, sort_keys=True)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, identity_path)
                finally:
                    temporary.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
