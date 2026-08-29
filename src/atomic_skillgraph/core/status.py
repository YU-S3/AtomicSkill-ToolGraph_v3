"""Persistent lifecycle and runtime provenance states."""

from enum import Enum


class SkillStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SHADOW = "shadow"
    SUPPRESSED = "suppressed"
    RETIRED = "retired"


class ToolStatus(str, Enum):
    DRAFT = "draft"
    ADMISSION_PENDING = "admission_pending"
    SHADOW = "shadow"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    PREFERRED = "preferred"
    SUPPRESSED = "suppressed"
    RETIRED = "retired"


class RuntimeMode(str, Enum):
    ONLINE = "online"
    FROZEN = "frozen"


class ExecutionProvenance(str, Enum):
    DIRECT = "direct"
    SEEDED = "seeded"
    DYNAMIC = "dynamic"


def skill_status_usable(status: SkillStatus | str, mode: RuntimeMode | str) -> bool:
    status = SkillStatus(status)
    mode = RuntimeMode(mode)
    allowed = {SkillStatus.ACTIVE}
    if mode is RuntimeMode.ONLINE:
        allowed.add(SkillStatus.CANDIDATE)
    return status in allowed


def tool_status_usable(status: ToolStatus | str, mode: RuntimeMode | str) -> bool:
    status = ToolStatus(status)
    mode = RuntimeMode(mode)
    allowed = {ToolStatus.ACTIVE, ToolStatus.PREFERRED}
    if mode is RuntimeMode.ONLINE:
        allowed.add(ToolStatus.CANDIDATE)
    return status in allowed
