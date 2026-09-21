"""Explicit deployment diagnostics, never inferred from tasks or outcomes."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DeploymentIntervention:
    condition: str = "C11"
    presentation: str = "new"
    programs: bool = True
    automatic_entry: bool = True

    @classmethod
    def from_config(cls, config):
        raw = config.get("r103_interventions")
        if not raw:
            return cls()
        if (config.get("repair_revision") != "R10.3"
                or config.get("experiment", {}).get("experiment_kind") != "diagnostic"
                or not isinstance(raw, dict) or set(raw) != {"condition"}):
            raise ValueError("deployment intervention requires an explicit R10.3 diagnostic condition")
        condition = raw["condition"]
        if condition not in {"C11", "C01", "C10", "C00", "A0"}:
            raise ValueError("unsupported deployment diagnostic condition")
        return cls(condition, "old" if condition in {"C01", "C00"} else "new",
                   condition not in {"C10", "C00"}, condition in {"C11", "C01"})

    def to_dict(self):
        return asdict(self)


def policy(ctx):
    raw = getattr(ctx, "runtime_config", {}).get("r103_deployment_intervention")
    return DeploymentIntervention(**raw) if raw else DeploymentIntervention()
