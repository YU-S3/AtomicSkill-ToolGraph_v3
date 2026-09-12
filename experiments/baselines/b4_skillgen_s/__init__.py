"""B4 SkillGen-S baseline adapter.

The adapter keeps the pinned SkillGen graph, TD(lambda), extraction, and
retrieval implementation authoritative while binding it to the common
Train/Test manifests and audit protocol.
"""

__all__ = ["SkillGenBaselineDriver"]


def __getattr__(name: str):
    """Keep worker package import independent of controller-only modules."""

    if name == "SkillGenBaselineDriver":
        from .driver import SkillGenBaselineDriver

        return SkillGenBaselineDriver
    raise AttributeError(name)
