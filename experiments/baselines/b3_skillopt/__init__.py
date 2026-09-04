"""SkillOpt (B3) baseline adapter: common-manifest EnvAdapter + worker.

This package is deliberately import-light: modules that require the upstream
``skillopt`` package (``common_alfworld_adapter``, ``episode_runner``,
``worker``) are imported explicitly only inside the per-method worker venv.
The controller-side ``driver``/``freeze`` modules stay importable without
the upstream package.
"""
