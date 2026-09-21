"""Independent-training ablations, retaining complete correctness checks."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_identity_enhanced = ContextVar("r103_identity_enhanced", default=True)


def enhanced():
    return _identity_enhanced.get()


@contextmanager
def identity_scope(enabled=True):
    token = _identity_enhanced.set(enabled)
    try:
        yield
    finally:
        _identity_enhanced.reset(token)


def condition(config):
    value = config.get("r103_learning_intervention", "Full")
    if value == "Full":
        return value
    experiment = config.get("experiment", {})
    if (value not in {"L-identity-support", "L-generalization"}
            or config.get("repair_revision") != "R10.3"
            or experiment.get("experiment_kind") != "learning_ablation"
            or config.get("r103_interventions")):
        raise ValueError("learning ablations require independent R10.3 learning_ablation config")
    return value


def training_source(source):
    """Explicit training authority; free-standing deployment diagnostics fail."""
    return source.get("experiment_kind") == "formal" or (
        source.get("experiment_kind") == "learning_ablation"
        and source.get("learning_condition") in {"L-identity-support", "L-generalization"})


def system_scope(method):
    @wraps(method)
    def scoped(self, *args, **kwargs):
        from .identity_audit import capture
        with identity_scope(condition(self.config) != "L-identity-support"), capture():
            return method(self, *args, **kwargs)
    return scoped


def full_proof(method):
    @wraps(method)
    def scoped(*args, **kwargs):
        with identity_scope():
            return method(*args, **kwargs)
    return scoped


def bind_bank(system):
    selected = condition(system.config)
    row = system.database.execute("SELECT value FROM metadata WHERE key='r103_learning_condition'").fetchone()
    if row is not None:
        if row[0] != selected:
            raise RuntimeError("learning condition cannot change or import another condition's bank")
    elif selected != "Full":
        if system.readonly or not system.is_empty_knowledge_bank():
            raise RuntimeError("learning ablation must begin with its own new empty bank")
        system.database.execute("INSERT INTO metadata(key,value) VALUES('r103_learning_condition',?)", (selected,))
        system.database.connection.commit()
