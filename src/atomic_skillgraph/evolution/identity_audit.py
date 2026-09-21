"""Context-local match diagnostics; never part of match or admission authority."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

_audit = ContextVar("r103_identity_audit", default=None)


@contextmanager
def capture():
    token = _audit.set({"layers":{},"pairs":set(),"proof_samples":[],"nonexact_samples":[]})
    try:
        yield
    finally:
        _audit.reset(token)


def snapshot():
    state = _audit.get()
    if state is None:
        return {}
    return {"identity_exact_different_unknown_by_layer":deepcopy(state["layers"]),
            "identity_source_pairs_checked":len(state["pairs"]),
            "proof_samples":deepcopy(state["proof_samples"]),
            "nonexact_samples":deepcopy(state["nonexact_samples"]),
            "counting_unit":"matcher invocations (including verification); distinct pairs reported separately"}


def observe(layer):
    def decorate(method):
        @wraps(method)
        def match(source, target, *args, **kwargs):
            result = method(source,target,*args,**kwargs)
            state = _audit.get()
            if state is not None:
                from .identity_matching import raw_hash
                from ..core.serialization import to_primitive
                counts = state["layers"].setdefault(layer,{"exact":0,"different":0,"unknown":0,"search_states":0})
                counts[result.status] += 1
                counts["search_states"] += result.search_states
                pair = (layer,raw_hash(source),raw_hash(target))
                state["pairs"].add(pair)
                sample = {"layer":layer,"source_raw_hash":pair[1],"target_raw_hash":pair[2],
                          "status":result.status,"reason":result.reason}
                if result.proof is not None:
                    samples = state["proof_samples"]
                    sample["proof"] = to_primitive(result.proof)
                else:
                    samples = state["nonexact_samples"]
                if len(samples) < 9 and sample not in samples:
                    samples.append(sample)
            return result
        return match
    return decorate
