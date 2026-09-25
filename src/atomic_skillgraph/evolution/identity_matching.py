"""Typed, bounded identity proofs. Refinement is a filter, never authority.

No environment calls, status changes, admission decisions or evidence credit
belong here. In particular, a search limit is UNKNOWN, not a failed contract.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Callable, Iterator, Literal

from ..core.contracts import AbstractAtomicSkill, ToolAsset, ImplementationAtom
from ..core.serialization import to_primitive
from .learning_interventions import enhanced, identity_scope
from .identity_audit import observe

IDENTITY_VERSION = "r103.identity.v1"
MAX_SEARCH_STATES = 100_000


def typed_json(value: Any) -> str:
    return json.dumps(to_primitive(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def raw_hash(value: Any) -> str:
    return sha256(typed_json(value).encode("utf-8")).hexdigest()


def bounded_matches(candidate, pool, matcher):
    """One search allowance for an entire retrieval scan, not per alias."""
    remaining = MAX_SEARCH_STATES
    for item in pool:
        result = matcher(candidate, item, max_states=remaining)
        remaining -= result.search_states
        yield item, result
        if remaining <= 0:
            break


@dataclass(frozen=True)
class IdentityProof:
    layer: str
    source_raw_hash: str
    target_raw_hash: str
    source_ref: str
    target_ref: str
    input_role_map: dict[str, str] = field(default_factory=dict)
    output_role_map: dict[str, str] = field(default_factory=dict)
    local_symbol_map: dict[str, str] = field(default_factory=dict)
    node_id_map: dict[str, str] = field(default_factory=dict)
    constraint_id_map: dict[str, str] = field(default_factory=dict)
    binding_role_map: dict[str, str] = field(default_factory=dict)
    normalized_payload_hash: str = ""
    checked_fields_version: str = IDENTITY_VERSION
    identity_version: str = IDENTITY_VERSION
    tool_proofs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchResult:
    status: Literal["exact", "different", "unknown"]
    proof: IdentityProof | None = None
    reason: str = ""
    search_states: int = 0


@dataclass
class _Graph:
    labels: dict[str, str] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, label: Any) -> str:
        key = str(len(self.labels))
        self.labels[key] = typed_json(label)
        return key

    def refinement(self, anchors: dict[str, int] | None = None) -> dict[str, str]:
        anchors = anchors or {}
        colors = {v: raw_hash([label, anchors.get(v)]) for v, label in self.labels.items()}
        outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
        incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for a, b, e in self.edges:
            outgoing[a].append((e, b))
            incoming[b].append((e, a))
        # A bounded filter is sufficient: the complete payload is compared
        # after every proposed bijection, including all multiplicities.
        for _ in range(4):
            colors = {v: raw_hash([
                colors[v], sorted((e, colors[w]) for e, w in outgoing[v]),
                sorted((e, colors[w]) for e, w in incoming[v]),
            ]) for v in self.labels}
        return colors


def _limit(max_states: int) -> None:
    if type(max_states) is not int or not 0 <= max_states <= MAX_SEARCH_STATES:
        raise ValueError("identity search limit must be an integer in [0, 100000]")


def _atomic_payload(atomic: AbstractAtomicSkill, inputs: dict[str, str],
                    outputs: dict[str, str]) -> dict[str, Any]:
    from .contract_canonicalizer import _identity_payload
    return _identity_payload(atomic, inputs, outputs)


def _atomic_graph(atomic: AbstractAtomicSkill) -> tuple[_Graph, dict[tuple[str, str], str]]:
    """Encode the *complete* identity payload, not local role descriptors.

    Fresh sentinels are introduced only by the schema-aware role rewriter;
    pre-existing constants cannot collide with them. JSON fields not understood
    by the rewriter remain literal, so unknown fields are never discarded.
    """
    prefix = "__r103_role__"
    raw = typed_json(atomic)
    while prefix in raw:
        prefix += "_"
    roles = [(boundary, spec.name) for boundary, specs in
             (("input", atomic.inputs), ("output", atomic.outputs)) for spec in specs]
    sentinels = {role: f"{prefix}{i}" for i, role in enumerate(roles)}
    g = _Graph()
    vertices = {role: g.add(["role", role[0]]) for role in roles}
    references = {sentinels[role]: vertex for role, vertex in vertices.items()}
    payload = _atomic_payload(atomic,
        {name: sentinels[("input", name)] for name in (s.name for s in atomic.inputs)},
        {name: sentinels[("output", name)] for name in (s.name for s in atomic.outputs)})

    def encode(value: Any, path: tuple[str, ...]) -> str:
        if isinstance(value, str) and value in references:
            return references[value]
        if isinstance(value, str) and value.startswith("$") and value[1:] in references:
            node = g.add(["dollar_reference"])
            g.edges.append((node, references[value[1:]], "role"))
            return node
        if isinstance(value, dict):
            node = g.add(["object"])
            for key, child in value.items():
                entry = g.add(["field"])
                g.edges.extend([(node, entry, "entry"),
                    (entry, encode(key, path + ("key",)), "key"),
                    (entry, encode(child, path + (key,)), "value")])
            return node
        if isinstance(value, list):
            unordered = path in {("inputs",), ("outputs",), ("preconditions",),
                ("effects",), ("validator_contract", "output_identity")}
            node = g.add(["bag" if unordered else "array"])
            for index, child in enumerate(value):
                g.edges.append((node, encode(child, path + ("item",)),
                                "item" if unordered else str(index)))
            return node
        return g.add(["literal", value])

    root = g.add(["atomic_root", IDENTITY_VERSION])
    g.edges.append((root, encode(payload, ()), "contract"))
    return g, vertices


class _Exhausted(Exception):
    pass


@dataclass
class _Search:
    limit: int
    states: int = 0

    def tick(self) -> None:
        if self.states >= self.limit:
            raise _Exhausted
        self.states += 1


def _role_maps(source: AbstractAtomicSkill, target: AbstractAtomicSkill,
               search: _Search) -> Iterator[tuple[dict[str, str], dict[str, str]]]:
    if not enhanced():
        # Former descriptor/name ordering proposes one mapping. It does not
        # search relational symmetries, but still must pass the full contract.
        from .contract_canonicalizer import _boundary_role_map
        maps = []
        for boundary in ("inputs", "outputs"):
            a = _boundary_role_map(source, getattr(source, boundary), boundary[:-1])
            b = _boundary_role_map(target, getattr(target, boundary), boundary[:-1])
            if set(a.values()) != set(b.values()):
                return
            inverse = {v:k for k,v in b.items()}
            maps.append({k:inverse[v] for k,v in a.items()})
        search.tick()
        if typed_json(_atomic_payload(source, *maps)) == typed_json(_atomic_payload(target,
                {s.name:s.name for s in target.inputs}, {s.name:s.name for s in target.outputs})):
            yield tuple(maps)
        return
    a, va = _atomic_graph(source)
    b, vb = _atomic_graph(target)
    ca, cb = a.refinement(), b.refinement()
    if Counter(ca.values()) != Counter(cb.values()):
        return
    choices = {role: [other for other in vb if other[0] == role[0]
                         and ca[va[role]] == cb[vb[other]]] for role in va}
    order = sorted(choices, key=lambda r: (len(choices[r]), r))
    assigned: dict[tuple[str, str], tuple[str, str]] = {}
    used: set[tuple[str, str]] = set()

    def visit(index: int):
        if index == len(order):
            inputs = {k[1]: v[1] for k, v in assigned.items() if k[0] == "input"}
            outputs = {k[1]: v[1] for k, v in assigned.items() if k[0] == "output"}
            # Complete typed JSON comparison is the authority, never colors,
            # hashes, a local descriptor, or the first partial mapping.
            target_inputs = {s.name: s.name for s in target.inputs}
            target_outputs = {s.name: s.name for s in target.outputs}
            if typed_json(_atomic_payload(source, inputs, outputs)) == typed_json(
                    _atomic_payload(target, target_inputs, target_outputs)):
                yield inputs, outputs
            return
        role = order[index]
        for other in choices[role]:
            if other not in used:
                search.tick()
                assigned[role] = other
                used.add(other)
                # Individualization propagates already-mapped role edges into
                # neighboring relation nodes. Reject incompatible residual
                # neighborhoods before enumerating all symmetric role maps.
                # Colors remain only necessary conditions, never the proof.
                anchors_a = {va[r]: i for i, r in enumerate(assigned)}
                anchors_b = {vb[assigned[r]]: i for i, r in enumerate(assigned)}
                if Counter(a.refinement(anchors_a).values()) == Counter(b.refinement(anchors_b).values()):
                    yield from visit(index + 1)
                used.remove(other)
                del assigned[role]
    yield from visit(0)


@observe("atomic")
def match_atomic(source: AbstractAtomicSkill, target: AbstractAtomicSkill, *,
                 max_states: int = MAX_SEARCH_STATES,
                 compatible_mapping: Callable[[dict[str, str], dict[str, str]], bool] | None = None,
                 ) -> MatchResult:
    """Find a full bijection; a joint executable check may reject a symmetry.

    ``compatible_mapping`` must be pure. Rejecting one complete Atomic mapping
    continues searching other symmetries instead of breaking ties by role name.
    """
    _limit(max_states)
    search = _Search(max_states)
    try:
        for inputs, outputs in _role_maps(source, target, search):
            if compatible_mapping is not None and not compatible_mapping(inputs, outputs):
                continue
            proof = IdentityProof("atomic", raw_hash(source), raw_hash(target),
                str(source.ref), str(target.ref), inputs, outputs,
                normalized_payload_hash=raw_hash(_atomic_payload(source, inputs, outputs)))
            return MatchResult("exact", proof, "complete typed contract bijection", search.states)
    except _Exhausted:
        return MatchResult("unknown", reason="search_state_limit", search_states=search.states)
    return MatchResult("different", reason="no complete compatible bijection", search_states=search.states)


def verify_atomic_proof(source: AbstractAtomicSkill, target: AbstractAtomicSkill,
                        proof: IdentityProof) -> bool:
    """Revalidate bytes and the complete mapping when consuming a saved proof."""
    if (proof.identity_version != IDENTITY_VERSION or proof.layer != "atomic"
            or proof.checked_fields_version != IDENTITY_VERSION
            or proof.source_ref != str(source.ref) or proof.target_ref != str(target.ref)
            or proof.source_raw_hash != raw_hash(source) or proof.target_raw_hash != raw_hash(target)):
        return False
    for mapping, src, dst in ((proof.input_role_map, source.inputs, target.inputs),
                               (proof.output_role_map, source.outputs, target.outputs)):
        if (set(mapping) != {s.name for s in src} or set(mapping.values()) != {s.name for s in dst}
                or len(set(mapping.values())) != len(mapping)):
            return False
    left = _atomic_payload(source, proof.input_role_map, proof.output_role_map)
    right = _atomic_payload(target, {s.name: s.name for s in target.inputs},
                            {s.name: s.name for s in target.outputs})
    return typed_json(left) == typed_json(right) and raw_hash(left) == proof.normalized_payload_hash


def canonical_atomic_roles(atomic: AbstractAtomicSkill, *, max_states: int = MAX_SEARCH_STATES
                           ) -> tuple[dict[str, str], dict[str, str]] | None:
    """Canonical full-payload representative, with UNKNOWN represented by None.

    Refinement separates most roles without search; residual symmetries are
    enumerated. Names affect only traversal order, never the chosen payload.
    """
    _limit(max_states)
    if not enhanced():
        from .contract_canonicalizer import _boundary_role_map
        return (_boundary_role_map(atomic, atomic.inputs, "input"),
                _boundary_role_map(atomic, atomic.outputs, "output"))
    graph, vertices = _atomic_graph(atomic)
    colors = graph.refinement()
    from .contract_canonicalizer import _parameter_descriptor
    specs = {("input", s.name): s for s in atomic.inputs}
    specs.update({("output", s.name): s for s in atomic.outputs})
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for (boundary, name), vertex in vertices.items():
        descriptor = typed_json(_parameter_descriptor(atomic, specs[(boundary, name)]))
        groups[(boundary, descriptor, colors[vertex])].append(name)
    choices: list[tuple[str, str, list[str]]] = []
    counters: dict[str, int] = defaultdict(int)
    for (boundary, _, _color), names in sorted(groups.items()):
        slots = [f"{boundary}_{i:03d}" for i in range(counters[boundary], counters[boundary] + len(names))]
        counters[boundary] += len(names)
        choices.extend((boundary, name, slots) for name in names)
    search = _Search(max_states)
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    used: set[str] = set()
    best: tuple[str, dict[str, str], dict[str, str]] | None = None

    def visit(index: int) -> None:
        nonlocal best
        if index == len(choices):
            payload = typed_json(_atomic_payload(atomic, inputs, outputs))
            if best is None or payload < best[0]:
                best = (payload, dict(inputs), dict(outputs))
            return
        boundary, name, slots = choices[index]
        mapping = inputs if boundary == "input" else outputs
        for slot in slots:
            if slot not in used:
                search.tick()
                mapping[name] = slot
                used.add(slot)
                visit(index + 1)
                used.remove(slot)
                del mapping[name]
    try:
        visit(0)
    except _Exhausted:
        return None
    assert best is not None
    return best[1], best[2]


def _schema(value: Any, roles: dict[str, str] | None = None) -> Any:
    """Strip annotations at JSON-Schema nodes, never inside const/enum/default."""
    if not isinstance(value, dict):
        return to_primitive(value)
    result = {}
    for key, child in value.items():
        if key in {"description", "title"}:
            continue
        if key in {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"} and isinstance(child, dict):
            result[key] = {(roles.get(name, name) if roles is not None and key == "properties" else name):
                           _schema(spec) for name, spec in child.items()}
        elif key in {"allOf", "anyOf", "oneOf", "prefixItems"} and isinstance(child, list):
            result[key] = [_schema(item) for item in child]
        elif key in {"items", "additionalProperties", "contains", "not", "if", "then", "else", "propertyNames"}:
            result[key] = _schema(child)
        elif key == "required" and isinstance(child, list):
            result[key] = sorted(roles.get(name, name) if roles is not None else name for name in child)
        else:
            result[key] = to_primitive(child)
    return result


class _Unsupported(ValueError):
    pass


@dataclass
class ToolView:
    payload: dict[str, Any]
    node_ids: dict[str, str]
    local_symbols: dict[str, str]
    constraint_ids: dict[str, str]


def tool_view(tool: ToolAsset, inputs: dict[str, str] | None = None,
              outputs: dict[str, str] | None = None) -> ToolView:
    """Read-only alpha projection with the interpreter's lexical loop scopes.

    This is not admission: the original parser/static/replay checks remain
    mandatory. Unsupported/malformed programs cannot obtain a new proof.
    Unknown fields remain byte-significant; no peephole optimization is used.
    """
    from ..tooling.ir import walk_program_nodes
    from ..tooling.validator import _validate_program_node_shapes
    inputs = dict(inputs if inputs is not None else {n: n for n in tool.signature.get("properties", {})})
    outputs = dict(outputs if outputs is not None else
                   {n: n for n in tool.interface.get("output_schema", {}).get("properties", {})})
    artifact, interface = to_primitive(tool.artifact), to_primitive(tool.interface)
    node_ids: dict[str, str] = {}
    locals_map: dict[str, str] = {}
    scopes: dict[str, dict[str, str]] = {}
    constraints: dict[str, str] = {}
    for raw in interface.get("entry_contract", {}).get("grounding_constraints", []):
        name = raw.get("constraint_id")
        if not isinstance(name, str) or name in constraints:
            raise _Unsupported("duplicate or missing entry constraint ID")
        constraints[name] = f"constraint_{len(constraints):03d}"
    if tool.artifact_kind == "tool_ir_v1":
        visits = walk_program_nodes(artifact.get("program"))
        _validate_program_node_shapes(visits)
        for node in visits:
            name = node.get("node_id")
            if not isinstance(name, str) or name in node_ids:
                raise _Unsupported("duplicate or missing program node ID")
            node_ids[name] = f"node_{len(node_ids):03d}"
    elif tool.artifact_kind != "primitive_ir":
        raise _Unsupported("unrecognized executable language")

    def path(raw: str) -> str:
        if raw in node_ids:
            return node_ids[raw]
        # Actual interpreter path syntax, not arbitrary substring replacement.
        parts = raw.split("/")
        for i, token in enumerate(parts):
            if token in node_ids:
                parts[i] = node_ids[token]
                continue
            for suffix in (":then", ":else", ":body"):
                if token.endswith(suffix) and token[:-len(suffix)] in node_ids:
                    parts[i] = node_ids[token[:-len(suffix)]] + suffix
                    break
        return "/".join(parts)

    def ref(kind: str, name: str, scope: dict[str, str]) -> str:
        if kind in {"skill_input", "tool_input"}:
            if name not in inputs:
                raise _Unsupported(f"undefined Tool input: {name}")
            return inputs[name]
        if kind == "tool_output":
            if name not in outputs:
                raise _Unsupported(f"undefined Tool output: {name}")
            return outputs[name]
        if kind == "local_variable":
            if name not in scope:
                raise _Unsupported(f"local reference outside lexical scope: {name}")
            return scope[name]
        return name

    def rewrite(value: Any, scope: dict[str, str], *, predicates: bool = False) -> Any:
        if isinstance(value, list):
            return [rewrite(item, scope, predicates=predicates) for item in value]
        if not isinstance(value, dict):
            if predicates and isinstance(value, str) and value.startswith("$"):
                name = value[1:]
                # Interpreter resolves formal inputs before unqualified locals.
                return "$" + ref("tool_input" if name in inputs else
                    "local_variable" if name in scope else "tool_output", name, scope)
            return value
        kind = value.get("kind")
        if kind == "constant" or value.get("source") == "constant":
            return to_primitive(value)
        result = {key: to_primitive(child) for key, child in value.items()}
        # Only documented reference positions recurse. Opaque literal JSON
        # (condition.value, selector.where equality filters, annotations) stays.
        for key in {"condition", "collection_source", "match", "semantic_compatible_with"} & value.keys():
            result[key] = rewrite(value[key], scope)
        if isinstance(value.get("where"), dict) and "semantic_compatible_with" in value["where"]:
            result["where"]["semantic_compatible_with"] = rewrite(value["where"]["semantic_compatible_with"], scope)
        if kind in {"skill_input", "tool_output", "local_variable"}:
            name = value.get("source_role", "")
            effective_kind = kind
            if predicates and kind == "skill_input" and name not in inputs:
                effective_kind = "local_variable" if name in scope else "tool_output"
            result["source_role"] = ref(effective_kind, name, scope)
        source = value.get("source")
        if source == "bounded_count" and "count" in value:
            result["count"] = rewrite(value["count"], scope)
        if source in {"tool_input", "tool_output", "local_variable"} and "field" in value:
            result["field"] = ref(source, value["field"], scope)
        if "predicate" in value and isinstance(value.get("args"), dict):
            result["args"] = {key: rewrite(child, scope, predicates=True) for key, child in value["args"].items()}
        for key in {"expected_effects", "final_effects", "conditions", "effects"} & value.keys():
            result[key] = rewrite(value[key], scope, predicates=True)
        if isinstance(value.get("argument_mapping"), dict):
            # Native Harness argument keys are never Tool formal parameter names.
            result["argument_mapping"] = {key: rewrite(child, scope) for key, child in value["argument_mapping"].items()}
        for key in {"output_sources", "output_mapping"} & value.keys():
            result[key] = {outputs.get(name, name): rewrite(child, scope) for name, child in value[key].items()}
        if "constraint_id" in value:
            result["constraint_id"] = constraints.get(value["constraint_id"], value["constraint_id"])
        if isinstance(value.get("grounding_constraints"), list):
            result["grounding_constraints"] = rewrite(value["grounding_constraints"], scope)
        return result

    def program(nodes: list[dict[str, Any]], scope: dict[str, str]) -> list[dict[str, Any]]:
        result = []
        for node in nodes:
            original = node["node_id"]
            canonical = node_ids[original]
            scopes[original] = dict(scope)
            current = rewrite(node, scope)
            current["node_id"] = canonical
            if node["op"] == "FOR_EACH":
                variable = node.get("iteration_variable")
                if not isinstance(variable, str) or not variable:
                    raise _Unsupported("missing loop variable")
                symbol = canonical + ":local"
                locals_map[f"{original}::{variable}"] = symbol
                current["iteration_variable"] = symbol
                current["body"] = program(node["body"], {**scope, variable: symbol})
            elif node["op"] == "IF":
                for branch in ("then_branch", "else_branch"):
                    if branch in node:
                        current[branch] = program(node[branch], dict(scope))
            elif node["op"] not in {"ACTION", "STOP_WHEN", "RETURN"}:
                raise _Unsupported("unrecognized opcode")
            result.append(current)
        return result

    normalized = rewrite(artifact, {})
    if tool.artifact_kind == "tool_ir_v1":
        normalized["program"] = program(artifact["program"], {})
    elif "steps" in artifact:
        normalized["steps"] = rewrite(artifact["steps"], {})
    for key in ("path_expectations", "evidence_outputs"):
        if key not in artifact:
            continue
        normalized[key] = []
        for item in artifact[key]:
            node = item.get("node_id", item.get("source_step", ""))
            view = rewrite(item, scopes.get(node, {}))
            for field_name in ("path", "step", "node_id", "source_step"):
                if isinstance(item.get(field_name), str):
                    view[field_name] = path(item[field_name])
            if key == "evidence_outputs" and "role" in item:
                view["role"] = outputs.get(item["role"], item["role"])
            normalized[key].append(view)
    new_interface = dict(interface)
    if "output_schema" in interface:
        new_interface["output_schema"] = _schema(interface["output_schema"], outputs)
    if "entry_contract" in interface:
        new_interface["entry_contract"] = rewrite(interface["entry_contract"], {})
    return ToolView({"signature": _schema(tool.signature, inputs),
        "interface": new_interface, "artifact_kind": tool.artifact_kind,
        "artifact": normalized, "safety": to_primitive(tool.safety)}, node_ids, locals_map, constraints)


def _schema_maps(source: dict, target: dict, search: _Search,
                 required: dict[str, str] | None = None) -> Iterator[dict[str, str]]:
    src, dst = source.get("properties", {}), target.get("properties", {})
    if len(src) != len(dst):
        return
    required = required or {}
    if set(required) - set(src) or set(required.values()) - set(dst):
        return
    candidates = {name: [other for other in dst
        if typed_json(_schema(src[name])) == typed_json(_schema(dst[other]))
        and (name in source.get("required", [])) == (other in target.get("required", []))
        and (name not in required or required[name] == other)] for name in src}
    order = sorted(src, key=lambda name: (len(candidates[name]), name))
    mapping: dict[str, str] = {}
    used: set[str] = set()
    def visit(index):
        if index == len(order):
            yield dict(mapping)
            return
        name = order[index]
        for other in candidates[name]:
            if other not in used:
                search.tick()
                mapping[name] = other
                used.add(other)
                yield from visit(index + 1)
                used.remove(other)
                del mapping[name]
    yield from visit(0)


def _tool_matches(source: ToolAsset, target: ToolAsset, search: _Search,
                  inputs: dict[str, str] | None = None, outputs: dict[str, str] | None = None
                  ) -> Iterator[IdentityProof]:
    if not enhanced():
        fields = ("signature", "interface", "artifact_kind", "artifact", "safety")
        if typed_json([getattr(source,k) for k in fields]) != typed_json([getattr(target,k) for k in fields]):
            return
        identity_inputs = {k:k for k in source.signature.get("properties", {})}
        identity_outputs = {k:k for k in source.interface.get("output_schema", {}).get("properties", {})}
        if inputs is not None and inputs != identity_inputs or outputs is not None and outputs != identity_outputs:
            return
        inputs, outputs = identity_inputs, identity_outputs
    target_view = tool_view(target)
    for ins in _schema_maps(source.signature, target.signature, search, inputs):
        for outs in _schema_maps(source.interface.get("output_schema", {}),
                                target.interface.get("output_schema", {}), search, outputs):
            view = tool_view(source, ins, outs)
            if typed_json(view.payload) != typed_json(target_view.payload):
                continue
            def correspond(a, b):
                inverse = {value: key for key, value in b.items()}
                return {key: inverse[value] for key, value in a.items()}
            yield IdentityProof("tool", raw_hash(source), raw_hash(target), str(source.ref), str(target.ref),
                ins, outs, correspond(view.local_symbols, target_view.local_symbols),
                correspond(view.node_ids, target_view.node_ids),
                correspond(view.constraint_ids, target_view.constraint_ids),
                normalized_payload_hash=raw_hash(view.payload))


@observe("tool")
def match_tool(source: ToolAsset, target: ToolAsset, *, max_states: int = MAX_SEARCH_STATES,
               input_role_map: dict[str, str] | None = None, output_role_map: dict[str, str] | None = None,
               compatible_mapping: Callable[[IdentityProof], bool] | None = None,
               ) -> MatchResult:
    _limit(max_states)
    search = _Search(max_states)
    try:
        proof = next((p for p in _tool_matches(source, target, search, input_role_map, output_role_map)
                      if compatible_mapping is None or compatible_mapping(p)), None)
    except _Exhausted:
        return MatchResult("unknown", reason="search_state_limit", search_states=search.states)
    except (ValueError, TypeError, KeyError) as exc:
        return MatchResult("unknown", reason=f"unsupported_tool_shape: {exc}", search_states=search.states)
    return MatchResult("exact" if proof else "different", proof,
        "complete scoped AST comparison" if proof else "no compatible Tool AST", search.states)


def verify_tool_proof(source: ToolAsset, target: ToolAsset, proof: IdentityProof) -> bool:
    if (proof.layer != "tool" or proof.identity_version != IDENTITY_VERSION
            or proof.checked_fields_version != IDENTITY_VERSION
            or proof.source_raw_hash != raw_hash(source) or proof.target_raw_hash != raw_hash(target)
            or proof.source_ref != str(source.ref) or proof.target_ref != str(target.ref)):
        return False
    with identity_scope():
        result = match_tool(source, target, input_role_map=proof.input_role_map, output_role_map=proof.output_role_map)
    return result.status == "exact" and result.proof == proof


@observe("implementation")
def match_implementation(source: ImplementationAtom, target: ImplementationAtom, *,
                         source_atomic: AbstractAtomicSkill, target_atomic: AbstractAtomicSkill,
                         source_tools: dict[str, ToolAsset], target_tools: dict[str, ToolAsset],
                         max_states: int = MAX_SEARCH_STATES,
                         compatible_proof: Callable[[IdentityProof], bool] | None = None) -> MatchResult:
    """Joint Atomic/ALL-Tool/Impl identity under one total bounded search.

    Tool binding order and independent Tool formal roles are not conflated with
    Atomic roles. A failed mapping continues exploring the Atomic symmetries.
    """
    _limit(max_states)
    search = _Search(max_states)
    if not enhanced():
        fields = ("abstract_ref", "tool_bindings", "grounding_constraints", "execution_policy", "compatibility")
        if typed_json([getattr(source,k) for k in fields]) != typed_json([getattr(target,k) for k in fields]):
            return MatchResult("different", reason="baseline reference-specific Implementation identity")
    if len(source.tool_bindings) != len(target.tool_bindings):
        return MatchResult("different", reason="ToolBinding cardinality differs")
    if source.abstract_ref != source_atomic.ref or target.abstract_ref != target_atomic.ref:
        return MatchResult("unknown", reason="Atomic envelope ref mismatch")
    source_bindings = sorted(source.tool_bindings, key=lambda b: b.order)
    target_bindings = sorted(target.tool_bindings, key=lambda b: b.order)
    if any(len({b.order for b in bindings}) != len(bindings) for bindings in (source_bindings, target_bindings)):
        return MatchResult("unknown", reason="ambiguous ToolBinding execution order")

    def payload(impl, ins, outs, tool_proofs, *, is_source):
        from .contract_canonicalizer import _rewrite_nested
        ordered_bindings = sorted(impl.tool_bindings, key=lambda b: b.order)
        binding_roles = {b.role: f"binding_{i:03d}" for i, b in enumerate(ordered_bindings)}
        if len(binding_roles) != len(impl.tool_bindings):
            raise _Unsupported("ambiguous ToolBinding role")
        def expr(raw):
            value = to_primitive(raw)
            if not isinstance(value, dict) or value.get("kind") == "constant":
                return value
            if value.get("kind") == "skill_input":
                value["source_role"] = ins.get(value["source_role"], value["source_role"])
            elif value.get("kind") == "tool_output":
                binding = next((i for i, b in enumerate(ordered_bindings) if b.role == value["source_step"]), None)
                if binding is None:
                    raise _Unsupported("unresolved Tool output binding")
                if is_source:
                    value["source_role"] = tool_proofs[binding].output_role_map.get(value["source_role"], value["source_role"])
                value["source_step"] = binding_roles[value["source_step"]]
            return value
        bindings = []
        for index, binding in enumerate(ordered_bindings):
            formal = tool_proofs[index].input_role_map if is_source else {}
            bindings.append({"role": binding_roles[binding.role], "order": binding.order,
                "tool": index, "parameter_mapping": {formal.get(k, k): expr(v) for k, v in binding.parameter_mapping.items()}})
        constraints = []
        constraint_map = {}
        for constraint in impl.grounding_constraints:
            row = to_primitive(constraint)
            name = row["constraint_id"]
            if name in constraint_map:
                raise _Unsupported("duplicate Implementation constraint ID")
            constraint_map[name] = f"constraint_{len(constraint_map):03d}"
            row["constraint_id"] = constraint_map[name]
            row["argument_mapping"] = {k: expr(v) for k, v in row["argument_mapping"].items()}
            constraints.append(row)
        policy = to_primitive(impl.execution_policy)
        if "output_mapping" in policy:
            policy["output_mapping"] = {outs.get(k, k): expr(v) for k, v in policy["output_mapping"].items()}
        return {"tools": bindings, "constraints": constraints, "policy": policy,
                "compatibility": to_primitive(impl.compatibility)}, binding_roles, constraint_map

    try:
        for ins, outs in _role_maps(source_atomic, target_atomic, search):
            proofs: list[IdentityProof] = []
            def visit(index):
                if index == len(source.tool_bindings):
                    left, roles_a, constraints_a = payload(source, ins, outs, proofs, is_source=True)
                    right, roles_b, constraints_b = payload(target,
                        {s.name: s.name for s in target_atomic.inputs},
                        {s.name: s.name for s in target_atomic.outputs}, proofs, is_source=False)
                    if typed_json(left) == typed_json(right):
                        inverse_roles = {v: k for k, v in roles_b.items()}
                        inverse_constraints = {v: k for k, v in constraints_b.items()}
                        yield IdentityProof("implementation", raw_hash(source), raw_hash(target),
                            str(source.ref), str(target.ref), ins, outs,
                            constraint_id_map={k: inverse_constraints[v] for k, v in constraints_a.items()},
                            binding_role_map={k: inverse_roles[v] for k, v in roles_a.items()},
                            tool_proofs={binding.role: to_primitive(p)
                                         for binding, p in zip(source_bindings, proofs)},
                            normalized_payload_hash=raw_hash({"implementation": left,
                                "atomic": _atomic_payload(source_atomic, ins, outs),
                                "tool_payloads": [p.normalized_payload_hash for p in proofs]}))
                    return
                a, b = source_bindings[index], target_bindings[index]
                for proof in _tool_matches(source_tools[str(a.tool_ref)], target_tools[str(b.tool_ref)], search):
                    proofs.append(proof)
                    yield from visit(index + 1)
                    proofs.pop()
            proof = next((p for p in visit(0) if compatible_proof is None or compatible_proof(p)), None)
            if proof:
                return MatchResult("exact", proof, "joint complete route identity", search.states)
    except _Exhausted:
        return MatchResult("unknown", reason="search_state_limit", search_states=search.states)
    except (ValueError, TypeError, KeyError) as exc:
        return MatchResult("unknown", reason=f"unsupported_route_shape: {exc}", search_states=search.states)
    return MatchResult("different", reason="no compatible joint route mapping", search_states=search.states)


def verify_implementation_proof(source, target, proof, *, source_atomic, target_atomic, source_tools, target_tools):
    if (proof.layer != "implementation" or proof.identity_version != IDENTITY_VERSION
            or proof.source_raw_hash != raw_hash(source) or proof.target_raw_hash != raw_hash(target)):
        return False
    with identity_scope():
        result = match_implementation(source, target, source_atomic=source_atomic, target_atomic=target_atomic,
            source_tools=source_tools, target_tools=target_tools, compatible_proof=lambda p: p == proof)
    return result.status == "exact" and result.proof == proof
