"""Persistence regressions for SemanticPredicate.effect_domain."""

from __future__ import annotations

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CompositeSkill,
    EffectDomain,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef, content_hash
from atomic_skillgraph.core.serialization import atomic_write_json, read_json
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry


def _registry(tmp_path):
    database = StateDatabase(tmp_path / "state.sqlite3")
    store = ArtifactStore(tmp_path, database)
    return database, store, SkillRegistry(store, database)


def _atomic(ref: SkillRef, *, effect_domain: EffectDomain) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=ref,
        summary="inspect an entity",
        inputs=[],
        outputs=[],
        preconditions=[
            SemanticPredicate(
                "entity.available",
                {"entity": "$entity"},
                effect_domain=effect_domain,
            )
        ],
        effects=[
            SemanticPredicate(
                "entity.observed",
                {"entity": "$entity"},
                effect_domain=effect_domain,
            )
        ],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
    )


def _composite(ref: SkillRef, *, effect_domain: EffectDomain) -> CompositeSkill:
    return CompositeSkill(
        ref=ref,
        summary="inspect workflow",
        occurrences=[],
        control_sequence=[],
        data_edges=[],
        dependency_edges=[],
        goal_contract=TaskContract(target_effects=[
            SemanticPredicate(
                "entity.observed",
                {"entity": "$entity"},
                effect_domain=effect_domain,
            )
        ]),
        guideline={},
        insight={},
        validator_spec={},
        metadata={},
    )


def _remove_effect_domains(
    database: StateDatabase,
    store: ArtifactStore,
    *,
    kind: str,
    ref: SkillRef,
) -> None:
    """Emulate a pre-effect-domain payload while preserving store integrity."""

    path = store.path_for(kind, ref)
    payload = read_json(path)
    if kind == "atomic":
        predicates = payload["preconditions"] + payload["effects"]
    else:
        predicates = payload["goal_contract"]["target_effects"]
    for predicate in predicates:
        predicate.pop("effect_domain")
    atomic_write_json(path, payload)
    digest = content_hash(
        payload,
        exclude=("status", "quality", "statistics", "evidence"),
    )
    database.execute(
        "UPDATE artifact_index SET content_hash=? WHERE artifact_ref=?",
        (digest, str(ref)),
    )
    database.connection.commit()


def test_atomic_effect_domain_survives_artifact_round_trip(tmp_path) -> None:
    database, _store, registry = _registry(tmp_path)
    ref = SkillRef("atomic_evidence_round_trip", "1.0.0")
    registry.register_atomic(_atomic(ref, effect_domain=EffectDomain.EVIDENCE))

    loaded = registry.get_atomic(ref)

    assert loaded.preconditions[0].effect_domain is EffectDomain.EVIDENCE
    assert loaded.effects[0].effect_domain is EffectDomain.EVIDENCE
    database.close()


def test_composite_effect_domain_survives_artifact_round_trip(tmp_path) -> None:
    database, _store, registry = _registry(tmp_path)
    ref = SkillRef("composite_evidence_round_trip", "1.0.0")
    registry.register_composite(_composite(ref, effect_domain=EffectDomain.EVIDENCE))

    loaded = registry.get_composite(ref)

    assert (
        loaded.goal_contract.target_effects[0].effect_domain
        is EffectDomain.EVIDENCE
    )
    database.close()


def test_atomic_payload_without_effect_domain_defaults_to_world(tmp_path) -> None:
    database, store, registry = _registry(tmp_path)
    ref = SkillRef("atomic_legacy_domain_default", "1.0.0")
    registry.register_atomic(_atomic(ref, effect_domain=EffectDomain.WORLD))
    _remove_effect_domains(database, store, kind="atomic", ref=ref)

    loaded = registry.get_atomic(ref)

    assert loaded.preconditions[0].effect_domain is EffectDomain.WORLD
    assert loaded.effects[0].effect_domain is EffectDomain.WORLD
    database.close()


def test_composite_payload_without_effect_domain_defaults_to_world(tmp_path) -> None:
    database, store, registry = _registry(tmp_path)
    ref = SkillRef("composite_legacy_domain_default", "1.0.0")
    registry.register_composite(_composite(ref, effect_domain=EffectDomain.WORLD))
    _remove_effect_domains(database, store, kind="composite", ref=ref)

    loaded = registry.get_composite(ref)

    assert loaded.goal_contract.target_effects[0].effect_domain is EffectDomain.WORLD
    database.close()
