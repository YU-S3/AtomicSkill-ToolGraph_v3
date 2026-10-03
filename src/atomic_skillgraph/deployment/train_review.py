"""Read-only, benchmark-independent review of learned Train deployment routes.

This does not promote, rewrite assets, pool alias credit or run a model/replay.
Preference is inherited only from the existing ledger-backed Preferred policy.
Actual invocation still requires current input/state preflight.
"""
from ..core.serialization import to_primitive
from ..core.status import RuntimeMode
from ..evolution.learning_interventions import training_source
from ..runtime.invocation_compiler import InvocationCompiler
from .train_bank_compiler import _proof_aliases, qualification_inventory

VERSION = 'skillcompiler.train-deployment-review.v1'


def review_train_deployment(system):
    if not system.r103 or system.readonly or system.projection is None:
        raise ValueError('Review requires the current qualified Train bank before freeze')
    digest = system.knowledge_digest()
    sources = system.runtime_support_store.committed()
    for source in sources:
        identity = source['evidence']['source_identity']
        if identity.get('split') != 'train' or not training_source(identity):
            raise ValueError('Non-Train execution source cannot authorize deployment')
    compiler = InvocationCompiler(system.skills, system.tools, system.harness, mode=RuntimeMode.FROZEN)
    tools, aliases, proofs = _proof_aliases(system.skills, system.tools)
    assets, routes = [], []
    for row in system.database.rows('SELECT artifact_ref,artifact_kind,status FROM artifact_index ORDER BY artifact_ref'):
        stats = system.projection.stats(row['artifact_ref'], row['artifact_kind'])
        decision = system.lifecycle.policy.review(row['artifact_ref'], row['artifact_kind'], row['status'], stats)
        assets.append({'artifact_ref': row['artifact_ref'], 'effective_status': row['status'],
                       'decision': to_primitive(decision), 'stats': stats.to_dict()})
    for impl in system.skills.implementations():
        failures = []
        used = []
        try:
            atomic = system.skills.get_atomic(impl.abstract_ref)
            used = [system.tools.get(b.tool_ref) for b in impl.tool_bindings]
            compiler.compile(atomic, impl, used, {})
            for tool in used:
                report = system.tool_static_validator.validate_tool_asset(tool, atomic, system.harness)
                if not report.passed:
                    failures.extend(report.failure_codes)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
        preferred = bool(used) and not failures
        for tool in used:
            stats = system.projection.stats(str(tool.ref), 'tool')
            preferred = preferred and tool.status.value == 'preferred' and system.lifecycle.policy._preferred_tool(stats)
        routes.append({'atomic_ref': str(impl.abstract_ref), 'implementation_ref': str(impl.ref),
            'canonical_implementation_ref': aliases.get(str(impl.ref), str(impl.ref)),
            'frozen_compilable': not failures, 'failure_codes': failures,
            'ledger_backed_preferred': bool(preferred), 'current_ready': None,
            'required_input_roles': [p.name for p in system.skills.get_atomic(impl.abstract_ref).inputs if p.required]})
        if system.mechanism_profile:
            from .qualification import qualified
            routes[-1]['deployment_qualified'] = qualified(system.database, str(impl.ref))
    preferences = []
    for atomic_ref in sorted({r['atomic_ref'] for r in routes}):
        eligible = {r['canonical_implementation_ref'] for r in routes
                    if r['atomic_ref'] == atomic_ref and r['ledger_backed_preferred']}
        if len(eligible) == 1:
            preferences.append({'atomic_ref': atomic_ref, 'implementation_ref': next(iter(eligible)),
                'priority': 1, 'applicable_condition': 'ready',
                'reason': 'unique exact route with existing ledger-backed Preferred Tool(s)'})
    if system.knowledge_digest() != digest:
        raise RuntimeError('Read-only Train review changed raw Bank')
    return {'version': 'skillcompiler.train-deployment-review.v2' if system.mechanism_profile else VERSION,
        'source_bank_digest': digest,
        **({'mechanism_effective_profile': system.mechanism_profile,
            'deployment_certificates': qualification_inventory(system),
            'qualification_policy': 'old Active OR immutable two-Train production qualification; current preflight always required'}
           if system.mechanism_profile else {}),
        'source_execution_keys': [r['observation']['execution_key'] for r in sources],
        'canonical_tool_aliases': tools, 'canonical_implementation_aliases': aliases,
        'identity_proofs': proofs, 'assets': assets, 'routes': routes,
        'preferred_implementations': preferences, 'new_credit': False, 'new_llm_requests': 0}
