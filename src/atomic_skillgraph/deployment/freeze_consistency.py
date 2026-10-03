"""No-model governance convergence at one committed ledger watermark."""
from ..core.serialization import to_primitive


def converge(system):
    from .qualification import qualify_composites
    count = system.database.execute('SELECT COUNT(*) FROM artifact_index').fetchone()[0]
    reviews, qualification = [], []
    # A newly promoted child can make a graph deployable; its certificate in turn
    # can close a parent. Converge both changes before computing the frozen digest.
    for _ in range(2 * count + 2):
        system.projection.consume_new_events()
        review = system.lifecycle.review()
        reviews.append(to_primitive(review))
        certificates = qualify_composites(system)
        qualification.extend(certificates)
        system.projection.consume_new_events()
        if not review.changed_count and not any(c['status'] == 'qualified' for c in certificates):
            break
    else:
        raise RuntimeError('freeze dependency lifecycle did not converge within asset bound')
    watermark = system.ledger.max_rowid()
    if system.projection.checkpoint != watermark:
        raise RuntimeError('freeze projection has not consumed the committed ledger')
    decisions = {d.artifact_ref: d for d in review.decisions}
    assets = []
    for row in system.database.rows('SELECT artifact_ref,artifact_kind,status FROM artifact_index ORDER BY artifact_ref'):
        stats = system.projection.stats(row['artifact_ref'], row['artifact_kind'])
        decision = decisions[row['artifact_ref']]
        if decision.changed:
            raise RuntimeError('freeze contains an unprocessed promotable asset')
        assets.append({'artifact_ref': row['artifact_ref'], 'status': row['status'],
            'canonical_support_task_ids': stats.execution_support.get('registered_canonical_support_tasks', []),
            'union_support_tasks': stats.execution_support.get('union_support_tasks', []),
            'decision': to_primitive(decision), 'stats': stats.to_dict()})
    return {'ledger_watermark': watermark, 'projection_watermark': watermark,
            'thresholds': to_primitive(system.lifecycle.policy.thresholds),
            'reviews': reviews, 'assets': assets, 'composition_qualification': qualification,
            'converged': True, 'llm_requests': 0}
