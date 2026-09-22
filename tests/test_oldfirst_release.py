"""OF01-04: supplied original banks, real alpha proofs and declared role edges.

External source archives are optional for portable CI; installed local deliveries
are never substituted with invented assets or successful validator mocks.
"""
import copy
import json
from pathlib import Path
import pytest
from atomic_skillgraph.deployment.bank_release import _extract, _source_inventory
from atomic_skillgraph.deployment.oldfirst_release import _rebindings
from atomic_skillgraph.deployment.oldfirst_revision import revise_graph
from atomic_skillgraph.deployment.release_protocol import sha, OLDFIRST_SOURCE_ARCHIVES
from atomic_skillgraph.evolution.identity_matching import match_tool, verify_tool_proof
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.bindings import BindingExpression


@pytest.fixture(params=(42,43,44))
def original(request, tmp_path):
    seed = request.param
    archives = {
        42: '/home/yangchengyu/asg_r1021_seed42_hCszY6/runs/alfworld_train_full_120_r1021_seed42/data_v3.zip',
        43: '/mnt/d/T3S_exp/ASTRG_R103_bank_edit_20260922/backups/seed43_frozen_bank_original.zip',
        44: '/home/yangchengyu/asg_r1021_parallel_18EnMl/seed44/runs/alfworld_train_full_120_r1021_seed44/data_v3.zip'}
    plan_path = Path('/mnt/d/Download/Edge/R10.3_旧库优先重整_三Seed逐资产清单与Frozen修复/data')/f'seed{seed}_oldfirst_edit_plan.json'
    archive = Path(archives[seed])
    if not archive.is_file() or not plan_path.is_file():
        pytest.skip('original delivery not installed on this host')
    assert sha(archive) in OLDFIRST_SOURCE_ARCHIVES[seed]
    _extract(archive, tmp_path/'source')
    db, rows = _source_inventory(tmp_path/'source')
    try:
        yield json.loads(plan_path.read_text()), rows
    finally:
        db.close()


def test_original_refs_bytes_status_and_layered_rebinding(original, tmp_path):
    plan, rows = original
    expected = {a['ref']: a for a in plan['asset_dispositions']}
    assets = {str(a.ref): a for _, a, _, _ in rows}
    assert len(rows) == plan['source_asset_count'] == {42:185,43:169,44:183}[plan['seed']]
    for row, asset, payload, path in rows:
        item = expected[str(asset.ref)]
        assert sha(path) == item['source_sha256']
        assert row['status'] == item['source_status']
    before = to_primitive(assets)
    revised, tools, impls, blocked = _rebindings(plan, assets, tmp_path/'proofs')
    assert to_primitive(assets) == before
    assert len(plan['manual_publications']) == {42:11,43:4,44:13}[plan['seed']]
    assert all(p['execution_credit_delta'] == 0 for p in plan['manual_publications'].values())
    for kind, new, source, status in revised:
        assert new.abstract_ref == assets[source].abstract_ref
        assert not new.quality
        assert new.ref != assets[source].ref and impls[source] == str(new.ref)
    for group in plan['merge_groups']:
        left, right = assets[group['alias_tool_ref']], assets[group['canonical_tool_ref']]
        proof = match_tool(left, right).proof
        assert verify_tool_proof(left, right, proof)
        corrupt = copy.deepcopy(right)
        corrupt.signature['properties']['unapproved'] = {'type': 'boolean'}
        corrupt.signature.setdefault('required', []).append('unapproved')
        assert not verify_tool_proof(left, corrupt, proof)
    for job in plan['workflow_targets']:
        source = assets[job['source_ref']]
        result = revise_graph(job, source)
        if job['action'] != 'revise':
            assert to_primitive(result) == to_primitive(source)
            continue
        assert result.ref.logical_id == source.ref.logical_id
        for node, occurrence in zip(job['target_nodes'], result.occurrences):
            assert str(occurrence.node_ref) == node['atomic_ref']
            assert to_primitive(occurrence.binding_specs) == {
                key: to_primitive(BindingExpression.from_dict(raw))
                for key, raw in node['binding_specs'].items()}
