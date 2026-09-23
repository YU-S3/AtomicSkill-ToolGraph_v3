"""Rebuild from preserved original archives and the actual prior plan locks."""
import argparse
import copy
import json
import shutil
from pathlib import Path

from atomic_skillgraph.core.serialization import atomic_create_json
from atomic_skillgraph.deployment.release_protocol import ReleaseSpec, ReleaseError
from atomic_skillgraph.deployment.oldfirst_plan import compile_plan
from atomic_skillgraph.deployment.oldfirst_release import prepare_oldfirst, verify_oldfirst
from atomic_skillgraph.deployment.bank_release import freeze_release
from atomic_skillgraph.runtime.support_call_surface import VERSION as SUPPORT_VERSION
from atomic_skillgraph.harness.public_discovery import VERSION as PUBLIC_VERSION
from atomic_skillgraph.deployment.preparation_coverage import VERSION as COVERAGE_VERSION


def prepare(previous, package, output):
    previous, package, output = [Path(p).resolve() for p in (previous, package, output)]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(previous/'dev/declared_tasks.json', output/'prior_dev_entries.json')
    for seed in (42, 43, 44):
        actual_base = previous/f'seed{seed}/frozen/edit_plan.lock.json'
        supplied_base = package/f'plans/seed{seed}_base_oldfirst_plan.json'
        actual, supplied = [json.loads(p.read_text(encoding='utf-8')) for p in (actual_base, supplied_base)]
        # Commentary/provenance can differ; executable dispositions must not.
        for field in ('asset_dispositions', 'atomic_revision_jobs', 'new_atomic_jobs', 'program_realization_jobs',
                      'workflow_targets', 'identity_rebindings', 'merge_groups', 'manual_publications'):
            if actual[field] != supplied[field]:
                raise ReleaseError(f'actual/supplied base plan differs: seed{seed} {field}')
        plan = output/'plans'/f'seed{seed}.json'
        compile_plan(actual_base, package/f'plans/seed{seed}_preparation_delta.json', plan)
        config = copy.deepcopy(json.loads((previous/f'seed{seed}/prepare_manifest.json').read_text())['config'])
        config.pop('_config_path', None)
        config['runtime'].update(support_interface_version=SUPPORT_VERSION, preparation_coverage_version=COVERAGE_VERSION)
        config['harness']['public_discovery_version'] = PUBLIC_VERSION
        atomic_create_json(output/'configs'/f'seed{seed}_base.json', config)
        prepared = prepare_oldfirst(ReleaseSpec(seed, previous/f'seed{seed}/source/input.zip', output/f'seed{seed}', config), plan)
        checks = verify_oldfirst(prepared)
        freeze_release(prepared, checks)
        print(json.dumps({'seed': seed, 'stage': 'frozen', 'root': str(prepared.root)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-root', type=Path, required=True)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous_root, args.package_root, args.output_root)
