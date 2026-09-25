"""Benchmark identity/selection authority, separate from the task execution loop."""
from dataclasses import dataclass
from pathlib import Path
from atomic_skillgraph.harness.scienceworld_resource import verify_resource
from .scienceworld_manifest import load, task_from_entry

@dataclass(frozen=True)
class BenchmarkExperimentProtocol:
    benchmark: str
    manifest: dict

    @classmethod
    def scienceworld(cls, path):
        manifest = load(Path(path))
        verify_resource(manifest['resource_identity'])
        return cls('scienceworld', manifest)

    def selected(self, *, diagnostic_macros=False):
        rows = list(self.manifest['tasks'])
        if not diagnostic_macros:
            return rows
        # Fixed diagnostic subset, never chosen based on observed results.
        return list({r['macro_type']: next(t for t in rows if t['macro_type'] == r['macro_type'])
                     for r in rows}.values())

    def task(self, entry, harness):
        harness.initialize()
        harness._env.load(entry['task_name'], entry['variation_idx'], 'easy', generateGoldPath=False)
        _, info = harness._env.reset()
        return task_from_entry(entry, info['taskDesc'])

    def result(self, trace):
        if trace.official_score is None:
            raise RuntimeError('ScienceWorld Trace has no official final score')
        return {'official_score': trace.official_score, 'normalized_score': trace.official_score / 100.0,
                'perfect_success': trace.official_score == 100.0, 'environment_done': trace.environment_done,
                'benchmark_success': trace.benchmark_success,
                'infrastructure_failure': trace.infrastructure_failure}
