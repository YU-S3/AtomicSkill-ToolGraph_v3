import json
from pathlib import Path
from experiments.run_skillcompiler_pilot import select


def test_training_selection_is_source_order_two_per_family():
    path = Path(__file__).resolve().parents[1] / 'data/baseline_manifests/train_120.json'
    chosen, audit = select(path, 'train', 2)
    assert len(chosen) == 12 and set(audit['counts'].values()) == {2}
    source = json.loads(path.read_text(encoding='utf-8'))['tasks']
    assert chosen == [row for row in source if row['task_id'] in {x['task_id'] for x in chosen}]
