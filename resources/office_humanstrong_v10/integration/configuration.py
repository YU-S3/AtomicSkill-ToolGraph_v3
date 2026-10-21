"""Public inputs, inference and budgets remain identical across both arms."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[2]/'office_humanstrong_v8/integration/configuration.py'
spec = importlib.util.spec_from_file_location('office_v10_config', path)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
verify_resources, TASK_TOKENS = base.verify_resources, base.TASK_TOKENS


def configure(config, resources, split):
    cfg = base.configure(config, resources, split)
    cfg['experiment']['office_human_bank_version'] = 'office.humanstrong.scoped.v10.20261021'
    return cfg
