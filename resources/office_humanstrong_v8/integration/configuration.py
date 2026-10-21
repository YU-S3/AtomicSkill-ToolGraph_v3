"""Keep approved external inputs, model and budgets unchanged."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[2]/'office_humanstrong_v2/integration/configuration.py'
spec = importlib.util.spec_from_file_location('office_v8_config', path)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
verify_resources, TASK_TOKENS = base.verify_resources, base.TASK_TOKENS


def configure(config, resources, split):
    cfg = base.configure(config, resources, split)
    cfg['experiment'].update(office_human_bank_version='office.humanstrong.sparse.v8.20261021',
                             shared_answer_contract='office.explicit-output-contract.v2')
    return cfg
