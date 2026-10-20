"""Same aligned public inputs and finish capacity for both V3 observation arms."""
import importlib.util
from pathlib import Path

path=Path(__file__).resolve().parents[2]/'office_humanstrong_v2/integration/configuration.py'
spec=importlib.util.spec_from_file_location('office_v3_shared_configuration',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
verify_resources=base.verify_resources
TASK_TOKENS=base.TASK_TOKENS


def configure(config,resources,split):
    cfg=base.configure(config,resources,split)
    cfg['experiment']['office_human_bank_version']='office.humanstrong.programmatic.v3.20261020'
    return cfg
