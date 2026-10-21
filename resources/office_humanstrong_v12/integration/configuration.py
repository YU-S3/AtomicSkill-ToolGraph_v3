"""Keep V11 inference settings and approved external public inputs."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[2]/'office_humanstrong_v11/integration/configuration.py'
spec=importlib.util.spec_from_file_location('office_v12_config',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
verify_resources,TASK_TOKENS=base.verify_resources,base.TASK_TOKENS


def configure(config,resources,split):
    cfg=base.configure(config,resources,split)
    cfg['experiment']['office_human_bank_version']='office.humanstrong.atomic-directions.v12.20261022'
    return cfg
