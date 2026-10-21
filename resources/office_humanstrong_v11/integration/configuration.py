"""The same public inputs, model settings and budgets as the frozen V10 control."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[2]/'office_humanstrong_v10/integration/configuration.py'
spec=importlib.util.spec_from_file_location('office_v11_config',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
verify_resources,TASK_TOKENS=base.verify_resources,base.TASK_TOKENS


def configure(config,resources,split):
    cfg=base.configure(config,resources,split)
    cfg['experiment']['office_human_bank_version']='office.humanstrong.source-final.v11.20261021'
    return cfg
