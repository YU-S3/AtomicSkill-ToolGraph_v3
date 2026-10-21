"""Same public input, solver, tool and budget settings as the preceding comparison."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[2]/'office_humanstrong_v12/integration/configuration.py'
spec=importlib.util.spec_from_file_location('office_v13_config',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
verify_resources,TASK_TOKENS=base.verify_resources,base.TASK_TOKENS


def configure(config,resources,split):
    cfg=base.configure(config,resources,split)
    cfg['experiment']['office_human_bank_version']='office.humanstrong.final-display.v13.20261022'
    return cfg
