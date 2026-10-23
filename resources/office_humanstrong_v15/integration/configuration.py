"""New immutable Bank and list-submission version, with unchanged task limits."""
import importlib.util
from pathlib import Path
from atomic_skillgraph.empirical.final_submission import LIST_VERSION

path=Path(__file__).resolve().parents[2]/'office_humanstrong_v14/integration/configuration.py'
spec=importlib.util.spec_from_file_location('office_v15_config_parent',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
verify_resources,TASK_TOKENS=base.verify_resources,base.TASK_TOKENS


def configure(config,resources,split):
    cfg=base.configure(config,resources,split)
    cfg['experiment']['office_human_bank_version']='office.humanstrong.operation-bound.v15.20261024'
    cfg['runtime']['final_submission_version']=LIST_VERSION
    return cfg
