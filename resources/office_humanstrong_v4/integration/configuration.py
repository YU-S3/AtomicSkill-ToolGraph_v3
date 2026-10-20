"""Frozen external input snapshot and unchanged task/finish caps."""
import importlib.util
from pathlib import Path

path=Path(__file__).resolve().parents[2]/'office_humanstrong_v3/integration/configuration.py'
spec=importlib.util.spec_from_file_location('office_v4_shared_configuration',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
verify_resources=base.verify_resources
TASK_TOKENS=base.TASK_TOKENS


def configure(config,resources,split):
    cfg=base.configure(config,resources,split)
    cfg['experiment']['office_human_bank_version']='office.humanstrong.programmatic.v4.20261020'
    cfg['experiment']['shared_answer_contract']='office.scalar-or-explicit-template.v1'
    return cfg
