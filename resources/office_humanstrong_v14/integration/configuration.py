"""Explicit new submission protocol; original task/input/budget settings retained."""
import importlib.util
from pathlib import Path
from atomic_skillgraph.empirical.final_submission import VERSION as SUBMISSION

path=Path(__file__).resolve().parents[2]/'office_humanstrong_v13/integration/configuration.py'
spec=importlib.util.spec_from_file_location('office_v14_config_parent',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
verify_resources,TASK_TOKENS=base.verify_resources,base.TASK_TOKENS


def configure(config,resources,split):
    cfg=base.configure(config,resources,split)
    cfg['experiment']['office_human_bank_version']='office.humanstrong.submission-facts.v14.20261023'
    cfg['runtime']['final_submission_version']=SUBMISSION
    cfg['llm']['purpose_overrides']['finish_only']['protocol']['thinking_type']='enabled'
    return cfg
