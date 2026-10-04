"""The single supported production factory."""
from pathlib import Path
import yaml
from .empirical.system import EmpiricalSystem, validate_config


def load_config(value):
    config = value if isinstance(value, dict) else yaml.safe_load(Path(value).read_text(encoding='utf-8'))
    return validate_config(config)


def create_system(config, **kwargs):
    return EmpiricalSystem(load_config(config), **kwargs)
