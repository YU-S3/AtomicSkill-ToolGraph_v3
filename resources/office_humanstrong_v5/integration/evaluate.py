"""Reuse the native frozen evaluator, binding only this version's Bank and config."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


base = load('office_v5_native_evaluate', ROOT.parents[1]/'office_humanstrong_v2/integration/evaluate.py')
base.frontier = load('office_v5_evaluation_frontier', ROOT/'frontier.py')
base.configuration = load('office_v5_evaluation_configuration', ROOT/'configuration.py')
evaluate = base.evaluate
