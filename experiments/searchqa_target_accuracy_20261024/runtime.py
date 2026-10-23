"""Frozen external HumanFrontier Test1400, with native scoring and cumulative billing."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
SOURCE = Path('/home/yangchengyu/asg_searchqa_gemini38_release_20261023')
ROOT = Path('/home/yangchengyu/searchqa_human_frontier_test1400_seed42_20261015')
PACKAGE = Path('/mnt/d/T3S_exp/resources/searchqa_human_frontier_20261015/SearchQA_HumanFrontier_Bank_20261015')
ZIP = Path('/mnt/d/Download/Edge/SearchQA_HumanFrontier_Bank_20261015.zip')
PARENT = Path('/home/yangchengyu/searchqa_manual_v2_test1400_seed42_20261015')
R3 = Path('/home/yangchengyu/searchqa_trace_r3_seed42_sources_20261015')
AUTHORITY = Path('/home/yangchengyu/searchqa_trace_r2_seed42_20261014_a418866')
DATASETS = Path('/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1/searchqa')
os.chdir(SOURCE); sys.path.insert(0, str(SOURCE / 'src'))
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.system import validate_config
from atomic_skillgraph.experiments.canonical_manifest import verify
from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity, utc
from atomic_skillgraph.experiments.run_empirical import code_identity, load_env, run, write_json
from atomic_skillgraph.experiments.run_searchqa_reference_test import status

read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bridge(package):
    path = package / 'integration/frontier.py'
    spec = importlib.util.spec_from_file_location('external_frontier_' + sha(path), path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
