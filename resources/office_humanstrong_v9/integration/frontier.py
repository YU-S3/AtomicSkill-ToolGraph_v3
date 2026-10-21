"""One specific principle after V8 pruning; preserve the sparse native bridge."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('office_v9_sparse_bridge', ROOT.parent/'office_humanstrong_v8/integration/frontier.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
VERSION = 'office.humanstrong.sparse.v9.20261021'
base.VERSION = base.base.VERSION = VERSION
base.SPECS = SPECS = json.loads((ROOT/'principles.json').read_text())
ANSWER_CONTRACT = base.ANSWER_CONTRACT
KINDS, native_import, input_protocol = base.KINDS, base.native_import, base.input_protocol
HumanBankView, bridge, matching = base.HumanBankView, base.bridge, base.matching
