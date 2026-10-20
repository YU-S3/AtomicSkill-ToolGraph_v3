"""Reuse the native V2 bridge with a separately loaded, versioned Bank view."""
import importlib.util
from pathlib import Path

path=Path(__file__).resolve().parents[2]/'office_humanstrong_v2/integration/frontier.py'
spec=importlib.util.spec_from_file_location('office_v3_native_bridge',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VERSION='office.humanstrong.programmatic.v3.20261020'
base.VERSION=VERSION
previous=base.selection.callable_programs
base.selection.callable_programs=lambda *a,**kw:previous(*a,**kw)+['polynomial_forecast.py']
KINDS=base.KINDS
HumanBankView=base.HumanBankView
native_import=base.native_import
bridge=base.bridge
input_protocol=base.input_protocol
