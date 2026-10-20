"""Versioned V4 Bank; all inference runs through the native Executor."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path

path=Path(__file__).resolve().parents[2]/'office_humanstrong_v3/integration/frontier.py'
spec=importlib.util.spec_from_file_location('office_v4_native_bridge',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VERSION='office.humanstrong.programmatic.v4.20261020'
base.base.VERSION=VERSION
previous=base.base.selection.callable_programs
base.base.selection.callable_programs=lambda *a,**kw:previous(*a,**kw)+['table_calculate.py']
KINDS=base.KINDS
HumanBankView=base.HumanBankView
native_import=base.native_import
input_protocol=base.input_protocol

# Shared output contract, also sent to NoSkill. It contains no target answer.
ANSWER_CONTRACT=('Return only the requested answer. For a numeric result, output the '
    'number at the requested scale and precision; add a unit or percent-sign suffix '
    'only if an explicit answer template requires it. For dates or lists follow '
    'the requested form. Do not add explanation.')


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    with base.bridge():
        original=EmpiricalSystem.agent
        original_finish=EmpiricalSystem.prepare_finish
        def agent(s,stage,prompt,*args,**kwargs):
            if isinstance(s.bank,HumanBankView) and stage=='runtime':
                prompt=ANSWER_CONTRACT+'\n'+prompt
            return original(s,stage,prompt,*args,**kwargs)
        def prepare_finish(s,materials,prompt,**kwargs):
            if isinstance(s.bank,HumanBankView):
                prompt=ANSWER_CONTRACT+'\n'+prompt
            return original_finish(s,materials,prompt,**kwargs)
        EmpiricalSystem.agent=agent
        EmpiricalSystem.prepare_finish=prepare_finish
        try:yield
        finally:
            EmpiricalSystem.agent=original
            EmpiricalSystem.prepare_finish=original_finish
