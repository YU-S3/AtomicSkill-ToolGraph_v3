"""SkillCompiler empirical API."""
__version__ = "3.1.0"

def __getattr__(name):
    if name == "EmpiricalSystem":
        from .empirical.system import EmpiricalSystem
        return EmpiricalSystem
    raise AttributeError(name)
