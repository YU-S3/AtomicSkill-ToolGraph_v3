from __future__ import annotations

from pathlib import Path

from experiments.baselines.common.source_identity import hash_code


def test_hash_code_ignores_all_virtual_environment_variants(tmp_path: Path) -> None:
    source = tmp_path / "experiments" / "driver.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = hash_code(tmp_path)

    for name in (
        ".venv",
        ".venv_b3_skillopt",
        ".venv_b4_skillgen",
        ".venv_b5_gepa_polluted_20260912",
    ):
        runtime_file = tmp_path / name / "lib" / "runtime.py"
        runtime_file.parent.mkdir(parents=True)
        runtime_file.write_text(f"RUNTIME = {name!r}\n", encoding="utf-8")

    assert hash_code(tmp_path) == before


def test_hash_code_changes_for_experiment_source(tmp_path: Path) -> None:
    source = tmp_path / "experiments" / "driver.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = hash_code(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert hash_code(tmp_path) != before
