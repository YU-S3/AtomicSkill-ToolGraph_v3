from pathlib import Path
import pytest
from scripts.finalize_budget_report import checked_overlay


def test_report_overlay_allows_only_reviewed_change():
    fixed = (Path(__file__).resolve().parents[1]/'experiments/report.py').read_text()
    old = fixed.replace('        "budget_exhausted",\n', '', 1)
    checked_overlay(old, fixed)
    with pytest.raises(ValueError, match='beyond'):
        checked_overlay(old, fixed.replace('"aborted",', '"arbitrary_success",', 1))
    with pytest.raises(ValueError, match='already fixed'):
        checked_overlay(fixed, fixed)
