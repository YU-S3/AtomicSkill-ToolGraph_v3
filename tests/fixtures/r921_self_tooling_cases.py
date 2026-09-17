"""R9.2.1 provider-boundary fixtures shared with the isolated live driver."""
from experiments.self_tooling_targeted import (
    CORE_TWO, CandidateHarness, RouteCase, RouteProvider, fixed_draft,
    fixture_task, install_parent, make_plan, parent_atomic, run_node_case, scripted_proposal,
)


def fixture_config(tmp_path):
    import yaml
    from pathlib import Path
    config = yaml.safe_load(Path("configs/alfworld_train_full_120_r102_seed42.yaml").read_text())
    config["data_dir"] = str(tmp_path / "bank")
    config["trace_data_dir"] = str(tmp_path / "traces")
    config["experiment"] = {**config["experiment"], "output_dir": str(tmp_path / "run")}
    return config
