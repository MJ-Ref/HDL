"""Tests for Modal baseline experiment launcher helpers."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "modal" / "run_baseline_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("run_baseline_experiment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_experiment_command = MODULE.build_experiment_command
find_artifact = MODULE.find_artifact


@pytest.mark.parametrize(
    "experiment,script_name,expected_flag",
    [
        ("E1", "run_llm_experiment.py", "--protocols"),
        ("E2", "run_e2_sweep.py", "--protocols"),
        ("E3", "run_cipher_experiment.py", "--top_k"),
        ("E4", "run_activation_experiment.py", "--combine"),
    ],
)
def test_build_experiment_command_shape(
    experiment: str, script_name: str, expected_flag: str
) -> None:
    cmd = build_experiment_command(
        experiment=experiment,
        model="Qwen/Qwen2.5-3B-Instruct",
        base_seed=1000,
        n_episodes=100,
        output_dir="/tmp/out",
    )
    assert cmd[0] == "python"
    assert script_name in cmd[1]
    assert expected_flag in cmd
    assert "/tmp/out" in cmd


def test_find_artifact_selects_latest_match(tmp_path: Path) -> None:
    older_dir = tmp_path / "cipher_e0_20250101_000000"
    newer_dir = tmp_path / "cipher_e0_20250101_000001"
    older_dir.mkdir()
    newer_dir.mkdir()
    (older_dir / "summary.json").write_text(json.dumps({"id": "old"}))
    (newer_dir / "summary.json").write_text(json.dumps({"id": "new"}))

    artifact = find_artifact("E3", tmp_path)
    assert artifact == newer_dir / "summary.json"


def test_find_artifact_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_artifact("E4", tmp_path)
