"""Tests for publication-facing report/package generators."""

from pathlib import Path

import pytest

from scripts.analysis.build_repro_package import (
    build_reproduce_script,
    normalize_copy_plan,
)
from scripts.analysis.generate_paper_pack import (
    build_main_table_rows,
    extract_gate_status,
)


def test_build_main_table_rows_uses_best_e2_and_gate() -> None:
    suite_rows = [
        {
            "model": "qwen_3b",
            "experiment": "E1",
            "metric": "P0.success_rate",
            "success_rate": 0.2,
            "success_ci": [0.12, 0.31],
            "n_episodes": 100,
        },
        {
            "model": "qwen_3b",
            "experiment": "E1",
            "metric": "P1.success_rate",
            "success_rate": 0.42,
            "success_ci": [0.33, 0.52],
            "n_episodes": 100,
        },
        {
            "model": "qwen_3b",
            "experiment": "E2",
            "metric": "P2_64B.success_rate",
            "success_rate": 0.36,
            "success_ci": [0.27, 0.46],
            "n_episodes": 100,
        },
        {
            "model": "qwen_3b",
            "experiment": "E2",
            "metric": "P5_64B.success_rate",
            "success_rate": 0.58,
            "success_ci": [0.48, 0.67],
            "n_episodes": 100,
        },
    ]
    gate_report = {
        "rows": [
            {
                "model": "qwen_3b",
                "summary": {
                    "available": True,
                    "criteria": {
                        "normal_gt_p0_ci_backed": True,
                        "shuffle_lt_p0_ci_backed": True,
                        "all_pass": True,
                    },
                },
            }
        ]
    }
    gate_status = extract_gate_status(gate_report)
    rows = build_main_table_rows(suite_rows, gate_status)
    assert len(rows) == 1
    row = rows[0]
    assert row["e2_best_metric"] == "P5_64B.success_rate"
    assert row["e2_best_success"] == 0.58
    assert row["e1_delta_p1_minus_p0"] == pytest.approx(0.22)
    assert row["m2_gate_all_pass"] is True


def test_normalize_copy_plan_includes_expected_targets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "outputs" / "full_suite" / "run_abc123"
    run_dir.mkdir(parents=True)
    manifest = {
        "commands": [
            {"log_path": "logs/global/aggregate.log"},
            {"log_path": "logs/global/paper.log"},
        ]
    }

    copy_plan = normalize_copy_plan(run_dir, repo_root, manifest)
    destinations = {str(dst) for _, dst in copy_plan}
    assert "run/manifest.json" in destinations
    assert "reports/suite_report.json" in destinations
    assert "scripts/analysis/generate_paper_pack.py" in destinations
    assert "run/logs/global/aggregate.log" in destinations
    assert "run/logs/global/paper.log" in destinations


def test_build_reproduce_script_contains_report_commands() -> None:
    script = build_reproduce_script("run_xyz")
    assert "generate_suite_report.py" in script
    assert "generate_paper_pack.py" in script
    assert "run_xyz" in script
