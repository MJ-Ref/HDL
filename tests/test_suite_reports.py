"""Tests for suite and M2 gate report helpers."""

from pathlib import Path

from scripts.analysis.generate_m2_gate_report import compute_gate_summary
from scripts.analysis.generate_suite_report import extract_rows


def test_extract_rows_parses_e1_summary(tmp_path: Path) -> None:
    artifact = tmp_path / "summary.json"
    artifact.write_text(
        """{
  "results": {
    "P0": {"success_rate": 0.2, "success_ci": [0.1, 0.3], "n_episodes": 50},
    "P1": {"success_rate": 0.7, "success_ci": [0.6, 0.8], "n_episodes": 50}
  }
}"""
    )
    manifest = {
        "commands": [
            {
                "model": "qwen_3b",
                "experiment": "E1",
                "artifacts": [str(artifact)],
            }
        ]
    }
    rows, warnings = extract_rows(manifest, tmp_path)
    assert len(rows) == 2
    assert not warnings
    metrics = {r["metric"] for r in rows}
    assert "P0.success_rate" in metrics
    assert "P1.success_rate" in metrics


def test_compute_gate_summary_ci_logic() -> None:
    eval_data = {
        "normal": {"successes": 45, "n_episodes": 100},
        "shuffle": {"successes": 2, "n_episodes": 100},
    }
    summary = compute_gate_summary(eval_data, p0_baseline=0.20)
    assert summary["available"] is True
    assert bool(summary["criteria"]["normal_gt_p0_ci_backed"]) is True
    assert bool(summary["criteria"]["shuffle_lt_p0_ci_backed"]) is True
    assert bool(summary["criteria"]["all_pass"]) is True
