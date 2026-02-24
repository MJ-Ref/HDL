"""Tests for PaperBanana figure pack generator helpers."""

from scripts.analysis.generate_paperbanana_figure_pack import build_figure_briefs


def test_build_figure_briefs_with_metrics_and_gates() -> None:
    manifest = {
        "run_id": "run_abc",
        "models": ["qwen_3b"],
        "stages_completed": ["prepare", "run", "aggregate", "gate", "publish"],
        "commands": [],
    }
    suite_report = {
        "rows": [
            {
                "model": "qwen_3b",
                "metric": "P0.success_rate",
                "success_rate": 0.2,
            },
            {
                "model": "qwen_3b",
                "metric": "P1.success_rate",
                "success_rate": 0.45,
            },
        ]
    }
    gate_report = {
        "rows": [
            {
                "model": "qwen_3b",
                "summary": {
                    "normal": {"successes": 30, "n": 100},
                    "shuffle": {"successes": 8, "n": 100},
                    "criteria": {
                        "normal_gt_p0_ci_backed": True,
                        "shuffle_lt_p0_ci_backed": True,
                        "all_pass": True,
                    },
                },
            }
        ]
    }

    briefs = build_figure_briefs(manifest, suite_report, gate_report)

    assert len(briefs) == 3
    gate_brief = next(b for b in briefs if b.figure_id == "fig2_gate_evidence")
    assert "qwen_3b: P0=20.0%, P1=45.0%" in gate_brief.content
    assert "all_pass=True" in gate_brief.content


def test_build_figure_briefs_handles_missing_reports() -> None:
    manifest = {
        "run_id": "run_empty",
        "models": ["qwen_3b"],
        "stages_completed": ["prepare"],
        "commands": [],
    }
    briefs = build_figure_briefs(manifest, None, None)
    gate_brief = next(b for b in briefs if b.figure_id == "fig2_gate_evidence")
    assert "No suite rows available" in gate_brief.content
    assert "Gate report unavailable for this run." in gate_brief.content
