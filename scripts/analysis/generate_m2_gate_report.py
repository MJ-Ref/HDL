#!/usr/bin/env python3
"""
Generate CI-backed M2 gate report from full-suite manifest artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lpca.core.metrics import MetricsCalculator


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def wilson_from_counts(successes: int, n: int) -> Tuple[float, float]:
    calc = MetricsCalculator()
    return calc.wilson_ci(successes, n)


def infer_counts(condition: Dict[str, Any]) -> Tuple[int, int]:
    n = int(condition.get("n_episodes", 0))
    if "successes" in condition:
        successes = int(condition["successes"])
    else:
        successes = int(round(float(condition.get("success_rate", 0.0)) * n))
    return successes, n


def compute_gate_summary(
    eval_data: Dict[str, Any],
    p0_baseline: float,
) -> Dict[str, Any]:
    normal = eval_data.get("normal", {})
    shuffle = eval_data.get("shuffle", {})
    if not isinstance(normal, dict) or not isinstance(shuffle, dict):
        return {"available": False, "reason": "normal/shuffle conditions missing"}

    normal_successes, normal_n = infer_counts(normal)
    shuffle_successes, shuffle_n = infer_counts(shuffle)
    if normal_n == 0 or shuffle_n == 0:
        return {"available": False, "reason": "n_episodes is zero"}

    normal_ci = wilson_from_counts(normal_successes, normal_n)
    shuffle_ci = wilson_from_counts(shuffle_successes, shuffle_n)
    p0_successes = int(round(p0_baseline * normal_n))
    p0_ci = wilson_from_counts(p0_successes, normal_n)

    normal_gt_p0 = normal_ci[0] > p0_ci[1]
    shuffle_lt_p0 = shuffle_ci[1] < p0_ci[0]

    return {
        "available": True,
        "normal": {
            "successes": normal_successes,
            "n": normal_n,
            "ci": list(normal_ci),
        },
        "shuffle": {
            "successes": shuffle_successes,
            "n": shuffle_n,
            "ci": list(shuffle_ci),
        },
        "p0_reference": {
            "baseline": p0_baseline,
            "successes": p0_successes,
            "n": normal_n,
            "ci": list(p0_ci),
        },
        "criteria": {
            "normal_gt_p0_ci_backed": normal_gt_p0,
            "shuffle_lt_p0_ci_backed": shuffle_lt_p0,
            "all_pass": normal_gt_p0 and shuffle_lt_p0,
        },
    }


def extract_m2_gate_rows(
    manifest: Dict[str, Any],
    p0_baseline: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for command in manifest.get("commands", []):
        if command.get("experiment") != "M2":
            continue
        model = command.get("model", "unknown")
        seed_set = command.get("seed_set", "primary")
        for artifact in command.get("artifacts", []):
            artifact_path = Path(artifact)
            if not artifact_path.exists():
                warnings.append(f"Missing artifact: {artifact}")
                continue
            data = load_json(artifact_path)
            if "eval" in data and isinstance(data["eval"], dict):
                eval_data = data["eval"]
            elif "normal" in data and isinstance(data["normal"], dict):
                eval_data = data
            else:
                warnings.append(f"Unsupported M2 artifact format: {artifact}")
                continue

            row = {
                "model": model,
                "seed_set": seed_set,
                "artifact": str(artifact_path),
                "summary": compute_gate_summary(eval_data, p0_baseline),
                "existing_gates": data.get("gates"),
            }
            rows.append(row)

    return rows, warnings


def render_markdown(report: Dict[str, Any], repo_root: Path) -> str:
    lines: List[str] = []
    lines.append("# M2 Gate Report")
    lines.append("")
    lines.append(f"Run ID: `{report.get('run_id', 'unknown')}`")
    lines.append("")
    lines.append(
        "| Model | Seed Set | Normal CI | Shuffle CI | P0 CI | Normal>P0 | Shuffle<P0 | All Pass | Artifact |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for row in report.get("rows", []):
        summary = row.get("summary", {})
        if not summary.get("available"):
            lines.append(
                f"| {row.get('model')} | {row.get('seed_set', 'primary')} | - | - | - | - | - | - | {row.get('artifact')} |"
            )
            continue
        normal_ci = summary["normal"]["ci"]
        shuffle_ci = summary["shuffle"]["ci"]
        p0_ci = summary["p0_reference"]["ci"]
        crit = summary["criteria"]
        artifact_path = Path(row["artifact"])
        artifact_rel = (
            str(artifact_path.relative_to(repo_root))
            if artifact_path.is_absolute()
            and str(artifact_path).startswith(str(repo_root))
            else str(artifact_path)
        )
        lines.append(
            f"| {row['model']} | "
            f"{row.get('seed_set', 'primary')} | "
            f"[{normal_ci[0]:.3f}, {normal_ci[1]:.3f}] | "
            f"[{shuffle_ci[0]:.3f}, {shuffle_ci[1]:.3f}] | "
            f"[{p0_ci[0]:.3f}, {p0_ci[1]:.3f}] | "
            f"{crit['normal_gt_p0_ci_backed']} | "
            f"{crit['shuffle_lt_p0_ci_backed']} | "
            f"{crit['all_pass']} | "
            f"{artifact_rel} |"
        )

    warnings = report.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CI-backed M2 gate report")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--p0-baseline", type=float, default=0.20)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    run_dir = manifest_path.parent
    repo_root = Path(__file__).resolve().parents[2]
    manifest = load_json(manifest_path)

    rows, warnings = extract_m2_gate_rows(manifest, args.p0_baseline)
    report = {
        "run_id": manifest.get("run_id"),
        "suite_name": manifest.get("suite_name"),
        "p0_baseline": args.p0_baseline,
        "rows": rows,
        "warnings": warnings,
    }

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else run_dir / "m2_gate_report.json"
    )
    output_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else run_dir / "m2_gate_report.md"
    )

    output_json.write_text(json.dumps(report, indent=2))
    output_md.write_text(render_markdown(report, repo_root))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
