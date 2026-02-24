#!/usr/bin/env python3
"""
Generate a preregistered-stats-only suite report from full-suite manifest artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def parse_ci(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, list) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def row(
    model: str,
    seed_set: str,
    experiment: str,
    metric: str,
    success_rate: float,
    success_ci: Optional[Tuple[float, float]],
    n_episodes: Optional[int],
    artifact: str,
) -> Dict[str, Any]:
    return {
        "model": model,
        "seed_set": seed_set,
        "experiment": experiment,
        "metric": metric,
        "success_rate": float(success_rate),
        "success_ci": list(success_ci) if success_ci else None,
        "n_episodes": int(n_episodes) if n_episodes is not None else None,
        "artifact": artifact,
    }


def parse_e1(
    model: str, seed_set: str, artifact_path: Path, rel_artifact: str
) -> List[Dict[str, Any]]:
    data = load_json(artifact_path)
    results = data.get("results", {})
    rows: List[Dict[str, Any]] = []
    for protocol in ("P0", "P1"):
        protocol_result = results.get(protocol)
        if not isinstance(protocol_result, dict):
            continue
        rows.append(
            row(
                model=model,
                seed_set=seed_set,
                experiment="E1",
                metric=f"{protocol}.success_rate",
                success_rate=protocol_result.get("success_rate", 0.0),
                success_ci=parse_ci(protocol_result.get("success_ci")),
                n_episodes=protocol_result.get("n_episodes"),
                artifact=rel_artifact,
            )
        )
    return rows


def parse_e2(
    model: str, seed_set: str, artifact_path: Path, rel_artifact: str
) -> List[Dict[str, Any]]:
    data = load_json(artifact_path)
    entries = data.get("results", [])
    rows: List[Dict[str, Any]] = []
    if not isinstance(entries, list):
        return rows
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config_str", "unknown")
        rows.append(
            row(
                model=model,
                seed_set=seed_set,
                experiment="E2",
                metric=f"{cfg}.success_rate",
                success_rate=entry.get("success_rate", 0.0),
                success_ci=parse_ci(entry.get("success_ci")),
                n_episodes=entry.get("n_episodes"),
                artifact=rel_artifact,
            )
        )
    return rows


def parse_e3(
    model: str, seed_set: str, artifact_path: Path, rel_artifact: str
) -> List[Dict[str, Any]]:
    data = load_json(artifact_path)
    e0 = data.get("results", {}).get("E0")
    if not isinstance(e0, dict):
        return []
    return [
        row(
            model=model,
            seed_set=seed_set,
            experiment="E3",
            metric="E0.success_rate",
            success_rate=e0.get("success_rate", 0.0),
            success_ci=parse_ci(e0.get("success_ci")),
            n_episodes=e0.get("n_episodes"),
            artifact=rel_artifact,
        )
    ]


def parse_e4(
    model: str, seed_set: str, artifact_path: Path, rel_artifact: str
) -> List[Dict[str, Any]]:
    data = load_json(artifact_path)
    main = data.get("results", {}).get("main")
    if not isinstance(main, dict):
        return []
    return [
        row(
            model=model,
            seed_set=seed_set,
            experiment="E4",
            metric="A0.success_rate",
            success_rate=main.get("success_rate", 0.0),
            success_ci=parse_ci(main.get("success_ci")),
            n_episodes=main.get("n_episodes"),
            artifact=rel_artifact,
        )
    ]


PARSERS = {
    "E1": parse_e1,
    "E2": parse_e2,
    "E3": parse_e3,
    "E4": parse_e4,
}


def extract_rows(
    manifest: Dict[str, Any], repo_root: Path
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for command in manifest.get("commands", []):
        model = command.get("model", "unknown")
        seed_set = command.get("seed_set", "primary")
        experiment = command.get("experiment", "")
        if experiment not in PARSERS:
            continue
        parser = PARSERS[experiment]
        for artifact in command.get("artifacts", []):
            artifact_path = Path(artifact)
            if not artifact_path.exists():
                warnings.append(f"Missing artifact: {artifact}")
                continue
            rel = (
                str(artifact_path.relative_to(repo_root))
                if artifact_path.is_absolute()
                and str(artifact_path).startswith(str(repo_root))
                else str(artifact_path)
            )
            parsed = parser(model, seed_set, artifact_path, rel)
            if not parsed:
                warnings.append(f"No parsable rows from {artifact} for {experiment}")
            rows.extend(parsed)
    return rows, warnings


def preregistered_stats_check(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_ci = [r for r in rows if r.get("success_ci") is None]
    return {
        "total_rows": len(rows),
        "rows_with_ci": len(rows) - len(missing_ci),
        "rows_missing_ci": len(missing_ci),
        "all_rows_have_ci": len(missing_ci) == 0,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Full Suite Report")
    lines.append("")
    lines.append(f"Run ID: `{report.get('run_id', 'unknown')}`")
    lines.append("")
    stats_check = report.get("checks", {}).get("preregistered_stats", {})
    lines.append("## Preregistered Stats Check")
    lines.append("")
    lines.append(f"- Total rows: {stats_check.get('total_rows', 0)}")
    lines.append(f"- Rows with CI: {stats_check.get('rows_with_ci', 0)}")
    lines.append(f"- Rows missing CI: {stats_check.get('rows_missing_ci', 0)}")
    lines.append(f"- All rows have CI: `{stats_check.get('all_rows_have_ci', False)}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "| Model | Seed Set | Experiment | Metric | Success | 95% CI | N | Artifact |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in report.get("rows", []):
        ci = r.get("success_ci")
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        n = r.get("n_episodes", "-")
        lines.append(
            f"| {r['model']} | {r.get('seed_set', 'primary')} | "
            f"{r['experiment']} | {r['metric']} | "
            f"{r['success_rate']:.3f} | {ci_str} | {n} | {r['artifact']} |"
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
    parser = argparse.ArgumentParser(
        description="Generate full-suite report from manifest"
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    run_dir = manifest_path.parent
    repo_root = Path(__file__).resolve().parents[2]

    manifest = load_json(manifest_path)
    rows, warnings = extract_rows(manifest, repo_root)
    checks = {"preregistered_stats": preregistered_stats_check(rows)}
    report = {
        "suite_name": manifest.get("suite_name"),
        "run_id": manifest.get("run_id"),
        "manifest_path": str(manifest_path),
        "rows": rows,
        "checks": checks,
        "warnings": warnings,
    }

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else run_dir / "suite_report.json"
    )
    output_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else run_dir / "suite_report.md"
    )

    output_json.write_text(json.dumps(report, indent=2))
    output_md.write_text(render_markdown(report))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
