#!/usr/bin/env python3
"""
Generate a markdown artifact index for headline LPCA results.

Writes docs/ARTIFACT_INDEX.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_latest(pattern: str, repo_root: Path) -> Optional[Path]:
    matches = sorted(repo_root.glob(pattern))
    return matches[-1] if matches else None


def fmt_ci(ci: List[float]) -> str:
    if not ci or len(ci) < 2:
        return "-"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def extract_e1(repo_root: Path) -> Tuple[List[str], List[List[str]]]:
    headers = ["Experiment", "Metric", "Value", "95% CI", "N", "Artifact"]
    rows: List[List[str]] = []

    summary_path = find_latest("results/e1_tightened/*/summary.json", repo_root)
    if summary_path is None:
        return headers, rows

    data = load_json(summary_path)
    results = data.get("results", {})
    for key in ["P0", "P1"]:
        if key not in results:
            continue
        r = results[key]
        rows.append(
            [
                "E1 Tightened",
                f"{key} success_rate",
                f"{r.get('success_rate', 0):.3f}",
                fmt_ci(r.get("success_ci", [])),
                str(r.get("n_episodes", "-")),
                str(summary_path.relative_to(repo_root)),
            ]
        )
    return headers, rows


def extract_e2(repo_root: Path) -> Tuple[List[str], List[List[str]]]:
    headers = ["Experiment", "Config", "Success", "95% CI", "N", "Artifact"]
    rows: List[List[str]] = []

    results_path = find_latest("results/e2_sweep_*/results.json", repo_root)
    if results_path is None:
        return headers, rows

    data = load_json(results_path)
    entries = data.get("results", [])
    wanted = {"P2_16B", "P2_64B", "P2_256B", "P5_16B", "P5_64B", "P5_256B"}
    for entry in entries:
        cfg = entry.get("config_str")
        if cfg not in wanted:
            continue
        rows.append(
            [
                "E2-min",
                cfg,
                f"{entry.get('success_rate', 0):.3f}",
                fmt_ci(entry.get("success_ci", [])),
                str(entry.get("n_episodes", "-")),
                str(results_path.relative_to(repo_root)),
            ]
        )
    return headers, rows


def extract_e3_e4(repo_root: Path) -> Tuple[List[str], List[List[str]]]:
    headers = ["Experiment", "Metric", "Value", "95% CI", "N", "Artifact"]
    rows: List[List[str]] = []

    e3_path = find_latest("results/cipher_e0_*/summary.json", repo_root)
    if e3_path is not None:
        data = load_json(e3_path)
        e3 = data.get("results", {}).get("E0", {})
        rows.append(
            [
                "E3 CIPHER",
                "E0 success_rate",
                f"{e3.get('success_rate', 0):.3f}",
                fmt_ci(e3.get("success_ci", [])),
                str(e3.get("n_episodes", "-")),
                str(e3_path.relative_to(repo_root)),
            ]
        )

    e4_path = find_latest("results/activation_grafting_*/summary.json", repo_root)
    if e4_path is not None:
        data = load_json(e4_path)
        e4 = data.get("results", {}).get("main", {})
        rows.append(
            [
                "E4 Activation",
                "A0 success_rate",
                f"{e4.get('success_rate', 0):.3f}",
                fmt_ci(e4.get("success_ci", [])),
                str(e4.get("n_episodes", "-")),
                str(e4_path.relative_to(repo_root)),
            ]
        )
    return headers, rows


def extract_m2(repo_root: Path) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "Experiment",
        "Status",
        "Normal",
        "Null",
        "Random",
        "Shuffle",
        "Artifact",
    ]
    rows: List[List[str]] = []
    attempt_path = repo_root / "docs/experiments/gate1-attempt-02.md"
    if not attempt_path.exists():
        return headers, rows
    rows.append(
        [
            "M2-SCALE Gate 1",
            "FAILED",
            "22.0% (11/50)",
            "24.0% (12/50)",
            "28.0% (14/50)",
            "32.0% (16/50)",
            str(attempt_path.relative_to(repo_root)),
        ]
    )
    return headers, rows


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_No artifacts found._"
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(repo_root: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines.append("# LPCA Artifact Index")
    lines.append("")
    lines.append(f"Generated (UTC): {now}")
    lines.append("")
    lines.append(
        "This file maps headline numbers to concrete artifact files for traceability."
    )
    lines.append("")

    sections = [
        ("E1 Tightened Baselines", extract_e1(repo_root)),
        ("E2-min Budgeted Text Sweep", extract_e2(repo_root)),
        ("E3/E4 Latent Baselines", extract_e3_e4(repo_root)),
        ("M2 Gate Attempts", extract_m2(repo_root)),
    ]
    for title, (headers, rows) in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(render_table(headers, rows))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Some headline docs reference larger sample sizes than currently indexed "
        "result files for E3/E4; this index reflects files present in this repository."
    )
    lines.append("- Re-run this script after new experiments:")
    lines.append("  - `python scripts/analysis/generate_artifact_index.py`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "docs" / "ARTIFACT_INDEX.md"
    output_path.write_text(build_markdown(repo_root))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
