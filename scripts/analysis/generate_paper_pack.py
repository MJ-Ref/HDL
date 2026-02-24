#!/usr/bin/env python3
"""
Generate submission-facing tables and figure-ready data from suite artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def rows_by_model(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        grouped.setdefault(model, []).append(row)
    return grouped


def lookup_metric(
    model_rows: List[Dict[str, Any]],
    experiment: str,
    metric: str,
) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[int]]:
    for row in model_rows:
        if row.get("experiment") != experiment or row.get("metric") != metric:
            continue
        rate = row.get("success_rate")
        n = row.get("n_episodes")
        if rate is None:
            continue
        return float(rate), parse_ci(row.get("success_ci")), int(n) if n else None
    return None, None, None


def best_e2_config(
    model_rows: List[Dict[str, Any]],
) -> Tuple[
    Optional[str], Optional[float], Optional[Tuple[float, float]], Optional[int]
]:
    best_row: Optional[Dict[str, Any]] = None
    for row in model_rows:
        if row.get("experiment") != "E2":
            continue
        if best_row is None or float(row.get("success_rate", 0.0)) > float(
            best_row.get("success_rate", 0.0)
        ):
            best_row = row

    if best_row is None:
        return None, None, None, None
    return (
        str(best_row.get("metric")),
        float(best_row.get("success_rate", 0.0)),
        parse_ci(best_row.get("success_ci")),
        int(best_row.get("n_episodes", 0)) if best_row.get("n_episodes") else None,
    )


def extract_gate_status(gate_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    status_by_model: Dict[str, Dict[str, Any]] = {}
    for row in gate_report.get("rows", []):
        if not isinstance(row, dict):
            continue
        model = str(row.get("model", "unknown"))
        summary = row.get("summary", {})
        criteria = summary.get("criteria", {}) if isinstance(summary, dict) else {}
        status_by_model[model] = {
            "available": bool(summary.get("available"))
            if isinstance(summary, dict)
            else False,
            "all_pass": bool(criteria.get("all_pass"))
            if isinstance(criteria, dict)
            else False,
            "normal_gt_p0": bool(criteria.get("normal_gt_p0_ci_backed"))
            if isinstance(criteria, dict)
            else False,
            "shuffle_lt_p0": bool(criteria.get("shuffle_lt_p0_ci_backed"))
            if isinstance(criteria, dict)
            else False,
            "normal_ci": summary.get("normal", {}).get("ci")
            if isinstance(summary, dict)
            else None,
            "shuffle_ci": summary.get("shuffle", {}).get("ci")
            if isinstance(summary, dict)
            else None,
            "p0_ci": summary.get("p0_reference", {}).get("ci")
            if isinstance(summary, dict)
            else None,
        }
    return status_by_model


def ci_to_str(ci: Optional[Tuple[float, float]]) -> str:
    if ci is None:
        return "-"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def build_main_table_rows(
    suite_rows: List[Dict[str, Any]],
    gate_status_by_model: Dict[str, Dict[str, Any]],
    target_seed_set: Optional[str],
) -> List[Dict[str, Any]]:
    grouped = rows_by_model(suite_rows)
    all_models = sorted(set(grouped.keys()) | set(gate_status_by_model.keys()))

    output_rows: List[Dict[str, Any]] = []
    for model in all_models:
        model_rows = grouped.get(model, [])
        p0, p0_ci, p0_n = lookup_metric(model_rows, "E1", "P0.success_rate")
        p1, p1_ci, p1_n = lookup_metric(model_rows, "E1", "P1.success_rate")
        e0, e0_ci, e0_n = lookup_metric(model_rows, "E3", "E0.success_rate")
        a0, a0_ci, a0_n = lookup_metric(model_rows, "E4", "A0.success_rate")
        best_e2, best_e2_rate, best_e2_ci, best_e2_n = best_e2_config(model_rows)
        gate = gate_status_by_model.get(model, {})

        delta = (p1 - p0) if (p1 is not None and p0 is not None) else None
        row = {
            "model": model,
            "seed_set": target_seed_set,
            "e1_p0_success": p0,
            "e1_p0_ci": p0_ci,
            "e1_p0_n": p0_n,
            "e1_p1_success": p1,
            "e1_p1_ci": p1_ci,
            "e1_p1_n": p1_n,
            "e1_delta_p1_minus_p0": delta,
            "e2_best_metric": best_e2,
            "e2_best_success": best_e2_rate,
            "e2_best_ci": best_e2_ci,
            "e2_best_n": best_e2_n,
            "e3_e0_success": e0,
            "e3_e0_ci": e0_ci,
            "e3_e0_n": e0_n,
            "e4_a0_success": a0,
            "e4_a0_ci": a0_ci,
            "e4_a0_n": a0_n,
            "m2_gate_available": bool(gate.get("available", False)),
            "m2_gate_all_pass": bool(gate.get("all_pass", False)),
            "m2_normal_gt_p0_ci_backed": bool(gate.get("normal_gt_p0", False)),
            "m2_shuffle_lt_p0_ci_backed": bool(gate.get("shuffle_lt_p0", False)),
        }
        output_rows.append(row)

    return output_rows


def markdown_for_main_table(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Main Results Table")
    lines.append("")
    lines.append(
        "| Model | Seed Set | P0 | P1 | P1-P0 | Best E2 | E2 Success | E0 | A0 | M2 Gate Pass |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        p0 = "-" if row["e1_p0_success"] is None else f"{row['e1_p0_success']:.3f}"
        p1 = "-" if row["e1_p1_success"] is None else f"{row['e1_p1_success']:.3f}"
        delta = (
            "-"
            if row["e1_delta_p1_minus_p0"] is None
            else f"{row['e1_delta_p1_minus_p0']:.3f}"
        )
        best_e2 = row["e2_best_metric"] or "-"
        best_e2_success = (
            "-" if row["e2_best_success"] is None else f"{row['e2_best_success']:.3f}"
        )
        e0 = "-" if row["e3_e0_success"] is None else f"{row['e3_e0_success']:.3f}"
        a0 = "-" if row["e4_a0_success"] is None else f"{row['e4_a0_success']:.3f}"
        gate_pass = (
            "True" if row["m2_gate_available"] and row["m2_gate_all_pass"] else "False"
        )
        lines.append(
            f"| {row['model']} | {row['seed_set'] or '-'} | {p0} | {p1} | {delta} | {best_e2} | "
            f"{best_e2_success} | {e0} | {a0} | {gate_pass} |"
        )
    lines.append("")
    lines.append("## CI Detail")
    lines.append("")
    lines.append("| Model | P0 CI | P1 CI | Best E2 CI | E0 CI | A0 CI |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            f"| {row['model']} | {ci_to_str(row['e1_p0_ci'])} | "
            f"{ci_to_str(row['e1_p1_ci'])} | {ci_to_str(row['e2_best_ci'])} | "
            f"{ci_to_str(row['e3_e0_ci'])} | {ci_to_str(row['e4_a0_ci'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_figure_data(
    suite_rows: List[Dict[str, Any]],
    main_rows: List[Dict[str, Any]],
    gate_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    e2_rows: List[Dict[str, Any]] = []
    for row in suite_rows:
        if row.get("experiment") != "E2":
            continue
        e2_rows.append(
            {
                "model": row.get("model"),
                "config": row.get("metric"),
                "success_rate": row.get("success_rate"),
                "success_ci": row.get("success_ci"),
                "n_episodes": row.get("n_episodes"),
            }
        )

    e1_delta = [
        {
            "model": row["model"],
            "seed_set": row["seed_set"],
            "p0": row["e1_p0_success"],
            "p1": row["e1_p1_success"],
            "delta": row["e1_delta_p1_minus_p0"],
        }
        for row in main_rows
    ]

    latent_baselines = [
        {
            "model": row["model"],
            "seed_set": row["seed_set"],
            "e0_success": row["e3_e0_success"],
            "a0_success": row["e4_a0_success"],
        }
        for row in main_rows
    ]

    payload: Dict[str, Any] = {
        "e1_delta_by_model": e1_delta,
        "e2_protocol_sweep": e2_rows,
        "latent_baselines": latent_baselines,
    }
    if gate_report is not None:
        payload["m2_gate"] = gate_report.get("rows", [])
    return payload


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_for_m2_gate(gate_report: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# M2 Gate Table")
    lines.append("")
    if gate_report is None:
        lines.append("_M2 gate report not available for this run._")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "| Model | Normal CI | Shuffle CI | P0 CI | Normal>P0 | Shuffle<P0 | All Pass |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in gate_report.get("rows", []):
        if not isinstance(row, dict):
            continue
        summary = row.get("summary", {})
        if not isinstance(summary, dict) or not summary.get("available"):
            lines.append(f"| {row.get('model', 'unknown')} | - | - | - | - | - | - |")
            continue
        normal_ci = parse_ci(summary.get("normal", {}).get("ci"))
        shuffle_ci = parse_ci(summary.get("shuffle", {}).get("ci"))
        p0_ci = parse_ci(summary.get("p0_reference", {}).get("ci"))
        criteria = summary.get("criteria", {})
        lines.append(
            f"| {row.get('model', 'unknown')} | {ci_to_str(normal_ci)} | "
            f"{ci_to_str(shuffle_ci)} | {ci_to_str(p0_ci)} | "
            f"{bool(criteria.get('normal_gt_p0_ci_backed'))} | "
            f"{bool(criteria.get('shuffle_lt_p0_ci_backed'))} | "
            f"{bool(criteria.get('all_pass'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-facing tables/figures")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--suite-report", type=str, default=None)
    parser.add_argument("--gate-report", type=str, default=None)
    parser.add_argument(
        "--seed-set",
        type=str,
        default=None,
        help="Optional seed-set filter; defaults to manifest.primary_seed_set",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    run_dir = manifest_path.parent
    suite_report_path = (
        Path(args.suite_report).resolve()
        if args.suite_report
        else run_dir / "suite_report.json"
    )
    gate_report_path = (
        Path(args.gate_report).resolve()
        if args.gate_report
        else run_dir / "m2_gate_report.json"
    )
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else run_dir / "paper_pack"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(manifest_path)
    suite_report = load_json(suite_report_path)
    gate_report = load_json(gate_report_path) if gate_report_path.exists() else None

    suite_rows = suite_report.get("rows", [])
    if not isinstance(suite_rows, list):
        raise ValueError("suite_report.rows must be a list")
    target_seed_set = (
        args.seed_set
        if args.seed_set
        else (
            str(manifest["primary_seed_set"])
            if manifest.get("primary_seed_set") is not None
            else None
        )
    )
    if target_seed_set is not None:
        suite_rows = [
            row
            for row in suite_rows
            if str(row.get("seed_set", target_seed_set)) == target_seed_set
        ]

    gate_status = extract_gate_status(gate_report) if gate_report is not None else {}
    main_rows = build_main_table_rows(suite_rows, gate_status, target_seed_set)
    figure_data = build_figure_data(suite_rows, main_rows, gate_report)

    main_csv = output_dir / "main_table.csv"
    main_md = output_dir / "main_table.md"
    figures_json = output_dir / "figure_data.json"
    m2_md = output_dir / "m2_gate_table.md"
    pack_manifest = output_dir / "paper_pack_manifest.json"
    readme = output_dir / "README.md"

    write_csv(main_csv, main_rows)
    main_md.write_text(markdown_for_main_table(main_rows))
    figures_json.write_text(json.dumps(figure_data, indent=2))
    m2_md.write_text(markdown_for_m2_gate(gate_report))

    package_payload = {
        "run_id": manifest.get("run_id"),
        "suite_name": manifest.get("suite_name"),
        "files": [
            str(main_csv),
            str(main_md),
            str(figures_json),
            str(m2_md),
        ],
    }
    pack_manifest.write_text(json.dumps(package_payload, indent=2))
    readme.write_text(
        "\n".join(
            [
                "# Paper Pack",
                "",
                "Generated tables and figure-ready data for manuscript assembly.",
                "",
                "## Files",
                "",
                "- `main_table.csv`: machine-readable main results table.",
                "- `main_table.md`: manuscript-ready markdown main table.",
                "- `m2_gate_table.md`: CI-backed M2 gate table.",
                "- `figure_data.json`: data payload for plots/figures.",
                "- `paper_pack_manifest.json`: package metadata.",
                "",
            ]
        )
    )

    print(f"Wrote {main_csv}")
    print(f"Wrote {main_md}")
    print(f"Wrote {figures_json}")
    print(f"Wrote {m2_md}")
    print(f"Wrote {pack_manifest}")
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
