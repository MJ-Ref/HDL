#!/usr/bin/env python3
"""
Canonical LPCA benchmark orchestrator.

Runs staged execution for E1/E2/E3/E4/M2 and writes:
- per-command logs
- artifact discovery records
- manifest JSON
- run summary markdown
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_CONFIG_KEYS = [
    "suite_name",
    "seed_registry_path",
    "output_root",
    "default_device",
    "seed_usage",
    "episodes",
    "m2",
    "models",
    "experiments",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must parse to a mapping")
    return data


def canonical_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expand_seed_registry(seed_registry: Dict[str, Any]) -> Dict[str, List[int]]:
    seed_sets = seed_registry.get("seed_sets", {})
    if not isinstance(seed_sets, dict):
        raise ValueError("seed_registry.seed_sets must be a mapping")

    expanded: Dict[str, List[int]] = {}
    for name, spec in seed_sets.items():
        if isinstance(spec, list):
            if not all(isinstance(x, int) for x in spec):
                raise ValueError(f"Seed set {name} must contain only integers")
            expanded[name] = list(spec)
            continue

        if isinstance(spec, dict) and "start" in spec and "count" in spec:
            start = int(spec["start"])
            count = int(spec["count"])
            if count < 0:
                raise ValueError(f"Seed set {name} has negative count")
            expanded[name] = list(range(start, start + count))
            continue

        raise ValueError(
            f"Seed set {name} must be a list of ints or a dict with start/count"
        )

    return expanded


def get_git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def parse_csv(arg: str | None) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def resolve_seed_sets(
    config: Dict[str, Any],
    expanded_seed_registry: Dict[str, List[int]],
    requested_seed_sets: List[str],
) -> List[Tuple[str, int, int]]:
    configured = config.get("seed_usage", {}).get("matrix_sets")
    if configured is None:
        configured = [config["seed_usage"]["primary_set"]]
    if not isinstance(configured, list) or not configured:
        raise ValueError("seed_usage.matrix_sets must be a non-empty list")
    if not all(isinstance(x, str) for x in configured):
        raise ValueError("seed_usage.matrix_sets must contain only seed-set names")

    effective = configured
    if requested_seed_sets:
        effective = [x for x in configured if x in requested_seed_sets]
        missing = [x for x in requested_seed_sets if x not in configured]
        if missing:
            raise ValueError(
                "Requested seed sets are not configured in seed_usage.matrix_sets: "
                f"{missing}"
            )
        if not effective:
            raise ValueError("No seed sets selected after applying --seed-sets filter")

    resolved: List[Tuple[str, int, int]] = []
    for name in effective:
        values = expanded_seed_registry.get(name)
        if values is None:
            raise ValueError(f"Seed set {name} not found in registry")
        if not values:
            raise ValueError(f"Seed set {name} is empty")
        resolved.append((name, int(values[0]), len(values)))
    return resolved


def format_command(command_template: List[str], context: Dict[str, Any]) -> List[str]:
    return [str(token).format(**context) for token in command_template]


def discover_artifacts(repo_root: Path, pattern: str) -> List[str]:
    resolved_pattern = Path(pattern)
    if not resolved_pattern.is_absolute():
        resolved_pattern = repo_root / resolved_pattern
    matches = sorted(glob.glob(str(resolved_pattern)))
    return [str(Path(m).resolve()) for m in matches]


def run_command(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    execute: bool,
) -> Dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not execute:
        log_path.write_text("[DRY-RUN]\n" + " ".join(cmd) + "\n")
        return {
            "status": "dry_run",
            "returncode": 0,
            "elapsed_s": 0.0,
        }

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    output = (
        f"$ {' '.join(cmd)}\n\n"
        f"=== STDOUT ===\n{proc.stdout}\n\n"
        f"=== STDERR ===\n{proc.stderr}\n"
    )
    log_path.write_text(output)

    return {
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
    }


def validate_config(config: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if not config["models"]:
        raise ValueError("Config models list cannot be empty")
    if not config["experiments"]:
        raise ValueError("Config experiments mapping cannot be empty")


def validate_matrix_requirements(
    config: Dict[str, Any],
    selected_models: List[Dict[str, Any]],
    selected_seed_sets: List[Tuple[str, int, int]],
    runnable_experiments: Dict[str, Any],
) -> None:
    reqs = config.get("matrix_requirements", {})
    if not isinstance(reqs, dict):
        raise ValueError("matrix_requirements must be a mapping when provided")
    min_models = int(reqs.get("min_models", 3))
    min_seed_count = int(reqs.get("min_seed_count", 100))

    if len(selected_models) < min_models:
        raise ValueError(
            f"Model matrix requires at least {min_models} models; "
            f"got {len(selected_models)}"
        )
    for seed_set_name, _, seed_count in selected_seed_sets:
        if seed_count < min_seed_count:
            raise ValueError(
                f"Seed set {seed_set_name} has {seed_count} seeds; "
                f"requires >= {min_seed_count}"
            )

    exp_to_episode_key = {
        "E1": "e1",
        "E2": "e2",
        "E3": "e3",
        "E4": "e4",
        "M2": "m2_eval",
    }
    for exp_name in runnable_experiments:
        episode_key = exp_to_episode_key.get(exp_name)
        if episode_key is None:
            continue
        n_episodes = int(config["episodes"][episode_key])
        if n_episodes < min_seed_count:
            raise ValueError(
                f"Experiment {exp_name} has {n_episodes} episodes; "
                f"requires >= {min_seed_count} for matrix runs"
            )


def render_summary_markdown(manifest: Dict[str, Any], run_dir: Path) -> Path:
    lines: List[str] = []
    lines.append("# Full Suite Run Summary")
    lines.append("")
    lines.append(f"Run ID: `{manifest['run_id']}`")
    lines.append(f"Execute mode: `{manifest['execute']}`")
    lines.append(f"Git SHA: `{manifest['git_sha']}`")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append(
        "| Model | Seed Set | Experiment | Status | Return Code | Elapsed (s) | Artifact Count |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for rec in manifest.get("commands", []):
        lines.append(
            f"| {rec['model']} | {rec.get('seed_set', '-')} | {rec['experiment']} | "
            f"{rec['status']} | "
            f"{rec['returncode']} | {rec['elapsed_s']} | {len(rec['artifacts'])} |"
        )
    lines.append("")

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def write_manifest(manifest: Dict[str, Any], run_dir: Path) -> Tuple[Path, Path]:
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    summary_path = render_summary_markdown(manifest, run_dir)
    return manifest_path, summary_path


def build_context(
    config: Dict[str, Any],
    model: Dict[str, str],
    seed_set_name: str,
    output_dir: Path,
    base_seed: int,
    seed_count: int,
) -> Dict[str, Any]:
    episodes = config["episodes"]
    m2 = config["m2"]
    return {
        "model_name": model["name"],
        "model_id": model["model_id"],
        "seed_set_name": seed_set_name,
        "device": config["default_device"],
        "output_dir": str(output_dir),
        "e1_episodes": episodes["e1"],
        "e2_episodes": episodes["e2"],
        "e3_episodes": episodes["e3"],
        "e4_episodes": episodes["e4"],
        "m2_eval_episodes": episodes["m2_eval"],
        "m2_k": m2["k"],
        "m2_epochs": m2["epochs"],
        "m2_data_path": m2["data_path"],
        "primary_base_seed": base_seed,
        "primary_seed_count": seed_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical LPCA full suite")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_suite_frozen.yaml",
        help="Path to full-suite YAML config",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="prepare,run,aggregate,gate,paper,package,render,publish",
        help="Comma-separated stages to run",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated model names filter",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Optional comma-separated experiment keys filter",
    )
    parser.add_argument(
        "--seed-sets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated seed-set names filter "
            "(must be configured in seed_usage.matrix_sets)"
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id override",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output root override",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute commands; default is dry-run",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on first failed command",
    )
    parser.add_argument(
        "--skip-matrix-requirements",
        action="store_true",
        help="Allow partial matrix runs for quick/pilot execution",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    config = load_yaml(config_path)
    validate_config(config)

    seed_registry_path = (repo_root / config["seed_registry_path"]).resolve()
    seed_registry = json.loads(seed_registry_path.read_text())
    expanded_seed_registry = expand_seed_registry(seed_registry)
    primary_seed_set_name = config["seed_usage"]["primary_set"]
    if primary_seed_set_name not in expanded_seed_registry:
        raise ValueError(
            f"Primary seed set {primary_seed_set_name} not found in seed registry"
        )
    primary_seed_values = expanded_seed_registry[primary_seed_set_name]
    if not primary_seed_values:
        raise ValueError(f"Primary seed set {primary_seed_set_name} is empty")

    run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (repo_root / config["output_root"]).resolve()
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stages = parse_csv(args.stages)
    model_filter = set(parse_csv(args.models))
    experiment_filter = set(parse_csv(args.experiments))
    seed_set_filter = parse_csv(args.seed_sets)

    selected_models = [
        m for m in config["models"] if not model_filter or m["name"] in model_filter
    ]
    selected_experiments = {
        name: cfg
        for name, cfg in config["experiments"].items()
        if (not experiment_filter or name in experiment_filter)
    }
    runnable_experiments = {
        name: cfg
        for name, cfg in selected_experiments.items()
        if cfg.get("enabled", False)
    }
    selected_seed_sets = resolve_seed_sets(
        config=config,
        expanded_seed_registry=expanded_seed_registry,
        requested_seed_sets=seed_set_filter,
    )
    if not args.skip_matrix_requirements:
        validate_matrix_requirements(
            config=config,
            selected_models=selected_models,
            selected_seed_sets=selected_seed_sets,
            runnable_experiments=runnable_experiments,
        )

    manifest: Dict[str, Any] = {
        "suite_name": config["suite_name"],
        "run_id": run_id,
        "execute": args.execute,
        "git_sha": get_git_sha(repo_root),
        "config_path": str(config_path.relative_to(repo_root)),
        "seed_registry_path": str(seed_registry_path.relative_to(repo_root)),
        "config_hash": canonical_hash(config),
        "seed_hash": canonical_hash(expanded_seed_registry),
        "stages_requested": stages,
        "stages_completed": [],
        "models": [m["name"] for m in selected_models],
        "seed_sets": [seed_set for seed_set, _, _ in selected_seed_sets],
        "primary_seed_set": primary_seed_set_name,
        "experiments": list(runnable_experiments.keys()),
        "commands": [],
    }

    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "resolved_seed_registry.json").write_text(
        json.dumps(expanded_seed_registry, indent=2)
    )

    if "prepare" in stages:
        manifest["stages_completed"].append("prepare")

    if "run" in stages:
        for model in selected_models:
            for seed_set_name, seed_set_base_seed, seed_set_count in selected_seed_sets:
                for exp_name, exp_cfg in runnable_experiments.items():
                    output_dir = (
                        run_dir / "artifacts" / model["name"] / seed_set_name / exp_name
                    )
                    log_path = (
                        run_dir
                        / "logs"
                        / model["name"]
                        / seed_set_name
                        / exp_name
                        / "command.log"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)

                    context = build_context(
                        config=config,
                        model=model,
                        seed_set_name=seed_set_name,
                        output_dir=output_dir,
                        base_seed=seed_set_base_seed,
                        seed_count=seed_set_count,
                    )
                    command = format_command(exp_cfg["command"], context)

                    result = run_command(
                        cmd=command,
                        cwd=repo_root,
                        log_path=log_path,
                        execute=args.execute,
                    )

                    artifacts: List[str] = []
                    artifact_glob = exp_cfg.get("artifact_glob")
                    if artifact_glob:
                        artifacts = discover_artifacts(
                            repo_root,
                            artifact_glob.format(**context),
                        )

                    record = {
                        "model": model["name"],
                        "seed_set": seed_set_name,
                        "experiment": exp_name,
                        "command": command,
                        "log_path": str(log_path.relative_to(run_dir)),
                        "status": result["status"],
                        "returncode": result["returncode"],
                        "elapsed_s": result["elapsed_s"],
                        "artifacts": artifacts,
                    }
                    manifest["commands"].append(record)

                    if args.stop_on_error and result["status"] == "failed":
                        write_manifest(manifest, run_dir)
                        print("Stopping on first failed command", file=sys.stderr)
                        sys.exit(result["returncode"])

        manifest["stages_completed"].append("run")

    if "aggregate" in stages:
        # Persist intermediate manifest for downstream analysis scripts.
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        agg_log = run_dir / "logs" / "aggregate_artifact_index.log"
        agg_cmd = ["python", "scripts/analysis/generate_artifact_index.py"]
        agg_result = run_command(agg_cmd, repo_root, agg_log, args.execute)
        manifest["commands"].append(
            {
                "model": "global",
                "experiment": "aggregate_artifact_index",
                "command": agg_cmd,
                "log_path": str(agg_log.relative_to(run_dir)),
                "status": agg_result["status"],
                "returncode": agg_result["returncode"],
                "elapsed_s": agg_result["elapsed_s"],
                "artifacts": [str((repo_root / "docs/ARTIFACT_INDEX.md").resolve())],
            }
        )

        suite_log = run_dir / "logs" / "aggregate_suite_report.log"
        suite_cmd = [
            "python",
            "scripts/analysis/generate_suite_report.py",
            "--manifest",
            str(run_dir / "manifest.json"),
            "--output-json",
            str(run_dir / "suite_report.json"),
            "--output-md",
            str(run_dir / "suite_report.md"),
        ]
        suite_result = run_command(suite_cmd, repo_root, suite_log, args.execute)
        manifest["commands"].append(
            {
                "model": "global",
                "experiment": "aggregate_suite_report",
                "command": suite_cmd,
                "log_path": str(suite_log.relative_to(run_dir)),
                "status": suite_result["status"],
                "returncode": suite_result["returncode"],
                "elapsed_s": suite_result["elapsed_s"],
                "artifacts": [
                    str((run_dir / "suite_report.json").resolve()),
                    str((run_dir / "suite_report.md").resolve()),
                ],
            }
        )
        manifest["stages_completed"].append("aggregate")

    if "gate" in stages:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        gate_log = run_dir / "logs" / "gate_m2_report.log"
        gate_cmd = [
            "python",
            "scripts/analysis/generate_m2_gate_report.py",
            "--manifest",
            str(run_dir / "manifest.json"),
            "--output-json",
            str(run_dir / "m2_gate_report.json"),
            "--output-md",
            str(run_dir / "m2_gate_report.md"),
        ]
        gate_result = run_command(gate_cmd, repo_root, gate_log, args.execute)
        manifest["commands"].append(
            {
                "model": "global",
                "experiment": "gate_m2_report",
                "command": gate_cmd,
                "log_path": str(gate_log.relative_to(run_dir)),
                "status": gate_result["status"],
                "returncode": gate_result["returncode"],
                "elapsed_s": gate_result["elapsed_s"],
                "artifacts": [
                    str((run_dir / "m2_gate_report.json").resolve()),
                    str((run_dir / "m2_gate_report.md").resolve()),
                ],
            }
        )
        manifest["stages_completed"].append("gate")

    if "paper" in stages:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        paper_log = run_dir / "logs" / "paper_pack.log"
        paper_cmd = [
            "python",
            "scripts/analysis/generate_paper_pack.py",
            "--manifest",
            str(run_dir / "manifest.json"),
            "--suite-report",
            str(run_dir / "suite_report.json"),
            "--gate-report",
            str(run_dir / "m2_gate_report.json"),
            "--output-dir",
            str(run_dir / "paper_pack"),
        ]
        paper_result = run_command(paper_cmd, repo_root, paper_log, args.execute)
        manifest["commands"].append(
            {
                "model": "global",
                "experiment": "paper_pack",
                "command": paper_cmd,
                "log_path": str(paper_log.relative_to(run_dir)),
                "status": paper_result["status"],
                "returncode": paper_result["returncode"],
                "elapsed_s": paper_result["elapsed_s"],
                "artifacts": [
                    str((run_dir / "paper_pack" / "main_table.csv").resolve()),
                    str((run_dir / "paper_pack" / "main_table.md").resolve()),
                    str((run_dir / "paper_pack" / "m2_gate_table.md").resolve()),
                    str((run_dir / "paper_pack" / "figure_data.json").resolve()),
                    str(
                        (run_dir / "paper_pack" / "paper_pack_manifest.json").resolve()
                    ),
                ],
            }
        )
        manifest["stages_completed"].append("paper")

    if "package" in stages:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        package_log = run_dir / "logs" / "package_repro_bundle.log"
        package_cmd = [
            "python",
            "scripts/analysis/build_repro_package.py",
            "--manifest",
            str(run_dir / "manifest.json"),
            "--output-dir",
            str(run_dir / "repro_package"),
        ]
        package_result = run_command(package_cmd, repo_root, package_log, args.execute)
        manifest["commands"].append(
            {
                "model": "global",
                "experiment": "package_repro_bundle",
                "command": package_cmd,
                "log_path": str(package_log.relative_to(run_dir)),
                "status": package_result["status"],
                "returncode": package_result["returncode"],
                "elapsed_s": package_result["elapsed_s"],
                "artifacts": [
                    str((run_dir / "repro_package" / "RUNBOOK.md").resolve()),
                    str((run_dir / "repro_package" / "environment_lock.txt").resolve()),
                    str(
                        (
                            run_dir / "repro_package" / "reproduce_main_table.sh"
                        ).resolve()
                    ),
                    str(
                        (
                            run_dir / "repro_package" / "repro_package_manifest.json"
                        ).resolve()
                    ),
                ],
            }
        )
        manifest["stages_completed"].append("package")

    if "render" in stages:
        render_summary_markdown(manifest, run_dir)
        manifest["stages_completed"].append("render")

    if "publish" in stages:
        manifest["stages_completed"].append("publish")
        write_manifest(manifest, run_dir)

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
