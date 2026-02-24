#!/usr/bin/env python3
"""
Build a submission-grade reproducibility package from a full-suite run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_copy_plan(
    run_dir: Path,
    repo_root: Path,
    manifest: Dict[str, Any],
) -> List[Tuple[Path, Path]]:
    copy_plan: List[Tuple[Path, Path]] = []
    run_level = [
        ("manifest.json", "run/manifest.json"),
        ("summary.md", "run/summary.md"),
        ("resolved_config.json", "run/resolved_config.json"),
        ("resolved_seed_registry.json", "run/resolved_seed_registry.json"),
        ("suite_report.json", "reports/suite_report.json"),
        ("suite_report.md", "reports/suite_report.md"),
        ("m2_gate_report.json", "reports/m2_gate_report.json"),
        ("m2_gate_report.md", "reports/m2_gate_report.md"),
    ]
    for src_rel, dst_rel in run_level:
        copy_plan.append((run_dir / src_rel, Path(dst_rel)))

    paper_pack = run_dir / "paper_pack"
    if paper_pack.exists() and paper_pack.is_dir():
        for path in sorted(paper_pack.rglob("*")):
            if path.is_file():
                rel = path.relative_to(run_dir)
                copy_plan.append((path, Path("reports") / rel))

    tracked_repo_files = [
        "configs/full_suite_frozen.yaml",
        "configs/seeds/full_suite_seed_registry.json",
        "docs/ARTIFACT_INDEX.md",
        "scripts/run_full_suite.py",
        "scripts/analysis/generate_suite_report.py",
        "scripts/analysis/generate_m2_gate_report.py",
        "scripts/analysis/generate_paper_pack.py",
        "scripts/analysis/build_repro_package.py",
        "requirements.txt",
        "setup.py",
    ]
    for rel in tracked_repo_files:
        copy_plan.append((repo_root / rel, Path(rel)))

    for command in manifest.get("commands", []):
        if not isinstance(command, dict):
            continue
        log_rel = command.get("log_path")
        if isinstance(log_rel, str):
            copy_plan.append((run_dir / log_rel, Path("run") / log_rel))

    deduped: Dict[Path, Path] = {}
    for src, dst in copy_plan:
        deduped[dst] = src
    return sorted([(src, dst) for dst, src in deduped.items()], key=lambda x: str(x[1]))


def run_pip_freeze() -> str:
    try:
        freeze_output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        ).strip()
    except Exception as exc:
        freeze_output = f"# pip freeze unavailable: {exc}"

    lines = [
        f"# python_executable={sys.executable}",
        f"# python_version={sys.version.split()[0]}",
        f"# platform={platform.platform()}",
        "",
        freeze_output,
        "",
    ]
    return "\n".join(lines)


def build_runbook(
    manifest: Dict[str, Any],
    repo_root: Path,
    run_dir: Path,
) -> str:
    config_path = manifest.get("config_path", "configs/full_suite_frozen.yaml")
    run_id = manifest.get("run_id", "run_id")
    run_rel = (
        str(run_dir.relative_to(repo_root))
        if str(run_dir).startswith(str(repo_root))
        else str(run_dir)
    )
    lines = [
        "# Reproducibility Runbook",
        "",
        "## Environment",
        "",
        "```bash",
        "python -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt",
        "pip install -e .",
        "```",
        "",
        "## Reproduce Main Run",
        "",
        "```bash",
        "python scripts/run_full_suite.py \\",
        f"  --config {config_path} \\",
        f"  --run-id {run_id} \\",
        "  --stages prepare,run,aggregate,gate,paper,package,render,publish \\",
        "  --execute",
        "```",
        "",
        "## Reproduce Main Table Only",
        "",
        "```bash",
        f"bash {run_rel}/repro_package/reproduce_main_table.sh {run_id}",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_reproduce_script(run_id: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'PACKAGE_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"',
        'RUN_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"',
        'REPO_ROOT="$(cd "${RUN_ROOT}/../.." && pwd)"',
        "",
        f'RUN_ID="${{1:-{run_id}}}"',
        'MANIFEST="${RUN_ROOT}/manifest.json"',
        "",
        'python "${REPO_ROOT}/scripts/analysis/generate_suite_report.py" \\',
        '  --manifest "${MANIFEST}" \\',
        '  --output-json "${RUN_ROOT}/suite_report.json" \\',
        '  --output-md "${RUN_ROOT}/suite_report.md"',
        "",
        'python "${REPO_ROOT}/scripts/analysis/generate_paper_pack.py" \\',
        '  --manifest "${MANIFEST}" \\',
        '  --suite-report "${RUN_ROOT}/suite_report.json" \\',
        '  --gate-report "${RUN_ROOT}/m2_gate_report.json" \\',
        '  --output-dir "${RUN_ROOT}/paper_pack"',
        "",
        'echo "Main table regenerated under ${RUN_ROOT}/paper_pack"',
        "",
    ]
    return "\n".join(lines)


def copy_files(
    copy_plan: Sequence[Tuple[Path, Path]],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    entries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for src, dst_rel in copy_plan:
        if not src.exists() or not src.is_file():
            warnings.append(f"Missing source file: {src}")
            continue
        dst = output_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        entries.append(
            {
                "source": str(src),
                "path": str(dst.relative_to(output_dir)),
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )
    return entries, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full-suite reproducibility package"
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    run_dir = manifest_path.parent
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "repro_package"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(manifest_path)
    run_id = str(manifest.get("run_id", "run_id"))

    copy_plan = normalize_copy_plan(run_dir, repo_root, manifest)
    copied_entries, warnings = copy_files(copy_plan, output_dir)

    env_lock = output_dir / "environment_lock.txt"
    env_lock.write_text(run_pip_freeze())
    copied_entries.append(
        {
            "source": "generated",
            "path": str(env_lock.relative_to(output_dir)),
            "size_bytes": env_lock.stat().st_size,
            "sha256": sha256_file(env_lock),
        }
    )

    runbook = output_dir / "RUNBOOK.md"
    runbook.write_text(build_runbook(manifest, repo_root, run_dir))
    copied_entries.append(
        {
            "source": "generated",
            "path": str(runbook.relative_to(output_dir)),
            "size_bytes": runbook.stat().st_size,
            "sha256": sha256_file(runbook),
        }
    )

    reproduce_script = output_dir / "reproduce_main_table.sh"
    reproduce_script.write_text(build_reproduce_script(run_id))
    reproduce_script.chmod(0o755)
    copied_entries.append(
        {
            "source": "generated",
            "path": str(reproduce_script.relative_to(output_dir)),
            "size_bytes": reproduce_script.stat().st_size,
            "sha256": sha256_file(reproduce_script),
        }
    )

    package_manifest = output_dir / "repro_package_manifest.json"
    payload = {
        "run_id": run_id,
        "suite_name": manifest.get("suite_name"),
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "file_count": len(copied_entries),
        "files": copied_entries,
        "warnings": warnings,
    }
    package_manifest.write_text(json.dumps(payload, indent=2))

    print(f"Wrote {package_manifest}")
    print(f"Wrote {runbook}")
    print(f"Wrote {env_lock}")
    print(f"Wrote {reproduce_script}")
    if warnings:
        print(f"Warnings: {len(warnings)} missing files")


if __name__ == "__main__":
    main()
