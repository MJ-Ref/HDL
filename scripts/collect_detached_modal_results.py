#!/usr/bin/env python3
"""
Collect detached Modal function-call results into canonical LPCA artifact paths.

This script polls FunctionCall IDs from a submission manifest produced by
detached matrix launches and writes completed results into local output folders.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal


E1E4_RUN_ID = "run_e1e4_modal_matrix_detached"
M2_RUN_ID = "run_m2v2_matrix_modal_publication_detached"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def resolve_output_path(repo_root: Path, record: Dict[str, Any]) -> Path:
    model = record["model"]
    exp = record["experiment"]

    if exp in {"E1", "E2", "E3", "E4"}:
        file_name = "results.json" if exp == "E2" else "summary.json"
        return (
            repo_root
            / "outputs"
            / "full_suite"
            / E1E4_RUN_ID
            / "artifacts"
            / model
            / "key_comparisons"
            / exp
            / file_name
        )

    if exp == "M2":
        return (
            repo_root
            / "outputs"
            / "full_suite"
            / M2_RUN_ID
            / "artifacts"
            / model
            / "key_comparisons"
            / "M2"
            / "m2_eval.json"
        )

    raise ValueError(f"Unsupported experiment for collection: {exp}")


def write_payload(
    output_path: Path, record: Dict[str, Any], result: Dict[str, Any]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp = record["experiment"]
    if exp in {"E1", "E2", "E3", "E4"}:
        payload = result.get("artifact", result)
    elif exp == "M2":
        payload = {
            **result,
            "config": {
                "model_name": record["model_id"],
                "k_vectors": record.get("k", 16),
                "epochs": record.get("epochs", 10),
                "eval_episodes": record.get("eval_episodes", 100),
                "base_seed": record.get("base_seed", 1000),
                "codec_variant": "m2v2",
            },
        }
    else:
        raise ValueError(f"Unsupported experiment payload writer: {exp}")

    output_path.write_text(json.dumps(payload, indent=2, default=str))
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "call_id": record["call_id"],
                "experiment": exp,
                "model": record["model"],
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            default=str,
        )
    )


def collect_one(repo_root: Path, record: Dict[str, Any]) -> Tuple[str, str]:
    call_id = record["call_id"]
    call = modal.FunctionCall.from_id(call_id)
    output_path = resolve_output_path(repo_root, record)

    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return "pending", str(output_path)
    except Exception as exc:
        return f"error:{type(exc).__name__}", str(output_path)

    if not isinstance(result, dict):
        return "error:invalid_result", str(output_path)

    write_payload(output_path, record, result)
    return "collected", str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect detached Modal results")
    parser.add_argument(
        "--submission-manifest",
        type=str,
        default="outputs/full_suite/detached_submissions/submission_manifest_latest.json",
        help="Path to detached submission manifest",
    )
    parser.add_argument(
        "--status-output",
        type=str,
        default="outputs/full_suite/detached_submissions/submission_status_latest.json",
        help="Path to write collection status JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    submission_path = (repo_root / args.submission_manifest).resolve()
    status_output = (repo_root / args.status_output).resolve()

    submission = load_json(submission_path)
    records: List[Dict[str, Any]] = [
        *submission.get("e1e4", []),
        *submission.get("m2", []),
    ]

    statuses: List[Dict[str, Any]] = []
    counts = {"collected": 0, "pending": 0, "errors": 0}
    for record in records:
        state, output_path = collect_one(repo_root, record)
        statuses.append(
            {
                "call_id": record["call_id"],
                "experiment": record["experiment"],
                "model": record["model"],
                "state": state,
                "output_path": output_path,
            }
        )
        if state == "collected":
            counts["collected"] += 1
        elif state == "pending":
            counts["pending"] += 1
        else:
            counts["errors"] += 1

    summary = {
        "submission_manifest": str(submission_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "statuses": statuses,
    }
    status_output.parent.mkdir(parents=True, exist_ok=True)
    status_output.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {status_output}")
    print(
        f"collected={counts['collected']} pending={counts['pending']} "
        f"errors={counts['errors']}"
    )


if __name__ == "__main__":
    main()
