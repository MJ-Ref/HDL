#!/usr/bin/env python3
"""
Run E1-E4 baseline experiments on Modal and write canonical local artifacts.

This script is designed to be called by scripts/run_full_suite.py commands.
It executes one experiment per invocation and writes a single JSON artifact at
the provided --output path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    MODAL_AVAILABLE = False


EXPERIMENT_CHOICES = ("E1", "E2", "E3", "E4")
ARTIFACT_PATTERNS = {
    "E1": "llm_constraint_satisfaction_*/summary.json",
    "E2": "e2_sweep_*/results.json",
    "E3": "cipher_e0_*/summary.json",
    "E4": "activation_grafting_*/summary.json",
}


def tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_experiment_command(
    *,
    experiment: str,
    model: str,
    base_seed: int,
    n_episodes: int,
    output_dir: str,
    protocols: Optional[str] = None,
    layer: int = 18,
    combine: str = "replace",
    top_k: int = 100,
    n_soft_tokens: int = 1,
    device: str = "cuda",
) -> List[str]:
    if experiment == "E1":
        return [
            "python",
            "/root/scripts/run_llm_experiment.py",
            "--protocols",
            protocols or "P0,P1",
            "--n_episodes",
            str(n_episodes),
            "--base_seed",
            str(base_seed),
            "--model",
            model,
            "--device",
            device,
            "--output",
            output_dir,
        ]
    if experiment == "E2":
        return [
            "python",
            "/root/scripts/run_e2_sweep.py",
            "--protocols",
            protocols or "P2,P5",
            "--n_episodes",
            str(n_episodes),
            "--base_seed",
            str(base_seed),
            "--model",
            model,
            "--device",
            device,
            "--output",
            output_dir,
        ]
    if experiment == "E3":
        return [
            "python",
            "/root/scripts/run_cipher_experiment.py",
            "--n_episodes",
            str(n_episodes),
            "--base_seed",
            str(base_seed),
            "--model",
            model,
            "--device",
            device,
            "--top_k",
            str(top_k),
            "--n_soft_tokens",
            str(n_soft_tokens),
            "--output",
            output_dir,
        ]
    if experiment == "E4":
        return [
            "python",
            "/root/scripts/run_activation_experiment.py",
            "--layer",
            str(layer),
            "--combine",
            combine,
            "--n_episodes",
            str(n_episodes),
            "--base_seed",
            str(base_seed),
            "--model",
            model,
            "--device",
            device,
            "--output",
            output_dir,
        ]
    raise ValueError(f"Unsupported experiment: {experiment}")


def find_artifact(experiment: str, output_dir: Path) -> Path:
    pattern = ARTIFACT_PATTERNS[experiment]
    matches = sorted(output_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No artifact produced for {experiment}; expected pattern {pattern} in {output_dir}"
        )
    return matches[-1]


def run_experiment_payload(
    *,
    experiment: str,
    model: str,
    base_seed: int,
    n_episodes: int,
    protocols: Optional[str],
    layer: int,
    combine: str,
    top_k: int,
    n_soft_tokens: int,
    device: str,
) -> Dict[str, Any]:
    output_dir = Path(tempfile.mkdtemp(prefix=f"lpca_{experiment.lower()}_"))
    cmd = build_experiment_command(
        experiment=experiment,
        model=model,
        base_seed=base_seed,
        n_episodes=n_episodes,
        output_dir=str(output_dir),
        protocols=protocols,
        layer=layer,
        combine=combine,
        top_k=top_k,
        n_soft_tokens=n_soft_tokens,
        device=device,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{experiment} command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT tail:\n{tail(proc.stdout)}\n"
            f"STDERR tail:\n{tail(proc.stderr)}"
        )

    artifact_path = find_artifact(experiment, output_dir)
    artifact_data = json.loads(artifact_path.read_text())
    return {
        "experiment": experiment,
        "model": model,
        "base_seed": base_seed,
        "n_episodes": n_episodes,
        "command": cmd,
        "artifact": artifact_data,
        "artifact_basename": artifact_path.name,
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }


if MODAL_AVAILABLE:
    app = modal.App("lpca-baseline-suite")

    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            "torch>=2.2.0",
            "transformers>=4.38.0",
            "accelerate>=0.27.0",
            "datasets>=2.18.0",
            "safetensors>=0.4.0",
            "einops>=0.7.0",
            "numpy>=1.26.0",
            "scipy>=1.12.0",
            "pandas>=2.2.0",
            "pyarrow>=15.0.0",
            "pydantic>=2.6.0",
            "pyyaml>=6.0.0",
            "tqdm>=4.66.0",
        )
        .add_local_dir("lpca", remote_path="/root/lpca")
        .add_local_dir("scripts", remote_path="/root/scripts")
    )

    @app.function(gpu="A100", timeout=21600, image=image)
    def run_experiment_modal(
        experiment: str,
        model: str,
        base_seed: int,
        n_episodes: int,
        protocols: Optional[str] = None,
        layer: int = 18,
        combine: str = "replace",
        top_k: int = 100,
        n_soft_tokens: int = 1,
    ) -> Dict[str, Any]:
        os.environ.setdefault("PYTHONPATH", "/root")
        return run_experiment_payload(
            experiment=experiment,
            model=model,
            base_seed=base_seed,
            n_episodes=n_episodes,
            protocols=protocols,
            layer=layer,
            combine=combine,
            top_k=top_k,
            n_soft_tokens=n_soft_tokens,
            device="cuda",
        )


def write_output(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload["artifact"], indent=2, default=str))
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "experiment": payload["experiment"],
                "model": payload["model"],
                "base_seed": payload["base_seed"],
                "n_episodes": payload["n_episodes"],
                "command": payload["command"],
                "stdout_tail": payload.get("stdout_tail", ""),
                "stderr_tail": payload.get("stderr_tail", ""),
            },
            indent=2,
            default=str,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1-E4 on Modal")
    parser.add_argument(
        "--experiment", type=str, required=True, choices=EXPERIMENT_CHOICES
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--protocols", type=str, default=None)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--combine", type=str, default="replace")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--n-soft-tokens", type=int, default=1)
    parser.add_argument(
        "--local", action="store_true", help="Run locally instead of Modal"
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()

    if args.local:
        payload = run_experiment_payload(
            experiment=args.experiment,
            model=args.model,
            base_seed=args.base_seed,
            n_episodes=args.n_episodes,
            protocols=args.protocols,
            layer=args.layer,
            combine=args.combine,
            top_k=args.top_k,
            n_soft_tokens=args.n_soft_tokens,
            device="mps",
        )
        write_output(output_path, payload)
        print(f"Wrote {output_path}")
        return

    if not MODAL_AVAILABLE:
        raise RuntimeError(
            "Modal is not available. Install with `pip install modal>=1.1.0` "
            "or pass --local."
        )

    with app.run():
        payload = run_experiment_modal.remote(
            experiment=args.experiment,
            model=args.model,
            base_seed=args.base_seed,
            n_episodes=args.n_episodes,
            protocols=args.protocols,
            layer=args.layer,
            combine=args.combine,
            top_k=args.top_k,
            n_soft_tokens=args.n_soft_tokens,
        )
    write_output(output_path, payload)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
