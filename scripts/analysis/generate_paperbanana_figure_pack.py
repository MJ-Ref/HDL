#!/usr/bin/env python3
"""
Prepare (and optionally execute) a PaperBanana figure pack from HDL run artifacts.

This script builds three publication-oriented figure briefs:
1) method_overview
2) gate_evidence
3) reproducibility_pipeline

It can also call PaperBanana's multi-agent diagram generator to render candidate
images for each brief when --execute is supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image


@dataclass
class FigureBrief:
    figure_id: str
    title: str
    visual_intent: str
    content: str
    aspect_ratio: str = "16:9"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def find_latest_run_dir(repo_root: Path) -> Path:
    full_suite_root = repo_root / "outputs" / "full_suite"
    if not full_suite_root.exists():
        raise FileNotFoundError(f"No full-suite outputs found under {full_suite_root}")

    candidates = [
        p
        for p in full_suite_root.iterdir()
        if p.is_dir() and (p / "manifest.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run directories with manifest.json found under {full_suite_root}"
        )
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(str(t) for t in cmd)


def _main_results_lines(suite_rows: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    grouped: Dict[str, Dict[str, float]] = {}
    for row in suite_rows:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model", "unknown"))
        metric = str(row.get("metric", ""))
        if not metric.endswith(".success_rate"):
            continue
        grouped.setdefault(model, {})
        grouped[model][metric] = float(row.get("success_rate", 0.0))

    for model in sorted(grouped):
        metrics = grouped[model]
        p0 = metrics.get("P0.success_rate")
        p1 = metrics.get("P1.success_rate")
        e0 = metrics.get("E0.success_rate")
        a0 = metrics.get("A0.success_rate")
        delta = (p1 - p0) if (p0 is not None and p1 is not None) else None
        out.append(
            (
                f"- {model}: P0={_fmt_pct(p0)}, P1={_fmt_pct(p1)}, "
                f"delta(P1-P0)={_fmt_pct(delta)}, E0={_fmt_pct(e0)}, A0={_fmt_pct(a0)}"
            )
        )
    if not out:
        out.append("- No suite rows available in suite_report.json for this run.")
    return out


def _gate_lines(gate_report: Optional[Dict[str, Any]]) -> List[str]:
    if not gate_report:
        return ["- Gate report unavailable for this run."]
    rows = gate_report.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return ["- Gate report contains no rows."]

    out: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model", "unknown"))
        summary = row.get("summary", {})
        criteria = summary.get("criteria", {}) if isinstance(summary, dict) else {}
        normal = summary.get("normal", {}) if isinstance(summary, dict) else {}
        shuffle = summary.get("shuffle", {}) if isinstance(summary, dict) else {}
        out.append(
            (
                f"- {model}: normal={normal.get('successes', '-')}/{normal.get('n', '-')}, "
                f"shuffle={shuffle.get('successes', '-')}/{shuffle.get('n', '-')}, "
                f"normal_gt_p0={criteria.get('normal_gt_p0_ci_backed', False)}, "
                f"shuffle_lt_p0={criteria.get('shuffle_lt_p0_ci_backed', False)}, "
                f"all_pass={criteria.get('all_pass', False)}"
            )
        )
    return out


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def build_figure_briefs(
    manifest: Dict[str, Any],
    suite_report: Optional[Dict[str, Any]],
    gate_report: Optional[Dict[str, Any]],
) -> List[FigureBrief]:
    run_id = str(manifest.get("run_id", "unknown_run"))
    models = manifest.get("models", [])
    models_str = ", ".join(models) if isinstance(models, list) else str(models)
    stages = manifest.get("stages_completed", [])
    stages_str = " -> ".join(stages) if isinstance(stages, list) else str(stages)

    suite_rows: List[Dict[str, Any]] = []
    if suite_report and isinstance(suite_report.get("rows"), list):
        suite_rows = [r for r in suite_report["rows"] if isinstance(r, dict)]

    method_content = "\n".join(
        [
            "Project: Latent Protocol Communication Analysis (LPCA).",
            "Goal: test whether latent communication channels preserve useful semantics under strict controls.",
            f"Run ID: {run_id}.",
            f"Models in this run: {models_str}.",
            f"Executed stages: {stages_str}.",
            "Canonical execution path: prepare -> run -> aggregate -> gate -> paper -> render -> package -> publish.",
            "Core experiments: E1 constrained text baseline, E2 budget sweep, E3 cipher baseline, E4 activation grafting baseline, M2 learned codec.",
            "M2 gate policy: require CI-backed normal > P0 and shuffle < P0 with ablations.",
            "Figure should emphasize mechanism, controls, and falsification discipline rather than marketing framing.",
        ]
    )

    gate_content_lines = [
        "This figure summarizes quantitative outcomes and gate results.",
        f"Run ID: {run_id}.",
        "Main benchmark outcomes by model:",
        *_main_results_lines(suite_rows),
        "M2 gate outcomes by model:",
        *_gate_lines(gate_report),
        "The visual should clearly separate text baselines (E1/E2) from latent baselines (E3/E4) and learned codec outcomes (M2).",
        "Use explicit pass/fail markers for CI-backed criteria.",
    ]
    gate_content = "\n".join(gate_content_lines)

    commands = manifest.get("commands", [])
    cmd_lines: List[str] = []
    if isinstance(commands, list):
        for rec in commands[:12]:
            if not isinstance(rec, dict):
                continue
            exp = rec.get("experiment", "unknown")
            status = rec.get("status", "unknown")
            cmd = rec.get("command", [])
            cmd_lines.append(f"- {exp} [{status}]: {_format_cmd(cmd)}")

    repro_content = "\n".join(
        [
            "This figure presents the submission-grade reproducibility workflow.",
            f"Run ID: {run_id}.",
            "Artifacts: manifest, suite report, gate report, paper pack, reproducibility package.",
            "Repro package includes runbook, lockfile, and script to regenerate manuscript tables.",
            "Representative command sequence:",
            *cmd_lines,
            "Show this as an auditable pipeline with artifact handoffs between stages.",
        ]
    )

    return [
        FigureBrief(
            figure_id="fig1_method_overview",
            title="LPCA Mechanism and Falsification Pipeline",
            visual_intent=(
                "Figure 1: LPCA mechanism overview and evaluation pipeline with explicit control paths and gate checks."
            ),
            content=method_content,
            aspect_ratio="16:9",
        ),
        FigureBrief(
            figure_id="fig2_gate_evidence",
            title="Model-by-Model Gate Evidence",
            visual_intent=(
                "Figure 2: Quantitative outcomes and CI-backed M2 gate decisions across models."
            ),
            content=gate_content,
            aspect_ratio="16:9",
        ),
        FigureBrief(
            figure_id="fig3_repro_pipeline",
            title="Submission Reproducibility Pipeline",
            visual_intent=(
                "Figure 3: End-to-end reproducibility flow from run execution to publication artifacts."
            ),
            content=repro_content,
            aspect_ratio="21:9",
        ),
    ]


def write_briefs(briefs: List[FigureBrief], output_dir: Path) -> None:
    briefs_dir = output_dir / "briefs"
    briefs_dir.mkdir(parents=True, exist_ok=True)
    for brief in briefs:
        path = briefs_dir / f"{brief.figure_id}.json"
        path.write_text(json.dumps(asdict(brief), indent=2))

    md_lines = ["# PaperBanana Figure Briefs", ""]
    for brief in briefs:
        md_lines.extend(
            [
                f"## {brief.figure_id}",
                "",
                f"- Title: {brief.title}",
                f"- Visual intent: {brief.visual_intent}",
                f"- Aspect ratio: {brief.aspect_ratio}",
                "",
                "### Content",
                "",
                brief.content,
                "",
            ]
        )
    (output_dir / "briefs.md").write_text("\n".join(md_lines))


def _decode_base64_image(data: str) -> Image.Image:
    blob = base64.b64decode(data)
    return Image.open(io.BytesIO(blob))


def _select_final_image_key(
    result: Dict[str, Any], max_critic_rounds: int
) -> Optional[str]:
    for round_idx in range(max_critic_rounds - 1, -1, -1):
        k = f"target_diagram_critic_desc{round_idx}_base64_jpg"
        if result.get(k):
            return k
    if result.get("target_diagram_stylist_desc0_base64_jpg"):
        return "target_diagram_stylist_desc0_base64_jpg"
    if result.get("target_diagram_desc0_base64_jpg"):
        return "target_diagram_desc0_base64_jpg"
    return None


def _load_paperbanana_modules(paperbanana_root: Path):
    sys.path.insert(0, str(paperbanana_root))
    from agents.critic_agent import CriticAgent  # type: ignore
    from agents.planner_agent import PlannerAgent  # type: ignore
    from agents.polish_agent import PolishAgent  # type: ignore
    from agents.retriever_agent import RetrieverAgent  # type: ignore
    from agents.stylist_agent import StylistAgent  # type: ignore
    from agents.vanilla_agent import VanillaAgent  # type: ignore
    from agents.visualizer_agent import VisualizerAgent  # type: ignore
    from utils import config  # type: ignore
    from utils.paperviz_processor import PaperVizProcessor  # type: ignore

    return (
        config,
        PaperVizProcessor,
        VanillaAgent,
        PlannerAgent,
        VisualizerAgent,
        StylistAgent,
        CriticAgent,
        RetrieverAgent,
        PolishAgent,
    )


async def run_paperbanana_generation(
    briefs: List[FigureBrief],
    output_dir: Path,
    paperbanana_root: Path,
    exp_mode: str,
    retrieval_setting: str,
    candidates: int,
    max_critic_rounds: int,
    max_concurrent: int,
    model_name: str,
) -> Dict[str, Any]:
    (
        config,
        PaperVizProcessor,
        VanillaAgent,
        PlannerAgent,
        VisualizerAgent,
        StylistAgent,
        CriticAgent,
        RetrieverAgent,
        PolishAgent,
    ) = _load_paperbanana_modules(paperbanana_root)

    exp_config = config.ExpConfig(
        dataset_name="PaperBananaBench",
        task_name="diagram",
        split_name="custom",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        max_critic_rounds=max_critic_rounds,
        model_name=model_name,
        work_dir=paperbanana_root,
    )

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    generations_dir = output_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)

    pack_manifest: Dict[str, Any] = {
        "exp_mode": exp_mode,
        "retrieval_setting": retrieval_setting,
        "candidates_per_figure": candidates,
        "max_critic_rounds": max_critic_rounds,
        "figures": {},
    }

    for brief in briefs:
        figure_dir = generations_dir / brief.figure_id
        figure_dir.mkdir(parents=True, exist_ok=True)

        data_list: List[Dict[str, Any]] = []
        for i in range(candidates):
            data_list.append(
                {
                    "filename": f"{brief.figure_id}_candidate_{i}",
                    "candidate_id": i,
                    "caption": brief.visual_intent,
                    "content": brief.content,
                    "visual_intent": brief.visual_intent,
                    "additional_info": {"rounded_ratio": brief.aspect_ratio},
                    "max_critic_rounds": max_critic_rounds,
                }
            )

        results: List[Dict[str, Any]] = []
        async for result_data in processor.process_queries_batch(
            data_list, max_concurrent=max_concurrent, do_eval=False
        ):
            results.append(result_data)

        result_json = figure_dir / "results.json"
        result_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        selected_images: List[Dict[str, Any]] = []
        for r in results:
            candidate_id = int(r.get("candidate_id", len(selected_images)))
            final_key = _select_final_image_key(r, max_critic_rounds=max_critic_rounds)
            if not final_key:
                selected_images.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "missing_image",
                        "final_image_key": None,
                    }
                )
                continue

            try:
                img = _decode_base64_image(r[final_key])
                out_path = figure_dir / f"candidate_{candidate_id}.png"
                img.save(out_path, format="PNG")
                selected_images.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "ok",
                        "final_image_key": final_key,
                        "image_path": str(out_path),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive serialization path
                selected_images.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "decode_error",
                        "final_image_key": final_key,
                        "error": str(exc),
                    }
                )

        pack_manifest["figures"][brief.figure_id] = {
            "results_path": str(result_json),
            "selected_images": selected_images,
        }

    return pack_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PaperBanana-ready figure briefs and optional renders from HDL artifacts"
    )
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument(
        "--paperbanana-root",
        type=str,
        default="/Users/mj/Documents/PaperBanana/PaperBanana-main",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--exp-mode", type=str, default="dev_full")
    parser.add_argument("--retrieval-setting", type=str, default="none")
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--max-critic-rounds", type=int, default=3)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Optional text model override passed to PaperBanana",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute PaperBanana generation in addition to writing briefs",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_dir = (
        Path(args.run_dir).resolve() if args.run_dir else find_latest_run_dir(repo_root)
    )
    manifest_path = run_dir / "manifest.json"
    suite_report_path = run_dir / "suite_report.json"
    gate_report_path = run_dir / "m2_gate_report.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "paperbanana_figure_pack"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(manifest_path)
    suite_report = load_json(suite_report_path) if suite_report_path.exists() else None
    gate_report = load_json(gate_report_path) if gate_report_path.exists() else None

    briefs = build_figure_briefs(manifest, suite_report, gate_report)
    write_briefs(briefs, output_dir)

    pack_manifest: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "suite_report_path": str(suite_report_path)
        if suite_report_path.exists()
        else None,
        "gate_report_path": str(gate_report_path)
        if gate_report_path.exists()
        else None,
        "brief_count": len(briefs),
        "briefs": [asdict(b) for b in briefs],
        "execution": {"requested": bool(args.execute)},
    }

    if args.execute:
        paperbanana_root = Path(args.paperbanana_root).resolve()
        if not paperbanana_root.exists():
            raise FileNotFoundError(f"PaperBanana root not found: {paperbanana_root}")
        generation_manifest = asyncio.run(
            run_paperbanana_generation(
                briefs=briefs,
                output_dir=output_dir,
                paperbanana_root=paperbanana_root,
                exp_mode=args.exp_mode,
                retrieval_setting=args.retrieval_setting,
                candidates=max(1, args.candidates),
                max_critic_rounds=max(1, args.max_critic_rounds),
                max_concurrent=max(1, args.max_concurrent),
                model_name=args.model_name,
            )
        )
        pack_manifest["execution"].update({"requested": True, **generation_manifest})

    manifest_out = output_dir / "paperbanana_figure_pack_manifest.json"
    manifest_out.write_text(json.dumps(pack_manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {output_dir / 'briefs.md'}")
    print(f"Wrote {manifest_out}")


if __name__ == "__main__":
    main()
