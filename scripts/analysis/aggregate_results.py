#!/usr/bin/env python3
"""
Results Aggregation Utilities for LPCA.

Combines results from multiple experiment runs, generates summary tables,
and produces publication-ready outputs.

Usage:
    # Aggregate all results in outputs directory
    python scripts/analysis/aggregate_results.py --output_dir outputs

    # Aggregate specific experiments
    python scripts/analysis/aggregate_results.py --experiments e1_baseline e4_activation

    # Generate LaTeX tables
    python scripts/analysis/aggregate_results.py --output_dir outputs --latex
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lpca.core.metrics import MetricsCalculator


@dataclass
class ExperimentSummary:
    """Summary of a single experiment run."""
    experiment_id: str
    timestamp: str
    config: Dict[str, Any]
    protocols: List[str]
    results: Dict[str, Dict[str, float]]
    n_episodes: int
    task_type: str


@dataclass
class AggregatedResults:
    """Aggregated results across multiple experiments."""
    experiments: List[ExperimentSummary]
    protocol_stats: Dict[str, Dict[str, float]]
    pairwise_comparisons: List[Dict]
    meta_statistics: Dict[str, float]


class ResultsAggregator:
    """Aggregates and summarizes experiment results."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.metrics = MetricsCalculator()
        self.experiments: List[ExperimentSummary] = []

    def discover_experiments(self) -> List[Path]:
        """Find all experiment directories."""
        if not self.output_dir.exists():
            return []

        experiment_dirs = []
        for path in self.output_dir.iterdir():
            if path.is_dir():
                # Check for summary.json or config.json
                if (path / "summary.json").exists() or (path / "config.json").exists():
                    experiment_dirs.append(path)

        return sorted(experiment_dirs, key=lambda p: p.name)

    def load_experiment(self, exp_dir: Path) -> Optional[ExperimentSummary]:
        """Load a single experiment's results."""
        summary_path = exp_dir / "summary.json"
        config_path = exp_dir / "config.json"

        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)

            return ExperimentSummary(
                experiment_id=data.get('experiment_id', exp_dir.name),
                timestamp=data.get('timestamp', ''),
                config=data.get('config', {}),
                protocols=list(data.get('results', {}).keys()),
                results=data.get('results', {}),
                n_episodes=data.get('config', {}).get('n_episodes', 0),
                task_type=data.get('config', {}).get('task_type', 'unknown'),
            )

        elif config_path.exists():
            # Try to reconstruct from individual result files
            with open(config_path) as f:
                config = json.load(f)

            results = {}
            for result_file in exp_dir.glob("results_*.json"):
                protocol = result_file.stem.replace("results_", "")
                with open(result_file) as f:
                    results[protocol] = json.load(f)

            if results:
                return ExperimentSummary(
                    experiment_id=exp_dir.name,
                    timestamp='',
                    config=config,
                    protocols=list(results.keys()),
                    results=results,
                    n_episodes=config.get('n_episodes', 0),
                    task_type=config.get('task_type', 'unknown'),
                )

        return None

    def load_all(self, experiment_filter: Optional[List[str]] = None) -> int:
        """Load all experiments, optionally filtering by name prefix."""
        exp_dirs = self.discover_experiments()

        for exp_dir in exp_dirs:
            # Apply filter if specified
            if experiment_filter:
                if not any(exp_dir.name.startswith(f) for f in experiment_filter):
                    continue

            exp = self.load_experiment(exp_dir)
            if exp:
                self.experiments.append(exp)

        return len(self.experiments)

    def aggregate_by_protocol(self) -> Dict[str, Dict[str, List[float]]]:
        """Aggregate results by protocol across all experiments."""
        protocol_data = defaultdict(lambda: defaultdict(list))

        for exp in self.experiments:
            for protocol, results in exp.results.items():
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        protocol_data[protocol][metric].append(value)

        return dict(protocol_data)

    def compute_protocol_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each protocol."""
        protocol_data = self.aggregate_by_protocol()
        stats = {}

        for protocol, metrics in protocol_data.items():
            stats[protocol] = {}

            for metric, values in metrics.items():
                if values:
                    arr = np.array(values)
                    stats[protocol][f"{metric}_mean"] = float(np.mean(arr))
                    stats[protocol][f"{metric}_std"] = float(np.std(arr))
                    stats[protocol][f"{metric}_min"] = float(np.min(arr))
                    stats[protocol][f"{metric}_max"] = float(np.max(arr))
                    stats[protocol][f"{metric}_n"] = len(values)

        return stats

    def compute_pairwise_comparisons(self) -> List[Dict]:
        """Compute pairwise protocol comparisons."""
        protocol_data = self.aggregate_by_protocol()
        comparisons = []

        protocols = list(protocol_data.keys())
        for i, p1 in enumerate(protocols):
            for p2 in protocols[i + 1:]:
                if 'success_rate' in protocol_data[p1] and 'success_rate' in protocol_data[p2]:
                    sr1 = np.array(protocol_data[p1]['success_rate'])
                    sr2 = np.array(protocol_data[p2]['success_rate'])

                    if len(sr1) > 0 and len(sr2) > 0:
                        diff = np.mean(sr1) - np.mean(sr2)
                        # Simple effect size
                        pooled_std = np.sqrt((np.var(sr1) + np.var(sr2)) / 2)
                        effect_size = diff / pooled_std if pooled_std > 0 else 0

                        comparisons.append({
                            'protocol_1': p1,
                            'protocol_2': p2,
                            'mean_diff': float(diff),
                            'effect_size': float(effect_size),
                            'n_experiments': min(len(sr1), len(sr2)),
                        })

        return comparisons

    def compute_meta_statistics(self) -> Dict[str, float]:
        """Compute meta-level statistics."""
        return {
            'n_experiments': len(self.experiments),
            'n_protocols': len(set(p for e in self.experiments for p in e.protocols)),
            'total_episodes': sum(e.n_episodes for e in self.experiments),
            'task_types': len(set(e.task_type for e in self.experiments)),
        }

    def aggregate(self) -> AggregatedResults:
        """Run full aggregation."""
        return AggregatedResults(
            experiments=self.experiments,
            protocol_stats=self.compute_protocol_statistics(),
            pairwise_comparisons=self.compute_pairwise_comparisons(),
            meta_statistics=self.compute_meta_statistics(),
        )

    def generate_summary_table(self) -> str:
        """Generate markdown summary table."""
        stats = self.compute_protocol_statistics()

        if not stats:
            return "No results to aggregate."

        lines = [
            "# Aggregated Results Summary",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Experiments:** {len(self.experiments)}",
            "",
            "## Protocol Performance",
            "",
            "| Protocol | Success Rate | Std Dev | Partial Credit | Avg Turns | N |",
            "|----------|-------------|---------|----------------|-----------|---|",
        ]

        for protocol in sorted(stats.keys()):
            s = stats[protocol]
            sr = s.get('success_rate_mean', 0)
            sr_std = s.get('success_rate_std', 0)
            pc = s.get('partial_credit_mean_mean', s.get('partial_credit_mean', 0))
            turns = s.get('avg_turns_mean', 0)
            n = int(s.get('success_rate_n', 0))

            lines.append(
                f"| {protocol} | {sr:.3f} | {sr_std:.3f} | {pc:.3f} | {turns:.1f} | {n} |"
            )

        # Add comparisons
        comparisons = self.compute_pairwise_comparisons()
        if comparisons:
            lines.extend([
                "",
                "## Pairwise Comparisons",
                "",
                "| Comparison | Mean Diff | Effect Size | N |",
                "|------------|-----------|-------------|---|",
            ])

            for comp in comparisons:
                p1, p2 = comp['protocol_1'], comp['protocol_2']
                diff = comp['mean_diff']
                es = comp['effect_size']
                n = comp['n_experiments']

                sign = "+" if diff > 0 else ""
                lines.append(f"| {p1} vs {p2} | {sign}{diff:.3f} | {es:.2f} | {n} |")

        return "\n".join(lines)

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for publication."""
        stats = self.compute_protocol_statistics()

        if not stats:
            return "% No results to aggregate."

        lines = [
            "% Auto-generated LaTeX table",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Protocol Performance Comparison}",
            "\\label{tab:protocol_results}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Protocol & Success Rate & Partial Credit & Avg Turns & N \\\\",
            "\\midrule",
        ]

        for protocol in sorted(stats.keys()):
            s = stats[protocol]
            sr = s.get('success_rate_mean', 0)
            sr_std = s.get('success_rate_std', 0)
            pc = s.get('partial_credit_mean_mean', s.get('partial_credit_mean', 0))
            turns = s.get('avg_turns_mean', 0)
            n = int(s.get('success_rate_n', 0))

            lines.append(
                f"{protocol} & ${sr:.3f} \\pm {sr_std:.3f}$ & {pc:.3f} & {turns:.1f} & {n} \\\\"
            )

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def generate_json_export(self) -> Dict:
        """Generate JSON export of all aggregated data."""
        agg = self.aggregate()

        return {
            'generated': datetime.now().isoformat(),
            'meta': agg.meta_statistics,
            'protocol_stats': agg.protocol_stats,
            'pairwise_comparisons': agg.pairwise_comparisons,
            'experiments': [
                {
                    'id': e.experiment_id,
                    'task_type': e.task_type,
                    'n_episodes': e.n_episodes,
                    'protocols': e.protocols,
                }
                for e in agg.experiments
            ],
        }

    def save_aggregated(self, output_path: Path):
        """Save aggregated results to files."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Markdown summary
        summary_md = self.generate_summary_table()
        with open(output_path / "aggregated_summary.md", 'w') as f:
            f.write(summary_md)

        # LaTeX table
        latex = self.generate_latex_table()
        with open(output_path / "results_table.tex", 'w') as f:
            f.write(latex)

        # JSON export
        json_data = self.generate_json_export()
        with open(output_path / "aggregated_results.json", 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved aggregated results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate LPCA experiment results")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory containing experiment outputs",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        help="Filter to specific experiment prefixes",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default="outputs/aggregated",
        help="Directory to save aggregated results",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX tables",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON to stdout",
    )

    args = parser.parse_args()

    aggregator = ResultsAggregator(output_dir=args.output_dir)
    n_loaded = aggregator.load_all(experiment_filter=args.experiments)

    if n_loaded == 0:
        print(f"No experiments found in {args.output_dir}")
        return

    print(f"Loaded {n_loaded} experiments")

    if args.json:
        print(json.dumps(aggregator.generate_json_export(), indent=2))
    elif args.latex:
        print(aggregator.generate_latex_table())
    else:
        print(aggregator.generate_summary_table())

    # Save to files
    aggregator.save_aggregated(Path(args.save_to))


if __name__ == "__main__":
    main()
