#!/usr/bin/env python3
"""
Generate plots from LPCA experiment results.

Usage:
    python scripts/analysis/plot_results.py --results results/experiment_001
    python scripts/analysis/plot_results.py --results results/experiment_001 --output figures/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Lazy import for matplotlib to avoid dependency issues
def get_plt():
    import matplotlib.pyplot as plt
    return plt


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load episode results from JSONL file."""
    episodes = []
    jsonl_path = results_dir / "episodes.jsonl"

    if jsonl_path.exists():
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))

    return episodes


def aggregate_by_protocol(episodes: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Aggregate episode results by protocol."""
    protocols = {}

    for ep in episodes:
        protocol = ep.get('protocol', 'unknown')
        if protocol not in protocols:
            protocols[protocol] = {
                'successes': [],
                'partial_credits': [],
                'turns': [],
                'bits': [],
                'episodes': [],
            }

        protocols[protocol]['successes'].append(ep.get('success', False))
        protocols[protocol]['partial_credits'].append(ep.get('partial_credit', 0.0))
        protocols[protocol]['turns'].append(ep.get('n_turns', 0))
        protocols[protocol]['bits'].append(ep.get('total_bits', 0))
        protocols[protocol]['episodes'].append(ep)

    # Calculate summary statistics
    for protocol, data in protocols.items():
        n = len(data['successes'])
        data['n_episodes'] = n
        data['success_rate'] = np.mean(data['successes']) if n > 0 else 0.0
        data['success_ci'] = wilson_ci(sum(data['successes']), n)
        data['partial_credit_mean'] = np.mean(data['partial_credits']) if n > 0 else 0.0
        data['partial_credit_std'] = np.std(data['partial_credits']) if n > 0 else 0.0
        data['turns_mean'] = np.mean(data['turns']) if n > 0 else 0.0
        data['bits_mean'] = np.mean(data['bits']) if n > 0 else 0.0

    return protocols


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """Calculate Wilson score confidence interval."""
    if total == 0:
        return (0.0, 0.0)

    from scipy import stats
    import math

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)
    ) / denominator

    return (max(0, center - margin), min(1, center + margin))


def plot_success_rates(protocols: Dict, output_path: Optional[Path] = None):
    """Plot success rates by protocol with confidence intervals."""
    plt = get_plt()

    # Sort protocols by success rate
    sorted_protocols = sorted(
        protocols.items(),
        key=lambda x: x[1]['success_rate'],
        reverse=True
    )

    names = [p[0] for p in sorted_protocols]
    rates = [p[1]['success_rate'] for p in sorted_protocols]
    ci_lower = [p[1]['success_ci'][0] for p in sorted_protocols]
    ci_upper = [p[1]['success_ci'][1] for p in sorted_protocols]

    # Calculate error bars
    yerr_lower = [r - l for r, l in zip(rates, ci_lower)]
    yerr_upper = [u - r for r, u in zip(rates, ci_upper)]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(names))
    bars = ax.bar(x, rates, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.8)

    # Color code by protocol type
    colors = []
    for name in names:
        if name.startswith('P'):
            colors.append('#3498db')  # Blue for text
        elif name.startswith('E') or name.startswith('A'):
            colors.append('#e74c3c')  # Red for latent
        else:
            colors.append('#95a5a6')  # Gray for others

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Protocol')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate by Protocol (95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)

    # Add value labels
    for i, (rate, ci_u) in enumerate(zip(rates, ci_upper)):
        ax.annotate(f'{rate:.2f}', xy=(i, ci_u + 0.02), ha='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / 'success_rates.png', dpi=150)
        print(f"Saved: {output_path / 'success_rates.png'}")
    else:
        plt.show()

    plt.close()


def plot_capability_vs_bits(protocols: Dict, output_path: Optional[Path] = None):
    """Plot capability vs communication cost (bits)."""
    plt = get_plt()

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in protocols.items():
        bits = data['bits_mean']
        success = data['success_rate']
        ci = data['success_ci']

        # Marker style by protocol type
        if name.startswith('P'):
            marker, color = 'o', '#3498db'
        elif name.startswith('E') or name.startswith('A'):
            marker, color = 's', '#e74c3c'
        else:
            marker, color = '^', '#95a5a6'

        ax.errorbar(
            bits, success,
            yerr=[[success - ci[0]], [ci[1] - success]],
            marker=marker, markersize=10, color=color,
            label=name, capsize=5
        )

    ax.set_xlabel('Average Bits per Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Capability vs Communication Cost')
    ax.legend(loc='best')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(left=-50)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / 'capability_vs_bits.png', dpi=150)
        print(f"Saved: {output_path / 'capability_vs_bits.png'}")
    else:
        plt.show()

    plt.close()


def plot_partial_credit_distribution(protocols: Dict, output_path: Optional[Path] = None):
    """Plot partial credit distribution by protocol."""
    plt = get_plt()

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    data_to_plot = []
    labels = []

    for i, (name, data) in enumerate(sorted(protocols.items())):
        positions.append(i + 1)
        data_to_plot.append(data['partial_credits'])
        labels.append(name)

    bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True)

    # Color by protocol type
    for i, (patch, name) in enumerate(zip(bp['boxes'], labels)):
        if name.startswith('P'):
            patch.set_facecolor('#3498db')
        elif name.startswith('E') or name.startswith('A'):
            patch.set_facecolor('#e74c3c')
        else:
            patch.set_facecolor('#95a5a6')
        patch.set_alpha(0.7)

    ax.set_xlabel('Protocol')
    ax.set_ylabel('Partial Credit')
    ax.set_title('Partial Credit Distribution by Protocol')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / 'partial_credit_dist.png', dpi=150)
        print(f"Saved: {output_path / 'partial_credit_dist.png'}")
    else:
        plt.show()

    plt.close()


def plot_turns_comparison(protocols: Dict, output_path: Optional[Path] = None):
    """Plot turns to completion by protocol."""
    plt = get_plt()

    sorted_protocols = sorted(protocols.items(), key=lambda x: x[0])

    names = [p[0] for p in sorted_protocols]
    turns_mean = [p[1]['turns_mean'] for p in sorted_protocols]
    turns_std = [np.std(p[1]['turns']) if p[1]['turns'] else 0 for p in sorted_protocols]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(names))
    bars = ax.bar(x, turns_mean, yerr=turns_std, capsize=5, alpha=0.8)

    # Color by protocol type
    for bar, name in zip(bars, names):
        if name.startswith('P'):
            bar.set_color('#3498db')
        elif name.startswith('E') or name.startswith('A'):
            bar.set_color('#e74c3c')
        else:
            bar.set_color('#95a5a6')

    ax.set_xlabel('Protocol')
    ax.set_ylabel('Average Turns')
    ax.set_title('Turns to Completion by Protocol')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path / 'turns_comparison.png', dpi=150)
        print(f"Saved: {output_path / 'turns_comparison.png'}")
    else:
        plt.show()

    plt.close()


def generate_summary_table(protocols: Dict) -> str:
    """Generate markdown summary table."""
    lines = [
        "| Protocol | Episodes | Success Rate | 95% CI | Partial Credit | Avg Turns | Avg Bits |",
        "|----------|----------|--------------|--------|----------------|-----------|----------|",
    ]

    for name in sorted(protocols.keys()):
        data = protocols[name]
        ci = data['success_ci']
        line = (
            f"| {name} | {data['n_episodes']} | "
            f"{data['success_rate']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{data['partial_credit_mean']:.3f} ± {data['partial_credit_std']:.3f} | "
            f"{data['turns_mean']:.1f} | {data['bits_mean']:.0f} |"
        )
        lines.append(line)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate plots from LPCA results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results directory containing episodes.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for figures (default: show interactively)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for figures",
    )

    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    # Load and aggregate results
    print(f"Loading results from: {results_dir}")
    episodes = load_results(results_dir)

    if not episodes:
        print("No episodes found. Generating demo data...")
        # Generate demo data for testing
        episodes = generate_demo_data()

    print(f"Loaded {len(episodes)} episodes")

    protocols = aggregate_by_protocol(episodes)
    print(f"Found protocols: {list(protocols.keys())}")

    # Generate plots
    print("\nGenerating plots...")
    plot_success_rates(protocols, output_path)
    plot_capability_vs_bits(protocols, output_path)
    plot_partial_credit_distribution(protocols, output_path)
    plot_turns_comparison(protocols, output_path)

    # Generate summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(generate_summary_table(protocols))

    if output_path:
        # Save summary to markdown
        summary_path = output_path / "summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Experiment Results Summary\n\n")
            f.write(generate_summary_table(protocols))
            f.write("\n\n## Figures\n\n")
            f.write("- `success_rates.png` - Success rates by protocol\n")
            f.write("- `capability_vs_bits.png` - Capability vs communication cost\n")
            f.write("- `partial_credit_dist.png` - Partial credit distribution\n")
            f.write("- `turns_comparison.png` - Turns to completion\n")
        print(f"\nSummary saved to: {summary_path}")


def generate_demo_data() -> List[Dict]:
    """Generate demo data for testing plots."""
    np.random.seed(42)
    episodes = []

    for protocol, base_success, avg_bits in [
        ('P0', 0.0, 0),
        ('P1', 0.4, 500),
        ('P2', 0.35, 250),
        ('A0', 0.55, 1000),
        ('E0', 0.45, 300),
    ]:
        for i in range(50):
            success = np.random.random() < base_success
            episodes.append({
                'protocol': protocol,
                'success': success,
                'partial_credit': base_success + np.random.normal(0, 0.1),
                'n_turns': np.random.randint(2, 15),
                'total_bits': int(avg_bits * (1 + np.random.normal(0, 0.2))),
            })

    return episodes


if __name__ == "__main__":
    main()
