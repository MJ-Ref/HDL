#!/usr/bin/env python3
"""
Statistical comparison of LPCA protocols.

Performs pre-registered statistical tests comparing protocols:
- Paired t-tests for success rate differences
- Cohen's d effect sizes
- Bonferroni-corrected multiple comparisons
- Bootstrap confidence intervals

Usage:
    python scripts/analysis/statistical_tests.py --results results/experiment_001
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load episode results from JSONL file."""
    episodes = []
    jsonl_path = results_dir / "episodes.jsonl"

    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))

    return episodes


def ensure_nonempty_episodes(episodes: List[Dict[str, Any]], results_dir: Path) -> None:
    """Fail fast if no episode data is present."""
    if episodes:
        return
    msg = (
        f"No episode data found at {results_dir}/episodes.jsonl. "
        "Statistical analysis requires real episode logs."
    )
    raise FileNotFoundError(msg)


def aggregate_by_protocol(episodes: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Aggregate episode results by protocol."""
    protocols = {}

    for ep in episodes:
        protocol = ep.get("protocol", "unknown")
        if protocol not in protocols:
            protocols[protocol] = {
                "successes": [],
                "partial_credits": [],
            }

        protocols[protocol]["successes"].append(1 if ep.get("success", False) else 0)
        protocols[protocol]["partial_credits"].append(ep.get("partial_credit", 0.0))

    return protocols


def proportion_test(s1: int, n1: int, s2: int, n2: int) -> Tuple[float, float]:
    """Two-proportion z-test."""
    p1 = s1 / n1 if n1 > 0 else 0
    p2 = s2 / n2 if n2 > 0 else 0
    p_pool = (s1 + s2) / (n1 + n2) if (n1 + n2) > 0 else 0

    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(
    data: List[float], n_bootstrap: int = 10000, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for mean."""
    if not data:
        return 0.0, 0.0

    data_arr = np.array(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def run_pairwise_comparisons(
    protocols: Dict[str, Dict], baseline: str = "P1", alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """Run pairwise comparisons against baseline."""
    if baseline not in protocols:
        print(f"Warning: Baseline {baseline} not found")
        return []

    baseline_data = protocols[baseline]
    results = []

    other_protocols = [p for p in protocols.keys() if p != baseline]

    for protocol in other_protocols:
        data = protocols[protocol]

        # Success rate comparison
        s1 = sum(baseline_data["successes"])
        n1 = len(baseline_data["successes"])
        s2 = sum(data["successes"])
        n2 = len(data["successes"])

        z, p_value = proportion_test(s1, n1, s2, n2)

        # Effect size
        d = cohens_d(baseline_data["successes"], data["successes"])

        # Success rates
        sr_baseline = s1 / n1 if n1 > 0 else 0
        sr_protocol = s2 / n2 if n2 > 0 else 0

        results.append(
            {
                "comparison": f"{protocol} vs {baseline}",
                "protocol": protocol,
                "baseline": baseline,
                "sr_baseline": sr_baseline,
                "sr_protocol": sr_protocol,
                "difference": sr_protocol - sr_baseline,
                "z_statistic": z,
                "p_value": p_value,
                "cohens_d": d,
                "n_baseline": n1,
                "n_protocol": n2,
            }
        )

    # Apply Bonferroni correction
    n_comparisons = len(results)
    adjusted_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    for r in results:
        r["bonferroni_alpha"] = adjusted_alpha
        r["significant"] = r["p_value"] < adjusted_alpha

    return results


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def format_comparison_table(results: List[Dict]) -> str:
    """Format comparison results as markdown table."""
    lines = [
        "| Comparison | Δ Success | p-value | Cohen's d | Effect | Significant |",
        "|------------|-----------|---------|-----------|--------|-------------|",
    ]

    for r in sorted(results, key=lambda x: x["p_value"]):
        sig = "Yes" if r["significant"] else "No"
        effect = interpret_effect_size(r["cohens_d"])
        line = (
            f"| {r['comparison']} | "
            f"{r['difference']:+.3f} | "
            f"{r['p_value']:.4f} | "
            f"{r['cohens_d']:.3f} | "
            f"{effect} | "
            f"{sig} |"
        )
        lines.append(line)

    return "\n".join(lines)


def run_hypothesis_tests(protocols: Dict[str, Dict]) -> Dict[str, Any]:
    """Run pre-registered hypothesis tests."""
    results = {}

    # H1: Latent > Text
    text_protocols = [p for p in protocols.keys() if p.startswith("P")]
    latent_protocols = [
        p for p in protocols.keys() if p.startswith("A") or p.startswith("E")
    ]

    if text_protocols and latent_protocols:
        best_text = max(
            text_protocols, key=lambda p: np.mean(protocols[p]["successes"])
        )
        best_latent = max(
            latent_protocols, key=lambda p: np.mean(protocols[p]["successes"])
        )

        sr_text = np.mean(protocols[best_text]["successes"])
        sr_latent = np.mean(protocols[best_latent]["successes"])

        # Test H1: latent > text by at least 15%
        h1_threshold = 0.15
        h1_met = (sr_latent - sr_text) >= h1_threshold

        z, p = proportion_test(
            sum(protocols[best_latent]["successes"]),
            len(protocols[best_latent]["successes"]),
            sum(protocols[best_text]["successes"]),
            len(protocols[best_text]["successes"]),
        )

        results["H1"] = {
            "hypothesis": "Latent ≥ Text + 15%",
            "best_text": best_text,
            "best_latent": best_latent,
            "sr_text": sr_text,
            "sr_latent": sr_latent,
            "difference": sr_latent - sr_text,
            "threshold": h1_threshold,
            "met": h1_met,
            "p_value": p,
            "status": "SUPPORTED" if h1_met and p < 0.05 else "NOT SUPPORTED",
        }

    # Communication matters check (P1 > P0)
    if "P0" in protocols and "P1" in protocols:
        sr_p0 = np.mean(protocols["P0"]["successes"])
        sr_p1 = np.mean(protocols["P1"]["successes"])

        z, p = proportion_test(
            sum(protocols["P1"]["successes"]),
            len(protocols["P1"]["successes"]),
            sum(protocols["P0"]["successes"]),
            len(protocols["P0"]["successes"]),
        )

        results["Communication"] = {
            "hypothesis": "P1 > P0 (communication matters)",
            "sr_p0": sr_p0,
            "sr_p1": sr_p1,
            "difference": sr_p1 - sr_p0,
            "p_value": p,
            "status": "CONFIRMED"
            if sr_p1 > sr_p0 + 0.1 and p < 0.05
            else "NOT CONFIRMED",
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of LPCA results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results directory",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="P1",
        help="Baseline protocol for comparisons",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results",
    )

    args = parser.parse_args()

    results_dir = Path(args.results)

    # Load data
    print(f"Loading results from: {results_dir}")
    episodes = load_results(results_dir)

    try:
        ensure_nonempty_episodes(episodes, results_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)

    print(f"Loaded {len(episodes)} episodes")

    protocols = aggregate_by_protocol(episodes)
    print(f"Protocols: {list(protocols.keys())}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for name, data in sorted(protocols.items()):
        n = len(data["successes"])
        sr = np.mean(data["successes"])
        ci = bootstrap_ci(data["successes"])
        print(f"{name}: n={n}, success={sr:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")

    # Pairwise comparisons
    print("\n" + "=" * 60)
    print(f"PAIRWISE COMPARISONS (vs {args.baseline})")
    print("=" * 60)

    comparisons = run_pairwise_comparisons(protocols, args.baseline, args.alpha)
    if comparisons:
        print(format_comparison_table(comparisons))
    else:
        print("No comparisons available")

    # Hypothesis tests
    print("\n" + "=" * 60)
    print("PRE-REGISTERED HYPOTHESIS TESTS")
    print("=" * 60)

    hypothesis_results = run_hypothesis_tests(protocols)
    for name, result in hypothesis_results.items():
        print(f"\n{name}: {result['hypothesis']}")
        print(f"  Status: {result['status']}")
        if "difference" in result:
            print(f"  Difference: {result['difference']:+.3f}")
        if "p_value" in result:
            print(f"  p-value: {result['p_value']:.4f}")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write("# Statistical Analysis Results\n\n")
            f.write("## Pairwise Comparisons\n\n")
            f.write(format_comparison_table(comparisons))
            f.write("\n\n## Hypothesis Tests\n\n")
            for name, result in hypothesis_results.items():
                f.write(f"### {name}\n")
                f.write(f"**Hypothesis:** {result['hypothesis']}\n\n")
                f.write(f"**Status:** {result['status']}\n\n")
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
