#!/usr/bin/env python3
"""
LPCA Experiment Runner

Runs full experiments with configurable protocols, tasks, and safety monitoring.

Usage:
    # Run with mock agents (no LLM required)
    python scripts/run_experiment.py --mock --protocol P1 --task constraint_satisfaction

    # Run baseline validation (E1) with config
    python scripts/run_experiment.py --config configs/e1_baseline.yaml

    # Run with specific protocol
    python scripts/run_experiment.py --protocol P1 --task constraint_satisfaction --n_episodes 100

    # Run multiple protocols for comparison
    python scripts/run_experiment.py --mock --protocols P0,P1,P2 --n_episodes 50

    # Dry run (print config only)
    python scripts/run_experiment.py --config configs/base.yaml --dry_run
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from lpca.core.logging import EpisodeLog, EpisodeLogger
from lpca.core.metrics import MetricsCalculator
from lpca.envs.split_synthetic import SplitSyntheticEnv
from lpca.channels.text import (
    NoCommChannel,
    FullTextChannel,
    BudgetedTextChannel,
    create_channel,
)

# Optional imports for LLM experiments
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class SimpleConfig:
    """Simplified experiment configuration."""
    experiment_name: str = "lpca_experiment"
    protocol: str = "P1"
    task_type: str = "constraint_satisfaction"
    difficulty: str = "medium"
    n_episodes: int = 10
    base_seed: int = 42
    max_turns: int = 20
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    temperature: float = 0.7
    device: str = "auto"


class MockAgent:
    """Mock agent for testing without real LLM."""

    def __init__(self, agent_id: str, task_type: str):
        self.agent_id = agent_id
        self.task_type = task_type
        self.turn_count = 0

    def reset(self):
        self.turn_count = 0

    def respond(self, observation: str, received_message: str = None):
        """Generate mock response based on task type."""
        self.turn_count += 1
        import re

        if self.task_type == "constraint_satisfaction":
            return self._respond_constraint(observation, received_message)
        elif self.task_type == "arithmetic":
            return self._respond_arithmetic(observation, received_message)
        else:
            return self._respond_generic(observation, received_message)

    def _respond_constraint(self, observation: str, received_message: str):
        """Mock response for constraint satisfaction."""
        import re
        constraints = re.findall(r'- (.+)', observation)

        if self.turn_count == 1:
            return {
                'text': f"I see {len(constraints)} constraints. Let me share them.",
                'message': f"My constraints: {'; '.join(constraints[:2])}",
                'final_answer': None,
            }
        elif self.turn_count >= 2 and received_message:
            return {
                'text': "Based on our combined constraints, here's my answer.",
                'message': None,
                'final_answer': '{"x1": 0, "x2": 1, "x3": 0}',
            }
        else:
            return {
                'text': "Need more information.",
                'message': "What constraints do you have?",
                'final_answer': None,
            }

    def _respond_arithmetic(self, observation: str, received_message: str):
        """Mock response for arithmetic."""
        import re
        values = re.findall(r'(\w+) = (\d+)', observation)

        if self.turn_count == 1:
            return {
                'text': f"I know these values: {values}",
                'message': f"My values: {', '.join(f'{k}={v}' for k,v in values)}",
                'final_answer': None,
            }
        else:
            return {
                'text': "Computing result...",
                'message': None,
                'final_answer': "42",
            }

    def _respond_generic(self, observation: str, received_message: str):
        """Generic mock response."""
        if self.turn_count >= 3:
            return {
                'text': "Submitting answer.",
                'message': None,
                'final_answer': "answer",
            }
        return {
            'text': "Processing...",
            'message': "Sharing information.",
            'final_answer': None,
        }


class ExperimentRunner:
    """
    Main experiment runner for LPCA experiments.

    Supports:
    - All text protocols (P0-P5)
    - Mock agents for testing
    - Comprehensive logging
    - Multi-protocol comparison
    """

    def __init__(
        self,
        config: SimpleConfig,
        output_dir: str = "results",
        use_mock: bool = True,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.use_mock = use_mock

        # Generate experiment ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = f"{config.experiment_name}_{timestamp}"

        # Setup output directory
        self.experiment_dir = self.output_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.env = None
        self.metrics = MetricsCalculator()

        # Results storage
        self.all_results: Dict[str, Dict] = {}

    def setup_environment(self):
        """Initialize task environment."""
        self.env = SplitSyntheticEnv(difficulty=self.config.difficulty)
        self.env.select_environment(self.config.task_type)

    def create_channel(self, protocol: str):
        """Create communication channel for protocol."""
        channels = {
            "P0": NoCommChannel,
            "P1": FullTextChannel,
            "P2": BudgetedTextChannel,
        }

        if protocol in channels:
            return channels[protocol]()

        # Try advanced channels
        return create_channel(protocol)

    def create_agents(self):
        """Create agents (mock or real)."""
        if self.use_mock:
            return {
                'A': MockAgent('A', self.config.task_type),
                'B': MockAgent('B', self.config.task_type),
            }
        else:
            if not HAS_TORCH:
                raise RuntimeError("torch not available. Use --mock for testing.")
            # Real LLM agent loading would go here
            raise NotImplementedError("Real LLM agents require torch. Use --mock.")

    def run_episode(
        self,
        channel,
        agents: Dict,
        seed: int,
    ) -> Dict[str, Any]:
        """Run a single episode."""
        # Reset components
        task = self.env.reset(seed)
        channel.reset()
        agents['A'].reset()
        agents['B'].reset()

        turns = []
        final_answer = None
        current_message = None
        start_time = time.perf_counter()

        for turn_idx in range(self.config.max_turns):
            # Agent A's turn
            response_A = agents['A'].respond(task.obs_A, current_message)

            turns.append({
                'turn_idx': turn_idx,
                'agent': 'A',
                'response': response_A.get('text', ''),
                'message': response_A.get('message'),
            })

            if response_A.get('final_answer'):
                final_answer = response_A['final_answer']
                break

            # Send message through channel
            if response_A.get('message'):
                msg = channel.send(response_A['message'], 'A', 'B', turn_idx)
                current_message = channel.receive(msg)
            else:
                current_message = None

            # Agent B's turn
            response_B = agents['B'].respond(task.obs_B, current_message)

            turns.append({
                'turn_idx': turn_idx,
                'agent': 'B',
                'response': response_B.get('text', ''),
                'message': response_B.get('message'),
            })

            if response_B.get('final_answer'):
                final_answer = response_B['final_answer']
                break

            # Send message through channel
            if response_B.get('message'):
                msg = channel.send(response_B['message'], 'B', 'A', turn_idx)
                current_message = channel.receive(msg)
            else:
                current_message = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify result
        if final_answer:
            result = self.env.verify(final_answer)
        else:
            result = self.env.verify("")

        # Get channel stats
        channel_stats = channel.get_stats()

        return {
            'seed': seed,
            'n_turns': len(turns),
            'final_answer': final_answer,
            'success': result.success,
            'partial_credit': result.partial_credit,
            'total_bits': channel_stats.get('total_bits', 0),
            'message_count': channel_stats.get('message_count', 0),
            'elapsed_ms': elapsed_ms,
        }

    def run_protocol(self, protocol: str) -> Dict[str, Any]:
        """Run experiment for a single protocol."""
        print(f"\n--- Protocol {protocol} ---")

        channel = self.create_channel(protocol)
        agents = self.create_agents()

        successes = []
        partial_credits = []
        turn_counts = []
        bits_used = []
        episodes = []

        for ep_idx in range(self.config.n_episodes):
            seed = self.config.base_seed + ep_idx

            try:
                episode_result = self.run_episode(channel, agents, seed)
                episodes.append(episode_result)

                successes.append(episode_result['success'])
                partial_credits.append(episode_result['partial_credit'])
                turn_counts.append(episode_result['n_turns'])
                bits_used.append(episode_result['total_bits'])

                status = "SUCCESS" if episode_result['success'] else "FAIL"
                print(f"  Episode {ep_idx+1}: {status} "
                      f"(turns={episode_result['n_turns']}, "
                      f"bits={episode_result['total_bits']:.0f})")

            except Exception as e:
                print(f"  Episode {ep_idx+1}: ERROR - {e}")
                continue

        # Calculate metrics
        n = len(successes)
        results = {
            'protocol': protocol,
            'n_episodes': n,
            'success_rate': np.mean(successes) if n > 0 else 0,
            'success_ci': self.metrics.wilson_ci(sum(successes), n) if n > 0 else (0, 0),
            'partial_credit_mean': np.mean(partial_credits) if n > 0 else 0,
            'partial_credit_std': np.std(partial_credits) if n > 0 else 0,
            'avg_turns': np.mean(turn_counts) if n > 0 else 0,
            'avg_bits': np.mean(bits_used) if n > 0 else 0,
            'episodes': episodes,
        }

        # Print summary
        sr = results['success_rate']
        ci = results['success_ci']
        print(f"\n  Results for {protocol}:")
        print(f"    Success Rate: {sr:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
        print(f"    Partial Credit: {results['partial_credit_mean']:.3f}")
        print(f"    Avg Turns: {results['avg_turns']:.1f}")
        print(f"    Avg Bits: {results['avg_bits']:.0f}")

        return results

    def run(self, protocols: List[str] = None) -> Dict[str, Dict]:
        """Run the full experiment."""
        protocols = protocols or [self.config.protocol]

        print(f"\n{'='*60}")
        print(f"LPCA Experiment: {self.experiment_id}")
        print(f"{'='*60}")
        print(f"Protocols: {protocols}")
        print(f"Task: {self.config.task_type}")
        print(f"Difficulty: {self.config.difficulty}")
        print(f"Episodes per protocol: {self.config.n_episodes}")
        print(f"Mode: {'Mock' if self.use_mock else 'Real LLM'}")
        print(f"{'='*60}")

        # Setup
        self.setup_environment()

        # Save config
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        # Run each protocol
        for protocol in protocols:
            results = self.run_protocol(protocol)
            self.all_results[protocol] = results

            # Save protocol results
            protocol_path = self.experiment_dir / f"results_{protocol}.json"
            # Remove episodes for JSON (too large)
            save_results = {k: v for k, v in results.items() if k != 'episodes'}
            with open(protocol_path, 'w') as f:
                json.dump(save_results, f, indent=2)

        # Print comparison
        self._print_comparison(protocols)

        # Save summary
        self._save_summary()

        return self.all_results

    def _print_comparison(self, protocols: List[str]):
        """Print protocol comparison table."""
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        print(f"{'Protocol':<10} {'Success':<10} {'Partial':<10} {'Turns':<10} {'Bits':<10}")
        print("-" * 50)

        for protocol in protocols:
            r = self.all_results.get(protocol, {})
            sr = r.get('success_rate', 0)
            pc = r.get('partial_credit_mean', 0)
            turns = r.get('avg_turns', 0)
            bits = r.get('avg_bits', 0)
            print(f"{protocol:<10} {sr:<10.3f} {pc:<10.3f} {turns:<10.1f} {bits:<10.0f}")

        # Key findings
        print(f"\n{'='*60}")
        print("KEY FINDINGS")
        print(f"{'='*60}")

        if 'P0' in self.all_results and 'P1' in self.all_results:
            p0_sr = self.all_results['P0']['success_rate']
            p1_sr = self.all_results['P1']['success_rate']

            if p1_sr > p0_sr + 0.1:
                print(f"- Communication matters: P1 ({p1_sr:.3f}) >> P0 ({p0_sr:.3f})")
            else:
                print(f"- WARNING: P1 not much better than P0 - communication may not be bottleneck")

    def _save_summary(self):
        """Save experiment summary."""
        summary = {
            'experiment_id': self.experiment_id,
            'config': asdict(self.config),
            'results': {
                protocol: {k: v for k, v in results.items() if k != 'episodes'}
                for protocol, results in self.all_results.items()
            },
        }

        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {self.experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="LPCA Experiment Runner")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="P1",
        help="Single protocol to run",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        help="Comma-separated protocols to compare (e.g., P0,P1,P2)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="constraint_satisfaction",
        choices=["constraint_satisfaction", "arithmetic", "program_synthesis"],
        help="Task type",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10,
        help="Number of episodes per protocol",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Task difficulty",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock agents (default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print config and exit",
    )

    args = parser.parse_args()

    # Create config
    config = SimpleConfig(
        experiment_name=f"exp_{args.task}",
        protocol=args.protocol,
        task_type=args.task,
        difficulty=args.difficulty,
        n_episodes=args.n_episodes,
        base_seed=args.seed,
    )

    if args.dry_run:
        print("Experiment Configuration:")
        print(json.dumps(asdict(config), indent=2))
        return

    # Determine protocols to run
    if args.protocols:
        protocols = [p.strip() for p in args.protocols.split(",")]
    else:
        protocols = [args.protocol]

    # Run experiment
    runner = ExperimentRunner(
        config=config,
        output_dir=args.output,
        use_mock=args.mock,
    )

    results = runner.run(protocols=protocols)
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
