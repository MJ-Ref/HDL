#!/usr/bin/env python3
"""
E2: Text Baseline Sweep

Sweeps through P0-P5 protocols to establish strong text baselines
and generate capability-vs-bits curves.

Usage:
    # Quick sweep (10 episodes each)
    python scripts/run_e2_sweep.py --n_episodes 10

    # Full sweep for publication (50+ episodes)
    python scripts/run_e2_sweep.py --n_episodes 50

    # Test specific protocols
    python scripts/run_e2_sweep.py --protocols P1,P2,P5 --n_episodes 20
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lpca.agents.llm_agent import LLMAgent
from lpca.agents.model_wrapper import ModelWrapper
from lpca.channels.text import create_channel
from lpca.core.metrics import MetricsCalculator
from lpca.envs.split_synthetic import SplitSyntheticEnv


@dataclass
class E2Config:
    """Configuration for E2 sweep."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    task_type: str = "constraint_satisfaction"
    difficulty: str = "easy"
    n_episodes: int = 20
    base_seed: int = 42
    max_turns: int = 6
    temperature: float = 0.3
    device: str = "mps"
    # Budget levels for P2-P4 (bytes)
    budget_levels: List[int] = None

    def __post_init__(self):
        if self.budget_levels is None:
            self.budget_levels = [64, 128, 256, 512]


class E2SweepRunner:
    """Runs E2 text baseline sweep."""

    def __init__(self, config: E2Config, output_dir: str = "results"):
        self.config = config
        self.output_dir = Path(output_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = f"e2_sweep_{timestamp}"
        self.experiment_dir = self.output_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.env = SplitSyntheticEnv(difficulty=config.difficulty)
        self.env.select_environment(config.task_type)
        self.metrics = MetricsCalculator()

        self._model = None
        self._tokenizer = None
        self._wrapper = None

        self.all_results = {}

    def _load_model(self):
        if self._model is not None:
            return

        print(f"Loading model: {self.config.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float32 if self.config.device == "mps" else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=None if self.config.device == "mps" else "auto",
            trust_remote_code=True,
        )
        if self.config.device == "mps":
            self._model = self._model.to("mps")

        self._wrapper = ModelWrapper(self._model, self._tokenizer, self.config.device)
        print(f"Model loaded: {self._wrapper.n_layers} layers")

    def create_agents(self) -> Dict[str, LLMAgent]:
        self._load_model()
        return {
            'A': LLMAgent('A', self._wrapper, self._tokenizer, temperature=self.config.temperature),
            'B': LLMAgent('B', self._wrapper, self._tokenizer, temperature=self.config.temperature),
        }

    def run_episode(self, agents, channel, seed: int) -> Dict[str, Any]:
        """Run a single episode."""
        task = self.env.reset(seed)
        channel.reset()
        agents['A'].reset()
        agents['B'].reset()

        turns = []
        final_answer = None
        current_message = None
        start_time = time.perf_counter()

        for turn_idx in range(self.config.max_turns):
            # Agent A
            response_A = agents['A'].respond(task.obs_A, current_message)
            turns.append({'agent': 'A', 'answer': response_A.final_answer})

            if response_A.final_answer:
                final_answer = response_A.final_answer
                break

            if response_A.message_to_partner:
                msg = channel.send(response_A.message_to_partner, 'A', 'B', turn_idx)
                current_message = channel.receive(msg)
            else:
                current_message = None

            # Agent B
            response_B = agents['B'].respond(task.obs_B, current_message)
            turns.append({'agent': 'B', 'answer': response_B.final_answer})

            if response_B.final_answer:
                final_answer = response_B.final_answer
                break

            if response_B.message_to_partner:
                msg = channel.send(response_B.message_to_partner, 'B', 'A', turn_idx)
                current_message = channel.receive(msg)
            else:
                current_message = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if final_answer:
            result = self.env.verify(final_answer)
        else:
            result = self.env.verify("")

        channel_stats = channel.get_stats()

        return {
            'seed': seed,
            'n_turns': len(turns),
            'success': result.success,
            'partial_credit': result.partial_credit,
            'total_bits': channel_stats.get('total_bits', 0),
            'elapsed_ms': elapsed_ms,
        }

    def run_protocol(self, protocol: str, channel_kwargs: Dict = None) -> Dict[str, Any]:
        """Run experiment for a single protocol configuration."""
        channel_kwargs = channel_kwargs or {}
        config_str = f"{protocol}"
        if 'max_bytes' in channel_kwargs:
            config_str += f"_{channel_kwargs['max_bytes']}B"

        print(f"\n  {config_str}...", end=" ", flush=True)

        channel = create_channel(protocol, **channel_kwargs)
        agents = self.create_agents()

        successes = []
        partial_credits = []
        bits_used = []
        times = []

        for ep_idx in range(self.config.n_episodes):
            seed = self.config.base_seed + ep_idx
            try:
                result = self.run_episode(agents, channel, seed)
                successes.append(result['success'])
                partial_credits.append(result['partial_credit'])
                bits_used.append(result['total_bits'])
                times.append(result['elapsed_ms'])
            except Exception as e:
                print(f"Error: {e}")
                continue

        n = len(successes)
        success_rate = np.mean(successes) if n > 0 else 0
        ci = self.metrics.wilson_ci(sum(successes), n) if n > 0 else (0, 0)

        print(f"{success_rate:.0%} ({sum(successes)}/{n})")

        return {
            'protocol': protocol,
            'config': channel_kwargs,
            'config_str': config_str,
            'n_episodes': n,
            'success_rate': float(success_rate),
            'success_ci': ci,
            'partial_credit_mean': float(np.mean(partial_credits)) if n > 0 else 0,
            'avg_bits': float(np.mean(bits_used)) if n > 0 else 0,
            'avg_time_ms': float(np.mean(times)) if n > 0 else 0,
        }

    def run_sweep(self, protocols: List[str] = None):
        """Run full E2 sweep."""
        if protocols is None:
            protocols = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']

        print("=" * 60)
        print("E2: Text Baseline Sweep")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Task: {self.config.task_type} ({self.config.difficulty})")
        print(f"Episodes per config: {self.config.n_episodes}")
        print(f"Protocols: {protocols}")
        print("=" * 60)

        results = []

        for protocol in protocols:
            print(f"\nProtocol {protocol}:")

            if protocol in ['P0', 'P1']:
                # No budget variants
                result = self.run_protocol(protocol)
                results.append(result)

            elif protocol in ['P2', 'P3', 'P4']:
                # Test at multiple budget levels
                for max_bytes in self.config.budget_levels:
                    result = self.run_protocol(protocol, {'max_bytes': max_bytes})
                    results.append(result)

            elif protocol == 'P5':
                # Structured workspace (optional budget)
                result = self.run_protocol(protocol)
                results.append(result)
                # Also test with budget
                for max_bytes in [128, 256]:
                    result = self.run_protocol(protocol, {'max_bytes': max_bytes})
                    results.append(result)

        # Print summary table
        self._print_summary(results)

        # Save results
        self._save_results(results)

        return results

    def _print_summary(self, results: List[Dict]):
        """Print summary table."""
        print("\n" + "=" * 70)
        print("E2 SWEEP RESULTS")
        print("=" * 70)
        print(f"{'Config':<20} {'Success':<12} {'95% CI':<18} {'Avg Bits':<12}")
        print("-" * 70)

        # Sort by success rate
        sorted_results = sorted(results, key=lambda x: x['success_rate'], reverse=True)

        for r in sorted_results:
            ci_str = f"[{r['success_ci'][0]:.1%}, {r['success_ci'][1]:.1%}]"
            print(f"{r['config_str']:<20} {r['success_rate']:<12.1%} {ci_str:<18} {r['avg_bits']:<12.0f}")

        # Key findings
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        # Find best protocol
        best = sorted_results[0]
        print(f"Best: {best['config_str']} at {best['success_rate']:.1%}")

        # Check if any budgeted protocol beats P1
        p1_result = next((r for r in results if r['protocol'] == 'P1'), None)
        if p1_result:
            p1_sr = p1_result['success_rate']
            better_than_p1 = [r for r in results
                            if r['success_rate'] > p1_sr and r['protocol'] != 'P1']
            if better_than_p1:
                print(f"Protocols beating P1 ({p1_sr:.1%}):")
                for r in better_than_p1:
                    print(f"  - {r['config_str']}: {r['success_rate']:.1%}")
            else:
                print(f"P1 (full text) is best or tied at {p1_sr:.1%}")

    def _save_results(self, results: List[Dict]):
        """Save results to file."""
        output = {
            'experiment_id': self.experiment_id,
            'config': asdict(self.config),
            'results': results,
        }

        output_path = self.experiment_dir / "results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")

        # Also save CSV for easy plotting
        csv_path = self.experiment_dir / "results.csv"
        with open(csv_path, 'w') as f:
            f.write("config,success_rate,ci_low,ci_high,avg_bits,n_episodes\n")
            for r in results:
                f.write(f"{r['config_str']},{r['success_rate']:.4f},"
                       f"{r['success_ci'][0]:.4f},{r['success_ci'][1]:.4f},"
                       f"{r['avg_bits']:.0f},{r['n_episodes']}\n")

        print(f"CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="E2: Text Baseline Sweep")
    parser.add_argument("--protocols", type=str, help="Comma-separated protocols")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="results")

    args = parser.parse_args()

    config = E2Config(
        model_name=args.model,
        difficulty=args.difficulty,
        n_episodes=args.n_episodes,
        device=args.device,
    )

    protocols = None
    if args.protocols:
        protocols = [p.strip() for p in args.protocols.split(",")]

    runner = E2SweepRunner(config, output_dir=args.output)
    runner.run_sweep(protocols=protocols)

    print("\nE2 sweep complete!")


if __name__ == "__main__":
    main()
