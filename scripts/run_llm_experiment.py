#!/usr/bin/env python3
"""
LLM Experiment Runner for LPCA.

Runs experiments with real LLM agents on constraint satisfaction tasks.

Usage:
    # Run baseline comparison (P0 vs P1)
    python scripts/run_llm_experiment.py --protocols P0,P1 --n_episodes 10

    # Run single protocol
    python scripts/run_llm_experiment.py --protocol P1 --n_episodes 5
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from lpca.agents.llm_agent import load_llm_agent, LLMAgent
from lpca.agents.model_wrapper import ModelWrapper
from lpca.channels.text import NoCommChannel, FullTextChannel, BudgetedTextChannel
from lpca.core.metrics import MetricsCalculator
from lpca.envs.split_synthetic import SplitSyntheticEnv


@dataclass
class LLMExperimentConfig:
    """Configuration for LLM experiments."""
    experiment_name: str = "llm_experiment"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # Better instruction following
    task_type: str = "constraint_satisfaction"
    difficulty: str = "easy"  # Start with easy
    n_episodes: int = 10
    base_seed: int = 42
    max_turns: int = 6
    temperature: float = 0.3
    device: str = "mps"


class LLMExperimentRunner:
    """Runs LPCA experiments with real LLM agents."""

    def __init__(self, config: LLMExperimentConfig, output_dir: str = "results"):
        self.config = config
        self.output_dir = Path(output_dir)

        # Generate experiment ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = f"{config.experiment_name}_{timestamp}"

        # Setup output directory
        self.experiment_dir = self.output_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize environment
        self.env = SplitSyntheticEnv(difficulty=config.difficulty)
        self.env.select_environment(config.task_type)

        # Metrics calculator
        self.metrics = MetricsCalculator()

        # Results storage
        self.all_results: Dict[str, Dict] = {}

        # Model and agents (loaded lazily)
        self._model = None
        self._tokenizer = None
        self._wrapper = None

    def _load_model(self):
        """Load model (once, shared between agents)."""
        if self._model is not None:
            return

        print(f"Loading model: {self.config.model_name}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float32 if self.config.device == "mps" else torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=None if self.config.device == "mps" else "auto",
        )

        if self.config.device == "mps":
            self._model = self._model.to("mps")

        self._wrapper = ModelWrapper(self._model, self._tokenizer, self.config.device)
        print(f"Model loaded: {self._wrapper.n_layers} layers, {self._wrapper.d_model} dim")

    def create_agents(self) -> Dict[str, LLMAgent]:
        """Create agent pair sharing the same model."""
        self._load_model()

        agent_A = LLMAgent(
            agent_id='A',
            model_wrapper=self._wrapper,
            tokenizer=self._tokenizer,
            temperature=self.config.temperature,
        )
        agent_B = LLMAgent(
            agent_id='B',
            model_wrapper=self._wrapper,
            tokenizer=self._tokenizer,
            temperature=self.config.temperature,
        )

        return {'A': agent_A, 'B': agent_B}

    def create_channel(self, protocol: str):
        """Create communication channel for protocol."""
        if protocol == "P0":
            return NoCommChannel()
        elif protocol == "P1":
            return FullTextChannel()
        elif protocol == "P2":
            return BudgetedTextChannel(max_bytes=256)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    def run_episode(
        self,
        agents: Dict[str, LLMAgent],
        channel,
        seed: int,
    ) -> Dict[str, Any]:
        """Run a single episode."""
        # Reset
        task = self.env.reset(seed)
        channel.reset()
        agents['A'].reset()
        agents['B'].reset()

        turns = []
        final_answer = None
        current_message = None
        start_time = time.perf_counter()
        total_tokens = 0

        for turn_idx in range(self.config.max_turns):
            # Agent A's turn
            response_A = agents['A'].respond(task.obs_A, current_message)
            total_tokens += response_A.input_tokens + response_A.output_tokens

            turns.append({
                'turn': turn_idx,
                'agent': 'A',
                'text': response_A.text[:500],  # Truncate for storage
                'message': response_A.message_to_partner,
                'answer': response_A.final_answer,
            })

            if response_A.final_answer:
                final_answer = response_A.final_answer
                break

            # Send message through channel
            if response_A.message_to_partner:
                msg = channel.send(response_A.message_to_partner, 'A', 'B', turn_idx)
                current_message = channel.receive(msg)
            else:
                current_message = None

            # Agent B's turn
            response_B = agents['B'].respond(task.obs_B, current_message)
            total_tokens += response_B.input_tokens + response_B.output_tokens

            turns.append({
                'turn': turn_idx,
                'agent': 'B',
                'text': response_B.text[:500],
                'message': response_B.message_to_partner,
                'answer': response_B.final_answer,
            })

            if response_B.final_answer:
                final_answer = response_B.final_answer
                break

            # Send message through channel
            if response_B.message_to_partner:
                msg = channel.send(response_B.message_to_partner, 'B', 'A', turn_idx)
                current_message = channel.receive(msg)
            else:
                current_message = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify result
        if final_answer:
            result = self.env.verify(final_answer)
        else:
            result = self.env.verify("")

        channel_stats = channel.get_stats()

        return {
            'seed': seed,
            'n_turns': len(turns),
            'final_answer': final_answer,
            'success': result.success,
            'partial_credit': result.partial_credit,
            'total_bits': channel_stats.get('total_bits', 0),
            'message_count': channel_stats.get('message_count', 0),
            'total_tokens': total_tokens,
            'elapsed_ms': elapsed_ms,
            'turns': turns,
        }

    def run_protocol(self, protocol: str) -> Dict[str, Any]:
        """Run experiment for a single protocol."""
        print(f"\n{'='*50}")
        print(f"Protocol: {protocol}")
        print(f"{'='*50}")

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
                episode = self.run_episode(agents, channel, seed)
                episodes.append(episode)

                successes.append(episode['success'])
                partial_credits.append(episode['partial_credit'])
                turn_counts.append(episode['n_turns'])
                bits_used.append(episode['total_bits'])

                status = "SUCCESS" if episode['success'] else "FAIL"
                print(f"  Episode {ep_idx+1}/{self.config.n_episodes}: {status} "
                      f"(turns={episode['n_turns']}, bits={episode['total_bits']:.0f}, "
                      f"tokens={episode['total_tokens']}, time={episode['elapsed_ms']:.0f}ms)")

                # Print answer if any
                if episode['final_answer']:
                    print(f"    Answer: {episode['final_answer'][:80]}")

            except Exception as e:
                print(f"  Episode {ep_idx+1}: ERROR - {e}")
                import traceback
                traceback.print_exc()
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
        print(f"\n  Summary for {protocol}:")
        print(f"    Success Rate: {sr:.1%} (95% CI: [{ci[0]:.1%}, {ci[1]:.1%}])")
        print(f"    Partial Credit: {results['partial_credit_mean']:.3f}")
        print(f"    Avg Turns: {results['avg_turns']:.1f}")
        print(f"    Avg Bits: {results['avg_bits']:.0f}")

        return results

    def run(self, protocols: List[str]) -> Dict[str, Dict]:
        """Run the full experiment."""
        print(f"\n{'='*60}")
        print(f"LPCA LLM Experiment: {self.experiment_id}")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_name}")
        print(f"Task: {self.config.task_type} ({self.config.difficulty})")
        print(f"Protocols: {protocols}")
        print(f"Episodes per protocol: {self.config.n_episodes}")
        print(f"Device: {self.config.device}")
        print(f"{'='*60}")

        # Save config
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        # Run each protocol
        for protocol in protocols:
            results = self.run_protocol(protocol)
            self.all_results[protocol] = results

            # Save protocol results (without full episodes for size)
            protocol_path = self.experiment_dir / f"results_{protocol}.json"
            save_results = {k: v for k, v in results.items() if k != 'episodes'}
            with open(protocol_path, 'w') as f:
                json.dump(save_results, f, indent=2)

        # Print comparison
        self._print_comparison(protocols)

        # Save summary
        self._save_summary()

        return self.all_results

    def _print_comparison(self, protocols: List[str]):
        """Print protocol comparison."""
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Protocol':<10} {'Success':<12} {'Partial':<10} {'Turns':<8} {'Bits':<8}")
        print("-" * 50)

        for protocol in protocols:
            r = self.all_results.get(protocol, {})
            sr = r.get('success_rate', 0)
            pc = r.get('partial_credit_mean', 0)
            turns = r.get('avg_turns', 0)
            bits = r.get('avg_bits', 0)
            print(f"{protocol:<10} {sr:<12.1%} {pc:<10.3f} {turns:<8.1f} {bits:<8.0f}")

        # Key findings
        if 'P0' in self.all_results and 'P1' in self.all_results:
            p0_sr = self.all_results['P0']['success_rate']
            p1_sr = self.all_results['P1']['success_rate']

            print(f"\nKey Finding:")
            if p1_sr > p0_sr + 0.05:
                print(f"  Communication helps! P1 ({p1_sr:.1%}) > P0 ({p0_sr:.1%})")
            elif p1_sr > p0_sr:
                print(f"  Slight benefit from communication: P1 ({p1_sr:.1%}) vs P0 ({p0_sr:.1%})")
            else:
                print(f"  No clear communication benefit: P1 ({p1_sr:.1%}) vs P0 ({p0_sr:.1%})")

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
    parser = argparse.ArgumentParser(description="LPCA LLM Experiment Runner")

    parser.add_argument("--protocol", type=str, default="P1", help="Single protocol")
    parser.add_argument("--protocols", type=str, help="Comma-separated protocols (e.g., P0,P1)")
    parser.add_argument("--n_episodes", type=int, default=10, help="Episodes per protocol")
    parser.add_argument("--task", type=str, default="constraint_satisfaction")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="results")

    args = parser.parse_args()

    # Create config
    config = LLMExperimentConfig(
        experiment_name=f"llm_{args.task}",
        model_name=args.model,
        task_type=args.task,
        difficulty=args.difficulty,
        n_episodes=args.n_episodes,
        device=args.device,
    )

    # Determine protocols
    if args.protocols:
        protocols = [p.strip() for p in args.protocols.split(",")]
    else:
        protocols = [args.protocol]

    # Run experiment
    runner = LLMExperimentRunner(config, output_dir=args.output)
    runner.run(protocols=protocols)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
