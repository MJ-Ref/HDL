#!/usr/bin/env python3
"""
LPCA End-to-End Demo Experiment

Demonstrates the full experimental pipeline:
1. Load model and create agents
2. Generate split-information tasks
3. Run episodes with different protocols
4. Compare results

Usage:
    # Run with mock model (fast, for testing)
    python scripts/demo_experiment.py --mock

    # Run with real model (requires GPU/MPS)
    python scripts/demo_experiment.py --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only what we need for mock experiments (avoid torch dependencies)
from lpca.core.metrics import MetricsCalculator
from lpca.envs.split_synthetic import SplitSyntheticEnv
from lpca.channels.text import (
    NoCommChannel,
    FullTextChannel,
    BudgetedTextChannel,
)


def create_channel(protocol: str):
    """Create channel by protocol identifier (P0-P2 only for mock demo)."""
    channels = {
        "P0": NoCommChannel,
        "P1": FullTextChannel,
        "P2": BudgetedTextChannel,
    }
    if protocol not in channels:
        raise ValueError(f"Unknown protocol for mock demo: {protocol}. Use P0, P1, or P2.")
    return channels[protocol]()


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

        # Extract key info from observation
        if self.task_type == "constraint_satisfaction":
            return self._respond_constraint(observation, received_message)
        elif self.task_type == "arithmetic":
            return self._respond_arithmetic(observation, received_message)
        else:
            return self._respond_generic(observation, received_message)

    def _respond_constraint(self, observation: str, received_message: str):
        """Mock response for constraint satisfaction."""
        import re

        # Extract constraints
        constraints = re.findall(r'- (.+)', observation)

        if self.turn_count == 1:
            return {
                'text': f"I see {len(constraints)} constraints. Let me share them.",
                'message': f"My constraints: {'; '.join(constraints[:2])}",
                'final_answer': None,
            }
        elif self.turn_count >= 2 and received_message:
            # Try to solve
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

        # Extract values
        values = re.findall(r'(\w+) = (\d+)', observation)

        if self.turn_count == 1:
            return {
                'text': f"I know these values: {values}",
                'message': f"My values: {', '.join(f'{k}={v}' for k,v in values)}",
                'final_answer': None,
            }
        else:
            # Guess an answer
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


def run_mock_episode(
    env,
    channel,
    task,
    agent_A,
    agent_B,
    max_turns: int = 10,
) -> Dict[str, Any]:
    """Run a single episode with mock agents."""
    channel.reset()
    agent_A.reset()
    agent_B.reset()

    turns = []
    final_answer = None
    current_message = None

    start_time = time.perf_counter()

    for turn_idx in range(max_turns):
        # Agent A
        response_A = agent_A.respond(task.obs_A, current_message)
        turns.append({
            'turn_idx': turn_idx,
            'agent': 'A',
            'response': response_A['text'],
            'message': response_A['message'],
        })

        if response_A['final_answer']:
            final_answer = response_A['final_answer']
            break

        # Send message
        if response_A['message']:
            msg = channel.send(response_A['message'], 'A', 'B', turn_idx)
            current_message = channel.receive(msg)
        else:
            current_message = None

        # Agent B
        response_B = agent_B.respond(task.obs_B, current_message)
        turns.append({
            'turn_idx': turn_idx,
            'agent': 'B',
            'response': response_B['text'],
            'message': response_B['message'],
        })

        if response_B['final_answer']:
            final_answer = response_B['final_answer']
            break

        # Send message
        if response_B['message']:
            msg = channel.send(response_B['message'], 'B', 'A', turn_idx)
            current_message = channel.receive(msg)
        else:
            current_message = None

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Verify
    if final_answer:
        result = env.verify(final_answer)
    else:
        result = env.verify("")

    return {
        'turns': turns,
        'n_turns': len(turns),
        'final_answer': final_answer,
        'success': result.success,
        'partial_credit': result.partial_credit,
        'channel_stats': channel.get_stats(),
        'elapsed_ms': elapsed_ms,
    }


def run_experiment(
    task_type: str = "constraint_satisfaction",
    protocols: List[str] = ["P0", "P1", "P2"],
    n_episodes: int = 10,
    difficulty: str = "easy",
    use_mock: bool = True,
    model_name: str = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run experiment comparing protocols.

    Returns results dictionary with metrics per protocol.
    """
    print(f"\n{'='*60}")
    print(f"LPCA Experiment: {task_type}")
    print(f"{'='*60}")
    print(f"Protocols: {protocols}")
    print(f"Episodes: {n_episodes}")
    print(f"Difficulty: {difficulty}")
    print(f"Mode: {'Mock' if use_mock else 'Real LLM'}")
    print(f"{'='*60}\n")

    # Setup environment
    env = SplitSyntheticEnv(difficulty=difficulty)
    env.select_environment(task_type)

    results = {}

    for protocol in protocols:
        print(f"\n--- Protocol {protocol} ---")

        # Create channel
        channel = create_channel(protocol)

        # Create agents
        if use_mock:
            agent_A = MockAgent("A", task_type)
            agent_B = MockAgent("B", task_type)
        else:
            # Would load real LLM here
            raise NotImplementedError("Real LLM not implemented in demo")

        # Run episodes
        successes = []
        partial_credits = []
        turn_counts = []
        bits_used = []

        for ep_idx in range(n_episodes):
            seed = 42 + ep_idx
            task = env.reset(seed)

            episode_result = run_mock_episode(
                env, channel, task, agent_A, agent_B
            )

            successes.append(episode_result['success'])
            partial_credits.append(episode_result['partial_credit'])
            turn_counts.append(episode_result['n_turns'])
            bits_used.append(episode_result['channel_stats']['total_bits'])

            status = "SUCCESS" if episode_result['success'] else "FAIL"
            print(f"  Episode {ep_idx+1}: {status} "
                  f"(turns={episode_result['n_turns']}, "
                  f"bits={episode_result['channel_stats']['total_bits']})")

        # Calculate metrics
        import numpy as np
        metrics = MetricsCalculator()

        results[protocol] = {
            'success_rate': np.mean(successes),
            'partial_credit': np.mean(partial_credits),
            'avg_turns': np.mean(turn_counts),
            'avg_bits': np.mean(bits_used),
            'n_episodes': n_episodes,
        }

        sr = results[protocol]['success_rate']
        sr_ci = metrics.wilson_ci(sum(successes), len(successes))

        print(f"\n  Results for {protocol}:")
        print(f"    Success Rate: {sr:.3f} (95% CI: [{sr_ci[0]:.3f}, {sr_ci[1]:.3f}])")
        print(f"    Partial Credit: {results[protocol]['partial_credit']:.3f}")
        print(f"    Avg Turns: {results[protocol]['avg_turns']:.1f}")
        print(f"    Avg Bits: {results[protocol]['avg_bits']:.0f}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Protocol':<10} {'Success':<10} {'Partial':<10} {'Turns':<10} {'Bits':<10}")
    print("-" * 50)
    for protocol in protocols:
        r = results[protocol]
        print(f"{protocol:<10} {r['success_rate']:<10.3f} {r['partial_credit']:<10.3f} "
              f"{r['avg_turns']:<10.1f} {r['avg_bits']:<10.0f}")

    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")

    if len(protocols) >= 2:
        p0_sr = results.get('P0', {}).get('success_rate', 0)
        p1_sr = results.get('P1', {}).get('success_rate', 0)

        if p1_sr > p0_sr + 0.1:
            print(f"- Communication matters: P1 ({p1_sr:.3f}) >> P0 ({p0_sr:.3f})")
        else:
            print(f"- WARNING: P1 not much better than P0 - communication may not be bottleneck")

    return results


def main():
    parser = argparse.ArgumentParser(description="LPCA Demo Experiment")
    parser.add_argument(
        "--task",
        type=str,
        default="constraint_satisfaction",
        choices=["constraint_satisfaction", "arithmetic", "program_synthesis"],
        help="Task type",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        default="P0,P1,P2",
        help="Comma-separated list of protocols",
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
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Task difficulty",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock agents (default)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for real LLM (disables mock)",
    )

    args = parser.parse_args()

    protocols = [p.strip() for p in args.protocols.split(",")]
    use_mock = args.mock and args.model is None

    results = run_experiment(
        task_type=args.task,
        protocols=protocols,
        n_episodes=args.n_episodes,
        difficulty=args.difficulty,
        use_mock=use_mock,
        model_name=args.model,
    )

    print("\n\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
