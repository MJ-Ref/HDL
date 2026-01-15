#!/usr/bin/env python3
"""
M2 Training Dataset Generator (Option A: Replace Sender Text Message)

Generates per-turn training samples:
- inputs: (receiver_prompt_context, sender_text_message)
- target: receiver behavior under full text message

Data composition (from PLAN.md):
- ~70% successful episodes
- ~20% partial-credit episodes
- ~10% failed episodes

Usage:
    # Generate training dataset (500 episodes)
    python scripts/generate_m2_dataset.py --n_episodes 500 --output data/m2_train.jsonl

    # Generate with specific seed range
    python scripts/generate_m2_dataset.py --seed_start 42 --seed_end 542 --output data/m2_train.jsonl
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lpca.agents.llm_agent import LLMAgent
from lpca.agents.model_wrapper import ModelWrapper
from lpca.channels.text import create_channel
from lpca.envs.split_synthetic import SplitSyntheticEnv


@dataclass
class TurnSample:
    """A single turn sample for codec training."""
    episode_seed: int
    turn_idx: int
    sender_agent: str  # 'A' or 'B'
    receiver_agent: str  # 'B' or 'A'

    # Inputs
    receiver_context: str  # Receiver's observation + history (without current message)
    sender_message: str    # The text message to encode

    # Target
    receiver_output: str   # Full receiver response (message + optional answer)
    receiver_tokens: List[int]  # Tokenized receiver output

    # Metadata
    episode_success: bool
    episode_partial_credit: float
    turn_success: bool  # Did this turn lead to correct answer?


@dataclass
class DatasetStats:
    """Statistics for the generated dataset."""
    total_episodes: int
    total_turns: int
    success_episodes: int
    partial_episodes: int
    failed_episodes: int
    avg_message_length: float
    avg_response_length: float


class M2DatasetGenerator:
    """Generates per-turn training data for M2 codec."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "mps",
        temperature: float = 0.3,
        max_turns: int = 6,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_turns = max_turns

        self.env = SplitSyntheticEnv(difficulty="easy")
        self.env.select_environment("constraint_satisfaction")

        self._model = None
        self._tokenizer = None
        self._wrapper = None

    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is not None:
            return

        print(f"Loading model: {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float32 if self.device == "mps" else torch.float16
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=None if self.device == "mps" else "auto",
            trust_remote_code=True,
        )
        if self.device == "mps":
            self._model = self._model.to("mps")

        self._wrapper = ModelWrapper(self._model, self._tokenizer, self.device)
        print(f"Model loaded: {self._wrapper.n_layers} layers")

    def _create_agents(self) -> Dict[str, LLMAgent]:
        """Create fresh agents."""
        self._load_model()
        return {
            'A': LLMAgent('A', self._wrapper, self._tokenizer, temperature=self.temperature),
            'B': LLMAgent('B', self._wrapper, self._tokenizer, temperature=self.temperature),
        }

    def run_episode_and_collect(self, seed: int) -> Tuple[List[TurnSample], Dict]:
        """
        Run a P1 episode and collect per-turn samples.

        Returns:
            samples: List of TurnSample objects
            episode_result: Dict with episode-level metrics
        """
        task = self.env.reset(seed)
        channel = create_channel("P1")  # Full text communication
        agents = self._create_agents()

        channel.reset()
        agents['A'].reset()
        agents['B'].reset()

        samples = []
        current_message = None
        final_answer = None

        for turn_idx in range(self.max_turns):
            # Agent A's turn
            # Build receiver context (what B would see before A's message)
            receiver_context_for_B = self._build_receiver_context(
                task.obs_B, agents['B'].conversation_history
            )

            response_A = agents['A'].respond(task.obs_A, current_message)

            if response_A.final_answer:
                final_answer = response_A.final_answer
                break

            if response_A.message_to_partner:
                # This is a training sample: A sends to B
                msg = channel.send(response_A.message_to_partner, 'A', 'B', turn_idx)
                received_msg = channel.receive(msg)

                # Get B's response to this message (the target)
                response_B = agents['B'].respond(task.obs_B, received_msg)

                # Create sample
                sample = TurnSample(
                    episode_seed=seed,
                    turn_idx=turn_idx,
                    sender_agent='A',
                    receiver_agent='B',
                    receiver_context=receiver_context_for_B,
                    sender_message=response_A.message_to_partner,
                    receiver_output=response_B.text,
                    receiver_tokens=self._tokenizer.encode(response_B.text),
                    episode_success=False,  # Updated later
                    episode_partial_credit=0.0,  # Updated later
                    turn_success=response_B.final_answer is not None,
                )
                samples.append(sample)

                if response_B.final_answer:
                    final_answer = response_B.final_answer
                    break

                if response_B.message_to_partner:
                    # B sends to A - another training sample
                    receiver_context_for_A = self._build_receiver_context(
                        task.obs_A, agents['A'].conversation_history
                    )

                    msg = channel.send(response_B.message_to_partner, 'B', 'A', turn_idx)
                    current_message = channel.receive(msg)

                    # Get A's response (target)
                    next_response_A = agents['A'].respond(task.obs_A, current_message)

                    sample = TurnSample(
                        episode_seed=seed,
                        turn_idx=turn_idx,
                        sender_agent='B',
                        receiver_agent='A',
                        receiver_context=receiver_context_for_A,
                        sender_message=response_B.message_to_partner,
                        receiver_output=next_response_A.text,
                        receiver_tokens=self._tokenizer.encode(next_response_A.text),
                        episode_success=False,
                        episode_partial_credit=0.0,
                        turn_success=next_response_A.final_answer is not None,
                    )
                    samples.append(sample)

                    if next_response_A.final_answer:
                        final_answer = next_response_A.final_answer
                        break

                    if next_response_A.message_to_partner:
                        msg = channel.send(next_response_A.message_to_partner, 'A', 'B', turn_idx)
                        current_message = channel.receive(msg)
                    else:
                        current_message = None
                else:
                    current_message = None
            else:
                current_message = None

        # Verify final answer
        if final_answer:
            result = self.env.verify(final_answer)
        else:
            result = self.env.verify("")

        # Update samples with episode-level info
        for sample in samples:
            sample.episode_success = result.success
            sample.episode_partial_credit = result.partial_credit

        episode_result = {
            'seed': seed,
            'success': result.success,
            'partial_credit': result.partial_credit,
            'n_turns': len(samples),
        }

        return samples, episode_result

    def _build_receiver_context(self, observation: str, history: List) -> str:
        """Build the receiver's context (observation + conversation history)."""
        context_parts = [observation]

        for entry in history:
            if entry.get('role') == 'user':
                # Previous messages received
                context_parts.append(f"Partner said: {entry.get('content', '')}")
            elif entry.get('role') == 'assistant':
                # Own previous responses
                context_parts.append(f"You said: {entry.get('content', '')}")

        return "\n\n".join(context_parts)

    def generate_dataset(
        self,
        n_episodes: int,
        seed_start: int = 42,
        output_path: Optional[str] = None,
    ) -> Tuple[List[TurnSample], DatasetStats]:
        """
        Generate the full training dataset.

        Args:
            n_episodes: Number of episodes to run
            seed_start: Starting seed
            output_path: Path to save JSONL output

        Returns:
            all_samples: List of all turn samples
            stats: Dataset statistics
        """
        all_samples = []
        episode_results = []

        print(f"\nGenerating M2 dataset: {n_episodes} episodes")
        print(f"Seeds: {seed_start} to {seed_start + n_episodes - 1}")
        print("=" * 60)

        for i in tqdm(range(n_episodes), desc="Episodes"):
            seed = seed_start + i
            try:
                samples, result = self.run_episode_and_collect(seed)
                all_samples.extend(samples)
                episode_results.append(result)
            except Exception as e:
                print(f"\nError in episode {seed}: {e}")
                continue

        # Compute statistics
        success_count = sum(1 for r in episode_results if r['success'])
        partial_count = sum(1 for r in episode_results
                          if not r['success'] and r['partial_credit'] > 0)
        failed_count = len(episode_results) - success_count - partial_count

        msg_lengths = [len(s.sender_message) for s in all_samples]
        resp_lengths = [len(s.receiver_output) for s in all_samples]

        stats = DatasetStats(
            total_episodes=len(episode_results),
            total_turns=len(all_samples),
            success_episodes=success_count,
            partial_episodes=partial_count,
            failed_episodes=failed_count,
            avg_message_length=sum(msg_lengths) / len(msg_lengths) if msg_lengths else 0,
            avg_response_length=sum(resp_lengths) / len(resp_lengths) if resp_lengths else 0,
        )

        # Save to file
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                for sample in all_samples:
                    f.write(json.dumps(asdict(sample)) + '\n')

            # Also save stats
            stats_path = output_path.with_suffix('.stats.json')
            with open(stats_path, 'w') as f:
                json.dump(asdict(stats), f, indent=2)

            print(f"\nDataset saved to: {output_path}")
            print(f"Stats saved to: {stats_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"Total episodes: {stats.total_episodes}")
        print(f"Total turns: {stats.total_turns}")
        if stats.total_episodes > 0:
            print(f"Success: {stats.success_episodes} ({100*stats.success_episodes/stats.total_episodes:.1f}%)")
            print(f"Partial: {stats.partial_episodes} ({100*stats.partial_episodes/stats.total_episodes:.1f}%)")
            print(f"Failed: {stats.failed_episodes} ({100*stats.failed_episodes/stats.total_episodes:.1f}%)")
        else:
            print("No episodes completed successfully")
        print(f"Avg message length: {stats.avg_message_length:.0f} chars")
        print(f"Avg response length: {stats.avg_response_length:.0f} chars")

        return all_samples, stats


def main():
    parser = argparse.ArgumentParser(description="Generate M2 training dataset")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/m2_train.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()

    generator = M2DatasetGenerator(
        model_name=args.model,
        device=args.device,
    )

    generator.generate_dataset(
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
        output_path=args.output,
    )

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
