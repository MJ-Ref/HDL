#!/usr/bin/env python3
"""
E3: CIPHER Expected Embedding Experiment Runner.

Tests latent communication via expected embeddings (CIPHER protocol).

Usage:
    python scripts/run_cipher_experiment.py --n_episodes 10
    python scripts/run_cipher_experiment.py --top_k 50 --n_soft_tokens 3
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
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from lpca.agents.model_wrapper import ModelWrapper
from lpca.agents.llm_agent import LLMAgent
from lpca.core.metrics import MetricsCalculator
from lpca.envs.split_synthetic import SplitSyntheticEnv


@dataclass
class CIPHERExperimentConfig:
    """Configuration for CIPHER experiments."""
    experiment_name: str = "cipher_e0"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    task_type: str = "constraint_satisfaction"
    difficulty: str = "easy"
    n_episodes: int = 10
    base_seed: int = 42
    max_turns: int = 6
    temperature: float = 0.3
    device: str = "mps"
    # CIPHER params
    top_k: int = 100
    n_soft_tokens: int = 1


class CIPHERExperimentRunner:
    """Runs CIPHER expected embedding experiments."""

    def __init__(self, config: CIPHERExperimentConfig, output_dir: str = "results"):
        self.config = config
        self.output_dir = Path(output_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = f"{config.experiment_name}_{timestamp}"

        self.experiment_dir = self.output_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Environment
        self.env = SplitSyntheticEnv(difficulty=config.difficulty)
        self.env.select_environment(config.task_type)

        self.metrics = MetricsCalculator()
        self.all_results: Dict[str, Dict] = {}

        # Model components
        self._model = None
        self._tokenizer = None
        self._wrapper = None
        self._embed_matrix = None

    def _load_model(self):
        """Load model once."""
        if self._model is not None:
            return

        print(f"Loading model: {self.config.model_name}...")

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

        # Get embedding matrix for CIPHER
        embed_layer = self._model.get_input_embeddings()
        self._embed_matrix = embed_layer.weight.detach()

        print(f"Model loaded: {self._wrapper.n_layers} layers, {self._wrapper.d_model} dim")
        print(f"Embedding matrix: {self._embed_matrix.shape}")

    def compute_expected_embedding(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute expected embedding from logits using top-k filtering."""
        # Apply temperature
        logits = logits / self.config.temperature

        # Top-k filtering
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
            logits = logits.masked_fill(mask == 0, float('-inf'))

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Expected embedding: sum(prob_i * embed_i)
        expected = torch.matmul(probs, self._embed_matrix)

        return expected

    def create_agents(self) -> Dict[str, LLMAgent]:
        """Create agent pair."""
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

    def run_episode_with_cipher(
        self,
        agents: Dict[str, LLMAgent],
        seed: int,
    ) -> Dict[str, Any]:
        """
        Run episode with CIPHER expected embedding communication.

        Agent A generates soft tokens (expected embeddings) that are
        prepended to Agent B's input embedding sequence.
        """
        task = self.env.reset(seed)
        agents['A'].reset()
        agents['B'].reset()

        turns = []
        final_answer = None
        start_time = time.perf_counter()
        total_tokens = 0
        total_soft_tokens = 0

        for turn_idx in range(self.config.max_turns):
            # Agent A processes observation and generates soft tokens
            prompt_A = agents['A'].format_prompt(task.obs_A, None)

            if hasattr(self._tokenizer, 'apply_chat_template'):
                messages_A = agents['A'].format_chat_prompt(task.obs_A, None)
                prompt_A = self._tokenizer.apply_chat_template(
                    messages_A, tokenize=False, add_generation_prompt=True
                )

            inputs_A = self._tokenizer(prompt_A, return_tensors='pt', truncation=True, max_length=1024)
            inputs_A = {k: v.to(self._wrapper.device) for k, v in inputs_A.items()}

            # Get A's output logits
            with torch.no_grad():
                outputs_A = self._model(**inputs_A)
                logits_A = outputs_A.logits[:, -self.config.n_soft_tokens:, :]

            # Compute expected embeddings (soft tokens)
            soft_tokens = self.compute_expected_embedding(logits_A)
            total_soft_tokens += self.config.n_soft_tokens

            # Also generate A's text response for comparison
            response_A = agents['A'].respond(task.obs_A, None)
            total_tokens += response_A.input_tokens + response_A.output_tokens

            turns.append({
                'turn': turn_idx,
                'agent': 'A',
                'text': response_A.text[:300],
                'soft_token_shape': list(soft_tokens.shape),
            })

            if response_A.final_answer:
                final_answer = response_A.final_answer
                break

            # Agent B processes with soft tokens prepended
            prompt_B = agents['B'].format_prompt(task.obs_B, None)

            if hasattr(self._tokenizer, 'apply_chat_template'):
                messages_B = agents['B'].format_chat_prompt(task.obs_B, None)
                prompt_B = self._tokenizer.apply_chat_template(
                    messages_B, tokenize=False, add_generation_prompt=True
                )

            inputs_B = self._tokenizer(prompt_B, return_tensors='pt', truncation=True, max_length=1024)
            inputs_B = {k: v.to(self._wrapper.device) for k, v in inputs_B.items()}

            # Get B's input embeddings
            input_embeds_B = self._model.get_input_embeddings()(inputs_B['input_ids'])

            # Prepend soft tokens from A
            combined_embeds = torch.cat([soft_tokens, input_embeds_B], dim=1)

            # Create attention mask for combined input
            soft_mask = torch.ones(1, soft_tokens.shape[1], device=self._wrapper.device)
            combined_mask = torch.cat([soft_mask, inputs_B['attention_mask']], dim=1)

            # Generate from combined embeddings
            with torch.no_grad():
                outputs_B = self._model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                )
                logits_B = outputs_B.logits

            # Sample from output
            probs = F.softmax(logits_B[:, -1, :] / self.config.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Continue generation
            generated_ids = [next_token.item()]
            current_ids = next_token

            for _ in range(100):
                outputs = self._model(current_ids)
                next_probs = F.softmax(outputs.logits[:, -1, :] / self.config.temperature, dim=-1)
                next_token = torch.multinomial(next_probs, num_samples=1)
                generated_ids.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token], dim=1)

                if next_token.item() == self._tokenizer.eos_token_id:
                    break

            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            total_tokens += len(generated_ids)

            turns.append({
                'turn': turn_idx,
                'agent': 'B',
                'text': generated_text[:300],
                'cipher_input': True,
            })

            # Check for final answer
            final_match = agents['B']._extract_final_answer(generated_text)
            if final_match:
                final_answer = final_match
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify result
        if final_answer:
            result = self.env.verify(final_answer)
        else:
            result = self.env.verify("")

        # Calculate bits used (soft tokens have d_model * 32 bits per token for float32)
        bits_per_soft_token = self._wrapper.d_model * 32
        total_bits = total_soft_tokens * bits_per_soft_token

        return {
            'seed': seed,
            'n_turns': len(turns),
            'final_answer': final_answer,
            'success': result.success,
            'partial_credit': result.partial_credit,
            'total_soft_tokens': total_soft_tokens,
            'total_bits': total_bits,
            'total_tokens': total_tokens,
            'elapsed_ms': elapsed_ms,
        }

    def run(self) -> Dict[str, Any]:
        """Run CIPHER experiment."""
        self._load_model()

        print(f"\n{'='*60}")
        print(f"E3: CIPHER Expected Embedding Experiment")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_name}")
        print(f"Task: {self.config.task_type}")
        print(f"Top-k: {self.config.top_k}")
        print(f"Soft tokens: {self.config.n_soft_tokens}")
        print(f"Episodes: {self.config.n_episodes}")
        print(f"{'='*60}")

        agents = self.create_agents()

        successes = []
        partial_credits = []
        turn_counts = []
        bits_used = []
        episodes = []

        for ep_idx in range(self.config.n_episodes):
            seed = self.config.base_seed + ep_idx

            try:
                episode = self.run_episode_with_cipher(agents, seed)
                episodes.append(episode)

                successes.append(episode['success'])
                partial_credits.append(episode['partial_credit'])
                turn_counts.append(episode['n_turns'])
                bits_used.append(episode['total_bits'])

                status = "SUCCESS" if episode['success'] else "FAIL"
                print(f"  Episode {ep_idx+1}/{self.config.n_episodes}: {status} "
                      f"(turns={episode['n_turns']}, soft_tokens={episode['total_soft_tokens']}, "
                      f"bits={episode['total_bits']:.0f})")

                if episode['final_answer']:
                    print(f"    Answer: {episode['final_answer'][:60]}")

            except Exception as e:
                print(f"  Episode {ep_idx+1}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                continue

        n = len(successes)
        results = {
            'protocol': 'E0_CIPHER',
            'n_episodes': n,
            'success_rate': np.mean(successes) if n > 0 else 0,
            'success_ci': self.metrics.wilson_ci(sum(successes), n) if n > 0 else (0, 0),
            'partial_credit_mean': np.mean(partial_credits) if n > 0 else 0,
            'avg_turns': np.mean(turn_counts) if n > 0 else 0,
            'avg_bits': np.mean(bits_used) if n > 0 else 0,
        }

        sr = results['success_rate']
        ci = results['success_ci']
        print(f"\n  Summary for CIPHER E0:")
        print(f"    Success Rate: {sr:.1%} (95% CI: [{ci[0]:.1%}, {ci[1]:.1%}])")
        print(f"    Partial Credit: {results['partial_credit_mean']:.3f}")
        print(f"    Avg Turns: {results['avg_turns']:.1f}")
        print(f"    Avg Bits: {results['avg_bits']:.0f}")

        self.all_results['E0'] = results

        # Save results
        self._save_results()

        return results

    def _save_results(self):
        """Save experiment results."""
        summary = {
            'experiment_id': self.experiment_id,
            'config': asdict(self.config),
            'results': self.all_results,
        }

        with open(self.experiment_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nResults saved to: {self.experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="E3: CIPHER Experiments")

    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--n_soft_tokens", type=int, default=1)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="results")

    args = parser.parse_args()

    config = CIPHERExperimentConfig(
        model_name=args.model,
        n_episodes=args.n_episodes,
        top_k=args.top_k,
        n_soft_tokens=args.n_soft_tokens,
        device=args.device,
    )

    runner = CIPHERExperimentRunner(config, output_dir=args.output)
    runner.run()

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
