#!/usr/bin/env python3
"""
E4: Activation Grafting Experiment Runner.

Tests latent communication via activation injection between agents.

Usage:
    python scripts/run_activation_experiment.py --layer 18 --combine replace --n_episodes 10
    python scripts/run_activation_experiment.py --sweep_layers --n_episodes 5
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from lpca.agents.model_wrapper import (
    ModelWrapper, ActivationHook,
    combine_replace, combine_add, combine_average, combine_weighted
)
from lpca.agents.llm_agent import LLMAgent
from lpca.core.metrics import MetricsCalculator
from lpca.envs.split_synthetic import SplitSyntheticEnv


@dataclass
class ActivationExperimentConfig:
    """Configuration for activation grafting experiments."""
    experiment_name: str = "activation_grafting"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    task_type: str = "constraint_satisfaction"
    difficulty: str = "easy"
    n_episodes: int = 10
    base_seed: int = 42
    max_turns: int = 6
    temperature: float = 0.3
    device: str = "mps"
    # Activation grafting params
    graft_layer: int = 18  # Default to middle layer
    combine_fn: str = "replace"  # replace, add, average, weighted


class ActivationGraftingRunner:
    """Runs activation grafting experiments."""

    def __init__(self, config: ActivationExperimentConfig, output_dir: str = "results"):
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

        # Model (loaded once)
        self._model = None
        self._tokenizer = None
        self._wrapper = None

    def _load_model(self):
        """Load model once, shared between agents."""
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
        print(f"Model loaded: {self._wrapper.n_layers} layers, {self._wrapper.d_model} dim")

    def get_combine_fn(self, name: str) -> Callable:
        """Get combine function by name."""
        fns = {
            "replace": combine_replace,
            "add": combine_add,
            "average": combine_average,
            "weighted_0.3": combine_weighted(0.3),
            "weighted_0.5": combine_weighted(0.5),
            "weighted_0.7": combine_weighted(0.7),
        }
        return fns.get(name, combine_replace)

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

    def run_episode_with_grafting(
        self,
        agents: Dict[str, LLMAgent],
        seed: int,
        graft_layer: int,
        combine_fn: Callable,
    ) -> Dict[str, Any]:
        """
        Run episode with activation grafting.

        Instead of text messages, Agent A's activation is grafted into Agent B's
        computation at the specified layer.
        """
        task = self.env.reset(seed)
        agents['A'].reset()
        agents['B'].reset()

        turns = []
        final_answer = None
        start_time = time.perf_counter()
        total_tokens = 0
        total_grafts = 0

        for turn_idx in range(self.config.max_turns):
            # Agent A processes its observation
            # We capture the activation at the graft layer
            prompt_A = agents['A'].format_prompt(task.obs_A, None)

            if hasattr(self._tokenizer, 'apply_chat_template'):
                messages_A = agents['A'].format_chat_prompt(task.obs_A, None)
                prompt_A = self._tokenizer.apply_chat_template(
                    messages_A, tokenize=False, add_generation_prompt=True
                )

            inputs_A = self._tokenizer(prompt_A, return_tensors='pt', truncation=True, max_length=1024)
            inputs_A = {k: v.to(self._wrapper.device) for k, v in inputs_A.items()}

            # Capture A's activation
            activation_A = self._wrapper.capture_activation(inputs_A['input_ids'], layer_idx=graft_layer)

            # Generate A's response (for logging, but we use activation for communication)
            response_A = agents['A'].respond(task.obs_A, None)
            total_tokens += response_A.input_tokens + response_A.output_tokens

            turns.append({
                'turn': turn_idx,
                'agent': 'A',
                'text': response_A.text[:300],
                'activation_shape': list(activation_A.shape),
            })

            if response_A.final_answer:
                final_answer = response_A.final_answer
                break

            # Agent B processes with grafted activation from A
            prompt_B = agents['B'].format_prompt(task.obs_B, None)

            if hasattr(self._tokenizer, 'apply_chat_template'):
                messages_B = agents['B'].format_chat_prompt(task.obs_B, None)
                prompt_B = self._tokenizer.apply_chat_template(
                    messages_B, tokenize=False, add_generation_prompt=True
                )

            inputs_B = self._tokenizer(prompt_B, return_tensors='pt', truncation=True, max_length=1024)
            inputs_B = {k: v.to(self._wrapper.device) for k, v in inputs_B.items()}

            # Inject A's activation into B's forward pass
            # Note: activation shapes may differ, we use last token activation
            activation_to_inject = activation_A[:, -1:, :].expand(-1, inputs_B['input_ids'].shape[1], -1)

            logits_B = self._wrapper.inject_activation(
                inputs_B['input_ids'],
                graft_layer,
                activation_to_inject,
                combine_fn
            )
            total_grafts += 1

            # Generate from modified logits
            with torch.no_grad():
                # Sample from modified logits
                probs = torch.softmax(logits_B[:, -1, :] / self.config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Continue generation
                generated_ids = [next_token.item()]
                current_ids = torch.cat([inputs_B['input_ids'], next_token], dim=1)

                for _ in range(100):  # Max generation length
                    outputs = self._model(current_ids)
                    next_probs = torch.softmax(outputs.logits[:, -1, :] / self.config.temperature, dim=-1)
                    next_token = torch.multinomial(next_probs, num_samples=1)
                    generated_ids.append(next_token.item())
                    current_ids = torch.cat([current_ids, next_token], dim=1)

                    if next_token.item() == self._tokenizer.eos_token_id:
                        break

            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            total_tokens += len(generated_ids)

            # Parse B's response
            agents['B'].add_to_history("assistant", generated_text)

            turns.append({
                'turn': turn_idx,
                'agent': 'B',
                'text': generated_text[:300],
                'grafted': True,
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

        return {
            'seed': seed,
            'n_turns': len(turns),
            'final_answer': final_answer,
            'success': result.success,
            'partial_credit': result.partial_credit,
            'total_grafts': total_grafts,
            'total_tokens': total_tokens,
            'elapsed_ms': elapsed_ms,
            'graft_layer': graft_layer,
            'combine_fn': self.config.combine_fn,
        }

    def run_config(self, graft_layer: int, combine_name: str) -> Dict[str, Any]:
        """Run experiment for a specific layer/combine configuration."""
        config_name = f"L{graft_layer}_{combine_name}"
        print(f"\n{'='*50}")
        print(f"Config: {config_name}")
        print(f"{'='*50}")

        combine_fn = self.get_combine_fn(combine_name)
        agents = self.create_agents()

        successes = []
        partial_credits = []
        turn_counts = []
        episodes = []

        for ep_idx in range(self.config.n_episodes):
            seed = self.config.base_seed + ep_idx

            try:
                episode = self.run_episode_with_grafting(
                    agents, seed, graft_layer, combine_fn
                )
                episodes.append(episode)

                successes.append(episode['success'])
                partial_credits.append(episode['partial_credit'])
                turn_counts.append(episode['n_turns'])

                status = "SUCCESS" if episode['success'] else "FAIL"
                print(f"  Episode {ep_idx+1}/{self.config.n_episodes}: {status} "
                      f"(turns={episode['n_turns']}, grafts={episode['total_grafts']})")

                if episode['final_answer']:
                    print(f"    Answer: {episode['final_answer'][:60]}")

            except Exception as e:
                print(f"  Episode {ep_idx+1}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                continue

        n = len(successes)
        results = {
            'config': config_name,
            'graft_layer': graft_layer,
            'combine_fn': combine_name,
            'n_episodes': n,
            'success_rate': np.mean(successes) if n > 0 else 0,
            'success_ci': self.metrics.wilson_ci(sum(successes), n) if n > 0 else (0, 0),
            'partial_credit_mean': np.mean(partial_credits) if n > 0 else 0,
            'avg_turns': np.mean(turn_counts) if n > 0 else 0,
        }

        sr = results['success_rate']
        ci = results['success_ci']
        print(f"\n  Summary for {config_name}:")
        print(f"    Success Rate: {sr:.1%} (95% CI: [{ci[0]:.1%}, {ci[1]:.1%}])")
        print(f"    Partial Credit: {results['partial_credit_mean']:.3f}")

        return results

    def run_layer_sweep(self) -> Dict[str, Dict]:
        """Sweep across different layer positions."""
        n_layers = self._wrapper.n_layers if self._wrapper else 36

        # Test at n/4, n/3, n/2, 2n/3, 3n/4
        layer_fractions = [0.25, 0.33, 0.5, 0.67, 0.75]
        layers = [int(f * n_layers) for f in layer_fractions]

        print(f"\nLayer sweep: {layers}")

        for layer in layers:
            results = self.run_config(layer, self.config.combine_fn)
            self.all_results[f"L{layer}"] = results

        return self.all_results

    def run_combine_sweep(self) -> Dict[str, Dict]:
        """Sweep across different combine functions."""
        combines = ["replace", "add", "average", "weighted_0.5"]

        print(f"\nCombine sweep at layer {self.config.graft_layer}: {combines}")

        for combine in combines:
            results = self.run_config(self.config.graft_layer, combine)
            self.all_results[combine] = results

        return self.all_results

    def run(self) -> Dict[str, Dict]:
        """Run single configuration."""
        self._load_model()

        print(f"\n{'='*60}")
        print(f"E4: Activation Grafting Experiment")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_name}")
        print(f"Task: {self.config.task_type}")
        print(f"Layer: {self.config.graft_layer}")
        print(f"Combine: {self.config.combine_fn}")
        print(f"Episodes: {self.config.n_episodes}")
        print(f"{'='*60}")

        results = self.run_config(self.config.graft_layer, self.config.combine_fn)
        self.all_results['main'] = results

        # Save results
        self._save_results()

        return self.all_results

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
    parser = argparse.ArgumentParser(description="E4: Activation Grafting Experiments")

    parser.add_argument("--layer", type=int, default=18, help="Graft layer index")
    parser.add_argument("--combine", type=str, default="replace",
                       choices=["replace", "add", "average", "weighted_0.3", "weighted_0.5", "weighted_0.7"])
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--sweep_layers", action="store_true", help="Sweep across layers")
    parser.add_argument("--sweep_combines", action="store_true", help="Sweep across combine functions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="results")

    args = parser.parse_args()

    config = ActivationExperimentConfig(
        model_name=args.model,
        n_episodes=args.n_episodes,
        graft_layer=args.layer,
        combine_fn=args.combine,
        device=args.device,
    )

    runner = ActivationGraftingRunner(config, output_dir=args.output)

    if args.sweep_layers:
        runner.run_layer_sweep()
    elif args.sweep_combines:
        runner.run_combine_sweep()
    else:
        runner.run()

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
