#!/usr/bin/env python3
"""
E4: Quick Activation Grafting Test.

Simplified version to test layer effects quickly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import numpy as np
import torch
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from lpca.agents.model_wrapper import ModelWrapper, combine_replace, combine_average
from lpca.core.metrics import MetricsCalculator
from lpca.envs.split_synthetic import SplitSyntheticEnv


def run_grafted_episode(
    model, tokenizer, wrapper, env, seed, graft_layer, combine_fn, temperature=0.3
):
    """Run single episode with activation grafting."""
    task = env.reset(seed)

    # Encode observations
    obs_A_ids = tokenizer(task.obs_A, return_tensors='pt', truncation=True, max_length=512)
    obs_A_ids = obs_A_ids['input_ids'].to(wrapper.device)

    obs_B_ids = tokenizer(task.obs_B, return_tensors='pt', truncation=True, max_length=512)
    obs_B_ids = obs_B_ids['input_ids'].to(wrapper.device)

    # Capture A's activation
    act_A = wrapper.capture_activation(obs_A_ids, layer_idx=graft_layer)

    # Inject into B's forward pass (use last token activation)
    act_to_inject = act_A[:, -1:, :].expand(-1, obs_B_ids.shape[1], -1)

    # Get modified logits for B
    logits_B = wrapper.inject_activation(obs_B_ids, graft_layer, act_to_inject, combine_fn)

    # Generate from B with system prompt for answering
    system = "Solve for x1,x2,x3 (0 or 1). Reply ANSWER: {json}"
    full_prompt = f"{system}\n{task.obs_B}\nPartner info embedded.\nANSWER:"

    input_ids = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = input_ids['input_ids'].to(wrapper.device)

    # Inject A's activation at generation start too
    act_A_gen = wrapper.capture_activation(obs_A_ids, layer_idx=graft_layer)
    act_inject_gen = act_A_gen[:, -1:, :].expand(-1, input_ids.shape[1], -1)

    logits = wrapper.inject_activation(input_ids, graft_layer, act_inject_gen, combine_fn)

    # Generate answer
    generated = []
    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated.append(next_token.item())

    current_ids = torch.cat([input_ids, next_token], dim=1)

    for _ in range(50):
        with torch.no_grad():
            out = model(current_ids)
        probs = torch.softmax(out.logits[:, -1, :] / temperature, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        generated.append(next_tok.item())
        current_ids = torch.cat([current_ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(generated, skip_special_tokens=True)

    # Extract answer
    import re
    json_match = re.search(r'\{[^}]+\}', response)
    final_answer = json_match.group(0) if json_match else None

    # Verify
    if final_answer:
        result = env.verify(final_answer)
    else:
        result = env.verify("")

    return {
        'success': result.success,
        'partial_credit': result.partial_credit,
        'answer': final_answer,
        'response': response[:200],
    }


def main():
    print("E4: Quick Activation Grafting Test")
    print("="*50)

    # Load model
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to("mps")

    wrapper = ModelWrapper(model, tokenizer, "mps")
    print(f"Model: {wrapper.n_layers} layers, {wrapper.d_model} dim")

    # Environment
    env = SplitSyntheticEnv(difficulty="easy")
    env.select_environment("constraint_satisfaction")

    metrics = MetricsCalculator()

    # Test layers
    layers = [9, 18, 27]  # Early, middle, late
    n_episodes = 10

    results = {}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        successes = []

        for ep in range(n_episodes):
            seed = 42 + ep
            try:
                result = run_grafted_episode(
                    model, tokenizer, wrapper, env, seed,
                    graft_layer=layer,
                    combine_fn=combine_replace,
                )
                successes.append(result['success'])
                status = "SUCCESS" if result['success'] else "FAIL"
                print(f"  Ep {ep+1}: {status} | {result['answer']}")
            except Exception as e:
                print(f"  Ep {ep+1}: ERROR - {e}")
                successes.append(False)

        sr = np.mean(successes)
        ci = metrics.wilson_ci(sum(successes), len(successes))
        results[f"L{layer}"] = {
            'success_rate': sr,
            'ci': ci,
            'n': len(successes),
        }
        print(f"  Layer {layer}: {sr:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")

    # Save results
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"{'Layer':<10} {'Success':<12} {'95% CI'}")
    print("-"*40)
    for layer, r in results.items():
        ci = r['ci']
        print(f"{layer:<10} {r['success_rate']:<12.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")

    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(f"results/e4_quick_{timestamp}.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'layers': layers,
            'n_episodes': n_episodes,
            'results': {k: {**v, 'ci': list(v['ci'])} for k, v in results.items()},
        }, f, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
