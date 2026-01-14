#!/usr/bin/env python3
"""
Sanity checks to validate experimental setup.

These checks help identify whether failures are due to:
- Communication bottleneck (what we want to study)
- Model competence/prompting issues (artifact)
- Protocol/parsing bugs (artifact)

Run these BEFORE scaling up experiments.
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from lpca.envs.split_synthetic import SplitSyntheticEnv


@dataclass
class SanityCheckResult:
    """Result of a sanity check."""
    name: str
    passed: bool
    success_rate: float
    n_episodes: int
    details: str
    recommendation: str


def run_single_agent_full_info(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    n_episodes: int = 20,
    difficulty: str = "easy",
    device: str = "auto",
) -> SanityCheckResult:
    """
    Sanity Check 1: Single agent with full information.

    If a single agent with ALL constraints can't solve the task,
    then the problem isn't communication - it's model competence.

    Expected: Near-ceiling success rate (>80%) on easy tasks.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 1: Single Agent, Full Information")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Difficulty: {difficulty}")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return SanityCheckResult(
            name="single_agent_full_info",
            passed=False,
            success_rate=0,
            n_episodes=0,
            details="torch/transformers not available",
            recommendation="Install torch and transformers",
        )

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            actual_device = "cuda"
        elif torch.backends.mps.is_available():
            actual_device = "mps"
        else:
            actual_device = "cpu"
    else:
        actual_device = device

    dtype = torch.float32 if actual_device == "mps" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None if actual_device == "mps" else "auto",
        trust_remote_code=True,
    )
    if actual_device == "mps":
        model = model.to("mps")

    print(f"Model loaded on {actual_device}")

    # Initialize environment
    env = SplitSyntheticEnv(difficulty=difficulty)
    env.select_environment("constraint_satisfaction")

    successes = []
    wall_times = []

    for ep_idx in range(n_episodes):
        seed = 42 + ep_idx
        task = env.reset(seed)

        # Combine both observations into one
        full_observation = f"""You have a constraint satisfaction problem. Find values for variables that satisfy ALL constraints.

CONSTRAINTS (all of them):
{task.obs_A}

{task.obs_B}

Solve this problem. Respond with ANSWER: followed by a JSON object with variable assignments.
Example format: ANSWER: {{"x1": 0, "x2": 1, "x3": 0}}

Think step by step, then provide your answer."""

        # Format as chat
        messages = [
            {"role": "system", "content": "You are a constraint satisfaction solver. Analyze constraints carefully and find valid variable assignments."},
            {"role": "user", "content": full_observation},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate
        start_time = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        wall_times.append(elapsed_ms)

        # Extract answer
        import re
        answer = None

        # Try explicit ANSWER:
        match = re.search(r'ANSWER:\s*(\{[^}]+\})', response, re.IGNORECASE)
        if match:
            answer = match.group(1)
        else:
            # Fallback: last JSON object
            json_matches = re.findall(r'\{[^{}]+\}', response)
            if json_matches:
                answer = json_matches[-1]

        # Verify
        if answer:
            result = env.verify(answer)
            success = result.success
        else:
            success = False

        successes.append(success)
        status = "PASS" if success else "FAIL"
        print(f"  Episode {ep_idx+1}/{n_episodes}: {status} ({elapsed_ms:.0f}ms)")

    # Calculate results
    success_rate = np.mean(successes)
    avg_time = np.mean(wall_times)

    print(f"\n  Success Rate: {success_rate:.1%}")
    print(f"  Avg Wall Time: {avg_time:.0f}ms")

    # Determine pass/fail
    passed = success_rate >= 0.7  # Should be high on easy tasks

    if passed:
        recommendation = "Model is competent. Communication is likely the bottleneck in 2-agent setup."
    else:
        recommendation = "Model struggles even with full info. Fix prompts or try larger model before testing communication."

    return SanityCheckResult(
        name="single_agent_full_info",
        passed=passed,
        success_rate=success_rate,
        n_episodes=n_episodes,
        details=f"Avg wall time: {avg_time:.0f}ms",
        recommendation=recommendation,
    )


def run_injection_test(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "auto",
) -> SanityCheckResult:
    """
    Sanity Check 2: Activation injection actually changes output.

    If injecting activations doesn't change the logits distribution,
    then the E4 plumbing isn't connected properly.

    Expected: Output distribution should change measurably with injection.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 2: Activation Injection Changes Output")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return SanityCheckResult(
            name="injection_test",
            passed=False,
            success_rate=0,
            n_episodes=0,
            details="torch/transformers not available",
            recommendation="Install torch and transformers",
        )

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            actual_device = "cuda"
        elif torch.backends.mps.is_available():
            actual_device = "mps"
        else:
            actual_device = "cpu"
    else:
        actual_device = device

    dtype = torch.float32 if actual_device == "mps" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None if actual_device == "mps" else "auto",
        trust_remote_code=True,
    )
    if actual_device == "mps":
        model = model.to("mps")

    print(f"Model loaded on {actual_device}")
    n_layers = model.config.num_hidden_layers
    print(f"Layers: {n_layers}")

    # Test prompt
    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    # Get baseline output
    print("\n1. Getting baseline output (no injection)...")
    with torch.no_grad():
        baseline_outputs = model(**inputs, output_hidden_states=True)
        baseline_logits = baseline_outputs.logits[:, -1, :]
        baseline_probs = torch.softmax(baseline_logits, dim=-1)

    # Get output with random activation injection
    print("2. Getting output with random activation injection...")

    test_layer = n_layers // 2  # Middle layer
    captured_activation = baseline_outputs.hidden_states[test_layer].clone()

    # Create random noise with same shape
    noise = torch.randn_like(captured_activation) * 0.5

    # Injection hook
    def injection_hook(module, args, output):
        # Inject noise into the hidden state
        # For Qwen2 decoder layers, output is typically just a tensor
        if isinstance(output, torch.Tensor):
            return output + noise
        elif isinstance(output, tuple):
            hidden_state = output[0]
            modified = hidden_state + noise
            return (modified,) + output[1:]
        else:
            # Fallback for other types
            return output

    # Register hook
    layer = model.model.layers[test_layer]
    handle = layer.register_forward_hook(injection_hook)

    with torch.no_grad():
        injected_outputs = model(**inputs)
        injected_logits = injected_outputs.logits[:, -1, :]
        injected_probs = torch.softmax(injected_logits, dim=-1)

    handle.remove()

    # Compare distributions
    kl_div = torch.sum(baseline_probs * (torch.log(baseline_probs + 1e-10) - torch.log(injected_probs + 1e-10))).item()
    l2_dist = torch.norm(baseline_logits - injected_logits).item()

    # Check top tokens
    baseline_top = torch.topk(baseline_probs, k=5)
    injected_top = torch.topk(injected_probs, k=5)

    print(f"\n3. Comparing distributions:")
    print(f"   KL divergence: {kl_div:.4f}")
    print(f"   L2 distance (logits): {l2_dist:.4f}")
    print(f"\n   Baseline top tokens: {[tokenizer.decode([t]) for t in baseline_top.indices[0]]}")
    print(f"   Injected top tokens: {[tokenizer.decode([t]) for t in injected_top.indices[0]]}")

    # Pass if injection changed output meaningfully
    # KL > 0.1 or L2 > 1.0 indicates meaningful change
    passed = kl_div > 0.1 or l2_dist > 1.0

    if passed:
        recommendation = "Injection plumbing works. Output changes with injected activations."
    else:
        recommendation = "WARNING: Injection doesn't change output much. Check hook implementation."

    return SanityCheckResult(
        name="injection_test",
        passed=passed,
        success_rate=1.0 if passed else 0.0,
        n_episodes=1,
        details=f"KL={kl_div:.4f}, L2={l2_dist:.4f}",
        recommendation=recommendation,
    )


def run_parsing_test() -> SanityCheckResult:
    """
    Sanity Check 3: Answer parsing robustness.

    Test that various answer formats are correctly parsed.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK 3: Answer Parsing Robustness")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lpca.agents.llm_agent import LLMAgent

    # Create mock agent just for parsing test
    class MockWrapper:
        pass

    class MockTokenizer:
        pass

    agent = LLMAgent("test", MockWrapper(), MockTokenizer())

    test_cases = [
        # (input_text, expected_to_find_answer, description)
        ("ANSWER: {\"x1\": 1, \"x2\": 0}", True, "Standard format"),
        ("Let me think... ANSWER: {\"x1\": 1, \"x2\": 0}", True, "With preamble"),
        ("ANSWER: {\"x1\": 1, \"x2\": 0}\nDone.", True, "With epilogue"),
        ("answer: {\"x1\":1,\"x2\":0}", True, "Lowercase, no spaces"),
        ("FINAL ANSWER: {\"x1\": 1, \"x2\": 0}", True, "FINAL ANSWER prefix"),
        ("The answer is {\"x1\": 1}", False, "No ANSWER prefix - should NOT match"),
        ("Example ANSWER: {\"x1\": 1}", False, "In example context - should NOT match"),
        ("MESSAGE: sharing info\nANSWER: {\"x1\": 1}", True, "After message"),
        ("{\"x1\": 1}", False, "Raw JSON - should NOT match without keyword"),
        ("I think the ANSWER is: {\"x1\": 1, \"x2\": 0}", True, "Natural phrasing with keyword"),
    ]

    passed_count = 0
    for text, should_find, desc in test_cases:
        result = agent._extract_final_answer(text)
        found = result is not None

        if found == should_find:
            passed_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  {status}: {desc}")
        if found != should_find:
            print(f"       Input: {text[:50]}...")
            print(f"       Expected: {'found' if should_find else 'not found'}, Got: {'found' if found else 'not found'}")

    total = len(test_cases)
    success_rate = passed_count / total

    passed = success_rate >= 0.8

    return SanityCheckResult(
        name="parsing_test",
        passed=passed,
        success_rate=success_rate,
        n_episodes=total,
        details=f"{passed_count}/{total} test cases passed",
        recommendation="Parsing logic is robust" if passed else "Review parsing logic",
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run sanity checks")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--check", type=str, choices=["all", "single", "injection", "parsing"],
                        default="all", help="Which check to run")

    args = parser.parse_args()

    results = []

    # Always run parsing test first (fast, no model needed)
    if args.check in ["all", "parsing"]:
        results.append(run_parsing_test())

    # Run injection test
    if args.check in ["all", "injection"]:
        results.append(run_injection_test(args.model, args.device))

    # Run single-agent test
    if args.check in ["all", "single"]:
        results.append(run_single_agent_full_info(args.model, args.n_episodes, "easy", args.device))

    # Summary
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n{r.name}: {status}")
        print(f"  Success rate: {r.success_rate:.1%} ({r.n_episodes} episodes)")
        print(f"  Details: {r.details}")
        print(f"  Recommendation: {r.recommendation}")
        if not r.passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED - Ready to proceed with experiments")
    else:
        print("SOME CHECKS FAILED - Address issues before scaling up")
    print("=" * 60)

    # Save results
    output_path = Path("results/sanity_checks.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([{
            "name": r.name,
            "passed": bool(r.passed),  # Convert numpy bool to Python bool
            "success_rate": float(r.success_rate),
            "n_episodes": int(r.n_episodes),
            "details": r.details,
            "recommendation": r.recommendation,
        } for r in results], f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
