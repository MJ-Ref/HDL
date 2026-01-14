#!/usr/bin/env python3
"""
Run LPCA experiments.

Usage:
    python scripts/run_experiment.py --config configs/base.yaml
    python scripts/run_experiment.py --config configs/experiments/e1_baseline.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lpca.core.config import ExperimentConfig, load_config
from lpca.core.logging import EpisodeLogger, setup_logging, EpisodeLog, TurnLog
from lpca.core.metrics import MetricsCalculator
from lpca.core.budget import BudgetAccountant
from lpca.envs.split_synthetic import SplitSyntheticEnv
from lpca.channels.text import create_channel


def run_episode(
    env,
    channel,
    config: ExperimentConfig,
    seed: int,
    episode_idx: int,
    logger,
) -> EpisodeLog:
    """Run a single episode and return the log."""

    # Reset environment and channel
    task = env.reset(seed)
    channel.reset()

    # Initialize budget tracking
    budget = BudgetAccountant(
        max_bits_per_message=config.protocol.params.get("max_bits"),
        max_turns=config.max_turns,
    )
    budget.start_episode()

    # Create episode log
    episode_log = EpisodeLog(
        episode_id=f"{config.experiment_id}_{episode_idx:04d}",
        experiment_id=config.experiment_id,
        seed=seed,
        task_family=config.task.family,
        task_type=config.task.task_type,
        protocol=config.protocol.name,
        obs_A=task.obs_A,
        obs_B=task.obs_B,
        ground_truth=task.ground_truth,
        model_id=config.model.name,
        temperature=config.model.temperature,
    )

    print(f"  Episode {episode_idx}: seed={seed}, task={config.task.task_type}")

    # Simulate turns (placeholder - real implementation would use model)
    for turn in range(config.max_turns):
        # In real implementation:
        # 1. Get agent A response
        # 2. Send message through channel
        # 3. Get agent B response
        # 4. Check for final answer

        turn_log = TurnLog(
            turn_idx=turn,
            agent="A" if turn % 2 == 0 else "B",
            input_tokens=100,  # Placeholder
            output_tokens=50,  # Placeholder
            message_format=channel.channel_type,
            message_bits=0,
        )
        episode_log.turns.append(turn_log)

        # Placeholder: random completion after a few turns
        import random
        if turn >= 2 and random.random() < 0.3:
            break

    # Verify (placeholder)
    import random
    episode_log.verifier_result = random.random() < 0.6
    episode_log.partial_credit = random.random()

    # Budget summary
    summary = budget.get_episode_summary()
    episode_log.total_bits = summary.get("total_bits", 0)

    return episode_log


def main():
    parser = argparse.ArgumentParser(description="Run LPCA experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        help="Override protocol (P0-P5)",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=None,
        help="Override number of episodes",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print config and exit",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Apply overrides
    if args.protocol:
        config.protocol.name = args.protocol
    if args.n_episodes:
        config.task.n_episodes = args.n_episodes

    # Setup logging
    logger = setup_logging(level=config.logging.log_level)
    logger.info(f"Starting experiment: {config.experiment_id}")
    logger.info(f"Config: {config_path}")

    if args.dry_run:
        import json
        print(json.dumps(config.to_dict(), indent=2, default=str))
        return

    # Initialize components
    env = SplitSyntheticEnv(difficulty=config.task.difficulty)
    channel = create_channel(config.protocol.name)

    # Select task type
    if config.task.task_type in env.get_task_types():
        env.select_environment(config.task.task_type)

    # Create output directory
    output_dir = Path(config.logging.output_dir) / config.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(output_dir / "config.yaml")

    # Run episodes
    metrics_calc = MetricsCalculator()

    with EpisodeLogger(output_dir, config.experiment_id) as episode_logger:
        for episode_idx in range(config.task.n_episodes):
            seed = env.get_deterministic_seed(config.base_seed, episode_idx)

            episode_log = run_episode(
                env=env,
                channel=channel,
                config=config,
                seed=seed,
                episode_idx=episode_idx,
                logger=logger,
            )

            episode_logger.log_episode(episode_log)

            if (episode_idx + 1) % 10 == 0:
                logger.info(f"Completed {episode_idx + 1}/{config.task.n_episodes} episodes")

    # Compute final metrics
    outcomes = [ep.verifier_result for ep in episode_logger.episodes]
    partial_scores = [ep.partial_credit for ep in episode_logger.episodes]

    capability_metrics = metrics_calc.compute_capability_metrics(
        outcomes=outcomes,
        partial_scores=partial_scores,
        turns=[ep.n_turns for ep in episode_logger.episodes],
        had_retry=[False] * len(outcomes),  # Placeholder
    )

    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"Success Rate: {capability_metrics.success_rate:.3f} "
               f"(95% CI: {capability_metrics.success_rate_ci})")
    logger.info(f"Partial Credit: {capability_metrics.partial_credit_mean:.3f} "
               f"(std: {capability_metrics.partial_credit_std:.3f})")
    logger.info(f"Episodes: {capability_metrics.n_episodes}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
