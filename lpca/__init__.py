"""
LPCA: Latent-Path Communication for AI Agents

A research framework for evaluating machine-native communication
between multi-agent AI systems.
"""

__version__ = "0.1.0"
__author__ = "LPCA Research Team"

from lpca.core.config import ExperimentConfig, load_config
from lpca.core.logging import EpisodeLogger, setup_logging
from lpca.core.metrics import MetricsCalculator
from lpca.core.budget import BudgetAccountant

__all__ = [
    "ExperimentConfig",
    "load_config",
    "EpisodeLogger",
    "setup_logging",
    "MetricsCalculator",
    "BudgetAccountant",
]
