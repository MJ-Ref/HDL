"""
Logging infrastructure for LPCA experiments.

Provides structured logging for episodes, turns, and metrics
with support for JSONL and Parquet output formats.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys


@dataclass
class TurnLog:
    """Log entry for a single agent turn."""
    turn_idx: int
    agent: str  # 'A' or 'B'

    # Input
    input_tokens: int = 0
    input_text: str = ""

    # Message sent
    message_format: str = "text"  # text, json, embedding, activation, latent, discrete
    message_content: Any = None
    message_bits: int = 0

    # Output
    output_tokens: int = 0
    output_text: str = ""

    # Action taken
    action: Optional[str] = None
    action_result: Optional[str] = None

    # Timing
    inference_ms: int = 0

    # Monitor signals
    monitor_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeLog:
    """Complete log for a single episode."""
    # Identification (required)
    episode_id: str
    experiment_id: str
    seed: int

    # Task info (required)
    task_family: str
    task_type: str

    # Protocol (required)
    protocol: str

    # Optional fields with defaults
    task_params: Dict[str, Any] = field(default_factory=dict)
    protocol_params: Dict[str, Any] = field(default_factory=dict)

    # Task inputs (for reproducibility)
    obs_A: str = ""
    obs_B: str = ""
    ground_truth: Any = None

    # Execution
    turns: List[TurnLog] = field(default_factory=list)

    # Outcomes
    final_output: Any = None
    verifier_result: bool = False
    partial_credit: float = 0.0

    # Budget accounting
    total_bits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_flops: int = 0
    wall_clock_ms: int = 0

    # Model info
    model_id: str = ""
    temperature: float = 0.7

    # Safety signals
    compliance_condition: str = ""  # monitored/unmonitored
    monitor_flags: Dict[str, bool] = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def success(self) -> bool:
        return self.verifier_result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling nested dataclasses."""
        result = asdict(self)
        result["n_turns"] = self.n_turns
        result["success"] = self.success
        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class EpisodeLogger:
    """Logger for episode data with multiple output formats."""

    def __init__(
        self,
        output_dir: str | Path,
        experiment_id: str,
        save_format: str = "both",
    ):
        self.output_dir = Path(output_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.save_format = save_format

        self.episodes: List[EpisodeLog] = []
        self._jsonl_path = self.output_dir / "episodes.jsonl"
        self._parquet_path = self.output_dir / "episodes.parquet"

        # Initialize JSONL file
        if save_format in ["jsonl", "both"]:
            self._jsonl_file = open(self._jsonl_path, "a")

    def log_episode(self, episode: EpisodeLog):
        """Log a completed episode."""
        self.episodes.append(episode)

        # Write to JSONL immediately (append mode)
        if self.save_format in ["jsonl", "both"]:
            self._jsonl_file.write(episode.to_json() + "\n")
            self._jsonl_file.flush()

    def save_parquet(self):
        """Save all episodes to Parquet format."""
        if self.save_format not in ["parquet", "both"]:
            return

        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Flatten episodes for tabular format
            records = []
            for ep in self.episodes:
                record = {
                    "episode_id": ep.episode_id,
                    "experiment_id": ep.experiment_id,
                    "seed": ep.seed,
                    "task_family": ep.task_family,
                    "task_type": ep.task_type,
                    "protocol": ep.protocol,
                    "n_turns": ep.n_turns,
                    "success": ep.success,
                    "partial_credit": ep.partial_credit,
                    "total_bits": ep.total_bits,
                    "total_input_tokens": ep.total_input_tokens,
                    "total_output_tokens": ep.total_output_tokens,
                    "wall_clock_ms": ep.wall_clock_ms,
                    "model_id": ep.model_id,
                    "timestamp": ep.timestamp,
                }
                records.append(record)

            df = pd.DataFrame(records)
            df.to_parquet(self._parquet_path, index=False)

        except ImportError:
            logging.warning("pandas/pyarrow not available, skipping parquet save")

    def close(self):
        """Close file handles and finalize output."""
        if hasattr(self, "_jsonl_file"):
            self._jsonl_file.close()
        self.save_parquet()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Configure logging for LPCA experiments."""
    logger = logging.getLogger("lpca")
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class MetricsAggregator:
    """Aggregate metrics across episodes for reporting."""

    def __init__(self):
        self.episodes: List[EpisodeLog] = []

    def add_episode(self, episode: EpisodeLog):
        self.episodes.append(episode)

    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    def mean_partial_credit(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.partial_credit for e in self.episodes) / len(self.episodes)

    def mean_turns(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.n_turns for e in self.episodes) / len(self.episodes)

    def mean_bits(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.total_bits for e in self.episodes) / len(self.episodes)

    def summary(self) -> Dict[str, float]:
        return {
            "n_episodes": len(self.episodes),
            "success_rate": self.success_rate(),
            "mean_partial_credit": self.mean_partial_credit(),
            "mean_turns": self.mean_turns(),
            "mean_bits": self.mean_bits(),
        }
