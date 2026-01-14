"""
Budget accounting for LPCA experiments.

Tracks and enforces resource budgets including bits transmitted,
compute (FLOPs), and latency.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import json


@dataclass
class MessageBudget:
    """Budget tracking for a single message."""
    bits: int = 0
    bytes: int = 0
    tokens: int = 0
    format: str = "text"


@dataclass
class TurnBudget:
    """Budget tracking for a single turn."""
    input_tokens: int = 0
    output_tokens: int = 0
    message: MessageBudget = field(default_factory=MessageBudget)
    forward_passes: int = 1
    inference_ms: float = 0.0


@dataclass
class EpisodeBudget:
    """Aggregate budget for an episode."""
    turns: List[TurnBudget] = field(default_factory=list)

    @property
    def total_bits(self) -> int:
        return sum(t.message.bits for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(t.input_tokens + t.output_tokens for t in self.turns)

    @property
    def total_forward_passes(self) -> int:
        return sum(t.forward_passes for t in self.turns)

    @property
    def total_inference_ms(self) -> float:
        return sum(t.inference_ms for t in self.turns)

    def estimate_flops(self, n_params: int) -> int:
        """Estimate FLOPs using 2 * params * tokens approximation."""
        return 2 * n_params * self.total_tokens


class BudgetAccountant:
    """
    Track and enforce resource budgets for experiments.

    Supports both soft tracking (logging) and hard enforcement (limits).
    """

    def __init__(
        self,
        max_bits_per_message: Optional[int] = None,
        max_bits_per_episode: Optional[int] = None,
        max_tokens_per_turn: Optional[int] = None,
        max_turns: Optional[int] = None,
        model_params: int = 1_000_000_000,  # 1B default
    ):
        self.max_bits_per_message = max_bits_per_message
        self.max_bits_per_episode = max_bits_per_episode
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_turns = max_turns
        self.model_params = model_params

        self.current_episode: Optional[EpisodeBudget] = None
        self._turn_start_time: Optional[float] = None

    def start_episode(self):
        """Start tracking a new episode."""
        self.current_episode = EpisodeBudget()

    def start_turn(self):
        """Mark the start of a turn for timing."""
        self._turn_start_time = time.perf_counter()

    def end_turn(
        self,
        input_tokens: int,
        output_tokens: int,
        message: any,
        message_format: str,
        forward_passes: int = 1,
    ) -> TurnBudget:
        """Record a completed turn and return budget info."""
        inference_ms = 0.0
        if self._turn_start_time is not None:
            inference_ms = (time.perf_counter() - self._turn_start_time) * 1000

        message_budget = self.compute_message_budget(message, message_format)

        turn_budget = TurnBudget(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            message=message_budget,
            forward_passes=forward_passes,
            inference_ms=inference_ms,
        )

        if self.current_episode is not None:
            self.current_episode.turns.append(turn_budget)

        return turn_budget

    def compute_message_budget(
        self,
        message: any,
        format: str
    ) -> MessageBudget:
        """Compute budget for a message based on format."""
        if message is None:
            return MessageBudget(bits=0, bytes=0, tokens=0, format=format)

        if format == "text":
            if isinstance(message, str):
                byte_count = len(message.encode("utf-8"))
                # Rough token estimate: ~4 chars per token
                token_count = len(message) // 4
                return MessageBudget(
                    bits=byte_count * 8,
                    bytes=byte_count,
                    tokens=token_count,
                    format=format,
                )

        elif format == "json" or format == "structured":
            json_str = json.dumps(message, separators=(",", ":"))
            byte_count = len(json_str.encode("utf-8"))
            return MessageBudget(
                bits=byte_count * 8,
                bytes=byte_count,
                tokens=byte_count // 4,
                format=format,
            )

        elif format == "embedding":
            # Expected embedding: d_model floats
            d_model = getattr(message, "shape", [0])[-1] if hasattr(message, "shape") else 2048
            bits = d_model * 16  # fp16
            return MessageBudget(bits=bits, bytes=bits // 8, tokens=0, format=format)

        elif format == "activation":
            # Activation: seq_len * d_model floats
            if hasattr(message, "shape"):
                seq_len, d_model = message.shape[-2], message.shape[-1]
            else:
                seq_len, d_model = 1, 2048
            bits = seq_len * d_model * 16  # fp16
            return MessageBudget(bits=bits, bytes=bits // 8, tokens=0, format=format)

        elif format == "latent":
            # Continuous latent: k * d_model floats
            if hasattr(message, "shape"):
                k, d_model = message.shape[-2], message.shape[-1]
            else:
                k, d_model = 16, 2048
            bits = k * d_model * 16  # fp16
            return MessageBudget(bits=bits, bytes=bits // 8, tokens=0, format=format)

        elif format == "discrete":
            # Discrete latent: k indices into codebook
            if hasattr(message, "__len__"):
                k = len(message)
            else:
                k = 16
            # Assume codebook size from context or default
            codebook_size = getattr(message, "codebook_size", 1024)
            bits = k * math.ceil(math.log2(codebook_size))
            return MessageBudget(bits=bits, bytes=math.ceil(bits / 8), tokens=0, format=format)

        return MessageBudget(bits=0, bytes=0, tokens=0, format=format)

    def check_message_budget(self, message_budget: MessageBudget) -> bool:
        """Check if message is within budget. Returns True if OK."""
        if self.max_bits_per_message is not None:
            if message_budget.bits > self.max_bits_per_message:
                return False
        return True

    def check_episode_budget(self) -> bool:
        """Check if episode is within budget. Returns True if OK."""
        if self.current_episode is None:
            return True

        if self.max_bits_per_episode is not None:
            if self.current_episode.total_bits > self.max_bits_per_episode:
                return False

        if self.max_turns is not None:
            if len(self.current_episode.turns) > self.max_turns:
                return False

        return True

    def truncate_message_to_budget(
        self,
        message: str,
        max_bits: Optional[int] = None
    ) -> str:
        """Truncate text message to fit within bit budget."""
        if max_bits is None:
            max_bits = self.max_bits_per_message
        if max_bits is None:
            return message

        max_bytes = max_bits // 8
        encoded = message.encode("utf-8")

        if len(encoded) <= max_bytes:
            return message

        # Truncate at character boundary
        while len(message.encode("utf-8")) > max_bytes:
            message = message[:-1]

        return message

    def get_episode_summary(self) -> Dict:
        """Get summary of current episode budget."""
        if self.current_episode is None:
            return {}

        return {
            "total_bits": self.current_episode.total_bits,
            "total_tokens": self.current_episode.total_tokens,
            "total_forward_passes": self.current_episode.total_forward_passes,
            "total_inference_ms": self.current_episode.total_inference_ms,
            "estimated_flops": self.current_episode.estimate_flops(self.model_params),
            "n_turns": len(self.current_episode.turns),
            "within_budget": self.check_episode_budget(),
        }

    def get_turn_budgets(self) -> List[Dict]:
        """Get budget breakdown by turn."""
        if self.current_episode is None:
            return []

        return [
            {
                "turn": i,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                "message_bits": t.message.bits,
                "message_format": t.message.format,
                "forward_passes": t.forward_passes,
                "inference_ms": t.inference_ms,
            }
            for i, t in enumerate(self.current_episode.turns)
        ]


class BudgetEnforcer:
    """
    Decorator/context manager for enforcing budgets.

    Can be used to wrap message generation to ensure budget compliance.
    """

    def __init__(self, accountant: BudgetAccountant, strict: bool = False):
        self.accountant = accountant
        self.strict = strict

    def enforce_message(self, message: any, format: str) -> any:
        """Enforce budget on a message, potentially truncating."""
        budget = self.accountant.compute_message_budget(message, format)

        if not self.accountant.check_message_budget(budget):
            if self.strict:
                raise BudgetExceededError(
                    f"Message exceeds budget: {budget.bits} bits "
                    f"(max: {self.accountant.max_bits_per_message})"
                )
            elif format == "text" and isinstance(message, str):
                message = self.accountant.truncate_message_to_budget(message)

        return message


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded in strict mode."""
    pass
