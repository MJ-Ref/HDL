"""
Base classes for communication channels.

Channels define the message interface between agents, including
format, capacity constraints, and integration mechanism.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


@dataclass
class Message:
    """A message sent between agents."""
    content: Any
    format: str  # text, json, embedding, activation, latent, discrete
    sender: str  # 'A' or 'B'
    receiver: str
    turn_idx: int

    # Computed fields
    bits: int = 0
    bytes: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "content": str(self.content)[:1000] if self.content else None,
            "format": self.format,
            "sender": self.sender,
            "receiver": self.receiver,
            "turn_idx": self.turn_idx,
            "bits": self.bits,
            "bytes": self.bytes,
            "timestamp": self.timestamp,
        }


class BaseChannel(ABC):
    """
    Abstract base class for communication channels.

    A channel handles:
    - Message formatting/encoding
    - Capacity constraints
    - Message delivery
    - Logging
    """

    def __init__(
        self,
        name: str,
        channel_type: str,
        max_bits_per_message: Optional[int] = None,
        max_messages_per_episode: Optional[int] = None,
    ):
        self.name = name
        self.channel_type = channel_type
        self.max_bits_per_message = max_bits_per_message
        self.max_messages_per_episode = max_messages_per_episode

        self.message_history: List[Message] = []
        self._message_count = 0

    def reset(self):
        """Reset channel state for new episode."""
        self.message_history = []
        self._message_count = 0

    @abstractmethod
    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """
        Encode content into a message.

        Args:
            content: Raw content to send
            sender: 'A' or 'B'
            turn_idx: Current turn index

        Returns:
            Encoded Message
        """
        pass

    @abstractmethod
    def decode(self, message: Message) -> Any:
        """
        Decode a message for the receiver.

        Args:
            message: Message to decode

        Returns:
            Decoded content for receiver
        """
        pass

    def send(self, content: Any, sender: str, receiver: str, turn_idx: int) -> Message:
        """
        Send a message through the channel.

        Args:
            content: Content to send
            sender: Sending agent ('A' or 'B')
            receiver: Receiving agent ('A' or 'B')
            turn_idx: Current turn index

        Returns:
            The sent Message
        """
        # Check message limit
        if self.max_messages_per_episode is not None:
            if self._message_count >= self.max_messages_per_episode:
                raise ChannelLimitExceeded(
                    f"Message limit exceeded: {self.max_messages_per_episode}"
                )

        # Encode message
        message = self.encode(content, sender, turn_idx)
        message.receiver = receiver

        # Check bit limit
        if self.max_bits_per_message is not None:
            if message.bits > self.max_bits_per_message:
                message = self._truncate_message(message)

        # Record
        self.message_history.append(message)
        self._message_count += 1

        return message

    def receive(self, message: Message) -> Any:
        """
        Receive and decode a message.

        Args:
            message: Message to receive

        Returns:
            Decoded content
        """
        return self.decode(message)

    def _truncate_message(self, message: Message) -> Message:
        """Truncate message to fit within bit budget."""
        # Override in subclasses for format-specific truncation
        return message

    def get_history(self, agent: Optional[str] = None) -> List[Message]:
        """Get message history, optionally filtered by agent."""
        if agent is None:
            return self.message_history
        return [m for m in self.message_history
               if m.sender == agent or m.receiver == agent]

    def total_bits(self) -> int:
        """Get total bits transmitted."""
        return sum(m.bits for m in self.message_history)

    def get_stats(self) -> Dict[str, Any]:
        """Get channel statistics."""
        return {
            "name": self.name,
            "type": self.channel_type,
            "message_count": len(self.message_history),
            "total_bits": self.total_bits(),
            "avg_bits_per_message": (
                self.total_bits() / len(self.message_history)
                if self.message_history else 0
            ),
        }


class ChannelLimitExceeded(Exception):
    """Raised when a channel limit is exceeded."""
    pass
