"""
Base agent interface for LPCA experiments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentResponse:
    """Response from an agent."""
    text: str
    message_to_partner: Optional[str] = None
    action: Optional[str] = None
    final_answer: Optional[Any] = None
    is_done: bool = False

    # Metadata
    input_tokens: int = 0
    output_tokens: int = 0
    inference_ms: float = 0.0

    # Internal state (for latent protocols)
    hidden_state: Optional[Any] = None
    activations: Optional[Dict[int, Any]] = None


class BaseAgent(ABC):
    """
    Abstract base class for LPCA agents.

    An agent consumes observations and messages, and produces
    responses including messages to other agents.
    """

    def __init__(
        self,
        agent_id: str,
        role: str = "general",
    ):
        self.agent_id = agent_id
        self.role = role
        self.conversation_history: List[Dict[str, str]] = []

    def reset(self):
        """Reset agent state for new episode."""
        self.conversation_history = []

    @abstractmethod
    def respond(
        self,
        observation: str,
        received_message: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate a response given observation and optional message.

        Args:
            observation: Task observation for this agent
            received_message: Message from partner agent (if any)
            system_prompt: Optional system prompt override

        Returns:
            AgentResponse with text, message, and optional action
        """
        pass

    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()

    def format_input(
        self,
        observation: str,
        received_message: Optional[str] = None,
    ) -> str:
        """Format input for the agent."""
        parts = [f"Observation:\n{observation}"]

        if received_message:
            parts.append(f"\nMessage from partner:\n{received_message}")

        parts.append("\nRespond with your analysis and any message for your partner.")

        return "\n".join(parts)
