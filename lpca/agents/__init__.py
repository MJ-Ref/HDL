"""Agent implementations for LPCA experiments."""

from lpca.agents.base import BaseAgent, AgentResponse
from lpca.agents.model_wrapper import (
    ModelWrapper,
    ActivationHook,
    combine_replace,
    combine_add,
    combine_average,
    combine_weighted,
)
from lpca.agents.llm_agent import (
    LLMAgent,
    DualAgentRunner,
    load_llm_agent,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentResponse",
    # Model wrapper
    "ModelWrapper",
    "ActivationHook",
    "combine_replace",
    "combine_add",
    "combine_average",
    "combine_weighted",
    # LLM agent
    "LLMAgent",
    "DualAgentRunner",
    "load_llm_agent",
]
