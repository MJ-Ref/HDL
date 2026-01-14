"""Agent implementations for LPCA experiments."""

from lpca.agents.base import BaseAgent, AgentResponse

__all__ = [
    # Base
    "BaseAgent",
    "AgentResponse",
]

# Lazy imports for torch-dependent components
try:
    from lpca.agents.model_wrapper import (
        ModelWrapper,
        ActivationHook,
        combine_replace,
        combine_add,
        combine_average,
        combine_weighted,
    )
    __all__.extend([
        "ModelWrapper",
        "ActivationHook",
        "combine_replace",
        "combine_add",
        "combine_average",
        "combine_weighted",
    ])
except ImportError:
    pass  # torch not available

try:
    from lpca.agents.llm_agent import (
        LLMAgent,
        DualAgentRunner,
        load_llm_agent,
    )
    __all__.extend([
        "LLMAgent",
        "DualAgentRunner",
        "load_llm_agent",
    ])
except ImportError:
    pass  # torch/transformers not available
