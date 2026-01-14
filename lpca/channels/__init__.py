"""Communication channel implementations for LPCA."""

from lpca.channels.base import BaseChannel, Message
from lpca.channels.text import (
    NoCommChannel,
    FullTextChannel,
    BudgetedTextChannel,
    SummarizationChannel,
    RetrievalChannel,
    StructuredChannel,
    create_channel,
)

__all__ = [
    # Base
    "BaseChannel",
    "Message",
    # Text baselines (P0-P5)
    "NoCommChannel",
    "FullTextChannel",
    "BudgetedTextChannel",
    "SummarizationChannel",
    "RetrievalChannel",
    "StructuredChannel",
    "create_channel",
]

# Lazy imports for torch-dependent channels
try:
    from lpca.channels.cipher import (
        CIPHERChannel,
        MultiTokenCIPHER,
        create_cipher_channel,
    )
    __all__.extend([
        "CIPHERChannel",
        "MultiTokenCIPHER",
        "create_cipher_channel",
    ])
except ImportError:
    pass  # torch not available

try:
    from lpca.channels.activation import (
        ActivationGraftingChannel,
        ActivationGraftingProtocol,
        create_activation_channel,
        layer_sweep_configs,
        combine_fn_sweep_configs,
    )
    __all__.extend([
        "ActivationGraftingChannel",
        "ActivationGraftingProtocol",
        "create_activation_channel",
        "layer_sweep_configs",
        "combine_fn_sweep_configs",
    ])
except ImportError:
    pass  # torch not available
