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
from lpca.channels.cipher import (
    CIPHERChannel,
    MultiTokenCIPHER,
    create_cipher_channel,
)
from lpca.channels.activation import (
    ActivationGraftingChannel,
    ActivationGraftingProtocol,
    create_activation_channel,
    layer_sweep_configs,
    combine_fn_sweep_configs,
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
    # CIPHER (E0)
    "CIPHERChannel",
    "MultiTokenCIPHER",
    "create_cipher_channel",
    # Activation grafting (A0)
    "ActivationGraftingChannel",
    "ActivationGraftingProtocol",
    "create_activation_channel",
    "layer_sweep_configs",
    "combine_fn_sweep_configs",
]
