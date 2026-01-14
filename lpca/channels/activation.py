"""
Activation Grafting Channel (A0).

Implements mid-layer activation communication where the sender's
activation at layer ℓ is combined with the receiver's activation.

Based on: "Communicating Activations Between Language Model Agents" (2025)
https://arxiv.org/abs/2501.14082

Key findings from paper:
- Up to 27% improvement over natural language communication
- Achieves this with <1/4 the compute
- Zero additional parameters needed
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from lpca.channels.base import BaseChannel, Message


@dataclass
class ActivationMessage:
    """Container for activation-based message."""
    activation: torch.Tensor
    layer_idx: int
    seq_len: int
    combine_fn: str
    metadata: Dict[str, Any]


class ActivationGraftingChannel(BaseChannel):
    """
    A0: Activation Grafting Channel

    Communication via mid-layer activation injection:
    1. Sender runs forward pass, captures activation at layer ℓ
    2. Receiver runs forward pass until layer ℓ
    3. Sender's activation is combined with receiver's via function f
    4. Receiver continues forward pass from combined activation

    This is the "strong latent baseline" - if this doesn't improve
    over text, latent communication may not be beneficial for the task.
    """

    def __init__(
        self,
        model: Any = None,
        layer_idx: Optional[int] = None,
        combine_fn: str = "average",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(
            name="A0",
            channel_type="activation",
        )
        self.model = model
        self.layer_idx = layer_idx
        self.combine_fn_name = combine_fn
        self.combine_fn = self._get_combine_fn(combine_fn)
        self.dtype = dtype

        # Model info (set when attached)
        self.n_layers: int = 0
        self.d_model: int = 0

        # Captured activation storage
        self._captured_activation: Optional[torch.Tensor] = None
        self._hook_handle: Optional[Any] = None

    def _get_combine_fn(self, name: str) -> Callable:
        """Get combination function by name."""
        fns = {
            'replace': lambda r, s: s,
            'add': lambda r, s: r + s,
            'average': lambda r, s: (r + s) / 2,
            'weighted_0.3': lambda r, s: 0.7 * r + 0.3 * s,
            'weighted_0.5': lambda r, s: 0.5 * r + 0.5 * s,
            'weighted_0.7': lambda r, s: 0.3 * r + 0.7 * s,
            'weighted_0.9': lambda r, s: 0.1 * r + 0.9 * s,
        }
        return fns.get(name, fns['average'])

    def attach_model(self, model: Any):
        """Attach model and configure layer index."""
        self.model = model

        # Get model info
        if hasattr(model, 'config'):
            self.n_layers = getattr(model.config, 'num_hidden_layers',
                                   getattr(model.config, 'n_layer', 32))
            self.d_model = getattr(model.config, 'hidden_size',
                                  getattr(model.config, 'd_model', 2048))

        # Default to middle layer if not specified
        if self.layer_idx is None:
            self.layer_idx = self.n_layers // 2

    def _get_layers(self) -> Any:
        """Get transformer layers module."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        raise AttributeError("Could not find transformer layers")

    def capture_sender_activation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run sender forward pass and capture activation at target layer.

        Args:
            input_ids: Sender's input tokens
            attention_mask: Optional attention mask

        Returns:
            Captured activation tensor (batch, seq, d_model)
        """
        if self.model is None:
            raise RuntimeError("Model not attached")

        layers = self._get_layers()
        captured = None

        def capture_hook(module, input, output):
            nonlocal captured
            if isinstance(output, tuple):
                captured = output[0].clone().detach()
            else:
                captured = output.clone().detach()

        # Register hook
        handle = layers[self.layer_idx].register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            handle.remove()

        self._captured_activation = captured
        return captured

    def inject_to_receiver(
        self,
        input_ids: torch.Tensor,
        sender_activation: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run receiver forward pass with activation injection.

        Args:
            input_ids: Receiver's input tokens
            sender_activation: Sender's captured activation
            attention_mask: Optional attention mask

        Returns:
            Output logits after injection
        """
        if self.model is None:
            raise RuntimeError("Model not attached")

        layers = self._get_layers()

        def injection_hook(module, input, output):
            if isinstance(output, tuple):
                receiver_act = output[0]
            else:
                receiver_act = output

            # Handle sequence length mismatch
            sender_len = sender_activation.shape[1]
            receiver_len = receiver_act.shape[1]

            if sender_len != receiver_len:
                # Align by taking overlapping portion from end
                min_len = min(sender_len, receiver_len)
                sender_aligned = sender_activation[:, -min_len:, :]
                receiver_aligned = receiver_act[:, -min_len:, :]

                # Combine
                combined = self.combine_fn(receiver_aligned, sender_aligned)

                # Replace aligned portion
                new_output = receiver_act.clone()
                new_output[:, -min_len:, :] = combined
            else:
                new_output = self.combine_fn(receiver_act, sender_activation)

            if isinstance(output, tuple):
                return (new_output,) + output[1:]
            return new_output

        # Register injection hook
        handle = layers[self.layer_idx].register_forward_hook(injection_hook)

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            handle.remove()

        return outputs.logits

    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """
        Encode activation into message.

        Content should be the activation tensor or input for capture.
        """
        if content is None:
            activation = torch.zeros(1, 1, self.d_model, dtype=self.dtype)
            seq_len = 1
        elif isinstance(content, torch.Tensor):
            if content.dim() == 2:
                # Input IDs - need to capture activation
                activation = self.capture_sender_activation(content)
            else:
                # Already an activation
                activation = content
            seq_len = activation.shape[1]
        else:
            activation = torch.zeros(1, 1, self.d_model, dtype=self.dtype)
            seq_len = 1

        # Calculate bits
        bits_per_element = 16 if self.dtype == torch.float16 else 32
        total_bits = seq_len * self.d_model * bits_per_element

        return Message(
            content={
                'activation': activation,
                'layer_idx': self.layer_idx,
                'seq_len': seq_len,
            },
            format="activation",
            sender=sender,
            receiver="",
            turn_idx=turn_idx,
            bits=total_bits,
            bytes=total_bits // 8,
            metadata={
                'combine_fn': self.combine_fn_name,
                'layer_idx': self.layer_idx,
                'd_model': self.d_model,
            },
        )

    def decode(self, message: Message) -> Dict[str, Any]:
        """Decode activation message."""
        if message.content is None:
            return {'activation': None}

        return {
            'activation': message.content.get('activation'),
            'layer_idx': message.content.get('layer_idx'),
            'seq_len': message.content.get('seq_len'),
        }

    def bits_per_message(self, seq_len: int = 1) -> int:
        """Get bits per message for given sequence length."""
        bits_per_element = 16 if self.dtype == torch.float16 else 32
        return seq_len * self.d_model * bits_per_element


class ActivationGraftingProtocol:
    """
    Full protocol for activation grafting communication.

    Manages the capture-transmit-inject cycle between two agents.
    Uses a single model instance for memory efficiency.
    """

    def __init__(
        self,
        model: Any,
        layer_idx: Optional[int] = None,
        combine_fn: str = "average",
    ):
        self.model = model
        self.channel = ActivationGraftingChannel(
            model=model,
            layer_idx=layer_idx,
            combine_fn=combine_fn,
        )
        self.channel.attach_model(model)

    def communicate(
        self,
        sender_input_ids: torch.Tensor,
        receiver_input_ids: torch.Tensor,
        sender_attention_mask: Optional[torch.Tensor] = None,
        receiver_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute full communication cycle.

        Args:
            sender_input_ids: Sender's input tokens
            receiver_input_ids: Receiver's input tokens
            sender_attention_mask: Optional mask for sender
            receiver_attention_mask: Optional mask for receiver

        Returns:
            Tuple of (receiver_logits, metadata)
        """
        # Step 1: Capture sender's activation
        sender_activation = self.channel.capture_sender_activation(
            sender_input_ids,
            sender_attention_mask,
        )

        # Step 2: Inject into receiver and get output
        receiver_logits = self.channel.inject_to_receiver(
            receiver_input_ids,
            sender_activation,
            receiver_attention_mask,
        )

        # Create message for logging
        message = self.channel.encode(sender_activation, "A", 0)

        metadata = {
            'bits_transmitted': message.bits,
            'layer_idx': self.channel.layer_idx,
            'combine_fn': self.channel.combine_fn_name,
            'sender_seq_len': sender_activation.shape[1],
            'receiver_seq_len': receiver_input_ids.shape[1],
        }

        return receiver_logits, metadata


def layer_sweep_configs(n_layers: int) -> List[Dict[str, Any]]:
    """
    Generate configurations for layer sweep ablation.

    Returns configs for layers at n/4, n/3, n/2, 2n/3, 3n/4.
    """
    positions = [
        ('quarter', n_layers // 4),
        ('third', n_layers // 3),
        ('half', n_layers // 2),
        ('two_thirds', 2 * n_layers // 3),
        ('three_quarters', 3 * n_layers // 4),
    ]

    return [
        {'name': name, 'layer_idx': idx}
        for name, idx in positions
    ]


def combine_fn_sweep_configs() -> List[Dict[str, Any]]:
    """
    Generate configurations for combination function sweep.
    """
    return [
        {'name': 'replace', 'combine_fn': 'replace'},
        {'name': 'add', 'combine_fn': 'add'},
        {'name': 'average', 'combine_fn': 'average'},
        {'name': 'weighted_0.3', 'combine_fn': 'weighted_0.3'},
        {'name': 'weighted_0.5', 'combine_fn': 'weighted_0.5'},
        {'name': 'weighted_0.7', 'combine_fn': 'weighted_0.7'},
    ]


def create_activation_channel(
    model: Any = None,
    layer_idx: Optional[int] = None,
    combine_fn: str = "average",
) -> ActivationGraftingChannel:
    """Factory function to create activation grafting channel."""
    channel = ActivationGraftingChannel(
        model=model,
        layer_idx=layer_idx,
        combine_fn=combine_fn,
    )

    if model is not None:
        channel.attach_model(model)

    return channel
