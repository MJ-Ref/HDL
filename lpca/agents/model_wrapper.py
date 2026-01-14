"""
Model wrapper with activation hooks for LPCA experiments.

Provides white-box access to model internals for:
- Activation capture and injection
- Hidden state extraction
- Gradient computation (for codec training)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class ActivationHook:
    """Hook for capturing or modifying activations."""
    layer_idx: int
    capture: bool = True
    inject: bool = False
    injection_value: Optional[torch.Tensor] = None
    combine_fn: Optional[Callable] = None
    captured_value: Optional[torch.Tensor] = None


class ModelWrapper:
    """
    Wrapper for transformer models with activation hooks.

    Provides unified interface for:
    - Standard inference
    - Activation capture at any layer
    - Activation injection for grafting
    - Hidden state extraction
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Model info
        self.n_layers = self._get_n_layers()
        self.d_model = self._get_d_model()

        # Hook management
        self._hooks: Dict[int, ActivationHook] = {}
        self._active_handles: List[Any] = []

    def _get_n_layers(self) -> int:
        """Get number of transformer layers."""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "num_hidden_layers"):
                return self.model.config.num_hidden_layers
            if hasattr(self.model.config, "n_layer"):
                return self.model.config.n_layer
        # Try to infer from model structure
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        return 32  # Default

    def _get_d_model(self) -> int:
        """Get model hidden dimension."""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "hidden_size"):
                return self.model.config.hidden_size
            if hasattr(self.model.config, "d_model"):
                return self.model.config.d_model
        return 2048  # Default

    def _get_layers(self) -> Any:
        """Get the transformer layers module."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError("Could not find transformer layers")

    def register_hook(self, hook: ActivationHook):
        """Register an activation hook."""
        self._hooks[hook.layer_idx] = hook

    def clear_hooks(self):
        """Clear all hooks."""
        for handle in self._active_handles:
            handle.remove()
        self._active_handles = []
        self._hooks = {}

    def _install_hooks(self):
        """Install forward hooks for registered layers."""
        layers = self._get_layers()

        for layer_idx, hook in self._hooks.items():
            if layer_idx >= len(layers):
                continue

            def make_hook_fn(h: ActivationHook):
                def hook_fn(module, input, output):
                    # Handle different output formats
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Capture
                    if h.capture:
                        h.captured_value = hidden_states.clone().detach()

                    # Inject
                    if h.inject and h.injection_value is not None:
                        if h.combine_fn is not None:
                            combined = h.combine_fn(hidden_states, h.injection_value)
                        else:
                            combined = h.injection_value

                        if isinstance(output, tuple):
                            return (combined,) + output[1:]
                        return combined

                    return output
                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook_fn(hook))
            self._active_handles.append(handle)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text response.

        Returns:
            Tuple of (generated_text, metadata)
        """
        start_time = time.perf_counter()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings - max_new_tokens
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Install hooks if any registered
        if self._hooks:
            self._install_hooks()

        try:
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode
            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

        finally:
            # Clean up hooks
            for handle in self._active_handles:
                handle.remove()
            self._active_handles = []

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        metadata = {
            "input_tokens": input_length,
            "output_tokens": len(generated_ids),
            "inference_ms": elapsed_ms,
            "captured_activations": {
                idx: hook.captured_value
                for idx, hook in self._hooks.items()
                if hook.captured_value is not None
            },
        }

        return generated_text, metadata

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run forward pass and return logits.

        Returns:
            Tuple of (logits, metadata with captured activations)
        """
        # Install hooks if any registered
        if self._hooks:
            self._install_hooks()

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            logits = outputs.logits

        finally:
            # Clean up hooks
            for handle in self._active_handles:
                handle.remove()
            self._active_handles = []

        metadata = {
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "captured_activations": {
                idx: hook.captured_value
                for idx, hook in self._hooks.items()
                if hook.captured_value is not None
            },
        }

        return logits, metadata

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings."""
        embed_layer = self.model.get_input_embeddings()
        return embed_layer(input_ids)

    def forward_with_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward pass with custom input embeddings."""
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        return outputs.logits

    def capture_activation(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Capture activation at specific layer."""
        hook = ActivationHook(layer_idx=layer_idx, capture=True)
        self.register_hook(hook)

        try:
            self.forward(input_ids)
            return hook.captured_value
        finally:
            self.clear_hooks()

    def inject_activation(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        injection: torch.Tensor,
        combine_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Run forward with activation injection at specific layer."""
        hook = ActivationHook(
            layer_idx=layer_idx,
            capture=False,
            inject=True,
            injection_value=injection,
            combine_fn=combine_fn,
        )
        self.register_hook(hook)

        try:
            logits, _ = self.forward(input_ids)
            return logits
        finally:
            self.clear_hooks()


# Standard combine functions for activation grafting
def combine_replace(receiver: torch.Tensor, sender: torch.Tensor) -> torch.Tensor:
    """Replace receiver activation with sender's."""
    return sender


def combine_add(receiver: torch.Tensor, sender: torch.Tensor) -> torch.Tensor:
    """Add sender activation to receiver's."""
    return receiver + sender


def combine_average(receiver: torch.Tensor, sender: torch.Tensor) -> torch.Tensor:
    """Average receiver and sender activations."""
    return (receiver + sender) / 2


def combine_weighted(alpha: float = 0.5):
    """Create weighted combination function."""
    def combine(receiver: torch.Tensor, sender: torch.Tensor) -> torch.Tensor:
        return (1 - alpha) * receiver + alpha * sender
    return combine
