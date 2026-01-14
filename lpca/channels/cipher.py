"""
CIPHER Expected Embedding Channel (E0).

Implements token-space softening where instead of sampling discrete tokens,
we communicate via the expectation of output embeddings across the vocabulary.

Based on: "CIPHER: Communicating Inter-Model Protocol Through Embedding
Representation" (ICLR 2024) - https://arxiv.org/abs/2310.06272
"""

import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from lpca.channels.base import BaseChannel, Message


class CIPHERChannel(BaseChannel):
    """
    E0: CIPHER Expected Embedding Channel

    Instead of generating discrete tokens, produces expected embedding
    vectors that encode distributional beliefs across the vocabulary.

    Key mechanism:
    1. Generate logits at a position
    2. Apply softmax (optionally with top-k filtering)
    3. Compute expected embedding = sum(prob_i * embed_i)
    4. Receiver consumes as soft token (prepended embedding)
    """

    def __init__(
        self,
        model: Any = None,
        top_k: int = 100,
        temperature: float = 1.0,
        n_soft_tokens: int = 1,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(
            name="E0",
            channel_type="embedding",
        )
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.n_soft_tokens = n_soft_tokens
        self.dtype = dtype

        # Will be set when model is attached
        self.embed_matrix: Optional[torch.Tensor] = None
        self.d_model: int = 0
        self.vocab_size: int = 0

    def attach_model(self, model: Any):
        """Attach model and extract embedding matrix."""
        self.model = model

        # Get embedding layer
        embed_layer = model.get_input_embeddings()
        self.embed_matrix = embed_layer.weight.detach()
        self.d_model = self.embed_matrix.shape[1]
        self.vocab_size = self.embed_matrix.shape[0]

    def compute_expected_embedding(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expected embedding from logits.

        Args:
            logits: Shape (batch, vocab_size) or (batch, seq, vocab_size)

        Returns:
            expected_embedding: Shape (batch, d_model) or (batch, seq, d_model)
        """
        if self.embed_matrix is None:
            raise RuntimeError("Model not attached. Call attach_model() first.")

        # Handle sequence dimension
        if logits.dim() == 3:
            # Take last token logits
            logits = logits[:, -1, :]

        # Apply temperature
        scaled_logits = logits / self.temperature

        # Top-k filtering for tractability
        if self.top_k < self.vocab_size:
            top_values, top_indices = scaled_logits.topk(self.top_k, dim=-1)

            # Softmax over top-k only
            probs = F.softmax(top_values, dim=-1)  # (batch, k)

            # Get embeddings for top-k tokens
            top_embeddings = self.embed_matrix[top_indices]  # (batch, k, d_model)

            # Compute expected embedding
            expected = torch.einsum('bk,bkd->bd', probs, top_embeddings)
        else:
            # Full vocabulary softmax
            probs = F.softmax(scaled_logits, dim=-1)  # (batch, vocab)
            expected = torch.matmul(probs, self.embed_matrix)  # (batch, d_model)

        return expected.to(self.dtype)

    def encode_from_text(
        self,
        text: str,
        tokenizer: Any,
    ) -> torch.Tensor:
        """
        Generate expected embedding from text input.

        This runs the model on the text and extracts the expected
        embedding at the last position.
        """
        if self.model is None:
            raise RuntimeError("Model not attached")

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (batch, seq, vocab)

        # Compute expected embedding for last position
        expected_emb = self.compute_expected_embedding(logits)

        return expected_emb

    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """
        Encode content into CIPHER message.

        Content can be:
        - torch.Tensor (logits or pre-computed embedding)
        - str (text to process through model)
        - dict with 'embedding' key
        """
        if content is None:
            # Empty message
            embedding = torch.zeros(1, self.d_model, dtype=self.dtype)
        elif isinstance(content, torch.Tensor):
            if content.shape[-1] == self.d_model:
                # Already an embedding
                embedding = content
            else:
                # Assume logits
                embedding = self.compute_expected_embedding(content)
        elif isinstance(content, dict) and 'embedding' in content:
            embedding = content['embedding']
        else:
            # Placeholder - in real use, would process through model
            embedding = torch.zeros(1, self.d_model, dtype=self.dtype)

        # Calculate bits
        bits_per_element = 16 if self.dtype == torch.float16 else 32
        total_bits = self.d_model * bits_per_element * self.n_soft_tokens

        return Message(
            content={'embedding': embedding, 'shape': embedding.shape},
            format="embedding",
            sender=sender,
            receiver="",
            turn_idx=turn_idx,
            bits=total_bits,
            bytes=total_bits // 8,
            metadata={
                'top_k': self.top_k,
                'temperature': self.temperature,
                'dtype': str(self.dtype),
            },
        )

    def decode(self, message: Message) -> Dict[str, Any]:
        """
        Decode CIPHER message for receiver.

        Returns the embedding and metadata needed for injection.
        """
        if message.content is None:
            return {'embedding': None}

        return {
            'embedding': message.content.get('embedding'),
            'inject_as': 'prefix',  # Default injection method
            'n_tokens': self.n_soft_tokens,
        }

    def inject_embedding(
        self,
        receiver_input_ids: torch.Tensor,
        message_embedding: torch.Tensor,
        model: Any,
    ) -> torch.Tensor:
        """
        Inject expected embedding as prefix to receiver's input.

        Args:
            receiver_input_ids: Token IDs for receiver input
            message_embedding: Expected embedding to inject (batch, d_model)
            model: Model to get embeddings from

        Returns:
            Combined input embeddings with message prepended
        """
        # Get receiver's input embeddings
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(receiver_input_ids)  # (batch, seq, d_model)

        # Ensure message embedding has sequence dimension
        if message_embedding.dim() == 2:
            message_embedding = message_embedding.unsqueeze(1)  # (batch, 1, d_model)

        # Prepend message embedding
        combined = torch.cat([message_embedding, input_embeds], dim=1)

        return combined

    def bits_per_message(self) -> int:
        """Get bits per message for this configuration."""
        bits_per_element = 16 if self.dtype == torch.float16 else 32
        return self.d_model * bits_per_element * self.n_soft_tokens


class MultiTokenCIPHER(CIPHERChannel):
    """
    Extended CIPHER that generates multiple soft tokens.

    Instead of a single expected embedding, generates a sequence
    of n soft tokens for richer communication.
    """

    def __init__(
        self,
        model: Any = None,
        top_k: int = 100,
        temperature: float = 1.0,
        n_soft_tokens: int = 4,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(
            model=model,
            top_k=top_k,
            temperature=temperature,
            n_soft_tokens=n_soft_tokens,
            dtype=dtype,
        )

    def compute_multi_token_embedding(
        self,
        input_ids: torch.Tensor,
        n_tokens: int,
    ) -> torch.Tensor:
        """
        Generate multiple expected embeddings autoregressively.

        Args:
            input_ids: Starting input tokens
            n_tokens: Number of soft tokens to generate

        Returns:
            Tensor of shape (batch, n_tokens, d_model)
        """
        if self.model is None or self.embed_matrix is None:
            raise RuntimeError("Model not attached")

        embeddings = []
        current_ids = input_ids

        with torch.no_grad():
            for _ in range(n_tokens):
                # Forward pass
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]  # Last position

                # Compute expected embedding
                expected = self.compute_expected_embedding(logits)
                embeddings.append(expected)

                # For next iteration, sample a token (for context)
                # But we transmit the expected embedding, not the token
                probs = F.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                current_ids = torch.cat([current_ids, next_token], dim=1)

        return torch.stack(embeddings, dim=1)  # (batch, n_tokens, d_model)


def create_cipher_channel(
    model: Any = None,
    top_k: int = 100,
    temperature: float = 1.0,
    n_soft_tokens: int = 1,
) -> CIPHERChannel:
    """Factory function to create CIPHER channel."""
    if n_soft_tokens > 1:
        channel = MultiTokenCIPHER(
            model=model,
            top_k=top_k,
            temperature=temperature,
            n_soft_tokens=n_soft_tokens,
        )
    else:
        channel = CIPHERChannel(
            model=model,
            top_k=top_k,
            temperature=temperature,
            n_soft_tokens=n_soft_tokens,
        )

    if model is not None:
        channel.attach_model(model)

    return channel
