"""
Text-based communication channels (P0-P5).

Implements the text baseline protocols from BASELINES.md:
- P0: No Communication
- P1: Full Text
- P2: Budgeted Text
- P3: Text + Summarization
- P4: Text + Retrieval
- P5: Structured Workspace
"""

import json
from typing import Any, Dict, List, Optional
import numpy as np

from lpca.channels.base import BaseChannel, Message


class NoCommChannel(BaseChannel):
    """
    P0: No Communication

    Agents cannot communicate. Lower bound baseline.
    """

    def __init__(self):
        super().__init__(
            name="P0",
            channel_type="none",
            max_bits_per_message=0,
        )

    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """No encoding - returns empty message."""
        return Message(
            content=None,
            format="none",
            sender=sender,
            receiver="",
            turn_idx=turn_idx,
            bits=0,
            bytes=0,
        )

    def decode(self, message: Message) -> Any:
        """No decoding - returns empty string."""
        return ""


class FullTextChannel(BaseChannel):
    """
    P1: Full Text Communication

    Unconstrained natural language messages. Upper text reference.
    """

    def __init__(self, max_tokens: int = 2048):
        super().__init__(
            name="P1",
            channel_type="text",
        )
        self.max_tokens = max_tokens

    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """Encode as text message."""
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        # Rough truncation by tokens (~4 chars per token)
        max_chars = self.max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]

        byte_count = len(text.encode("utf-8"))

        return Message(
            content=text,
            format="text",
            sender=sender,
            receiver="",
            turn_idx=turn_idx,
            bits=byte_count * 8,
            bytes=byte_count,
        )

    def decode(self, message: Message) -> str:
        """Decode text message."""
        return message.content if message.content else ""


class BudgetedTextChannel(BaseChannel):
    """
    P2: Budgeted Text Communication

    Text communication under strict byte/bit constraints.
    """

    def __init__(self, max_bytes: int = 256):
        super().__init__(
            name="P2",
            channel_type="text",
            max_bits_per_message=max_bytes * 8,
        )
        self.max_bytes = max_bytes

    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """Encode and truncate to budget."""
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        # Truncate to byte budget
        text = self._truncate_to_bytes(text, self.max_bytes)
        byte_count = len(text.encode("utf-8"))

        return Message(
            content=text,
            format="text",
            sender=sender,
            receiver="",
            turn_idx=turn_idx,
            bits=byte_count * 8,
            bytes=byte_count,
        )

    def decode(self, message: Message) -> str:
        """Decode text message."""
        return message.content if message.content else ""

    def _truncate_to_bytes(self, text: str, max_bytes: int) -> str:
        """Truncate text to fit within byte budget."""
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text

        # Truncate at character boundary
        while len(text.encode("utf-8")) > max_bytes:
            text = text[:-1]

        return text


class SummarizationChannel(BaseChannel):
    """
    P3: Text + Summarization

    Uses LLM-based compression to maximize information density.
    """

    def __init__(
        self,
        max_bytes: int = 256,
        summarizer: Optional[Any] = None,
    ):
        super().__init__(
            name="P3",
            channel_type="text",
            max_bits_per_message=max_bytes * 8,
        )
        self.max_bytes = max_bytes
        self.summarizer = summarizer
        self._summarization_prompt = """Summarize the following message in at most {max_bytes} bytes.
Preserve all critical information needed for task completion.

Original message:
{message}

Compressed message:"""

    def set_summarizer(self, summarizer: Any):
        """Set the summarization model."""
        self.summarizer = summarizer

    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """Encode with summarization."""
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        # Summarize if over budget and summarizer available
        if len(text.encode("utf-8")) > self.max_bytes:
            if self.summarizer is not None:
                text = self._summarize(text)
            else:
                text = self._truncate_to_bytes(text, self.max_bytes)

        # Final truncation safety
        text = self._truncate_to_bytes(text, self.max_bytes)
        byte_count = len(text.encode("utf-8"))

        return Message(
            content=text,
            format="text",
            sender=sender,
            receiver="",
            turn_idx=turn_idx,
            bits=byte_count * 8,
            bytes=byte_count,
            metadata={"summarized": True},
        )

    def decode(self, message: Message) -> str:
        """Decode text message."""
        return message.content if message.content else ""

    def _summarize(self, text: str) -> str:
        """Summarize text using the model."""
        prompt = self._summarization_prompt.format(
            max_bytes=self.max_bytes,
            message=text,
        )

        try:
            # Assume summarizer has generate() method
            response = self.summarizer.generate(prompt, max_tokens=self.max_bytes // 2)
            return response.strip()
        except Exception:
            return text

    def _truncate_to_bytes(self, text: str, max_bytes: int) -> str:
        """Truncate text to byte budget."""
        while len(text.encode("utf-8")) > max_bytes:
            text = text[:-1]
        return text


class RetrievalChannel(BaseChannel):
    """
    P4: Text + Retrieval Memory

    Episodic memory with retrieval for efficient long-context coordination.
    """

    def __init__(
        self,
        max_bytes: int = 256,
        memory_size: int = 100,
        retrieval_k: int = 3,
    ):
        super().__init__(
            name="P4",
            channel_type="text",
            max_bits_per_message=max_bytes * 8,
        )
        self.max_bytes = max_bytes
        self.memory_size = memory_size
        self.retrieval_k = retrieval_k

        # Simple memory store
        self.memory: List[Dict[str, Any]] = []

    def reset(self):
        """Reset channel and memory."""
        super().reset()
        self.memory = []

    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """Encode and store in memory."""
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        # Store in memory
        self._add_to_memory(text, sender, turn_idx)

        # Truncate for transmission
        text = self._truncate_to_bytes(text, self.max_bytes)
        byte_count = len(text.encode("utf-8"))

        return Message(
            content=text,
            format="text",
            sender=sender,
            receiver="",
            turn_idx=turn_idx,
            bits=byte_count * 8,
            bytes=byte_count,
        )

    def decode(self, message: Message) -> str:
        """Decode with retrieval augmentation."""
        current = message.content if message.content else ""

        # Store received message
        self._add_to_memory(current, message.sender, message.turn_idx)

        # Retrieve relevant history
        retrieved = self._retrieve(current)

        if retrieved:
            augmented = f"{current}\n\n[Relevant history:\n"
            augmented += "\n".join(f"- {r}" for r in retrieved)
            augmented += "]"
            return augmented

        return current

    def _add_to_memory(self, content: str, sender: str, turn_idx: int):
        """Add content to memory."""
        if not content:
            return

        entry = {
            "content": content,
            "sender": sender,
            "turn_idx": turn_idx,
            "embedding": self._simple_embedding(content),
        }

        self.memory.append(entry)

        # Trim if over limit
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

    def _retrieve(self, query: str) -> List[str]:
        """Retrieve most relevant entries."""
        if not self.memory or not query:
            return []

        query_emb = self._simple_embedding(query)

        # Compute similarities
        scores = []
        for entry in self.memory:
            sim = self._cosine_similarity(query_emb, entry["embedding"])
            scores.append((sim, entry["content"]))

        # Get top-k (excluding exact match if present)
        scores.sort(reverse=True)
        results = []
        for sim, content in scores:
            if content != query and len(results) < self.retrieval_k:
                results.append(content[:200])  # Truncate for display

        return results

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple bag-of-words embedding (placeholder for real embeddings)."""
        # Use character n-grams as simple features
        text = text.lower()
        ngrams = set()
        for i in range(len(text) - 2):
            ngrams.add(text[i:i+3])

        # Create fixed-size vector via hashing
        vec = np.zeros(256)
        for ng in ngrams:
            idx = hash(ng) % 256
            vec[idx] += 1

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b))

    def _truncate_to_bytes(self, text: str, max_bytes: int) -> str:
        """Truncate text to byte budget."""
        while len(text.encode("utf-8")) > max_bytes:
            text = text[:-1]
        return text


class StructuredChannel(BaseChannel):
    """
    P5: Structured Workspace

    Non-NL structured state for coordination via JSON schema.
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        max_bytes: Optional[int] = None,
    ):
        super().__init__(
            name="P5",
            channel_type="structured",
            max_bits_per_message=max_bytes * 8 if max_bytes else None,
        )
        self.schema = schema or {}
        self.max_bytes = max_bytes
        self.workspace: Dict[str, Any] = {}

    def reset(self):
        """Reset channel and workspace."""
        super().reset()
        self.workspace = {}

    def encode(self, content: Any, sender: str, turn_idx: int) -> Message:
        """Encode structured data."""
        if content is None:
            data = {}
        elif isinstance(content, dict):
            data = content
        elif isinstance(content, str):
            # Try to parse as JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {"text": content}
        else:
            data = {"value": content}

        # Validate against schema if provided
        if self.schema:
            data = self._validate_schema(data)

        # Update workspace
        self._merge_workspace(data)

        # Serialize
        json_str = json.dumps(data, separators=(",", ":"))
        byte_count = len(json_str.encode("utf-8"))

        # Truncate if needed
        if self.max_bytes and byte_count > self.max_bytes:
            data = self._truncate_data(data, self.max_bytes)
            json_str = json.dumps(data, separators=(",", ":"))
            byte_count = len(json_str.encode("utf-8"))

        return Message(
            content=data,
            format="json",
            sender=sender,
            receiver="",
            turn_idx=turn_idx,
            bits=byte_count * 8,
            bytes=byte_count,
        )

    def decode(self, message: Message) -> Dict[str, Any]:
        """Decode structured message and return workspace."""
        if message.content:
            self._merge_workspace(message.content)
        return self.workspace.copy()

    def _validate_schema(self, data: Dict) -> Dict:
        """Validate data against schema (basic validation)."""
        # For now, just filter to known keys
        if not self.schema:
            return data

        valid = {}
        for key in self.schema.get("properties", {}).keys():
            if key in data:
                valid[key] = data[key]

        return valid if valid else data

    def _merge_workspace(self, updates: Dict):
        """Merge updates into workspace."""
        for key, value in updates.items():
            if key in self.workspace and isinstance(self.workspace[key], dict):
                if isinstance(value, dict):
                    self.workspace[key].update(value)
                else:
                    self.workspace[key] = value
            else:
                self.workspace[key] = value

    def _truncate_data(self, data: Dict, max_bytes: int) -> Dict:
        """Truncate data to fit within byte budget."""
        # Simple strategy: remove keys until it fits
        json_str = json.dumps(data, separators=(",", ":"))

        while len(json_str.encode("utf-8")) > max_bytes and data:
            # Remove longest value
            if not data:
                break
            longest_key = max(data.keys(),
                            key=lambda k: len(json.dumps(data[k])))
            del data[longest_key]
            json_str = json.dumps(data, separators=(",", ":"))

        return data

    def get_workspace(self) -> Dict[str, Any]:
        """Get current workspace state."""
        return self.workspace.copy()


# Factory function for creating channels by protocol ID
def create_channel(
    protocol: str,
    **kwargs
) -> BaseChannel:
    """
    Create a channel by protocol identifier.

    Args:
        protocol: Protocol ID (P0-P5)
        **kwargs: Protocol-specific arguments

    Returns:
        Configured channel instance
    """
    channels = {
        "P0": NoCommChannel,
        "P1": FullTextChannel,
        "P2": BudgetedTextChannel,
        "P3": SummarizationChannel,
        "P4": RetrievalChannel,
        "P5": StructuredChannel,
    }

    if protocol not in channels:
        raise ValueError(f"Unknown protocol: {protocol}")

    return channels[protocol](**kwargs)
