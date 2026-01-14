"""
LLM-based agent for LPCA experiments.

Wraps a language model to provide the agent interface, handling
prompt formatting, generation, and response parsing.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from lpca.agents.base import BaseAgent, AgentResponse
from lpca.agents.model_wrapper import ModelWrapper


class LLMAgent(BaseAgent):
    """
    LLM-based agent using HuggingFace transformers.

    Handles:
    - Prompt construction with observation and messages
    - Generation with configurable parameters
    - Response parsing to extract messages and actions
    - Activation capture for latent protocols
    """

    def __init__(
        self,
        agent_id: str,
        model_wrapper: ModelWrapper,
        tokenizer: Any,
        role: str = "general",
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        super().__init__(agent_id, role)
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.default_system_prompt = system_prompt or self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the agent."""
        return f"""You are Agent {self.agent_id}. You have PARTIAL information about a constraint satisfaction problem. Your partner has the REST.

IMPORTANT: You CANNOT solve this alone. You MUST share info first.

Response format:
- First turn: Share your constraints with MESSAGE: <your constraints>
- After receiving partner's info: Solve and respond with ANSWER: followed by a JSON object

Example first turn:
MESSAGE: My constraints are x1+x2>=1 and x3=0

Example final answer (only after you have ALL constraints):
ANSWER: the solution as a JSON object with variable assignments

ALWAYS share your constraints before attempting to answer. Only submit ANSWER when you have enough information to solve."""

    def format_prompt(
        self,
        observation: str,
        received_message: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format the full prompt for the model."""
        sys_prompt = system_prompt or self.default_system_prompt

        parts = [f"System: {sys_prompt}\n"]

        # Add conversation history
        for msg in self.conversation_history:
            parts.append(f"{msg['role'].capitalize()}: {msg['content']}\n")

        # Add current observation
        parts.append(f"Your observation:\n{observation}\n")

        # Add received message if any
        if received_message:
            parts.append(f"Message from partner:\n{received_message}\n")

        parts.append("Your response:")

        return "\n".join(parts)

    def format_chat_prompt(
        self,
        observation: str,
        received_message: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Format as chat messages for chat models."""
        sys_prompt = system_prompt or self.default_system_prompt

        messages = [{"role": "system", "content": sys_prompt}]

        # Add history
        for msg in self.conversation_history:
            messages.append(msg)

        # Build user message
        user_content = f"Your observation:\n{observation}"
        if received_message:
            user_content += f"\n\nMessage from partner:\n{received_message}"

        messages.append({"role": "user", "content": user_content})

        return messages

    def respond(
        self,
        observation: str,
        received_message: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """
        Generate a response given observation and optional message.
        """
        start_time = time.perf_counter()

        # Format prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = self.format_chat_prompt(observation, received_message, system_prompt)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = self.format_prompt(observation, received_message, system_prompt)

        # Generate
        generated_text, metadata = self.model_wrapper.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parse response
        message_to_partner = self._extract_message(generated_text)
        final_answer = self._extract_final_answer(generated_text)
        is_done = final_answer is not None

        # Update history
        self.add_to_history("assistant", generated_text)

        return AgentResponse(
            text=generated_text,
            message_to_partner=message_to_partner,
            action=None,
            final_answer=final_answer,
            is_done=is_done,
            input_tokens=metadata.get('input_tokens', 0),
            output_tokens=metadata.get('output_tokens', 0),
            inference_ms=elapsed_ms,
            activations=metadata.get('captured_activations'),
        )

    def respond_with_activation_capture(
        self,
        observation: str,
        received_message: Optional[str] = None,
        system_prompt: Optional[str] = None,
        capture_layer: Optional[int] = None,
    ) -> Tuple[AgentResponse, Optional[torch.Tensor]]:
        """
        Generate response and capture activation at specified layer.

        Used for activation grafting protocol.
        """
        from lpca.agents.model_wrapper import ActivationHook

        # Set up hook if layer specified
        if capture_layer is not None:
            hook = ActivationHook(layer_idx=capture_layer, capture=True)
            self.model_wrapper.register_hook(hook)

        try:
            response = self.respond(observation, received_message, system_prompt)

            activation = None
            if capture_layer is not None and response.activations:
                activation = response.activations.get(capture_layer)

        finally:
            self.model_wrapper.clear_hooks()

        return response, activation

    def _extract_message(self, text: str) -> Optional[str]:
        """Extract message to partner from response."""
        patterns = [
            r"MESSAGE:\s*(.+?)(?=ANSWER:|$)",
            r"MESSAGE TO PARTNER:\s*(.+?)(?=FINAL ANSWER:|ANSWER:|$)",
            r"Message:\s*(.+?)(?=Answer:|$)",
            r"\[To partner\]:\s*(.+?)(?=\[|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no explicit message marker and no answer, return the response as message
        if "ANSWER:" not in text.upper() and "FINAL ANSWER" not in text.upper():
            # Clean up the text - remove any partial tags
            cleaned = text.strip()
            if cleaned:
                return cleaned

        return None

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from response.

        Strategy:
        1. Look for explicit ANSWER: prefix (preferred)
        2. Fallback: find last valid JSON object in response (if ANSWER keyword present)

        Does NOT extract JSON from examples or explanations.
        """
        import json as json_module

        # Must have explicit ANSWER: prefix (not in examples)
        # Look for ANSWER: at start of line or after newline
        answer_match = re.search(r"(?:^|\n)\s*ANSWER:\s*(\{[^}]+\})", text, re.IGNORECASE)
        if answer_match:
            candidate = answer_match.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate

        # Also try FINAL ANSWER:
        final_match = re.search(r"(?:^|\n)\s*FINAL ANSWER:\s*(\{[^}]+\})", text, re.IGNORECASE)
        if final_match:
            candidate = final_match.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate

        # Fallback: if "ANSWER:" or "ANSWER is:" appears (but not "the answer is")
        # This catches "the ANSWER is:" but not "the answer is"
        if re.search(r"(?<![tT]he\s)\bANSWER\s*(?:is\s*)?[:\-]", text, re.IGNORECASE):
            # Find all JSON-like objects in the text
            json_candidates = re.findall(r'\{[^{}]+\}', text)
            if json_candidates:
                # Take the last one (most likely the actual answer)
                for candidate in reversed(json_candidates):
                    # Skip if it looks like an example (contains "example" nearby)
                    idx = text.rfind(candidate)
                    context = text[max(0, idx-50):idx].lower()
                    if "example" in context:
                        continue
                    if self._is_valid_json(candidate):
                        return candidate

        return None

    def _is_valid_json(self, text: str) -> bool:
        """Check if text is valid JSON with variable assignments."""
        import json as json_module
        try:
            obj = json_module.loads(text)
            # Must be a dict with at least one key
            if isinstance(obj, dict) and len(obj) > 0:
                # Keys should look like variable names (x1, x2, etc.)
                return any(re.match(r'^[a-zA-Z_]\w*$', k) for k in obj.keys())
            return False
        except (json_module.JSONDecodeError, ValueError):
            return False


class DualAgentRunner:
    """
    Runner for two-agent coordination tasks.

    Manages turn-taking between agents A and B with different
    communication protocols.
    """

    def __init__(
        self,
        agent_A: LLMAgent,
        agent_B: LLMAgent,
        channel: Any,
        max_turns: int = 10,
    ):
        self.agent_A = agent_A
        self.agent_B = agent_B
        self.channel = channel
        self.max_turns = max_turns

    def reset(self):
        """Reset agents and channel for new episode."""
        self.agent_A.reset()
        self.agent_B.reset()
        self.channel.reset()

    def run_episode(
        self,
        obs_A: str,
        obs_B: str,
    ) -> Dict[str, Any]:
        """
        Run a full episode of agent coordination.

        Returns episode results including final answer and metrics.
        """
        self.reset()

        turns = []
        final_answer = None
        current_message = None

        for turn_idx in range(self.max_turns):
            # Agent A's turn
            response_A = self.agent_A.respond(
                observation=obs_A,
                received_message=current_message,
            )

            turns.append({
                'turn': turn_idx,
                'agent': 'A',
                'response': response_A.text,
                'message': response_A.message_to_partner,
                'input_tokens': response_A.input_tokens,
                'output_tokens': response_A.output_tokens,
            })

            if response_A.is_done:
                final_answer = response_A.final_answer
                break

            # Send message through channel
            if response_A.message_to_partner:
                msg = self.channel.send(
                    response_A.message_to_partner,
                    sender='A',
                    receiver='B',
                    turn_idx=turn_idx,
                )
                current_message = self.channel.receive(msg)
            else:
                current_message = None

            # Agent B's turn
            response_B = self.agent_B.respond(
                observation=obs_B,
                received_message=current_message,
            )

            turns.append({
                'turn': turn_idx,
                'agent': 'B',
                'response': response_B.text,
                'message': response_B.message_to_partner,
                'input_tokens': response_B.input_tokens,
                'output_tokens': response_B.output_tokens,
            })

            if response_B.is_done:
                final_answer = response_B.final_answer
                break

            # Send message through channel
            if response_B.message_to_partner:
                msg = self.channel.send(
                    response_B.message_to_partner,
                    sender='B',
                    receiver='A',
                    turn_idx=turn_idx,
                )
                current_message = self.channel.receive(msg)
            else:
                current_message = None

        return {
            'turns': turns,
            'n_turns': len(turns),
            'final_answer': final_answer,
            'channel_stats': self.channel.get_stats(),
        }


def load_llm_agent(
    agent_id: str,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "auto",
    **kwargs
) -> Tuple[LLMAgent, ModelWrapper, Any]:
    """
    Load an LLM agent with model.

    Args:
        agent_id: Unique identifier for the agent
        model_name: HuggingFace model name (default: TinyLlama, open model)
        device: Device to use ('auto', 'mps', 'cuda', 'cpu')

    Returns:
        Tuple of (agent, model_wrapper, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Determine actual device
    if device == "auto":
        if torch.cuda.is_available():
            actual_device = "cuda"
        elif torch.backends.mps.is_available():
            actual_device = "mps"
        else:
            actual_device = "cpu"
    else:
        actual_device = device

    # Determine dtype - MPS works better with float32
    dtype = torch.float32 if actual_device == "mps" else torch.float16

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None if actual_device == "mps" else "auto",
        trust_remote_code=True,
    )

    if actual_device == "mps":
        model = model.to("mps")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create wrapper and agent
    wrapper = ModelWrapper(model, tokenizer, actual_device)
    agent = LLMAgent(agent_id, wrapper, tokenizer, **kwargs)

    return agent, wrapper, tokenizer
