"""
Base classes for LPCA task environments.

All environments implement a common interface for task generation,
agent interaction, and verification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random
import hashlib


@dataclass
class TaskInstance:
    """A single task instance with split observations."""
    task_id: str
    task_type: str
    seed: int

    # Split observations
    obs_A: str  # Agent A's observation
    obs_B: str  # Agent B's observation

    # Ground truth (hidden from agents)
    ground_truth: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            # Generate deterministic ID from content
            content = f"{self.task_type}_{self.seed}_{self.obs_A}_{self.obs_B}"
            self.task_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class VerifierResult:
    """Result from task verification."""
    success: bool
    partial_credit: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class StepResult:
    """Result from an environment step."""
    done: bool
    verifier_result: Optional[VerifierResult] = None
    feedback_A: str = ""
    feedback_B: str = ""


class BaseEnvironment(ABC):
    """
    Abstract base class for LPCA task environments.

    Environments generate tasks where two agents must coordinate
    to succeed, with each agent having partial observations.
    """

    def __init__(
        self,
        task_type: str,
        difficulty: str = "medium",
        params: Optional[Dict[str, Any]] = None,
    ):
        self.task_type = task_type
        self.difficulty = difficulty
        self.params = params or {}
        self.current_task: Optional[TaskInstance] = None
        self._rng = random.Random()

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = random.Random(seed)

    def get_deterministic_seed(self, base_seed: int, episode_idx: int) -> int:
        """Generate deterministic seed from base seed and episode index."""
        content = f"{base_seed}_{episode_idx}_{self.task_type}"
        return int(hashlib.md5(content.encode()).hexdigest()[:8], 16)

    @abstractmethod
    def generate_task(self, seed: int) -> TaskInstance:
        """
        Generate a new task instance.

        Args:
            seed: Random seed for reproducibility

        Returns:
            TaskInstance with split observations for agents A and B
        """
        pass

    @abstractmethod
    def verify(self, output: Any) -> VerifierResult:
        """
        Verify agent output against ground truth.

        Args:
            output: The combined output from agents

        Returns:
            VerifierResult with success status and partial credit
        """
        pass

    def reset(self, seed: int) -> TaskInstance:
        """
        Reset environment with a new task.

        Args:
            seed: Random seed for task generation

        Returns:
            New TaskInstance
        """
        self.set_seed(seed)
        self.current_task = self.generate_task(seed)
        return self.current_task

    def step(
        self,
        action_A: Optional[str],
        action_B: Optional[str],
        final_output: Optional[Any] = None,
    ) -> StepResult:
        """
        Process agent actions and check for completion.

        Args:
            action_A: Action from agent A (if any)
            action_B: Action from agent B (if any)
            final_output: Final answer to verify (if submitting)

        Returns:
            StepResult with done status and verification result
        """
        if final_output is not None:
            result = self.verify(final_output)
            return StepResult(done=True, verifier_result=result)

        return StepResult(done=False)

    def get_task_prompt(self, agent: str) -> str:
        """
        Get the task prompt for an agent.

        Args:
            agent: 'A' or 'B'

        Returns:
            Formatted prompt string
        """
        if self.current_task is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        obs = self.current_task.obs_A if agent == "A" else self.current_task.obs_B
        return self._format_prompt(obs, agent)

    def _format_prompt(self, observation: str, agent: str) -> str:
        """Format observation into agent prompt."""
        return f"""You are Agent {agent} in a collaborative task.

Your observation:
{observation}

Work with your partner to solve this task. Share relevant information
and coordinate to find the solution."""

    def get_difficulty_params(self) -> Dict[str, Any]:
        """Get parameters for current difficulty level."""
        difficulty_presets = {
            "easy": self._get_easy_params(),
            "medium": self._get_medium_params(),
            "hard": self._get_hard_params(),
        }
        return difficulty_presets.get(self.difficulty, self._get_medium_params())

    def _get_easy_params(self) -> Dict[str, Any]:
        """Override in subclass for easy difficulty settings."""
        return {}

    def _get_medium_params(self) -> Dict[str, Any]:
        """Override in subclass for medium difficulty settings."""
        return {}

    def _get_hard_params(self) -> Dict[str, Any]:
        """Override in subclass for hard difficulty settings."""
        return {}


class CompositeEnvironment:
    """
    Wrapper to combine multiple task types into a single environment.

    Useful for running diverse evaluation suites.
    """

    def __init__(self, environments: List[BaseEnvironment]):
        self.environments = environments
        self.current_env: Optional[BaseEnvironment] = None

    def select_environment(self, task_type: str) -> BaseEnvironment:
        """Select environment by task type."""
        for env in self.environments:
            if env.task_type == task_type:
                self.current_env = env
                return env
        raise ValueError(f"Unknown task type: {task_type}")

    def reset(self, task_type: str, seed: int) -> TaskInstance:
        """Reset with specific task type."""
        env = self.select_environment(task_type)
        return env.reset(seed)

    def verify(self, output: Any) -> VerifierResult:
        """Verify using current environment."""
        if self.current_env is None:
            raise RuntimeError("No environment selected")
        return self.current_env.verify(output)
