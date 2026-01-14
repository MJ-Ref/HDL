"""Tests for LPCA environments."""

import pytest
from lpca.envs.split_synthetic import (
    ConstraintSatisfactionTask,
    ArithmeticTask,
    ProgramSynthesisTask,
    SplitSyntheticEnv,
)


class TestConstraintSatisfactionTask:
    """Tests for constraint satisfaction environment."""

    def test_task_generation(self):
        """Test that tasks are generated correctly."""
        env = ConstraintSatisfactionTask(difficulty="easy")
        task = env.reset(seed=42)

        assert task.obs_A is not None
        assert task.obs_B is not None
        assert task.ground_truth is not None
        assert "CONSTRAINT" in task.obs_A

    def test_verification_correct(self):
        """Test verification with correct answer."""
        env = ConstraintSatisfactionTask(difficulty="easy")
        task = env.reset(seed=42)

        # Submit ground truth
        result = env.verify(task.ground_truth)
        assert result.success is True
        assert result.partial_credit == 1.0

    def test_verification_incorrect(self):
        """Test verification with incorrect answer."""
        env = ConstraintSatisfactionTask(difficulty="easy")
        task = env.reset(seed=42)

        # Submit wrong answer
        wrong_answer = {k: (v + 1) % 2 for k, v in task.ground_truth.items()}
        result = env.verify(wrong_answer)

        # Should not be fully successful
        assert result.partial_credit < 1.0

    def test_deterministic_generation(self):
        """Test that same seed produces same task."""
        env1 = ConstraintSatisfactionTask(difficulty="medium")
        env2 = ConstraintSatisfactionTask(difficulty="medium")

        task1 = env1.reset(seed=123)
        task2 = env2.reset(seed=123)

        assert task1.obs_A == task2.obs_A
        assert task1.obs_B == task2.obs_B
        assert task1.ground_truth == task2.ground_truth


class TestArithmeticTask:
    """Tests for arithmetic environment."""

    def test_task_generation(self):
        """Test arithmetic task generation."""
        env = ArithmeticTask(difficulty="easy")
        task = env.reset(seed=42)

        assert task.obs_A is not None
        assert task.obs_B is not None
        assert isinstance(task.ground_truth, int)

    def test_verification(self):
        """Test arithmetic verification."""
        env = ArithmeticTask(difficulty="easy")
        task = env.reset(seed=42)

        # Correct answer
        result = env.verify(task.ground_truth)
        assert result.success is True

        # Wrong answer
        result = env.verify(task.ground_truth + 100)
        assert result.success is False


class TestProgramSynthesisTask:
    """Tests for program synthesis environment."""

    def test_task_generation(self):
        """Test program synthesis task generation."""
        env = ProgramSynthesisTask(difficulty="easy")
        task = env.reset(seed=42)

        assert "PROGRAM SYNTHESIS" in task.obs_A
        assert task.ground_truth is not None

    def test_verification_with_code(self):
        """Test verification with code submission."""
        env = ProgramSynthesisTask(difficulty="easy")
        task = env.reset(seed=42)

        # Get the implementation from ground truth
        impl = task.ground_truth["impl"]
        code = f"def f(x): {impl}"

        result = env.verify(code)
        # Should pass most or all tests
        assert result.partial_credit > 0


class TestSplitSyntheticEnv:
    """Tests for composite synthetic environment."""

    def test_task_types(self):
        """Test that all task types are available."""
        env = SplitSyntheticEnv()
        types = env.get_task_types()

        assert "constraint_satisfaction" in types
        assert "arithmetic" in types
        assert "program_synthesis" in types

    def test_select_environment(self):
        """Test environment selection."""
        env = SplitSyntheticEnv()

        env.select_environment("constraint_satisfaction")
        task = env.reset(seed=42)
        assert task.task_type == "constraint_satisfaction"

        env.select_environment("arithmetic")
        task = env.reset(seed=42)
        assert task.task_type == "arithmetic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
