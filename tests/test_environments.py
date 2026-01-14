"""Tests for task environments."""

import pytest
from lpca.envs.split_synthetic import (
    ConstraintSatisfactionTask,
    ArithmeticTask,
    ProgramSynthesisTask,
    SplitSyntheticEnv,
)
from lpca.envs.base import TaskInstance, VerifierResult


class TestConstraintSatisfactionTask:
    """Tests for S1: Constraint Satisfaction."""

    def test_create(self):
        task = ConstraintSatisfactionTask(difficulty="easy")
        assert task.task_type == "constraint_satisfaction"

    def test_generate_task_returns_instance(self):
        task = ConstraintSatisfactionTask(difficulty="easy")
        instance = task.generate_task(seed=42)
        assert isinstance(instance, TaskInstance)
        assert instance.obs_A != ""
        assert instance.obs_B != ""

    def test_verify_correct_solution(self):
        task = ConstraintSatisfactionTask(difficulty="easy")
        instance = task.reset(seed=42)
        result = task.verify(instance.ground_truth)
        assert result.success
        assert result.partial_credit >= 0.9

    def test_verify_incorrect_solution(self):
        task = ConstraintSatisfactionTask(difficulty="easy")
        task.reset(seed=42)
        result = task.verify({"x1": 99, "x2": 99, "x3": 99})
        assert not result.success

    def test_deterministic_generation(self):
        task = ConstraintSatisfactionTask(difficulty="medium")
        instance1 = task.generate_task(seed=123)
        instance2 = task.generate_task(seed=123)
        assert instance1.obs_A == instance2.obs_A
        assert instance1.obs_B == instance2.obs_B
        assert instance1.ground_truth == instance2.ground_truth

    def test_different_seeds_different_tasks(self):
        task = ConstraintSatisfactionTask(difficulty="medium")
        instance1 = task.generate_task(seed=1)
        instance2 = task.generate_task(seed=2)
        assert instance1.ground_truth != instance2.ground_truth


class TestArithmeticTask:
    """Tests for S2: Arithmetic with Missing Operands."""

    def test_create(self):
        task = ArithmeticTask(difficulty="easy")
        assert task.task_type == "arithmetic"

    def test_generate_task(self):
        task = ArithmeticTask(difficulty="easy")
        instance = task.generate_task(seed=42)
        assert isinstance(instance, TaskInstance)
        # Check observations contain variable assignments
        assert "=" in instance.obs_A
        assert "=" in instance.obs_B

    def test_verify_correct_answer(self):
        task = ArithmeticTask(difficulty="easy")
        instance = task.reset(seed=42)
        result = task.verify(str(instance.ground_truth))
        assert result.success

    def test_verify_incorrect_answer(self):
        task = ArithmeticTask(difficulty="easy")
        task.reset(seed=42)
        result = task.verify("999999")
        assert not result.success


class TestProgramSynthesisTask:
    """Tests for S3: Program Synthesis (Toy)."""

    def test_create(self):
        task = ProgramSynthesisTask(difficulty="easy")
        assert task.task_type == "program_synthesis"

    def test_generate_task(self):
        task = ProgramSynthesisTask(difficulty="easy")
        instance = task.generate_task(seed=42)
        assert isinstance(instance, TaskInstance)
        # Check observations contain input-output examples
        assert "->" in instance.obs_A or "=>" in instance.obs_A

    def test_verify_with_function_string(self):
        task = ProgramSynthesisTask(difficulty="easy")
        task.reset(seed=42)
        # Try a simple function that might match
        result = task.verify("def f(x): return x")
        # Result depends on the generated task
        assert isinstance(result, VerifierResult)


class TestSplitSyntheticEnv:
    """Tests for composite environment."""

    def test_create(self):
        env = SplitSyntheticEnv(difficulty="medium")
        assert len(env.environments) == 3

    def test_get_task_types(self):
        env = SplitSyntheticEnv()
        types = env.get_task_types()
        assert "constraint_satisfaction" in types
        assert "arithmetic" in types
        assert "program_synthesis" in types

    def test_select_environment(self):
        env = SplitSyntheticEnv()
        selected = env.select_environment("arithmetic")
        assert selected.task_type == "arithmetic"

    def test_select_unknown_raises(self):
        env = SplitSyntheticEnv()
        with pytest.raises(ValueError):
            env.select_environment("unknown_task")

    def test_reset_after_select(self):
        env = SplitSyntheticEnv()
        env.select_environment("constraint_satisfaction")
        instance = env.reset(seed=42)
        assert isinstance(instance, TaskInstance)

    def test_reset_without_select_raises(self):
        env = SplitSyntheticEnv()
        with pytest.raises(RuntimeError):
            env.reset(seed=42)

    def test_verify_uses_current_env(self):
        env = SplitSyntheticEnv()
        env.select_environment("arithmetic")
        instance = env.reset(seed=42)
        result = env.verify(str(instance.ground_truth))
        assert result.success


class TestTaskDifficulty:
    """Tests for difficulty scaling."""

    def test_easy_vs_hard_constraint_size(self):
        easy = ConstraintSatisfactionTask(difficulty="easy")
        hard = ConstraintSatisfactionTask(difficulty="hard")

        easy_params = easy.get_difficulty_params()
        hard_params = hard.get_difficulty_params()

        # Hard should have more variables/constraints
        assert hard_params.get("n_vars", 3) >= easy_params.get("n_vars", 3)

    def test_easy_vs_hard_arithmetic(self):
        easy = ArithmeticTask(difficulty="easy")
        hard = ArithmeticTask(difficulty="hard")

        easy_params = easy.get_difficulty_params()
        hard_params = hard.get_difficulty_params()

        # Hard should have more terms
        assert hard_params.get("n_terms", 2) >= easy_params.get("n_terms", 2)
