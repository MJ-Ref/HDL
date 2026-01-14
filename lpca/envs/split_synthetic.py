"""
Synthetic split-information task environments.

Implements Task Family S from EXPERIMENTS.md:
- S1: Constraint Satisfaction
- S2: Arithmetic with Missing Operands
- S3: Program Synthesis (Toy)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
import random
import itertools

from lpca.envs.base import BaseEnvironment, TaskInstance, VerifierResult, CompositeEnvironment


class ConstraintSatisfactionTask(BaseEnvironment):
    """
    S1: Constraint Satisfaction Task

    Agent A receives half the constraints, Agent B receives the other half.
    Neither can solve alone; solution requires satisfying all constraints.
    """

    def __init__(
        self,
        difficulty: str = "medium",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("constraint_satisfaction", difficulty, params)

    def _get_easy_params(self) -> Dict[str, Any]:
        # Tightened: more variables & constraints to ensure communication is required
        # Old: n_vars=3, n_constraints=4, domain=2 gave P0=52% (too high)
        return {"n_variables": 4, "n_constraints": 8, "domain_size": 3}

    def _get_medium_params(self) -> Dict[str, Any]:
        return {"n_variables": 5, "n_constraints": 12, "domain_size": 3}

    def _get_hard_params(self) -> Dict[str, Any]:
        return {"n_variables": 6, "n_constraints": 16, "domain_size": 4}

    def generate_task(self, seed: int) -> TaskInstance:
        """Generate a constraint satisfaction problem split between agents."""
        self.set_seed(seed)
        params = {**self.get_difficulty_params(), **self.params}

        n_vars = params["n_variables"]
        n_constraints = params["n_constraints"]
        domain_size = params["domain_size"]

        # Generate variable names
        var_names = [f"x{i}" for i in range(1, n_vars + 1)]

        # Try multiple times to get a communication-requiring task
        max_attempts = 20
        for attempt in range(max_attempts):
            # Generate a satisfying assignment first
            solution = {v: self._rng.randint(0, domain_size - 1) for v in var_names}

            # Generate constraints that the solution satisfies
            constraints = self._generate_constraints(
                var_names, solution, n_constraints, domain_size
            )

            # Shuffle and split constraints
            self._rng.shuffle(constraints)
            mid = len(constraints) // 2
            constraints_A = constraints[:mid]
            constraints_B = constraints[mid:]

            # Check that neither agent alone can solve (communication required)
            solutions_A = self._count_valid_solutions(constraints_A, var_names, domain_size)
            solutions_B = self._count_valid_solutions(constraints_B, var_names, domain_size)

            # Require at least 2 solutions for each agent alone (ambiguity)
            if solutions_A >= 2 and solutions_B >= 2:
                break
            # If last attempt, use anyway (fallback)

        # Format observations
        obs_A = self._format_observation(var_names, constraints_A, domain_size, "A")
        obs_B = self._format_observation(var_names, constraints_B, domain_size, "B")

        return TaskInstance(
            task_id="",
            task_type=self.task_type,
            seed=seed,
            obs_A=obs_A,
            obs_B=obs_B,
            ground_truth=solution,
            metadata={
                "n_variables": n_vars,
                "n_constraints": n_constraints,
                "domain_size": domain_size,
                "all_constraints": constraints,
                "solutions_A_alone": solutions_A,
                "solutions_B_alone": solutions_B,
            },
        )

    def _count_valid_solutions(
        self,
        constraints: List[str],
        var_names: List[str],
        domain_size: int,
    ) -> int:
        """Count how many solutions satisfy a set of constraints."""
        count = 0
        # Enumerate all possible assignments
        for values in itertools.product(range(domain_size), repeat=len(var_names)):
            assignment = dict(zip(var_names, values))
            if all(self._check_constraint(c, assignment) for c in constraints):
                count += 1
        return count

    def _generate_constraints(
        self,
        var_names: List[str],
        solution: Dict[str, int],
        n_constraints: int,
        domain_size: int,
    ) -> List[str]:
        """Generate constraints satisfied by the solution."""
        constraints = []
        constraint_types = ["eq", "neq", "sum", "diff"]

        attempts = 0
        while len(constraints) < n_constraints and attempts < n_constraints * 10:
            attempts += 1
            ctype = self._rng.choice(constraint_types)

            if ctype == "eq":
                # x_i = value
                var = self._rng.choice(var_names)
                constraint = f"{var} = {solution[var]}"

            elif ctype == "neq":
                # x_i != value (pick a wrong value)
                var = self._rng.choice(var_names)
                wrong_values = [v for v in range(domain_size) if v != solution[var]]
                if wrong_values:
                    wrong = self._rng.choice(wrong_values)
                    constraint = f"{var} != {wrong}"
                else:
                    continue

            elif ctype == "sum":
                # x_i + x_j = value
                if len(var_names) >= 2:
                    v1, v2 = self._rng.sample(var_names, 2)
                    total = solution[v1] + solution[v2]
                    constraint = f"{v1} + {v2} = {total}"
                else:
                    continue

            elif ctype == "diff":
                # x_i - x_j = value
                if len(var_names) >= 2:
                    v1, v2 = self._rng.sample(var_names, 2)
                    diff = solution[v1] - solution[v2]
                    constraint = f"{v1} - {v2} = {diff}"
                else:
                    continue

            if constraint not in constraints:
                constraints.append(constraint)

        return constraints

    def _format_observation(
        self,
        var_names: List[str],
        constraints: List[str],
        domain_size: int,
        agent: str,
    ) -> str:
        """Format constraints into agent observation."""
        return f"""CONSTRAINT SATISFACTION PROBLEM

Variables: {', '.join(var_names)}
Domain: Each variable can be 0, 1, ..., {domain_size - 1}

Your constraints (you see {len(constraints)} of the total constraints):
{chr(10).join(f'  - {c}' for c in constraints)}

Your partner has additional constraints you cannot see.
Find values for all variables that satisfy ALL constraints (yours and theirs).

Submit your answer as: {{{', '.join(f'"{v}": <value>' for v in var_names)}}}"""

    def verify(self, output: Any) -> VerifierResult:
        """Verify the solution satisfies all constraints."""
        if self.current_task is None:
            return VerifierResult(success=False, error="No task loaded")

        solution = self.current_task.ground_truth
        all_constraints = self.current_task.metadata["all_constraints"]

        # Parse output if string
        if isinstance(output, str):
            try:
                import json
                import re
                # Try to extract JSON from string
                match = re.search(r'\{[^}]+\}', output)
                if match:
                    output = json.loads(match.group())
                else:
                    return VerifierResult(
                        success=False,
                        error="Could not parse output as JSON"
                    )
            except Exception as e:
                return VerifierResult(success=False, error=str(e))

        # Check each constraint
        satisfied = 0
        for constraint in all_constraints:
            if self._check_constraint(constraint, output):
                satisfied += 1

        partial = satisfied / len(all_constraints) if all_constraints else 0
        success = satisfied == len(all_constraints)

        return VerifierResult(
            success=success,
            partial_credit=partial,
            details={
                "satisfied": satisfied,
                "total": len(all_constraints),
                "submitted": output,
                "expected": solution,
            },
        )

    def _check_constraint(self, constraint: str, assignment: Dict) -> bool:
        """Check if assignment satisfies a constraint."""
        try:
            # Normalize variable names
            normalized = {}
            for k, v in assignment.items():
                # Handle both "x1" and "x_1" formats
                key = k.replace("_", "").lower()
                normalized[key] = int(v)

            # Parse and evaluate constraint
            if "!=" in constraint:
                var, val = constraint.split("!=")
                var = var.strip().replace("_", "").lower()
                val = int(val.strip())
                return normalized.get(var, -999) != val

            elif "+" in constraint and "=" in constraint:
                left, right = constraint.split("=")
                parts = left.split("+")
                v1 = parts[0].strip().replace("_", "").lower()
                v2 = parts[1].strip().replace("_", "").lower()
                expected = int(right.strip())
                return normalized.get(v1, 0) + normalized.get(v2, 0) == expected

            elif "-" in constraint and "=" in constraint:
                left, right = constraint.split("=")
                parts = left.split("-")
                v1 = parts[0].strip().replace("_", "").lower()
                v2 = parts[1].strip().replace("_", "").lower()
                expected = int(right.strip())
                return normalized.get(v1, 0) - normalized.get(v2, 0) == expected

            elif "=" in constraint:
                var, val = constraint.split("=")
                var = var.strip().replace("_", "").lower()
                val = int(val.strip())
                return normalized.get(var, -999) == val

        except Exception:
            pass

        return False


class ArithmeticTask(BaseEnvironment):
    """
    S2: Arithmetic with Missing Operands

    Agent A sees some operands, Agent B sees others.
    Together they must compute f(x1, x2, ..., xn).
    """

    def __init__(
        self,
        difficulty: str = "medium",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("arithmetic", difficulty, params)

    def _get_easy_params(self) -> Dict[str, Any]:
        return {"n_operands": 4, "depth": 2, "max_value": 10}

    def _get_medium_params(self) -> Dict[str, Any]:
        return {"n_operands": 6, "depth": 3, "max_value": 20}

    def _get_hard_params(self) -> Dict[str, Any]:
        return {"n_operands": 8, "depth": 4, "max_value": 50}

    def generate_task(self, seed: int) -> TaskInstance:
        """Generate arithmetic task with split operands."""
        self.set_seed(seed)
        params = {**self.get_difficulty_params(), **self.params}

        n_operands = params["n_operands"]
        depth = params["depth"]
        max_value = params["max_value"]

        # Generate operands
        operands = {f"x{i}": self._rng.randint(1, max_value)
                   for i in range(1, n_operands + 1)}

        # Generate expression tree
        expression, result = self._generate_expression(
            list(operands.keys()), depth
        )

        # Compute actual result
        result = self._evaluate_expression(expression, operands)

        # Split operands between agents
        var_list = list(operands.keys())
        self._rng.shuffle(var_list)
        mid = len(var_list) // 2
        vars_A = var_list[:mid]
        vars_B = var_list[mid:]

        obs_A = self._format_observation(vars_A, operands, expression, "A")
        obs_B = self._format_observation(vars_B, operands, expression, "B")

        return TaskInstance(
            task_id="",
            task_type=self.task_type,
            seed=seed,
            obs_A=obs_A,
            obs_B=obs_B,
            ground_truth=result,
            metadata={
                "operands": operands,
                "expression": expression,
                "vars_A": vars_A,
                "vars_B": vars_B,
            },
        )

    def _generate_expression(
        self,
        variables: List[str],
        depth: int
    ) -> Tuple[str, int]:
        """Generate a random arithmetic expression."""
        if depth <= 1 or len(variables) <= 2:
            # Base case: simple operation
            if len(variables) >= 2:
                v1, v2 = self._rng.sample(variables, 2)
                op = self._rng.choice(["+", "-", "*"])
                return f"({v1} {op} {v2})", 0
            else:
                return variables[0], 0

        # Recursive case: combine sub-expressions
        mid = len(variables) // 2
        left_vars = variables[:mid]
        right_vars = variables[mid:]

        left_expr, _ = self._generate_expression(left_vars, depth - 1)
        right_expr, _ = self._generate_expression(right_vars, depth - 1)

        op = self._rng.choice(["+", "-", "*"])
        return f"({left_expr} {op} {right_expr})", 0

    def _evaluate_expression(
        self,
        expression: str,
        operands: Dict[str, int]
    ) -> int:
        """Safely evaluate the expression."""
        # Replace variable names with values
        expr = expression
        for var, val in operands.items():
            expr = expr.replace(var, str(val))

        # Safe evaluation (only arithmetic)
        try:
            # Only allow digits, operators, parentheses, spaces
            allowed = set("0123456789+-*() ")
            if all(c in allowed for c in expr):
                return eval(expr)
        except Exception:
            pass
        return 0

    def _format_observation(
        self,
        visible_vars: List[str],
        all_operands: Dict[str, int],
        expression: str,
        agent: str,
    ) -> str:
        """Format arithmetic task for agent."""
        visible = {v: all_operands[v] for v in visible_vars}
        hidden = [v for v in all_operands if v not in visible_vars]

        return f"""ARITHMETIC TASK

Expression to evaluate: {expression}

Your known values:
{chr(10).join(f'  {v} = {val}' for v, val in visible.items())}

Variables you DON'T know: {', '.join(hidden)}
Your partner knows these values.

Share information to compute the final result.
Submit your answer as a single integer."""

    def verify(self, output: Any) -> VerifierResult:
        """Verify the arithmetic result."""
        if self.current_task is None:
            return VerifierResult(success=False, error="No task loaded")

        expected = self.current_task.ground_truth

        # Parse output
        try:
            if isinstance(output, str):
                import re
                numbers = re.findall(r'-?\d+', output)
                if numbers:
                    submitted = int(numbers[-1])  # Take last number
                else:
                    return VerifierResult(success=False, error="No number found")
            else:
                submitted = int(output)
        except Exception as e:
            return VerifierResult(success=False, error=str(e))

        success = submitted == expected

        # Partial credit based on how close
        if success:
            partial = 1.0
        else:
            diff = abs(submitted - expected)
            partial = max(0, 1 - diff / max(abs(expected), 1))

        return VerifierResult(
            success=success,
            partial_credit=partial,
            details={"submitted": submitted, "expected": expected},
        )


class ProgramSynthesisTask(BaseEnvironment):
    """
    S3: Program Synthesis (Toy)

    Agent A receives input-output examples (specification).
    Agent B receives additional test cases and type constraints.
    Together they must produce correct implementation.
    """

    def __init__(
        self,
        difficulty: str = "medium",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("program_synthesis", difficulty, params)

        # Define function templates
        self.templates = [
            {
                "name": "double",
                "signature": "def f(x: int) -> int",
                "impl": "return x * 2",
                "test_gen": lambda x: x * 2,
            },
            {
                "name": "square",
                "signature": "def f(x: int) -> int",
                "impl": "return x * x",
                "test_gen": lambda x: x * x,
            },
            {
                "name": "add_one",
                "signature": "def f(x: int) -> int",
                "impl": "return x + 1",
                "test_gen": lambda x: x + 1,
            },
            {
                "name": "sum_pair",
                "signature": "def f(x: int, y: int) -> int",
                "impl": "return x + y",
                "test_gen": lambda x, y: x + y,
            },
            {
                "name": "max_pair",
                "signature": "def f(x: int, y: int) -> int",
                "impl": "return max(x, y)",
                "test_gen": lambda x, y: max(x, y),
            },
            {
                "name": "abs_diff",
                "signature": "def f(x: int, y: int) -> int",
                "impl": "return abs(x - y)",
                "test_gen": lambda x, y: abs(x - y),
            },
            {
                "name": "is_even",
                "signature": "def f(x: int) -> bool",
                "impl": "return x % 2 == 0",
                "test_gen": lambda x: x % 2 == 0,
            },
            {
                "name": "is_positive",
                "signature": "def f(x: int) -> bool",
                "impl": "return x > 0",
                "test_gen": lambda x: x > 0,
            },
        ]

    def _get_easy_params(self) -> Dict[str, Any]:
        return {"n_examples": 2, "n_tests": 3, "unary_only": True}

    def _get_medium_params(self) -> Dict[str, Any]:
        return {"n_examples": 3, "n_tests": 5, "unary_only": False}

    def _get_hard_params(self) -> Dict[str, Any]:
        return {"n_examples": 4, "n_tests": 7, "unary_only": False}

    def generate_task(self, seed: int) -> TaskInstance:
        """Generate program synthesis task."""
        self.set_seed(seed)
        params = {**self.get_difficulty_params(), **self.params}

        # Filter templates
        if params.get("unary_only"):
            templates = [t for t in self.templates if "y:" not in t["signature"]]
        else:
            templates = self.templates

        template = self._rng.choice(templates)

        # Generate test inputs
        is_binary = "y:" in template["signature"]
        n_examples = params["n_examples"]
        n_tests = params["n_tests"]

        if is_binary:
            all_inputs = [(self._rng.randint(-10, 10), self._rng.randint(-10, 10))
                         for _ in range(n_examples + n_tests)]
            all_outputs = [template["test_gen"](x, y) for x, y in all_inputs]
        else:
            all_inputs = [(self._rng.randint(-10, 10),)
                         for _ in range(n_examples + n_tests)]
            all_outputs = [template["test_gen"](x[0]) for x in all_inputs]

        # Split into examples (A) and tests (B)
        examples = list(zip(all_inputs[:n_examples], all_outputs[:n_examples]))
        tests = list(zip(all_inputs[n_examples:], all_outputs[n_examples:]))

        obs_A = self._format_examples(template, examples)
        obs_B = self._format_tests(template, tests)

        return TaskInstance(
            task_id="",
            task_type=self.task_type,
            seed=seed,
            obs_A=obs_A,
            obs_B=obs_B,
            ground_truth={
                "signature": template["signature"],
                "impl": template["impl"],
                "test_gen": template["test_gen"],
                "all_tests": list(zip(all_inputs, all_outputs)),
            },
            metadata={"template_name": template["name"]},
        )

    def _format_examples(self, template: Dict, examples: List) -> str:
        """Format specification for Agent A."""
        example_strs = []
        for inputs, output in examples:
            if len(inputs) == 1:
                example_strs.append(f"  f({inputs[0]}) = {output}")
            else:
                example_strs.append(f"  f({inputs[0]}, {inputs[1]}) = {output}")

        return f"""PROGRAM SYNTHESIS TASK

Write a function that matches these input-output examples:

{chr(10).join(example_strs)}

Function signature: {template['signature']}

Your partner has additional test cases. Work together to find
the correct implementation.

Submit your answer as Python code: def f(...): return ..."""

    def _format_tests(self, template: Dict, tests: List) -> str:
        """Format test cases for Agent B."""
        test_strs = []
        for inputs, output in tests:
            if len(inputs) == 1:
                test_strs.append(f"  f({inputs[0]}) should return {output}")
            else:
                test_strs.append(f"  f({inputs[0]}, {inputs[1]}) should return {output}")

        return f"""PROGRAM SYNTHESIS TASK (TESTER)

Function signature: {template['signature']}

Additional test cases the solution must pass:

{chr(10).join(test_strs)}

Your partner has the specification examples. Help them find
an implementation that passes all these tests.

Submit your answer as Python code: def f(...): return ..."""

    def verify(self, output: Any) -> VerifierResult:
        """Verify the submitted implementation."""
        if self.current_task is None:
            return VerifierResult(success=False, error="No task loaded")

        ground_truth = self.current_task.ground_truth
        all_tests = ground_truth["all_tests"]
        test_gen = ground_truth["test_gen"]

        # Extract function from output
        if isinstance(output, str):
            func = self._extract_function(output)
            if func is None:
                return VerifierResult(
                    success=False,
                    error="Could not extract function from output"
                )
        else:
            func = output

        # Run tests
        passed = 0
        for inputs, expected in all_tests:
            try:
                if len(inputs) == 1:
                    result = func(inputs[0])
                else:
                    result = func(inputs[0], inputs[1])

                if result == expected:
                    passed += 1
            except Exception:
                pass

        partial = passed / len(all_tests) if all_tests else 0
        success = passed == len(all_tests)

        return VerifierResult(
            success=success,
            partial_credit=partial,
            details={"passed": passed, "total": len(all_tests)},
        )

    def _extract_function(self, code: str) -> Optional[callable]:
        """Extract function from code string."""
        import re

        # Try to find function definition
        match = re.search(r'def\s+f\s*\([^)]*\)\s*(?:->.*?)?:\s*(?:return\s+)?(.+)', code)
        if not match:
            # Try simpler patterns
            match = re.search(r'return\s+(.+)', code)

        if not match:
            return None

        try:
            # Create function in restricted namespace
            namespace = {"abs": abs, "max": max, "min": min}
            exec(code, namespace)
            return namespace.get("f")
        except Exception:
            return None


class SplitSyntheticEnv(CompositeEnvironment):
    """
    Composite environment for all synthetic split-information tasks.
    """

    def __init__(self, difficulty: str = "medium"):
        environments = [
            ConstraintSatisfactionTask(difficulty=difficulty),
            ArithmeticTask(difficulty=difficulty),
            ProgramSynthesisTask(difficulty=difficulty),
        ]
        super().__init__(environments)
        self.difficulty = difficulty

    def get_task_types(self) -> List[str]:
        """Return available task types."""
        return [env.task_type for env in self.environments]

    def reset(self, seed: int) -> TaskInstance:
        """
        Reset current environment with a new task.

        Call select_environment() first to choose task type.
        """
        if self.current_env is None:
            raise RuntimeError("No environment selected. Call select_environment() first.")
        return self.current_env.reset(seed)

