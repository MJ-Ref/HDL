"""Tests for metrics calculation."""

import pytest
import numpy as np
from lpca.core.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_create(self):
        assert self.calc is not None

    def test_wilson_ci_basic(self):
        # 50% success rate with 100 trials
        lower, upper = self.calc.wilson_ci(50, 100)
        assert 0.4 < lower < 0.5
        assert 0.5 < upper < 0.6

    def test_wilson_ci_all_success(self):
        lower, upper = self.calc.wilson_ci(100, 100)
        assert lower > 0.95
        assert upper == 1.0 or upper > 0.99

    def test_wilson_ci_all_failure(self):
        lower, upper = self.calc.wilson_ci(0, 100)
        assert lower == 0.0 or lower < 0.01
        assert upper < 0.05

    def test_wilson_ci_empty(self):
        lower, upper = self.calc.wilson_ci(0, 0)
        assert lower == 0.0
        assert upper == 0.0  # Returns (0.0, 0.0) for empty case

    def test_success_rate(self):
        outcomes = [True, True, False, True, False]
        rate = self.calc.success_rate(outcomes)
        assert abs(rate - 0.6) < 0.01

    def test_success_rate_empty(self):
        rate = self.calc.success_rate([])
        assert rate == 0.0


class TestPartialCredit:
    """Tests for partial credit calculations."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_partial_credit_basic(self):
        scores = [0.5, 1.0, 0.75, 0.25]
        mean, std = self.calc.partial_credit(scores)
        assert abs(mean - 0.625) < 0.01
        assert std > 0

    def test_partial_credit_empty(self):
        mean, std = self.calc.partial_credit([])
        assert mean == 0.0
        assert std == 0.0


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_bootstrap_ci_mean(self):
        np.random.seed(42)
        data = list(np.random.normal(100, 10, size=100))
        lower, upper = self.calc.bootstrap_ci(data, statistic="mean")
        assert lower < 100 < upper
        assert upper - lower < 10  # CI should be reasonable

    def test_bootstrap_ci_median(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # Has outlier
        lower, upper = self.calc.bootstrap_ci(data, statistic="median")
        assert lower < upper

    def test_bootstrap_ci_empty(self):
        lower, upper = self.calc.bootstrap_ci([])
        assert lower == 0.0
        assert upper == 0.0


class TestEffectSize:
    """Tests for effect size calculations."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_cohens_d_positive(self):
        treatment = [10.0, 11.0, 12.0, 13.0, 14.0]
        control = [5.0, 6.0, 7.0, 8.0, 9.0]
        d = self.calc.cohens_d(treatment, control)
        assert d > 0

    def test_cohens_d_same_data(self):
        same_data = [5.0, 5.0, 5.0, 5.0, 5.0]
        d = self.calc.cohens_d(same_data, same_data)
        assert abs(d) < 0.01

    def test_cohens_d_large_effect(self):
        treatment = [100.0, 101.0, 102.0]
        control = [1.0, 2.0, 3.0]
        d = self.calc.cohens_d(treatment, control)
        # Large effect (>0.8)
        assert d > 0.8


class TestStatisticalTests:
    """Tests for statistical comparison methods."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_independent_ttest(self):
        group1 = [10.0, 11.0, 12.0, 13.0, 14.0]
        group2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        t_stat, p_val = self.calc.independent_ttest(group1, group2)
        assert p_val < 0.05

    def test_bonferroni_correction(self):
        p_values = [0.005, 0.02, 0.03, 0.04, 0.05]
        significant = self.calc.bonferroni_correction(p_values, alpha=0.05)
        # With 5 tests, adjusted alpha = 0.01
        assert significant[0] == True  # 0.005 < 0.01
        assert significant[1] == False  # 0.02 > 0.01


class TestTurnsToSuccess:
    """Tests for turns to success calculations."""

    def setup_method(self):
        self.calc = MetricsCalculator()

    def test_turns_to_success_basic(self):
        turns = [3, 5, 7, 2, 10]
        outcomes = [True, True, True, False, True]
        median, iqr = self.calc.turns_to_success(turns, outcomes)
        assert median is not None
        assert iqr is not None

    def test_turns_to_success_no_successes(self):
        turns = [3, 5, 7]
        outcomes = [False, False, False]
        median, iqr = self.calc.turns_to_success(turns, outcomes)
        assert median is None
        assert iqr is None


class TestCapabilityMetrics:
    """Tests for capability metrics dataclass."""

    def test_capability_metrics_aggregation(self):
        calc = MetricsCalculator()
        outcomes = [True, True, False, True, False]
        partial_scores = [1.0, 0.8, 0.3, 1.0, 0.2]
        turns = [3, 5, 7, 2, 10]

        sr = calc.success_rate(outcomes)
        assert abs(sr - 0.6) < 0.01

        pc_mean, pc_std = calc.partial_credit(partial_scores)
        assert abs(pc_mean - 0.66) < 0.01

        median, iqr = calc.turns_to_success(turns, outcomes)
        assert median is not None
