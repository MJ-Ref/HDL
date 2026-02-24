"""Tests for analysis script safeguards."""

from pathlib import Path

import pytest

from scripts.analysis.statistical_tests import ensure_nonempty_episodes


def test_ensure_nonempty_episodes_passes_with_data():
    ensure_nonempty_episodes([{"protocol": "P1", "success": True}], Path("results/x"))


def test_ensure_nonempty_episodes_raises_on_empty():
    with pytest.raises(FileNotFoundError):
        ensure_nonempty_episodes([], Path("results/empty"))
