"""Tests for full-suite orchestration helpers."""

from scripts.run_full_suite import (
    canonical_hash,
    expand_seed_registry,
    format_command,
)


def test_canonical_hash_stable_for_dict_order() -> None:
    a = {"x": 1, "y": [3, 2], "z": {"k": "v"}}
    b = {"z": {"k": "v"}, "y": [3, 2], "x": 1}
    assert canonical_hash(a) == canonical_hash(b)


def test_expand_seed_registry_start_count() -> None:
    registry = {
        "seed_sets": {
            "pilot": {"start": 10, "count": 3},
            "fixed": [1, 2, 3],
        }
    }
    expanded = expand_seed_registry(registry)
    assert expanded["pilot"] == [10, 11, 12]
    assert expanded["fixed"] == [1, 2, 3]


def test_format_command_replaces_placeholders() -> None:
    template = [
        "python",
        "scripts/run_llm_experiment.py",
        "--model",
        "{model_id}",
        "--n_episodes",
        "{e1_episodes}",
    ]
    context = {"model_id": "Qwen/Qwen2.5-3B-Instruct", "e1_episodes": 100}
    cmd = format_command(template, context)
    assert cmd[-3:] == ["Qwen/Qwen2.5-3B-Instruct", "--n_episodes", "100"]
