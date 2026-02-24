"""Tests for full-suite orchestration helpers."""

from scripts.run_full_suite import (
    canonical_hash,
    expand_seed_registry,
    format_command,
    resolve_seed_sets,
    validate_matrix_requirements,
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


def test_resolve_seed_sets_defaults_to_primary() -> None:
    config = {"seed_usage": {"primary_set": "key_comparisons"}}
    expanded = {"key_comparisons": [1000, 1001, 1002]}
    resolved = resolve_seed_sets(config, expanded, [])
    assert resolved == [("key_comparisons", 1000, 3)]


def test_validate_matrix_requirements_passes() -> None:
    config = {
        "matrix_requirements": {"min_models": 3, "min_seed_count": 100},
        "episodes": {
            "e1": 100,
            "e2": 100,
            "e3": 100,
            "e4": 100,
            "m2_eval": 100,
        },
    }
    selected_models = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    selected_seed_sets = [("key_comparisons", 1000, 100)]
    runnable_experiments = {"E1": {}, "E2": {}, "E3": {}, "E4": {}}
    validate_matrix_requirements(
        config=config,
        selected_models=selected_models,
        selected_seed_sets=selected_seed_sets,
        runnable_experiments=runnable_experiments,
    )
