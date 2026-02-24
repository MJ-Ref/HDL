#!/usr/bin/env python3
"""
DEPRECATED: legacy Modal codec script.

This entrypoint is intentionally disabled. Use `modal/train_m2_codec.py`.
"""

import modal

DEPRECATION_MESSAGE = (
    "modal/train_codec.py is deprecated and disabled. "
    "Use modal/train_m2_codec.py (M2-SCALE) instead."
)

app = modal.App("lpca-codec-training-deprecated")
image = modal.Image.debian_slim(python_version="3.11")
volume = modal.Volume.from_name("lpca-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": volume})
def train_codec(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    k_vectors: int = 16,
    hidden_dim: int = 1024,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    data_path: str = "/data/p1_episodes.jsonl",
):
    del model_name, k_vectors, hidden_dim, epochs, batch_size, learning_rate, data_path
    raise RuntimeError(DEPRECATION_MESSAGE)


@app.function(image=image, volumes={"/data": volume})
def collect_p1_data(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    n_episodes: int = 100,
    output_path: str = "/data/p1_episodes.jsonl",
):
    del model_name, n_episodes, output_path
    raise RuntimeError(DEPRECATION_MESSAGE)


@app.local_entrypoint()
def main(
    k: list[int] = [16],
    collect_data: bool = False,
    n_episodes: int = 100,
):
    del k, collect_data, n_episodes
    raise SystemExit(DEPRECATION_MESSAGE)
