"""Setup script for LPCA package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read() if f else ""

setup(
    name="lpca",
    version="0.1.0",
    author="LPCA Research Team",
    description="Latent-Path Communication for AI Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MJ-Ref/HDL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "accelerate>=0.27.0",
        "datasets>=2.18.0",
        "safetensors>=0.4.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
        "einops>=0.7.0",
        "pydantic>=2.6.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
            "black>=24.0.0",
            "ruff>=0.2.0",
            "mypy>=1.8.0",
        ],
        "viz": [
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
        ],
        "full": [
            "hydra-core>=1.3.0",
            "wandb>=0.16.0",
            "scikit-learn>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lpca-run=scripts.run_experiment:main",
        ],
    },
)
