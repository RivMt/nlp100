"""Configure third-party model caches below the project res directory."""

from __future__ import annotations

import os
from pathlib import Path

from utils.path_solver import resource_path


def configure_gensim_cache() -> Path:
    cache = resource_path("gensim-data")
    os.environ.setdefault("GENSIM_DATA_DIR", str(cache))
    return cache


def configure_huggingface_cache() -> Path:
    cache = resource_path("huggingface")
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "transformers"))
    return cache

