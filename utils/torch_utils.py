"""PyTorch runtime helpers."""

from __future__ import annotations


def device() -> str:
    """Select CUDA when available, otherwise CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"

